#!/usr/bin/env python3
"""
Build the entities_Gene_Synonyms collection in Milvus.

Each row holds one gene symbol, synonym or full name together with its
MedCPT-Query-Encoder embedding. Rows point to the gene entity through
entity_name (GENE:<SYMBOL>), so a match on any of these strings resolves to the
entity stored in entities_Genes.

Sources:
1. entities_Genes: the symbol and the synonyms field of every gene entity
2. gene_info.gz: synonyms and the full name of human (tax_id 9606) protein-coding genes

Configuration is loaded from config/db_init.conf
"""

import argparse
import gzip
import hashlib
import re
import sys
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
from config_loader import PROJECT_ROOT, load_config
from embedding_utils import compute_embeddings

CONFIG = load_config(str(PROJECT_ROOT / "config" / "db_init.conf"))
DATA_DIR = CONFIG.get("DATA_DIR", str(PROJECT_ROOT / "data"))

COLLECTION_NAME = "entities_Gene_Synonyms"
GENE_COLLECTION_NAME = "entities_Genes"
LABEL_EMBEDDING_MODEL = CONFIG.get("LABEL_EMBEDDING_MODEL", "ncbi/MedCPT-Query-Encoder")
LABEL_EMBEDDING_DIM = int(CONFIG.get("LABEL_EMBEDDING_DIM", "768"))
MAX_LENGTH_LABEL = int(CONFIG.get("MAX_LENGTH_LABEL", "512"))
EMBEDDING_SUB_BATCH_SIZE = int(CONFIG.get("EMBEDDING_SUB_BATCH_SIZE", "64"))
MILVUS_HOST = CONFIG.get("MILVUS_HOST", "localhost")
MILVUS_PORT = CONFIG.get("MILVUS_PORT", "19530")
MILVUS_BATCH_SIZE = int(CONFIG.get("MILVUS_BATCH_SIZE", "5000"))
GENE_INFO_FILE = CONFIG.get("GENE_INFO_FILE", f"{DATA_DIR}/resources/genes/gene_info.gz")

HUMAN_TAX_ID = "9606"
SYNONYM_SPLIT = re.compile(r"[|;,]")
EMBED_CHUNK_ROWS = 50_000


def gene_key(symbol: str) -> str:
    return "GENE:" + symbol.upper()


def row_id(entity_name: str, text: str) -> str:
    return hashlib.md5(f"{entity_name}\t{text}".encode()).hexdigest()


def rows_from_gene_entities(gene_entities: Iterable[dict]) -> tuple[list[tuple], set[str]]:
    """Collect symbol and synonym rows from the entities_Genes records."""
    rows = []
    keys = set()
    for record in gene_entities:
        entity_name = record["entity_name"]
        keys.add(entity_name)
        rows.append((entity_name, entity_name.split(":", 1)[1], "symbol", "entity"))
        synonyms = record.get("synonyms") or ""
        parts = synonyms if isinstance(synonyms, list) else SYNONYM_SPLIT.split(synonyms)
        for synonym in parts:
            if synonym.strip():
                rows.append((entity_name, synonym.strip(), "synonym", "entity"))
    return rows, keys


def rows_from_gene_info(lines: Iterable[str], keys: set[str]) -> tuple[list[tuple], int]:
    """Collect synonym and full-name rows for human protein-coding genes in gene_info."""
    rows = []
    matched = 0
    for line in lines:
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 10 or fields[0] != HUMAN_TAX_ID or fields[9] != "protein-coding":
            continue
        entity_name = gene_key(fields[2])
        if entity_name not in keys:
            continue
        matched += 1
        if fields[4] != "-":
            for synonym in fields[4].split("|"):
                rows.append((entity_name, synonym.strip(), "synonym", "ncbi_human"))
        if fields[8] and fields[8] != "-":
            rows.append((entity_name, fields[8].strip(), "full_name", "ncbi_human"))
    return rows, matched


def deduplicate(rows: list[tuple]) -> list[dict]:
    """Drop empty texts and repeated (entity_name, text) pairs, and assign row ids."""
    seen = set()
    records = []
    for entity_name, text, text_type, source in rows:
        if not text or (entity_name, text) in seen:
            continue
        seen.add((entity_name, text))
        records.append(
            {
                "id": row_id(entity_name, text),
                "entity_name": entity_name,
                "label": text,
                "text_type": text_type,
                "source": source,
            }
        )
    return records


def read_gene_info(path: str) -> Iterator[str]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        next(handle)
        yield from handle


def build_records(gene_entities: Iterable[dict], gene_info_lines: Iterable[str]) -> list[dict]:
    entity_rows, keys = rows_from_gene_entities(gene_entities)
    info_rows, _ = rows_from_gene_info(gene_info_lines, keys)
    return deduplicate(entity_rows + info_rows)


def embed_texts(model, tokenizer, texts: list[str], device: str) -> np.ndarray:
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    vectors = np.zeros((len(texts), LABEL_EMBEDDING_DIM), dtype=np.float32)
    batch = 256
    for start in range(0, len(order), batch):
        idx = order[start : start + batch]
        vectors[idx] = compute_embeddings(
            model,
            tokenizer,
            [texts[i] for i in idx],
            device,
            batch_size=EMBEDDING_SUB_BATCH_SIZE,
            max_length=MAX_LENGTH_LABEL,
        )
    return vectors


def create_collection(client, drop_existing: bool) -> None:
    from pymilvus import DataType

    if client.has_collection(COLLECTION_NAME):
        if not drop_existing:
            raise SystemExit(f"{COLLECTION_NAME} already exists. Use --drop-existing to replace it.")
        client.drop_collection(COLLECTION_NAME)
        print(f"Dropped existing collection: {COLLECTION_NAME}")

    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("entity_name", DataType.VARCHAR, max_length=512)
    schema.add_field("label", DataType.VARCHAR, max_length=2048)
    schema.add_field("label_embedding", DataType.FLOAT_VECTOR, dim=LABEL_EMBEDDING_DIM)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="label_embedding",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 256},
    )
    index_params.add_index(field_name="entity_name", index_type="TRIE")
    client.create_collection(COLLECTION_NAME, schema=schema, index_params=index_params)
    print(f"Created collection: {COLLECTION_NAME}")


def read_gene_entities(client) -> Iterator[dict]:
    iterator = client.query_iterator(
        GENE_COLLECTION_NAME,
        filter="",
        output_fields=["entity_name", "synonyms"],
        batch_size=5000,
    )
    try:
        while True:
            batch = iterator.next()
            if not batch:
                break
            yield from batch
    finally:
        iterator.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the entities_Gene_Synonyms Milvus collection")
    parser.add_argument("--gene-info-file", default=GENE_INFO_FILE, help="Path to gene_info.gz")
    parser.add_argument("--milvus-host", default=MILVUS_HOST)
    parser.add_argument("--milvus-port", default=MILVUS_PORT)
    parser.add_argument("--device", default=None, help="Embedding device (default: cuda:0 if available, else cpu)")
    parser.add_argument("--batch-size", type=int, default=MILVUS_BATCH_SIZE, help="Rows per Milvus insert")
    parser.add_argument("--drop-existing", action="store_true", help="Drop and recreate the collection")
    args = parser.parse_args()

    if not Path(args.gene_info_file).exists():
        sys.exit(f"gene_info file not found: {args.gene_info_file}")

    import torch
    from pymilvus import MilvusClient
    from transformers import AutoModel, AutoTokenizer

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    client = MilvusClient(uri=f"http://{args.milvus_host}:{args.milvus_port}", timeout=300)

    print(f"Reading {GENE_COLLECTION_NAME}...")
    records = build_records(read_gene_entities(client), read_gene_info(args.gene_info_file))
    print(f"{len(records)} rows to embed")

    print(f"Loading {LABEL_EMBEDDING_MODEL} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(LABEL_EMBEDDING_MODEL)
    model = AutoModel.from_pretrained(LABEL_EMBEDDING_MODEL).to(device).eval()

    create_collection(client, args.drop_existing)

    start = time.time()
    inserted = 0
    for offset in range(0, len(records), EMBED_CHUNK_ROWS):
        chunk = records[offset : offset + EMBED_CHUNK_ROWS]
        vectors = embed_texts(model, tokenizer, [r["label"] for r in chunk], device)
        for first in range(0, len(chunk), args.batch_size):
            part = chunk[first : first + args.batch_size]
            rows = [
                {**record, "content": record["label"], "label_embedding": vector.tolist()}
                for record, vector in zip(part, vectors[first : first + args.batch_size], strict=True)
            ]
            client.insert(COLLECTION_NAME, rows)
            inserted += len(rows)
        print(f"Inserted {inserted}/{len(records)} rows ({time.time() - start:.0f}s)")

    client.flush(COLLECTION_NAME)
    client.load_collection(COLLECTION_NAME)
    print(f"Done: {inserted} rows in {COLLECTION_NAME}")


if __name__ == "__main__":
    main()
