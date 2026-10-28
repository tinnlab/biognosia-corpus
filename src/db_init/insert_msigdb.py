#!/usr/bin/env python3
"""
Insert MSigDB Gene Sets into Milvus.

This script:
1. Parses MSigDB SQLite database
2. Creates MSigDB_GeneSet entities in Milvus
3. Creates Gene->MSigDB_GeneSet relationships in Milvus
4. Filters by collection and species

Configuration loaded from config/db_init.conf
"""

import argparse
import csv
import hashlib
import json
import multiprocessing
import os
import sys
import time
from datetime import datetime
from multiprocessing import Manager
from pathlib import Path
from queue import Empty

import numpy as np
import torch
from config_loader import PROJECT_ROOT, load_config
from embedding_utils import masked_mean_pooling, normalize_embeddings
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from gene_id_mapper import GeneIDMapper
from msigdb_parser import MSigDBGeneSet, MSigDBParser

multiprocessing.set_start_method("spawn", force=True)


CONFIG_FILE = PROJECT_ROOT / "config" / "db_init.conf"
CONFIG = load_config(str(CONFIG_FILE))
DATA_DIR = CONFIG.get("DATA_DIR", str(PROJECT_ROOT / "data"))

MILVUS_HOST = CONFIG.get("MILVUS_HOST", "localhost")
MILVUS_PORT = CONFIG.get("MILVUS_PORT", "19530")
MILVUS_BATCH_SIZE = int(CONFIG.get("MILVUS_BATCH_SIZE", "5000"))
EMBEDDING_MODEL = CONFIG.get("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = int(CONFIG.get("EMBEDDING_DIM", "1024"))
LABEL_EMBEDDING_MODEL = CONFIG.get("LABEL_EMBEDDING_MODEL", "ncbi/MedCPT-Query-Encoder")
LABEL_EMBEDDING_DIM = int(CONFIG.get("LABEL_EMBEDDING_DIM", "768"))
MAX_LENGTH_CONTENT = int(CONFIG.get("MAX_LENGTH_CONTENT", "2048"))
MAX_LENGTH_LABEL = int(CONFIG.get("MAX_LENGTH_LABEL", "512"))
EMBEDDING_SUB_BATCH_SIZE = int(CONFIG.get("EMBEDDING_SUB_BATCH_SIZE", "64"))
WORKERS = int(CONFIG.get("WORKERS", "12"))
MAX_PENDING_BATCHES = int(CONFIG.get("MAX_PENDING_BATCHES", "10"))
GRAPH_FIELD_SEP = CONFIG.get("GRAPH_FIELD_SEP", "<SEP>")
METADATA_DIR = Path(CONFIG.get("METADATA_DIR", f"{DATA_DIR}/db-init"))
MSIGDB_METADATA_FILE = CONFIG.get("MSIGDB_METADATA_FILE", "msigdb_gene_sets.csv")


def embedding_worker(
    worker_id: int,
    gpu_id: int,
    content_model_name: str,
    label_model_name: str,
    sub_batch_size: int,
    max_length_content: int,
    max_length_label: int,
    input_queue,
    output_queue,
):
    """Worker process for generating dual embeddings on a specific GPU."""
    try:
        device = f"cuda:{gpu_id}"

        content_tokenizer = AutoTokenizer.from_pretrained(content_model_name, local_files_only=True)
        content_model = AutoModel.from_pretrained(content_model_name, local_files_only=True, use_safetensors=True).to(
            device
        )
        content_model.eval()

        label_tokenizer = AutoTokenizer.from_pretrained(label_model_name, local_files_only=True)
        label_model = AutoModel.from_pretrained(label_model_name, local_files_only=True, use_safetensors=True).to(
            device
        )
        label_model.eval()

        print(f"Worker {worker_id} loaded both models on {device}")

        while True:
            try:
                task = input_queue.get(timeout=1)
                if task is None:
                    break

                batch_id, contents, labels = task

                content_embeddings = []
                for i in range(0, len(contents), sub_batch_size):
                    sub_batch = contents[i : i + sub_batch_size]
                    inputs = content_tokenizer(
                        sub_batch, padding=True, truncation=True, max_length=max_length_content, return_tensors="pt"
                    ).to(device)
                    with torch.no_grad():
                        outputs = content_model(**inputs)
                        emb = masked_mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
                        emb = normalize_embeddings(emb)
                        emb = emb.cpu().numpy()
                        content_embeddings.append(emb)

                content_emb = np.vstack(content_embeddings)

                label_embeddings = []
                for i in range(0, len(labels), sub_batch_size):
                    sub_batch = labels[i : i + sub_batch_size]
                    inputs = label_tokenizer(
                        sub_batch, padding=True, truncation=True, max_length=max_length_label, return_tensors="pt"
                    ).to(device)
                    with torch.no_grad():
                        outputs = label_model(**inputs)
                        emb = masked_mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
                        emb = normalize_embeddings(emb)
                        emb = emb.cpu().numpy()
                        label_embeddings.append(emb)

                label_emb = np.vstack(label_embeddings)

                output_queue.put((batch_id, content_emb, label_emb))

            except Empty:
                continue
            except Exception as e:
                print(f"Worker {worker_id} error: {e}")
                output_queue.put((batch_id, None, None))

    except Exception as e:
        print(f"Worker {worker_id} initialization error: {e}")


class ParallelEmbeddingGenerator:
    """Generates dual embeddings in parallel across multiple GPUs."""

    def __init__(
        self,
        workers: int,
        content_model: str,
        label_model: str,
        sub_batch_size: int,
        max_length_content: int = MAX_LENGTH_CONTENT,
        max_length_label: int = MAX_LENGTH_LABEL,
    ):
        self.workers = workers
        self.content_model = content_model
        self.label_model = label_model
        self.sub_batch_size = sub_batch_size
        self.max_length_content = max_length_content
        self.max_length_label = max_length_label

        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")
        gpu_ids = [int(x) for x in cuda_devices.split(",") if x.strip()]

        print(f"Initializing {workers} embedding workers across {len(gpu_ids)} GPUs...")
        print(f"Content model: {content_model} (max_length={max_length_content})")
        print(f"Label model: {label_model} (max_length={max_length_label})")

        manager = Manager()
        self.input_queue = manager.Queue()
        self.output_queue = manager.Queue()

        self.processes = []
        for i in range(workers):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            p = multiprocessing.Process(
                target=embedding_worker,
                args=(
                    i,
                    gpu_id,
                    content_model,
                    label_model,
                    sub_batch_size,
                    max_length_content,
                    max_length_label,
                    self.input_queue,
                    self.output_queue,
                ),
            )
            p.start()
            self.processes.append(p)

    def embed_batch(self, batch_id: int, contents: list[str], labels: list[str]):
        """Submit a batch for embedding (non-blocking)."""
        self.input_queue.put((batch_id, contents, labels))

    def get_result(self, batch_id: int, timeout: float = 0.001) -> tuple | None:
        """Try to get a result (non-blocking with short timeout)."""
        try:
            result = self.output_queue.get(timeout=timeout)
            return result
        except Empty:
            return None

    def close(self):
        """Shutdown all workers."""
        for _ in range(self.workers):
            self.input_queue.put(None)
        for p in self.processes:
            p.join()
        print("All embedding workers closed")


class MilvusInserter:
    """Handles insertion of MSigDB gene sets and relationships into Milvus."""

    def __init__(
        self,
        host: str = MILVUS_HOST,
        port: str = MILVUS_PORT,
        batch_size: int = MILVUS_BATCH_SIZE,
        drop_existing: bool = True,
        skip_gene_sets: bool = False,
        skip_relationships: bool = False,
    ):
        self.host = host
        self.port = port
        self.batch_size = batch_size
        self.drop_existing = drop_existing
        self.skip_gene_sets = skip_gene_sets
        self.skip_relationships = skip_relationships

        connections.connect(host=host, port=port)

        if drop_existing:
            self._drop_msigdb_data()

        self._ensure_collections()

        if not skip_gene_sets:
            self.gene_set_collection = Collection("entities_MSigDB_GeneSet")
        if not skip_relationships:
            self.relationship_collection = Collection("relationships_Gene_MSigDB_GeneSet")

    @staticmethod
    def _compute_id(content: str, prefix: str = "") -> str:
        """Compute a prefixed MD5 hash ID."""
        return prefix + hashlib.md5(content.encode()).hexdigest()

    def _drop_msigdb_data(self):
        """Drop existing MSigDB collections."""
        if not self.skip_gene_sets and utility.has_collection("entities_MSigDB_GeneSet"):
            utility.drop_collection("entities_MSigDB_GeneSet")
            print("Dropped existing entities_MSigDB_GeneSet collection")

        if not self.skip_relationships and utility.has_collection("relationships_Gene_MSigDB_GeneSet"):
            utility.drop_collection("relationships_Gene_MSigDB_GeneSet")
            print("Dropped existing relationships_Gene_MSigDB_GeneSet collection")

    def _ensure_collections(self):
        """Create collections if they don't exist."""
        if not self.skip_gene_sets and not utility.has_collection("entities_MSigDB_GeneSet"):
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
                FieldSchema(name="label_embedding", dtype=DataType.FLOAT_VECTOR, dim=LABEL_EMBEDDING_DIM),
                FieldSchema(name="created_at", dtype=DataType.INT64),
                FieldSchema(name="entity_name", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=32768),
            ]
            schema = CollectionSchema(fields=fields, enable_dynamic_field=True, description="MSigDB Gene Set entities")
            Collection(name="entities_MSigDB_GeneSet", schema=schema)
            print("Created Milvus collection: entities_MSigDB_GeneSet")

        if not self.skip_relationships and not utility.has_collection("relationships_Gene_MSigDB_GeneSet"):
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
                FieldSchema(name="created_at", dtype=DataType.INT64),
                FieldSchema(name="src_id", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="tgt_id", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=32768),
            ]
            schema = CollectionSchema(
                fields=fields, enable_dynamic_field=True, description="Gene to MSigDB Gene Set relationships"
            )
            Collection(name="relationships_Gene_MSigDB_GeneSet", schema=schema)
            print("Created Milvus collection: relationships_Gene_MSigDB_GeneSet")

    def insert_gene_set_batch(
        self,
        gene_sets: list[MSigDBGeneSet],
        content_embeddings: np.ndarray,
        label_embeddings: np.ndarray,
    ):
        """Insert a batch of gene set entities."""
        timestamp = int(time.time() * 1000)

        entities = []
        for i, gene_set in enumerate(gene_sets):
            entity_name = f"MSigDB:{gene_set.standard_name}"
            entity_id = self._compute_id(entity_name, prefix="ent-")

            genes_str = ", ".join(gene_set.genes[:1000]) if gene_set.genes else ""
            content = genes_str

            if gene_set.description_brief:
                description = gene_set.description_brief
            elif gene_set.description_full:
                description = gene_set.description_full[:500]
            else:
                description = gene_set.standard_name

            entity = {
                "id": entity_id,
                "vector": content_embeddings[i].tolist(),
                "label_embedding": label_embeddings[i].tolist(),
                "created_at": timestamp,
                "entity_name": entity_name,
                "file_path": "msigdb_v2025.1",
                "content": content,
                "label": gene_set.standard_name,
                "description": description,
                "entity_type": "MSigDB_GeneSet",
                "entity_id": entity_name,
                "standard_name": gene_set.standard_name,
                "collection_name": gene_set.collection_name,
                "collection_full_name": gene_set.collection_full_name,
                "systematic_name": gene_set.systematic_name,
                "description_brief": gene_set.description_brief,
                "description_full": gene_set.description_full,
                "external_url": gene_set.external_url,
                "source_species": gene_set.source_species,
                "num_genes": gene_set.num_genes,
                "genes": json.dumps(gene_set.genes),
                "publication_pmid": gene_set.publication_pmid,
                "tags": gene_set.tags,
            }
            entities.append(entity)

        self.gene_set_collection.insert(entities)

    def insert_relationship_batch(
        self,
        gene_symbols: list[str],
        gene_set_names: list[str],
        collection_names: list[str],
        content_embeddings: np.ndarray,
    ):
        """Insert a batch of gene-gene_set relationships."""
        timestamp = int(time.time() * 1000)

        relationships = []
        for i, (gene, gene_set_name, collection_name) in enumerate(
            zip(gene_symbols, gene_set_names, collection_names, strict=False)
        ):
            src_id_key = f"GENE:{gene}"
            tgt_id_key = f"MSigDB:{gene_set_name}"
            composite_key = f"{src_id_key}{GRAPH_FIELD_SEP}{tgt_id_key}"
            rel_id = self._compute_id(composite_key, prefix="rel-")

            content = f"{gene} member_of {gene_set_name}"

            relationship = {
                "id": rel_id,
                "vector": content_embeddings[i].tolist(),
                "created_at": timestamp,
                "src_id": f"GENE:{gene}",
                "tgt_id": f"MSigDB:{gene_set_name}",
                "file_path": "msigdb_v2025.1",
                "content": content,
                "description": gene_set_name,
                "keywords": "member_of",
                "weight": 1.0,
                "source_id": "",
                "gene_symbol": gene,
                "gene_set_id": gene_set_name,
                "collection_name": collection_name,
            }
            relationships.append(relationship)

        self.relationship_collection.insert(relationships)

    def finalize_gene_sets(self):
        """Flush, create indexes, and load gene set collection."""
        if self.skip_gene_sets:
            return

        print("\nFinalizing gene sets in Milvus...")
        print("Flushing gene sets to Milvus...")
        self.gene_set_collection.flush()
        print("Gene sets flushed to Milvus")

        print("Creating indexes for gene set vector fields...")
        index_params = {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 200}}
        self.gene_set_collection.create_index(field_name="vector", index_params=index_params)
        print("Created index for gene sets content embeddings")

        self.gene_set_collection.create_index(field_name="label_embedding", index_params=index_params)
        print("Created index for gene sets label embeddings")

        self.gene_set_collection.load()
        print("Gene sets collection loaded and ready")

    def finalize_relationships(self):
        """Flush, create indexes, and load relationship collection."""
        if self.skip_relationships:
            return

        print("\nFinalizing relationships in Milvus...")
        print("Flushing relationships to Milvus...")
        self.relationship_collection.flush()
        print("Relationships flushed to Milvus")

        print("Creating indexes for relationship vector fields...")
        index_params = {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 200}}
        self.relationship_collection.create_index(field_name="vector", index_params=index_params)
        print("Created index for relationships content embeddings")

        self.relationship_collection.load()
        print("Relationships collection loaded and ready")


def process_gene_set_entities(
    gene_sets: dict[str, MSigDBGeneSet], embedder: ParallelEmbeddingGenerator, inserter: MilvusInserter, args
):
    """Process MSigDB gene set entities with three parallel streams."""
    if args.skip_gene_sets:
        return

    gene_set_list = list(gene_sets.values())
    if args.limit > 0:
        gene_set_list = gene_set_list[: args.limit]

    total_gene_sets = len(gene_set_list)
    print(f"Step 4: Processing {total_gene_sets:,} MSigDB gene set entities...")
    print(f"Max pending batches: {args.max_pending_batches}\n")

    pending_batches = {}
    next_batch_id = 0
    batch_idx = 0

    embed_pbar = tqdm(total=total_gene_sets, desc="Embedding gene sets", position=0)
    insert_pbar = tqdm(total=total_gene_sets, desc="Inserting to Milvus", position=1)

    batches = []
    for i in range(0, total_gene_sets, args.batch_size):
        batches.append(gene_set_list[i : i + args.batch_size])

    while batch_idx < len(batches) or pending_batches:
        while batch_idx < len(batches) and len(pending_batches) < args.max_pending_batches:
            batch = batches[batch_idx]
            batch_id = next_batch_id
            next_batch_id += 1
            batch_idx += 1

            contents = []
            labels = []
            for gs in batch:
                genes_str = ", ".join(gs.genes) if gs.genes else ""
                contents.append(genes_str)
                labels.append(gs.standard_name)

            embedder.embed_batch(batch_id, contents, labels)
            pending_batches[batch_id] = batch

        collected_any = False
        for bid in list(pending_batches.keys()):
            result = embedder.get_result(bid, timeout=0.001)
            if result is not None:
                result_bid, content_emb, label_emb = result
                if content_emb is not None:
                    batch_gene_sets = pending_batches.pop(result_bid)

                    inserter.insert_gene_set_batch(batch_gene_sets, content_emb, label_emb)

                    embed_pbar.update(len(batch_gene_sets))
                    insert_pbar.update(len(batch_gene_sets))
                    collected_any = True

        if not collected_any:
            time.sleep(0.01)

    embed_pbar.close()
    insert_pbar.close()

    inserter.finalize_gene_sets()


def process_relationships(
    relationships: list[tuple[str, str, str]],
    embedder: ParallelEmbeddingGenerator,
    inserter: MilvusInserter,
    args,
):
    """Process Gene-MSigDB_GeneSet relationships with three streams."""
    if args.skip_relationships:
        return

    if args.limit > 0:
        relationships = relationships[: args.limit * 100]

    total_relationships = len(relationships)
    print(f"\nStep 5: Processing {total_relationships:,} Gene-MSigDB_GeneSet relationships...")

    pending_batches = {}
    next_batch_id = 0
    batch_idx = 0

    embed_pbar = tqdm(total=total_relationships, desc="Embedding relationships", position=0)
    insert_pbar = tqdm(total=total_relationships, desc="Inserting to Milvus", position=1)

    batches = []
    for i in range(0, total_relationships, args.batch_size):
        batches.append(relationships[i : i + args.batch_size])

    while batch_idx < len(batches) or pending_batches:
        while batch_idx < len(batches) and len(pending_batches) < args.max_pending_batches:
            batch = batches[batch_idx]
            batch_id = next_batch_id
            next_batch_id += 1
            batch_idx += 1

            gene_symbols = [gene for gene, _, _ in batch]
            gene_set_names = [gene_set_name for _, gene_set_name, _ in batch]
            collection_names = [collection_name for _, _, collection_name in batch]

            contents = []
            labels = []
            for gene, gene_set_name, _ in batch:
                content = f"{gene} member_of {gene_set_name}"
                contents.append(content)
                labels.append(f"{gene}->{gene_set_name}")

            embedder.embed_batch(batch_id, contents, labels)
            pending_batches[batch_id] = (gene_symbols, gene_set_names, collection_names)

        collected_any = False
        for bid in list(pending_batches.keys()):
            result = embedder.get_result(bid, timeout=0.001)
            if result is not None:
                result_bid, content_emb, label_emb = result
                if content_emb is not None:
                    gene_symbols, gene_set_names, collection_names = pending_batches.pop(result_bid)

                    inserter.insert_relationship_batch(gene_symbols, gene_set_names, collection_names, content_emb)

                    embed_pbar.update(len(gene_symbols))
                    insert_pbar.update(len(gene_symbols))
                    collected_any = True

        if not collected_any:
            time.sleep(0.01)

    embed_pbar.close()
    insert_pbar.close()

    inserter.finalize_relationships()


def record_metadata(
    args, num_gene_sets: int, num_relationships: int, duration: float, collections: list[str], species: str
):
    """Record insertion metadata to CSV file."""
    metadata_file = METADATA_DIR / MSIGDB_METADATA_FILE

    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    file_exists = metadata_file.exists()

    with open(metadata_file, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(
                [
                    "timestamp",
                    "species",
                    "collections",
                    "num_gene_sets",
                    "num_relationships",
                    "batch_size",
                    "workers",
                    "embedding_model",
                    "label_model",
                    "milvus_batch_size",
                    "duration_seconds",
                ]
            )

        writer.writerow(
            [
                datetime.now().isoformat(),
                species,
                ",".join(collections) if collections else "ALL",
                num_gene_sets,
                num_relationships,
                args.batch_size,
                args.workers,
                EMBEDDING_MODEL,
                LABEL_EMBEDDING_MODEL,
                MILVUS_BATCH_SIZE,
                f"{duration:.2f}",
            ]
        )

    print(f"Recorded insertion metadata to {metadata_file}")


def main():
    parser = argparse.ArgumentParser(description="Insert MSigDB gene sets into Milvus")
    parser.add_argument(
        "--msigdb-db",
        type=str,
        default=CONFIG.get("MSIGDB_DB_HUMAN", f"{DATA_DIR}/resources/msigdb/msigdb_v2025.1.Hs.db"),
        help="Path to MSigDB SQLite database",
    )
    parser.add_argument(
        "--gene-info-file",
        type=str,
        default=CONFIG.get("GENE_INFO_FILE", f"{DATA_DIR}/resources/genes/gene_info.gz"),
        help="Path to gene_info.gz file",
    )
    parser.add_argument(
        "--species",
        type=str,
        default=CONFIG.get("MSIGDB_SPECIES", "HS"),
        help="Species code (HS for human, MM for mouse)",
    )
    parser.add_argument(
        "--collections",
        type=str,
        default=CONFIG.get("MSIGDB_COLLECTIONS", ""),
        help="Comma-separated collection names (empty for all)",
    )
    parser.add_argument(
        "--filter-taxid",
        type=str,
        default=CONFIG.get("GENE_FILTER_TAXID", "9606"),
        help="Filter genes by taxonomy ID (for gene mapping)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(CONFIG.get("MSIGDB_BATCH_SIZE", "1000")),
        help="Processing batch size",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=int(CONFIG.get("MSIGDB_LIMIT", "0")),
        help="Limit number of gene sets (0 = no limit)",
    )
    parser.add_argument("--workers", type=int, default=WORKERS, help="Number of parallel embedding workers")
    parser.add_argument("--max-pending-batches", type=int, default=MAX_PENDING_BATCHES, help="Maximum pending batches")
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        default=True,
        help="Drop existing MSigDB collections (default: True)",
    )
    parser.add_argument(
        "--no-drop-existing", action="store_false", dest="drop_existing", help="Keep existing MSigDB collections"
    )
    parser.add_argument("--skip-gene-sets", action="store_true", default=False, help="Skip gene set entities insertion")
    parser.add_argument("--skip-relationships", action="store_true", default=False, help="Skip relationships insertion")

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("MSigDB Gene Set Insertion")
    print("=" * 60)
    if args.drop_existing and not args.skip_gene_sets:
        print("WARNING: --drop-existing flag set - will delete all existing MSigDB data!")
    print("=" * 60)
    print(f"Configuration file: {CONFIG_FILE}")
    print(f"MSigDB database: {args.msigdb_db}")
    print(f"Species: {args.species}")
    print(f"Collections: {args.collections if args.collections else 'ALL'}")
    print(f"Batch size: {args.batch_size}")
    print(f"Limit: {args.limit}")
    print(f"Workers: {args.workers}")

    start_time = time.time()

    print("\nStep 1: Parsing MSigDB database...")
    with MSigDBParser(args.msigdb_db) as parser:
        collection_list = [c.strip() for c in args.collections.split(",") if c.strip()] if args.collections else None

        gene_sets = parser.get_gene_sets(collections=collection_list, species_code=args.species, limit=args.limit)

        print("\nStep 2: Loading gene members...")
        parser.populate_gene_members(gene_sets)

        print("\nStep 3: Loading valid gene symbols...")
        gene_mapper = GeneIDMapper(
            gene_info_file=args.gene_info_file, filter_taxid=[args.filter_taxid] if args.filter_taxid else []
        )
        gene_mapper.build_mapping()
        valid_genes = set(gene_mapper.mapping.values())
        print(f"Loaded {len(valid_genes):,} valid gene symbols")

        parser.filter_valid_genes(gene_sets, valid_genes)

    print("\nBuilding gene-gene_set relationships...")
    relationships = []
    for gene_set in gene_sets.values():
        for gene in gene_set.genes:
            relationships.append((gene, gene_set.standard_name, gene_set.collection_name))

    print(f"Built {len(relationships):,} gene-gene_set relationships")

    inserter = MilvusInserter(
        host=MILVUS_HOST,
        port=MILVUS_PORT,
        batch_size=MILVUS_BATCH_SIZE,
        drop_existing=args.drop_existing,
        skip_gene_sets=args.skip_gene_sets,
        skip_relationships=args.skip_relationships,
    )

    embedder = ParallelEmbeddingGenerator(
        workers=args.workers,
        content_model=EMBEDDING_MODEL,
        label_model=LABEL_EMBEDDING_MODEL,
        sub_batch_size=EMBEDDING_SUB_BATCH_SIZE,
    )

    process_gene_set_entities(gene_sets, embedder, inserter, args)

    process_relationships(relationships, embedder, inserter, args)

    embedder.close()

    duration = time.time() - start_time
    num_gene_sets = len(gene_sets) if args.limit == 0 else min(len(gene_sets), args.limit)
    num_relationships = len(relationships) if args.limit == 0 else min(len(relationships), args.limit * 100)
    record_metadata(args, num_gene_sets, num_relationships, duration, collection_list or [], args.species)

    print("\n" + "=" * 60)
    print(f"Inserted {num_gene_sets:,} MSigDB gene set entities")
    print(f"Inserted {num_relationships:,} Gene-MSigDB_GeneSet relationships")
    print(f"Duration: {duration:.2f} seconds ({duration / 60:.2f} minutes)")
    print("=" * 60)


if __name__ == "__main__":
    main()
