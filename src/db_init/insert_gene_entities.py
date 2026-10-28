#!/usr/bin/env python3
"""
Insert genes from gene_info.gz into Milvus.

This script:
1. Reads genes from NCBI gene_info.gz file
2. Filters by tax_id and/or type_of_gene
3. Generates embeddings using configured model
4. Inserts into Milvus entities_Genes collection

Configuration is loaded from config/db_init.conf
"""

import argparse
import csv
import gzip
import hashlib
import multiprocessing
import os
import sys
import time
from collections.abc import Generator
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Manager, Pool, Queue
from pathlib import Path
from queue import Empty
from typing import Optional

import numpy as np
import torch
from config_loader import PROJECT_ROOT, load_config
from embedding_utils import masked_mean_pooling, normalize_embeddings
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

multiprocessing.set_start_method("spawn", force=True)


CONFIG_FILE = PROJECT_ROOT / "config" / "db_init.conf"
CONFIG = load_config(str(CONFIG_FILE))
DATA_DIR = CONFIG.get("DATA_DIR", str(PROJECT_ROOT / "data"))

WORKSPACE = CONFIG.get("WORKSPACE", "lightrag")
GRAPH_FIELD_SEP = CONFIG.get("GRAPH_FIELD_SEP", "<SEP>")
EMBEDDING_MODEL = CONFIG.get("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = int(CONFIG.get("EMBEDDING_DIM", "1024"))
LABEL_EMBEDDING_MODEL = CONFIG.get("LABEL_EMBEDDING_MODEL", "ncbi/MedCPT-Query-Encoder")
LABEL_EMBEDDING_DIM = int(CONFIG.get("LABEL_EMBEDDING_DIM", "768"))
MAX_LENGTH_CONTENT = 512
MAX_LENGTH_LABEL = int(CONFIG.get("MAX_LENGTH_LABEL", "512"))
EMBEDDING_SUB_BATCH_SIZE = int(CONFIG.get("EMBEDDING_SUB_BATCH_SIZE", "64"))
WORKERS = int(CONFIG.get("WORKERS", "20"))
CUDA_DEVICES = CONFIG.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")
MAX_PENDING_BATCHES = int(CONFIG.get("MAX_PENDING_BATCHES", "10"))
INSERTION_STREAMS = int(CONFIG.get("INSERTION_STREAMS", "5"))

MILVUS_HOST = CONFIG.get("MILVUS_HOST", "localhost")
MILVUS_PORT = CONFIG.get("MILVUS_PORT", "19530")
MILVUS_BATCH_SIZE = int(CONFIG.get("MILVUS_BATCH_SIZE", "1000"))

METADATA_DIR = CONFIG.get("METADATA_DIR", f"{DATA_DIR}/db-init")
GENE_METADATA_FILE = CONFIG.get("GENE_METADATA_FILE", "gene.csv")

HF_TOKEN = CONFIG.get("HF_TOKEN", "")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN


@dataclass
class GeneRecord:
    """Represents a gene from gene_info."""

    tax_id: str
    gene_id: str
    symbol: str
    locus_tag: str
    synonyms: str
    dbxrefs: str
    chromosome: str
    map_location: str
    description: str
    type_of_gene: str
    symbol_from_authority: str
    full_name_from_authority: str
    nomenclature_status: str
    other_designations: str
    modification_date: str
    feature_type: str

    @classmethod
    def from_line(cls, line: str) -> Optional["GeneRecord"]:
        """Parse a line from gene_info file."""
        fields = line.strip().split("\t")
        if len(fields) != 16:
            return None

        return cls(
            tax_id=fields[0],
            gene_id=fields[1],
            symbol=fields[2].upper(),
            locus_tag=fields[3],
            synonyms=fields[4],
            dbxrefs=fields[5],
            chromosome=fields[6],
            map_location=fields[7],
            description=fields[8],
            type_of_gene=fields[9],
            symbol_from_authority=fields[10],
            full_name_from_authority=fields[11],
            nomenclature_status=fields[12],
            other_designations=fields[13],
            modification_date=fields[14],
            feature_type=fields[15],
        )

    def is_valid(self) -> bool:
        """Check if gene has minimum required data."""
        if self.symbol == "NEWENTRY":
            return False
        if not self.symbol or self.symbol == "-":
            return False
        if not self.gene_id or self.gene_id == "-":
            return False
        if self.symbol.startswith("LOC") and self.symbol[3:].isdigit():
            return False
        if "_" in self.symbol:
            return False
        if self.description and "uncharacterized protein" in self.description.lower():
            return False
        if not self.chromosome or self.chromosome == "-" or self.chromosome.lower() == "unknown":
            return False
        return True

    def to_entity_content(self) -> str:
        """Create content string for embedding."""
        parts = [self.symbol]

        if self.full_name_from_authority and self.full_name_from_authority != "-":
            parts.append(self.full_name_from_authority)

        if self.description and self.description != "-":
            parts.append(self.description)

        metadata = f"Gene ID: {self.gene_id}"
        if self.type_of_gene and self.type_of_gene != "-":
            metadata += f", Type: {self.type_of_gene}"
        parts.append(metadata)

        if self.synonyms and self.synonyms != "-":
            parts.append(f"Synonyms: {self.synonyms}")

        return "\n".join(parts)

    def to_entity_description(self) -> str:
        """Create description for Neo4j/metadata."""
        desc_parts = []

        if self.full_name_from_authority and self.full_name_from_authority != "-":
            desc_parts.append(self.full_name_from_authority)

        if self.description and self.description != "-":
            desc_parts.append(self.description)

        if not desc_parts:
            desc_parts.append(f"Gene {self.symbol}")

        return ". ".join(desc_parts)


class InsertionMetadataTracker:
    """Track gene insertions in a CSV file."""

    def __init__(self, metadata_dir: str = METADATA_DIR, metadata_file: str = GENE_METADATA_FILE):
        self.metadata_dir = Path(metadata_dir)
        self.metadata_path = self.metadata_dir / metadata_file
        self._ensure_metadata_dir()
        self._ensure_metadata_file()

    def _ensure_metadata_dir(self):
        """Create metadata directory if it doesn't exist."""
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_metadata_file(self):
        """Create metadata file with header if it doesn't exist."""
        if not self.metadata_path.exists():
            with open(self.metadata_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "filter_taxid",
                        "filter_gene_type",
                        "num_genes_inserted",
                        "batch_size",
                        "workers",
                        "embedding_model",
                        "milvus_batch_size",
                        "duration_seconds",
                    ]
                )

    def record_insertion(
        self,
        filter_taxid: list[str] | None,
        filter_gene_type: list[str] | None,
        num_genes_inserted: int,
        batch_size: int,
        workers: int,
        embedding_model: str,
        milvus_batch_size: int,
        duration_seconds: float,
    ):
        """Record an insertion run to the metadata file."""
        timestamp = datetime.now().isoformat()
        taxid_str = ",".join(filter_taxid) if filter_taxid else "ALL"
        gene_type_str = ",".join(filter_gene_type) if filter_gene_type else "ALL"

        with open(self.metadata_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    timestamp,
                    taxid_str,
                    gene_type_str,
                    num_genes_inserted,
                    batch_size,
                    workers,
                    embedding_model,
                    milvus_batch_size,
                    f"{duration_seconds:.2f}",
                ]
            )

        print(f"Recorded insertion metadata to {self.metadata_path}")


class GeneReader:
    """Read and filter genes from gene_info.gz."""

    def __init__(
        self, file_path: str, filter_taxid: list[str] | None = None, filter_gene_type: list[str] | None = None
    ):
        self.file_path = file_path
        self.filter_taxid = set(filter_taxid) if filter_taxid else None
        self.filter_gene_type = set(filter_gene_type) if filter_gene_type else None

    def read_genes(self, limit: int | None = None) -> Generator[GeneRecord, None, None]:
        """Read genes from file with optional filtering."""
        count = 0
        with gzip.open(self.file_path, "rt") as f:
            next(f)

            for line in f:
                gene = GeneRecord.from_line(line)
                if not gene or not gene.is_valid():
                    continue

                if self.filter_taxid and gene.tax_id not in self.filter_taxid:
                    continue

                if self.filter_gene_type and gene.type_of_gene not in self.filter_gene_type:
                    continue

                yield gene
                count += 1

                if limit and count >= limit:
                    break

    def count_genes(self) -> int:
        """Count total genes that match filters."""
        return sum(1 for _ in self.read_genes())


def _embedding_worker(
    worker_id: int,
    device_id: int,
    content_model_name: str,
    label_model_name: str,
    sub_batch_size: int,
    max_length_content: int,
    max_length_label: int,
    input_queue: Queue,
    output_queue: Queue,
):
    """Worker process for generating both content and label embeddings on a specific GPU."""
    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"

    try:
        content_tokenizer = AutoTokenizer.from_pretrained(content_model_name, local_files_only=True)
        content_model = AutoModel.from_pretrained(content_model_name, local_files_only=True, use_safetensors=True)
        content_model.to(device)
        content_model.eval()
    except Exception as e:
        print(f"Worker {worker_id} failed to load content model: {e}")
        output_queue.put(("WORKER_INIT_ERROR", str(e)))
        return

    try:
        label_tokenizer = AutoTokenizer.from_pretrained(label_model_name, local_files_only=True)
        label_model = AutoModel.from_pretrained(label_model_name, local_files_only=True, use_safetensors=True)
        label_model.to(device)
        label_model.eval()
        print(f"Worker {worker_id} loaded both models on {device}")
    except Exception as e:
        print(f"Worker {worker_id} failed to load label model: {e}")
        output_queue.put(("WORKER_INIT_ERROR", str(e)))
        return

    while True:
        try:
            item = input_queue.get(timeout=1)
            if item is None:
                break

            batch_id, contents, labels = item

            all_content_embeddings = []
            for i in range(0, len(contents), sub_batch_size):
                sub_texts = contents[i : i + sub_batch_size]

                inputs = content_tokenizer(
                    sub_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length_content
                )

                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = content_model(**inputs)
                    embeddings = masked_mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
                    embeddings = normalize_embeddings(embeddings)
                    embeddings = embeddings.cpu().numpy()
                    all_content_embeddings.append(embeddings)

            all_label_embeddings = []
            for i in range(0, len(labels), sub_batch_size):
                sub_labels = labels[i : i + sub_batch_size]

                inputs = label_tokenizer(
                    sub_labels, return_tensors="pt", padding=True, truncation=True, max_length=max_length_label
                )

                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)

                with torch.no_grad():
                    outputs = label_model(input_ids=input_ids, attention_mask=attention_mask)
                    embeddings = masked_mean_pooling(outputs.last_hidden_state, attention_mask)
                    embeddings = normalize_embeddings(embeddings)
                    embeddings = embeddings.cpu().numpy()
                    all_label_embeddings.append(embeddings)

            final_content_embeddings = (
                np.vstack(all_content_embeddings) if len(all_content_embeddings) > 1 else all_content_embeddings[0]
            )
            final_label_embeddings = (
                np.vstack(all_label_embeddings) if len(all_label_embeddings) > 1 else all_label_embeddings[0]
            )

            output_queue.put((batch_id, final_content_embeddings, final_label_embeddings))

        except Empty:
            continue
        except Exception as e:
            print(f"Worker {worker_id} error: {e}")
            output_queue.put((batch_id, None, None))


class ParallelEmbeddingGenerator:
    """Generate embeddings in parallel across multiple GPUs."""

    def __init__(
        self,
        num_workers: int = WORKERS,
        cuda_devices: str = CUDA_DEVICES,
        content_model_name: str = EMBEDDING_MODEL,
        label_model_name: str = LABEL_EMBEDDING_MODEL,
        sub_batch_size: int = EMBEDDING_SUB_BATCH_SIZE,
        max_length_content: int = MAX_LENGTH_CONTENT,
        max_length_label: int = MAX_LENGTH_LABEL,
    ):
        self.num_workers = num_workers
        self.content_model_name = content_model_name
        self.label_model_name = label_model_name
        self.sub_batch_size = sub_batch_size
        self.max_length_content = max_length_content
        self.max_length_label = max_length_label

        if cuda_devices and torch.cuda.is_available():
            self.devices = [int(d) for d in cuda_devices.split(",")]
        else:
            self.devices = [0]

        self.manager = Manager()
        self.input_queue = self.manager.Queue()
        self.output_queue = self.manager.Queue()
        self.workers = []

        print(f"Initializing {num_workers} embedding workers across {len(self.devices)} GPUs...")
        print(f"Content model: {content_model_name} (max_length={max_length_content})")
        print(f"Label model: {label_model_name} (max_length={max_length_label})")
        self._start_workers()

    def _start_workers(self):
        """Start worker processes."""
        for worker_id in range(self.num_workers):
            device_id = self.devices[worker_id % len(self.devices)]

            p = Pool(
                processes=1,
                initializer=_embedding_worker,
                initargs=(
                    worker_id,
                    device_id,
                    self.content_model_name,
                    self.label_model_name,
                    self.sub_batch_size,
                    self.max_length_content,
                    self.max_length_label,
                    self.input_queue,
                    self.output_queue,
                ),
            )
            self.workers.append(p)

    def embed_batch(self, batch_id: int, contents: list[str], labels: list[str]):
        """Submit a batch for dual embedding (content + label)."""
        self.input_queue.put((batch_id, contents, labels))

    def get_result(self, block: bool = True, timeout: float = None) -> tuple[int, np.ndarray, np.ndarray] | None:
        """Get completed embedding result (content embeddings, label embeddings).

        Args:
            block: If True, block until result available. If False, raise Empty if no result.
            timeout: Timeout in seconds (only used if block=True)

        Returns:
            Tuple of (batch_id, content_embeddings, label_embeddings) or None if non-blocking and empty
        """
        try:
            if timeout is not None:
                return self.output_queue.get(block=block, timeout=timeout)
            else:
                return self.output_queue.get(block=block)
        except Empty:
            return None

    def close(self):
        """Shutdown all workers."""
        for _ in range(self.num_workers):
            self.input_queue.put(None)

        for p in self.workers:
            p.close()
            p.join()

        print("All embedding workers closed")


class MilvusInserter:
    """Insert gene entities into Milvus with batching support and multiple insertion streams."""

    def __init__(
        self,
        host: str = MILVUS_HOST,
        port: str = MILVUS_PORT,
        batch_size: int = MILVUS_BATCH_SIZE,
        num_streams: int = INSERTION_STREAMS,
        drop_existing: bool = False,
    ):
        connections.connect(host=host, port=port)
        self.collection_name = "entities_Genes"
        self.batch_size = batch_size
        self.num_streams = num_streams
        self.stream_batch_size = batch_size // num_streams

        self.pending_streams = [[] for _ in range(num_streams)]
        self.current_stream = 0

        self.total_inserted = 0
        self.total_insert_time = 0.0

        self._drop_collection()

        self._ensure_collection()

    def _drop_collection(self):
        """Drop existing collection if it exists."""
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            print(f"Dropped existing Milvus collection: {self.collection_name}")

    def _ensure_collection(self):
        """Create entities collection if it doesn't exist."""
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
            FieldSchema(name="label_embedding", dtype=DataType.FLOAT_VECTOR, dim=LABEL_EMBEDDING_DIM),
            FieldSchema(name="created_at", dtype=DataType.INT64),
            FieldSchema(name="entity_name", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=32768),
        ]
        schema = CollectionSchema(fields, enable_dynamic_field=True)
        self.collection = Collection(self.collection_name, schema)
        print(
            f"Created Milvus collection: {self.collection_name} with dual embeddings (indexes will be created after insertion)"
        )

    def insert_batch(self, genes: list[GeneRecord], content_embeddings: np.ndarray, label_embeddings: np.ndarray):
        """Add genes to pending batch using round-robin across streams."""
        current_time = int(time.time())

        for gene, content_emb, label_emb in zip(genes, content_embeddings, label_embeddings, strict=False):
            entity_type = "Gene"
            entity_name = f"GENE:{gene.symbol}"
            pk_id = self._compute_id(entity_name, prefix="ent-")

            if gene.full_name_from_authority and gene.full_name_from_authority != "-":
                label = f"{gene.symbol}: {gene.full_name_from_authority}"
            elif gene.description and gene.description != "-":
                label = f"{gene.symbol}: {gene.description}"
            else:
                label = gene.symbol

            record = {
                "id": pk_id,
                "vector": content_emb.tolist(),
                "label_embedding": label_emb.tolist(),
                "created_at": current_time,
                "entity_name": entity_name,
                "file_path": "genes/gene_info.gz",
                "content": label,
                "label": label,
                "source_id": "",
                "description": gene.to_entity_description(),
                "entity_type": entity_type,
                "entity_id": gene.symbol,
                "gene_id": gene.gene_id,
                "tax_id": gene.tax_id,
                "synonyms": gene.synonyms if gene.synonyms != "-" else "",
                "chromosome": gene.chromosome if gene.chromosome != "-" else "",
                "gene_type": gene.type_of_gene if gene.type_of_gene != "-" else "",
                "dbxrefs": gene.dbxrefs if gene.dbxrefs != "-" else "",
            }

            stream_idx = self.current_stream
            self.pending_streams[stream_idx].append(record)
            self.current_stream = (self.current_stream + 1) % self.num_streams

            if len(self.pending_streams[stream_idx]) >= self.stream_batch_size:
                self._flush_stream(stream_idx)

    def _flush_stream(self, stream_idx: int):
        """Insert pending data from a specific stream to Milvus."""
        if not self.pending_streams[stream_idx]:
            return

        flush_start = time.time()

        self.collection.insert(self.pending_streams[stream_idx])

        flush_duration = time.time() - flush_start
        batch_count = len(self.pending_streams[stream_idx])
        self.total_inserted += batch_count
        self.total_insert_time += flush_duration

        self.pending_streams[stream_idx] = []

    def _flush_all_streams(self):
        """Flush all pending streams."""
        for stream_idx in range(self.num_streams):
            self._flush_stream(stream_idx)

    def finalize(self):
        """Flush pending data, create indexes, and load collection for searching."""
        self._flush_all_streams()

        print("Flushing all data to Milvus...")
        self.collection.flush()
        print("Data flushed to Milvus")

        print("Creating indexes for vector fields...")
        index_params = {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 256}}
        self.collection.create_index("vector", index_params)
        print("Created index for content embeddings (vector)")

        self.collection.create_index("label_embedding", index_params)
        print("Created index for label embeddings (label_embedding)")

        self.collection.load()
        print("Milvus collection loaded and ready")

    def close(self):
        """Flush pending data and disconnect from Milvus."""
        self._flush_all_streams()
        connections.disconnect("default")

    @staticmethod
    def _compute_id(content: str, prefix: str = "") -> str:
        """Compute a prefixed MD5 hash ID."""
        return prefix + hashlib.md5(content.encode()).hexdigest()


def process_genes(args):
    """Main processing function with parallel embedding generation."""
    start_time = time.time()

    print("=" * 60)
    print("Gene Entities Insertion")
    print("=" * 60)
    if args.drop_existing:
        print("WARNING: --drop-existing flag set - will delete all existing Gene data!")
        print("=" * 60)
    print(f"Configuration file: {CONFIG_FILE}")
    print(f"Input file: {args.input_file}")
    print(f"Batch size: {args.batch_size}")
    if args.filter_taxid:
        print(f"Filter tax_id: {args.filter_taxid}")
    if args.filter_gene_type:
        print(f"Filter gene type: {args.filter_gene_type}")
    if args.limit:
        print(f"Limit: {args.limit}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Embedding dimension: {EMBEDDING_DIM}")
    print(f"Workers: {WORKERS}")
    print(f"CUDA devices: {CUDA_DEVICES}")
    print(f"Sub-batch size: {EMBEDDING_SUB_BATCH_SIZE}")
    print()

    print("Step 1: Loading and filtering genes...")
    reader = GeneReader(args.input_file, filter_taxid=args.filter_taxid, filter_gene_type=args.filter_gene_type)

    all_genes = []
    for gene in tqdm(reader.read_genes(limit=args.limit), desc="Reading genes"):
        all_genes.append(gene)

    print(f"Loaded {len(all_genes):,} genes into memory")
    print()

    print("Step 1.5: Deduplicating genes by symbol...")
    gene_dict = {}
    duplicates_removed = 0

    for gene in all_genes:
        if gene.symbol not in gene_dict:
            gene_dict[gene.symbol] = gene
        else:
            duplicates_removed += 1

    all_genes = list(gene_dict.values())
    print(f"Removed {duplicates_removed:,} duplicate genes (same symbol after uppercase)")
    print(f"Final gene count: {len(all_genes):,} unique genes")
    print()

    if len(all_genes) == 0:
        print("No genes to process. Exiting.")
        return

    print("Step 2: Initializing Milvus and embedding workers...")
    embedder = ParallelEmbeddingGenerator()
    milvus = MilvusInserter(drop_existing=args.drop_existing)
    metadata_tracker = InsertionMetadataTracker()
    print(
        f"Milvus insertion streams: {INSERTION_STREAMS} (batch size per stream: {MILVUS_BATCH_SIZE // INSERTION_STREAMS})"
    )
    print()

    print(f"Step 3: Processing {len(all_genes):,} genes with independent parallel streams...")
    print(f"Max pending batches: {MAX_PENDING_BATCHES} (buffer size: {MAX_PENDING_BATCHES * args.batch_size:,} genes)")
    print()

    batch_id = 0
    next_batch_idx = 0
    total_batches = (len(all_genes) + args.batch_size - 1) // args.batch_size

    pending_batches = {}
    insertion_queue = []
    submitted_count = 0
    embedded_count = 0
    inserted_count = 0

    try:
        pbar_embed = tqdm(total=len(all_genes), desc="Embedding", position=0, leave=True)
        pbar_insert = tqdm(total=len(all_genes), desc="Inserting to Milvus", position=1, leave=True)

        while submitted_count < total_batches or pending_batches or insertion_queue:
            while submitted_count < total_batches and len(pending_batches) + len(insertion_queue) < MAX_PENDING_BATCHES:
                start_idx = next_batch_idx * args.batch_size
                end_idx = min(start_idx + args.batch_size, len(all_genes))
                batch_genes = all_genes[start_idx:end_idx]

                labels = []
                for g in batch_genes:
                    if g.full_name_from_authority and g.full_name_from_authority != "-":
                        label = f"{g.symbol}: {g.full_name_from_authority}"
                    elif g.description and g.description != "-":
                        label = f"{g.symbol}: {g.description}"
                    else:
                        label = g.symbol
                    labels.append(label)

                contents = labels

                embedder.embed_batch(batch_id, contents, labels)
                pending_batches[batch_id] = (batch_genes, contents)

                submitted_count += 1
                next_batch_idx += 1
                batch_id += 1

            if pending_batches:
                result = embedder.get_result(block=False)
                if result is not None:
                    result_batch_id, content_embeddings, label_embeddings = result

                    if result_batch_id == "WORKER_INIT_ERROR":
                        pbar_embed.close()
                        pbar_insert.close()
                        print(f"\nWorker initialization failed: {content_embeddings}")
                        print("Aborting insertion...")
                        raise RuntimeError(f"Worker failed to initialize: {content_embeddings}")

                    if (
                        content_embeddings is not None
                        and label_embeddings is not None
                        and result_batch_id in pending_batches
                    ):
                        genes, contents = pending_batches.pop(result_batch_id)

                        embedded_count += len(genes)
                        pbar_embed.update(len(genes))

                        insertion_queue.append((genes, content_embeddings, label_embeddings))

            if insertion_queue:
                genes, content_embeddings, label_embeddings = insertion_queue.pop(0)

                milvus.insert_batch(genes, content_embeddings, label_embeddings)

                inserted_count += len(genes)
                pbar_insert.update(len(genes))

        pbar_embed.close()
        pbar_insert.close()

        print()
        print("Finalizing Milvus...")
        milvus.finalize()

        end_time = time.time()
        duration = end_time - start_time

        print()
        print("=" * 60)
        print(f"Inserted {inserted_count:,} genes")
        print(f"Submitted {submitted_count:,} batches")
        print(f"Embedded {embedded_count:,} genes")
        print(f"Duration: {duration:.2f} seconds ({duration / 60:.2f} minutes)")
        print(f"Throughput: {inserted_count / duration:.1f} genes/second")
        print("=" * 60)

        metadata_tracker.record_insertion(
            filter_taxid=args.filter_taxid,
            filter_gene_type=args.filter_gene_type,
            num_genes_inserted=inserted_count,
            batch_size=args.batch_size,
            workers=WORKERS,
            embedding_model=EMBEDDING_MODEL,
            milvus_batch_size=MILVUS_BATCH_SIZE,
            duration_seconds=duration,
        )

    finally:
        embedder.close()
        milvus.close()


def main():
    parser = argparse.ArgumentParser(
        description="Insert genes into Milvus",
        epilog="Configuration is loaded from config/db_init.conf. Command-line arguments override config file values.",
    )
    parser.add_argument(
        "--input-file",
        default=CONFIG.get("GENE_INFO_FILE", f"{DATA_DIR}/resources/genes/gene_info.gz"),
        help="Path to gene_info.gz file (default from config)",
    )
    parser.add_argument(
        "--filter-taxid",
        nargs="+",
        default=None,
        help="Filter by taxonomy ID(s) (e.g., 9606 for human). Overrides config.",
    )
    parser.add_argument(
        "--filter-gene-type",
        nargs="+",
        default=None,
        help="Filter by gene type(s) (e.g., protein-coding). Overrides config.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for processing (default from config)")
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of genes to process (default from config)"
    )
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop all existing Gene data from Milvus and Neo4j before insertion",
    )

    args = parser.parse_args()

    if args.filter_taxid is None:
        taxid_str = CONFIG.get("GENE_FILTER_TAXID", "")
        if taxid_str:
            args.filter_taxid = [t.strip() for t in taxid_str.split(",")]

    if args.filter_gene_type is None:
        type_str = CONFIG.get("GENE_FILTER_TYPE", "")
        if type_str:
            args.filter_gene_type = [t.strip() for t in type_str.split(",")]

    if args.batch_size is None:
        args.batch_size = int(CONFIG.get("GENE_BATCH_SIZE", "1000"))

    if args.limit is None:
        limit_val = int(CONFIG.get("GENE_LIMIT", "0"))
        if limit_val > 0:
            args.limit = limit_val

    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)

    process_genes(args)


if __name__ == "__main__":
    main()
