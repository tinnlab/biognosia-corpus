#!/usr/bin/env python3
"""
Insert Gene Ontology annotations from gene2go.gz into Milvus.

This script:
1. Builds an Entrez ID -> Gene Symbol mapping from gene_info.gz
2. Reads GO annotations from gene2go.gz
3. Creates Gene_Ontology entities in Milvus
4. Creates relationships from Gene to Gene_Ontology in Milvus
5. Filters by tax_id and GO category

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
from collections import defaultdict
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
from gene_id_mapper import GeneIDMapper
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
MAX_LENGTH_CONTENT = int(CONFIG.get("MAX_LENGTH_CONTENT", "2048"))
MAX_LENGTH_LABEL = int(CONFIG.get("MAX_LENGTH_LABEL", "512"))
EMBEDDING_SUB_BATCH_SIZE = int(CONFIG.get("EMBEDDING_SUB_BATCH_SIZE", "64"))
WORKERS = int(CONFIG.get("WORKERS", "4"))
CUDA_DEVICES = CONFIG.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")
MAX_PENDING_BATCHES = int(CONFIG.get("MAX_PENDING_BATCHES", "10"))

MILVUS_HOST = CONFIG.get("MILVUS_HOST", "localhost")
MILVUS_PORT = CONFIG.get("MILVUS_PORT", "19530")
MILVUS_BATCH_SIZE = int(CONFIG.get("MILVUS_BATCH_SIZE", "10000"))

METADATA_DIR = CONFIG.get("METADATA_DIR", f"{DATA_DIR}/db-init")
GENE2GO_METADATA_FILE = CONFIG.get("GENE2GO_METADATA_FILE", "gene2go.csv")

HF_TOKEN = CONFIG.get("HF_TOKEN", "")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN


@dataclass
class Gene2GORecord:
    """Represents a Gene-to-GO annotation."""

    tax_id: str
    gene_id: str
    go_id: str
    evidence: str
    qualifier: str
    go_term: str
    pubmed: str
    category: str

    @classmethod
    def from_line(cls, line: str) -> Optional["Gene2GORecord"]:
        """Parse a line from gene2go file."""
        fields = line.strip().split("\t")
        if len(fields) != 8:
            return None

        return cls(
            tax_id=fields[0],
            gene_id=fields[1],
            go_id=fields[2],
            evidence=fields[3],
            qualifier=fields[4],
            go_term=fields[5],
            pubmed=fields[6],
            category=fields[7],
        )

    def is_valid(self) -> bool:
        """Check if record has minimum required data."""
        if not self.go_id or self.go_id == "-":
            return False
        if not self.go_term or self.go_term == "-":
            return False
        return True


class InsertionMetadataTracker:
    """Track gene2go insertions in a CSV file."""

    def __init__(self, metadata_dir: str = METADATA_DIR, metadata_file: str = GENE2GO_METADATA_FILE):
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
                        "filter_category",
                        "num_go_terms_inserted",
                        "num_relationships_inserted",
                        "batch_size",
                        "workers",
                        "embedding_model",
                        "neo4j_batch_size",
                        "milvus_batch_size",
                        "duration_seconds",
                    ]
                )

    def record_insertion(
        self,
        filter_taxid: list[str] | None,
        filter_category: list[str] | None,
        num_go_terms: int,
        num_relationships: int,
        batch_size: int,
        workers: int,
        embedding_model: str,
        neo4j_batch_size: int,
        milvus_batch_size: int,
        duration_seconds: float,
    ):
        """Record an insertion run to the metadata file."""
        timestamp = datetime.now().isoformat()
        taxid_str = ",".join(filter_taxid) if filter_taxid else "ALL"
        category_str = ",".join(filter_category) if filter_category else "ALL"

        with open(self.metadata_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    timestamp,
                    taxid_str,
                    category_str,
                    num_go_terms,
                    num_relationships,
                    batch_size,
                    workers,
                    embedding_model,
                    neo4j_batch_size,
                    milvus_batch_size,
                    f"{duration_seconds:.2f}",
                ]
            )

        print(f"Recorded insertion metadata to {self.metadata_path}")


class Gene2GOReader:
    """Read and filter Gene2GO annotations."""

    def __init__(
        self,
        file_path: str,
        gene_mapper: GeneIDMapper,
        filter_taxid: list[str] | None = None,
        filter_category: list[str] | None = None,
    ):
        self.file_path = file_path
        self.gene_mapper = gene_mapper
        self.filter_taxid = set(filter_taxid) if filter_taxid else None
        self.filter_category = set(filter_category) if filter_category else None

    def read_annotations(self, limit: int | None = None) -> Generator[tuple[Gene2GORecord, str], None, None]:
        """Read annotations from file with filtering. Yields (record, gene_symbol)."""
        count = 0
        with gzip.open(self.file_path, "rt") as f:
            next(f)

            for line in f:
                record = Gene2GORecord.from_line(line)
                if not record or not record.is_valid():
                    continue

                if self.filter_taxid and record.tax_id not in self.filter_taxid:
                    continue

                if self.filter_category and record.category not in self.filter_category:
                    continue

                gene_symbol = self.gene_mapper.get_symbol(record.gene_id)
                if not gene_symbol:
                    continue

                yield (record, gene_symbol)
                count += 1

                if limit and count >= limit:
                    break


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
    """Insert GO entities and relationships into Milvus."""

    def __init__(
        self,
        host: str = MILVUS_HOST,
        port: str = MILVUS_PORT,
        batch_size: int = MILVUS_BATCH_SIZE,
        drop_existing: bool = False,
        skip_go_terms: bool = False,
        skip_relationships: bool = False,
    ):
        connections.connect(host=host, port=port)
        self.batch_size = batch_size
        self.skip_go_terms = skip_go_terms
        self.skip_relationships = skip_relationships
        self.entity_pending = []
        self.relationship_pending = []
        self.entity_total_inserted = 0
        self.entity_total_time = 0.0
        self.rel_total_inserted = 0
        self.rel_total_time = 0.0

        self._drop_go_data()

        self._ensure_collections()

    def _drop_go_data(self):
        """Drop existing GO entities and relationships collections (conditionally)."""
        if not self.skip_go_terms:
            if utility.has_collection("entities_Gene_Ontology"):
                utility.drop_collection("entities_Gene_Ontology")
                print("Dropped existing Milvus collection: entities_Gene_Ontology")
        else:
            print("Skipping drop of entities_Gene_Ontology (--skip-go-terms flag)")

        if not self.skip_relationships:
            if utility.has_collection("relationships_Gene_Gene_Ontology"):
                utility.drop_collection("relationships_Gene_Gene_Ontology")
                print("Dropped existing Milvus collection: relationships_Gene_Gene_Ontology")
        else:
            print("Skipping drop of relationships_Gene_Gene_Ontology (--skip-relationships flag)")

    def _ensure_collections(self):
        """Ensure entities and relationships collections exist."""
        self.entities_collection_name = "entities_Gene_Ontology"
        if utility.has_collection(self.entities_collection_name):
            self.entities_collection = Collection(self.entities_collection_name)
            print(f"Using existing Milvus collection: {self.entities_collection_name}")
        else:
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
                FieldSchema(name="label_embedding", dtype=DataType.FLOAT_VECTOR, dim=LABEL_EMBEDDING_DIM),
                FieldSchema(name="created_at", dtype=DataType.INT64),
                FieldSchema(name="entity_name", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=32768),
            ]
            schema = CollectionSchema(fields, enable_dynamic_field=True)
            self.entities_collection = Collection(self.entities_collection_name, schema)
            print(f"Created Milvus collection: {self.entities_collection_name}")
            print("  (indexes will be created after insertion)")

        self.relationships_collection_name = "relationships_Gene_Gene_Ontology"
        if utility.has_collection(self.relationships_collection_name):
            self.relationships_collection = Collection(self.relationships_collection_name)
            print(f"Using existing Milvus collection: {self.relationships_collection_name}")
        else:
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
                FieldSchema(name="created_at", dtype=DataType.INT64),
                FieldSchema(name="src_id", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="tgt_id", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=32768),
            ]
            schema = CollectionSchema(fields, enable_dynamic_field=True)
            self.relationships_collection = Collection(self.relationships_collection_name, schema)
            print(f"Created Milvus collection: {self.relationships_collection_name}")
            print("  (indexes will be created after insertion)")

    def insert_entity_batch(
        self,
        go_ids: list[str],
        go_terms: list[str],
        go_categories: list[str],
        contents: list[str],
        content_embeddings: np.ndarray,
        label_embeddings: np.ndarray,
    ):
        """Add GO entities to pending batch."""
        current_time = int(time.time())

        for go_id, go_term, category, content, content_emb, label_emb in zip(
            go_ids, go_terms, go_categories, contents, content_embeddings, label_embeddings, strict=False
        ):
            entity_type = "Gene_Ontology"
            pk_id = self._compute_id(go_id, prefix="ent-")

            label = go_term

            if content:
                genes = content.split(", ")
                if len(genes) > 1000:
                    truncated_genes = genes[:1000]
                    content_for_storage = ", ".join(truncated_genes)
                else:
                    content_for_storage = content
            else:
                content_for_storage = content

            record = {
                "id": pk_id,
                "vector": content_emb.tolist(),
                "label_embedding": label_emb.tolist(),
                "created_at": current_time,
                "entity_name": go_id,
                "file_path": "genes/gene2go.gz",
                "content": content_for_storage,
                "label": label,
                "source_id": "",
                "description": go_term,
                "entity_type": entity_type,
                "entity_id": go_id,
                "go_id": go_id,
                "go_term": go_term,
                "go_category": category,
            }
            self.entity_pending.append(record)

            if len(self.entity_pending) >= self.batch_size:
                self._flush_entities()

    def insert_relationship_batch(
        self, src_symbols: list[str], tgt_go_ids: list[str], records: list[Gene2GORecord], embeddings: np.ndarray
    ):
        """Add Gene->GO relationships to pending batch."""
        current_time = int(time.time())

        for src_symbol, tgt_go_id, record, embedding in zip(src_symbols, tgt_go_ids, records, embeddings, strict=False):
            src_id = f"GENE:{src_symbol}"
            tgt_id = tgt_go_id

            composite_key = f"{src_id}{GRAPH_FIELD_SEP}{tgt_id}"
            pk_id = self._compute_id(composite_key, prefix="rel-")

            qualifier = record.qualifier if record.qualifier and record.qualifier != "-" else ""
            if qualifier:
                content = f"{src_symbol} {qualifier} {record.go_term} {tgt_go_id}"
            else:
                content = f"{src_symbol} {record.go_term} {tgt_go_id}"

            rel_record = {
                "id": pk_id,
                "vector": embedding.tolist(),
                "created_at": current_time,
                "src_id": src_id,
                "tgt_id": tgt_id,
                "file_path": "genes/gene2go.gz",
                "content": content,
                "description": record.go_term,
                "keywords": f"{record.evidence},{record.qualifier},{record.category}",
                "weight": 0.8,
                "source_id": "",
                "evidence": record.evidence,
                "qualifier": record.qualifier if record.qualifier != "-" else "",
                "category": record.category,
                "pubmed": record.pubmed if record.pubmed != "-" else "",
            }
            self.relationship_pending.append(rel_record)

            if len(self.relationship_pending) >= self.batch_size:
                self._flush_relationships()

    def _flush_entities(self):
        """Flush pending entities to Milvus using insert."""
        if not self.entity_pending:
            return

        flush_start = time.time()
        self.entities_collection.insert(self.entity_pending)

        flush_duration = time.time() - flush_start
        batch_count = len(self.entity_pending)
        self.entity_total_inserted += batch_count
        self.entity_total_time += flush_duration

        self.entity_pending = []

    def _flush_relationships(self):
        """Flush pending relationships to Milvus using insert."""
        if not self.relationship_pending:
            return

        flush_start = time.time()
        self.relationships_collection.insert(self.relationship_pending)

        flush_duration = time.time() - flush_start
        batch_count = len(self.relationship_pending)
        self.rel_total_inserted += batch_count
        self.rel_total_time += flush_duration

        self.relationship_pending = []

    def finalize_entities(self):
        """Flush entities data, create indexes, and load collection."""
        self._flush_entities()

        print("Flushing entities to Milvus...")
        self.entities_collection.flush()
        print("Entities flushed to Milvus")

        print("Creating indexes for entity vector fields...")
        index_params = {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 256}}

        self.entities_collection.create_index("vector", index_params)
        print("Created index for entities content embeddings (vector)")

        self.entities_collection.create_index("label_embedding", index_params)
        print("Created index for entities label embeddings (label_embedding)")

        self.entities_collection.load()
        print("Entities collection loaded and ready")

    def finalize_relationships(self):
        """Flush relationships data, create indexes, and load collection."""
        self._flush_relationships()

        print("Flushing relationships to Milvus...")
        self.relationships_collection.flush()
        print("Relationships flushed to Milvus")

        print("Creating indexes for relationship vector fields...")
        index_params = {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 256}}

        self.relationships_collection.create_index("vector", index_params)
        print("Created index for relationships embeddings (vector)")

        self.relationships_collection.load()
        print("Relationships collection loaded and ready")

    def close(self):
        """Flush and disconnect."""
        self._flush_entities()
        self._flush_relationships()
        connections.disconnect("default")

    @staticmethod
    def _compute_id(content: str, prefix: str = "") -> str:
        """Compute a prefixed MD5 hash ID."""
        return prefix + hashlib.md5(content.encode()).hexdigest()


def process_gene2go(args):
    """Main processing function."""
    start_time = time.time()

    print("=" * 60)
    print("Gene2GO Insertion")
    print("=" * 60)
    if args.drop_existing:
        print("WARNING: --drop-existing flag set - will delete all existing GO data!")
        print("=" * 60)
    print(f"Configuration file: {CONFIG_FILE}")
    print(f"Gene info file: {args.gene_info_file}")
    print(f"Gene2GO file: {args.gene2go_file}")
    print(f"Batch size: {args.batch_size}")
    if args.filter_taxid:
        print(f"Filter tax_id: {args.filter_taxid}")
    if args.filter_category:
        print(f"Filter category: {args.filter_category}")
    if args.limit:
        print(f"Limit: {args.limit}")
    print(f"Workers: {WORKERS}")
    print()

    gene_mapper = GeneIDMapper(args.gene_info_file, filter_taxid=args.filter_taxid)
    gene_mapper.build_mapping()

    reader = Gene2GOReader(
        args.gene2go_file, gene_mapper, filter_taxid=args.filter_taxid, filter_category=args.filter_category
    )

    embedder = ParallelEmbeddingGenerator()
    milvus = MilvusInserter(
        drop_existing=args.drop_existing, skip_go_terms=args.skip_go_terms, skip_relationships=args.skip_relationships
    )
    metadata_tracker = InsertionMetadataTracker()

    print("\nStep 3: Loading and filtering annotations...")
    go_terms_dict = {}
    relationships = []
    skipped_count = 0

    for record, gene_symbol in tqdm(reader.read_annotations(limit=args.limit), desc="Reading annotations"):
        if gene_symbol is None:
            skipped_count += 1
            continue

        if record.go_id not in go_terms_dict:
            go_terms_dict[record.go_id] = (record.go_term, record.category)

        relationships.append((gene_symbol, record.go_id, record))

    print(f"Found {len(go_terms_dict):,} unique GO terms")
    print(f"Found {len(relationships):,} Gene->GO relationships")
    if skipped_count > 0:
        print(f"Skipped {skipped_count:,} annotations (gene ID not mapped to symbol)")
    print()

    print("Building GO term to genes mapping...")
    go_id_to_genes = defaultdict(set)
    for gene_symbol, go_id, _record in relationships:
        go_id_to_genes[go_id].add(gene_symbol.upper())

    print(f"Mapped {len(go_id_to_genes):,} GO terms to their associated genes")
    print()

    if len(go_terms_dict) == 0:
        print("No GO terms to process. Exiting.")
        return

    if args.skip_go_terms:
        print()
        print("Skipping GO terms insertion (--skip-go-terms flag)")
        print()
    else:
        print(f"Step 4: Processing {len(go_terms_dict):,} GO terms with independent parallel streams...")
        print(
            f"Max pending batches: {MAX_PENDING_BATCHES} (buffer size: {MAX_PENDING_BATCHES * args.batch_size:,} terms)"
        )
        print()

    all_go_ids = list(go_terms_dict.keys())
    all_go_terms = [go_terms_dict[go_id][0] for go_id in all_go_ids]
    all_go_categories = [go_terms_dict[go_id][1] for go_id in all_go_ids]

    batch_id = 0
    next_batch_idx = 0
    total_batches = (len(all_go_ids) + args.batch_size - 1) // args.batch_size
    pending_batches = {}
    insertion_queue = []
    submitted_count = 0
    embedded_count = 0
    inserted_count = 0

    try:
        if not args.skip_go_terms:
            pbar_embed = tqdm(total=len(all_go_ids), desc="Embedding GO terms", position=0, leave=True)
            pbar_insert = tqdm(total=len(all_go_ids), desc="Inserting to Milvus", position=1, leave=True)

        while not args.skip_go_terms and (submitted_count < total_batches or pending_batches or insertion_queue):
            while submitted_count < total_batches and len(pending_batches) + len(insertion_queue) < MAX_PENDING_BATCHES:
                start_idx = next_batch_idx * args.batch_size
                end_idx = min(start_idx + args.batch_size, len(all_go_ids))
                batch_go_ids = all_go_ids[start_idx:end_idx]
                batch_go_terms = all_go_terms[start_idx:end_idx]
                batch_categories = all_go_categories[start_idx:end_idx]

                contents = []
                labels = []
                for go_id, term in zip(batch_go_ids, batch_go_terms, strict=False):
                    genes = sorted(go_id_to_genes.get(go_id, set()))
                    genes_str = ", ".join(genes)

                    content = genes_str
                    contents.append(content)

                    labels.append(term)

                embedder.embed_batch(batch_id, contents, labels)
                pending_batches[batch_id] = (batch_go_ids, batch_go_terms, batch_categories, contents)

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
                        raise RuntimeError(f"Worker failed to initialize: {content_embeddings}")

                    if (
                        content_embeddings is not None
                        and label_embeddings is not None
                        and result_batch_id in pending_batches
                    ):
                        batch_go_ids, batch_go_terms, batch_categories, batch_contents = pending_batches.pop(
                            result_batch_id
                        )

                        embedded_count += len(batch_go_ids)
                        pbar_embed.update(len(batch_go_ids))

                        insertion_queue.append(
                            (
                                batch_go_ids,
                                batch_go_terms,
                                batch_categories,
                                batch_contents,
                                content_embeddings,
                                label_embeddings,
                            )
                        )

            if insertion_queue:
                batch_go_ids, batch_go_terms, batch_categories, batch_contents, content_embeddings, label_embeddings = (
                    insertion_queue.pop(0)
                )

                milvus.insert_entity_batch(
                    batch_go_ids, batch_go_terms, batch_categories, batch_contents, content_embeddings, label_embeddings
                )

                inserted_count += len(batch_go_ids)
                pbar_insert.update(len(batch_go_ids))

        if not args.skip_go_terms:
            pbar_embed.close()
            pbar_insert.close()

            print()
            print("Finalizing GO entities in Milvus...")
            milvus.finalize_entities()

        if args.skip_relationships:
            print()
            print("Skipping relationships insertion (--skip-relationships flag)")
            print()
        else:
            print()
            print(f"Step 5: Processing {len(relationships):,} Gene->GO relationships")
            print("        with independent parallel streams...")
            buffer_size = MAX_PENDING_BATCHES * args.batch_size
            print(f"Max pending batches: {MAX_PENDING_BATCHES} (buffer size: {buffer_size:,} relationships)")
            print()

        if not args.skip_relationships:
            batch_id = 0
            next_batch_idx = 0
            total_batches = (len(relationships) + args.batch_size - 1) // args.batch_size
            pending_batches = {}
            insertion_queue = []
            submitted_count = 0
            embedded_count = 0
            inserted_count = 0

            pbar_embed = tqdm(total=len(relationships), desc="Embedding relationships", position=0, leave=True)
            pbar_insert = tqdm(total=len(relationships), desc="Inserting to Milvus", position=1, leave=True)

            while submitted_count < total_batches or pending_batches or insertion_queue:
                while (
                    submitted_count < total_batches
                    and len(pending_batches) + len(insertion_queue) < MAX_PENDING_BATCHES
                ):
                    start_idx = next_batch_idx * args.batch_size
                    end_idx = min(start_idx + args.batch_size, len(relationships))
                    batch_rels = relationships[start_idx:end_idx]
                    batch_src = [r[0] for r in batch_rels]
                    batch_tgt = [r[1] for r in batch_rels]
                    batch_records = [r[2] for r in batch_rels]

                    contents = []
                    labels = []
                    for src, tgt, rec in batch_rels:
                        qualifier = rec.qualifier if rec.qualifier and rec.qualifier != "-" else ""
                        if qualifier:
                            content = f"{src} {qualifier} {rec.go_term} {tgt}"
                        else:
                            content = f"{src} {rec.go_term} {tgt}"
                        contents.append(content)
                        labels.append(f"{src}->{tgt}")

                    embedder.embed_batch(batch_id, contents, labels)
                    pending_batches[batch_id] = (batch_src, batch_tgt, batch_records)

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
                            raise RuntimeError(f"Worker failed to initialize: {content_embeddings}")

                        if (
                            content_embeddings is not None
                            and label_embeddings is not None
                            and result_batch_id in pending_batches
                        ):
                            batch_src, batch_tgt, batch_records = pending_batches.pop(result_batch_id)

                            embedded_count += len(batch_src)
                            pbar_embed.update(len(batch_src))

                            insertion_queue.append((batch_src, batch_tgt, batch_records, content_embeddings))

                if insertion_queue:
                    batch_src, batch_tgt, batch_records, content_embeddings = insertion_queue.pop(0)

                    milvus.insert_relationship_batch(batch_src, batch_tgt, batch_records, content_embeddings)

                    inserted_count += len(batch_src)
                    pbar_insert.update(len(batch_src))

            pbar_embed.close()
            pbar_insert.close()

            print()
            print("Finalizing relationships in Milvus...")
            milvus.finalize_relationships()

        end_time = time.time()
        duration = end_time - start_time

        print()
        print("=" * 60)
        print(f"Inserted {len(go_terms_dict):,} GO terms")
        print(f"Inserted {len(relationships):,} Gene->GO relationships")
        print(f"Duration: {duration:.2f} seconds ({duration / 60:.2f} minutes)")
        print("=" * 60)

        metadata_tracker.record_insertion(
            filter_taxid=args.filter_taxid,
            filter_category=args.filter_category,
            num_go_terms=len(go_terms_dict),
            num_relationships=len(relationships),
            batch_size=args.batch_size,
            workers=WORKERS,
            embedding_model=EMBEDDING_MODEL,
            neo4j_batch_size=0,
            milvus_batch_size=MILVUS_BATCH_SIZE,
            duration_seconds=duration,
        )

    finally:
        embedder.close()
        milvus.close()


def main():
    parser = argparse.ArgumentParser(
        description="Insert Gene2GO annotations into Milvus",
        epilog="Configuration is loaded from config/db_init.conf. Command-line arguments override config.",
    )
    parser.add_argument(
        "--gene-info-file",
        default=CONFIG.get("GENE_INFO_FILE", f"{DATA_DIR}/resources/genes/gene_info.gz"),
        help="Path to gene_info.gz file (for GeneID mapping)",
    )
    parser.add_argument(
        "--gene2go-file",
        default=CONFIG.get("GENE2GO_FILE", f"{DATA_DIR}/resources/genes/gene2go.gz"),
        help="Path to gene2go.gz file",
    )
    parser.add_argument("--filter-taxid", nargs="+", default=None, help="Filter by taxonomy ID(s)")
    parser.add_argument(
        "--filter-category", nargs="+", default=None, help="Filter by GO category (Function, Process, Component)"
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for processing")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of annotations to process")
    parser.add_argument("--drop-existing", action="store_true", help="Drop all existing GO data from Milvus and Neo4j")
    parser.add_argument(
        "--skip-go-terms",
        action="store_true",
        help="Skip GO terms insertion (keep entities_Gene_Ontology collection intact)",
    )
    parser.add_argument(
        "--skip-relationships",
        action="store_true",
        help="Skip relationships insertion (keep relationships_Gene_Gene_Ontology collection intact)",
    )

    args = parser.parse_args()

    if args.filter_taxid is None:
        taxid_str = CONFIG.get("GENE2GO_FILTER_TAXID", "")
        if taxid_str:
            args.filter_taxid = [t.strip() for t in taxid_str.split(",")]

    if args.filter_category is None:
        cat_str = CONFIG.get("GENE2GO_FILTER_CATEGORY", "")
        if cat_str:
            args.filter_category = [c.strip() for c in cat_str.split(",")]

    if args.batch_size is None:
        args.batch_size = int(CONFIG.get("GENE2GO_BATCH_SIZE", "10000"))

    if args.limit is None:
        limit_val = int(CONFIG.get("GENE2GO_LIMIT", "0"))
        if limit_val > 0:
            args.limit = limit_val

    if not args.skip_go_terms:
        args.skip_go_terms = CONFIG.get("GENE2GO_SKIP_GO_TERMS", "false").lower() == "true"

    if not args.skip_relationships:
        args.skip_relationships = CONFIG.get("GENE2GO_SKIP_RELATIONSHIPS", "false").lower() == "true"

    if not os.path.exists(args.gene_info_file):
        print(f"Error: Gene info file not found: {args.gene_info_file}")
        sys.exit(1)

    if not os.path.exists(args.gene2go_file):
        print(f"Error: Gene2GO file not found: {args.gene2go_file}")
        sys.exit(1)

    process_gene2go(args)


if __name__ == "__main__":
    main()
