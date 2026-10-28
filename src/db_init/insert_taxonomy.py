#!/usr/bin/env python3
"""
Insert NCBI Taxonomy from taxonomy dump files into Milvus.

This script:
1. Parses NCBI Taxonomy dump files
2. Creates Taxonomy entities in Milvus
3. Creates Taxonomy->Taxonomy hierarchical relationships in Milvus
4. Filters by rank and division

Configuration loaded from config/db_init.conf
"""

import argparse
import csv
import hashlib
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
from taxonomy_parser import TaxonomyParser, TaxonomyRecord
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
WORKERS = int(CONFIG.get("WORKERS", "4"))
CUDA_DEVICES = CONFIG.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")
MAX_PENDING_BATCHES = int(CONFIG.get("MAX_PENDING_BATCHES", "10"))

MILVUS_HOST = CONFIG.get("MILVUS_HOST", "localhost")
MILVUS_PORT = CONFIG.get("MILVUS_PORT", "19530")
MILVUS_BATCH_SIZE = int(CONFIG.get("MILVUS_BATCH_SIZE", "10000"))

METADATA_DIR = CONFIG.get("METADATA_DIR", f"{DATA_DIR}/db-init")
TAXONOMY_METADATA_FILE = CONFIG.get("TAXONOMY_METADATA_FILE", "taxonomy.csv")

HF_TOKEN = CONFIG.get("HF_TOKEN", "")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN


class InsertionMetadataTracker:
    """Track taxonomy insertions in a CSV file."""

    def __init__(self, metadata_dir: str = METADATA_DIR, metadata_file: str = TAXONOMY_METADATA_FILE):
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
                        "num_entities",
                        "num_relationships",
                        "ranks_filter",
                        "divisions_filter",
                        "batch_size",
                        "workers",
                        "embedding_model",
                        "milvus_batch_size",
                        "duration_seconds",
                    ]
                )

    def record_insertion(
        self,
        num_entities: int,
        num_relationships: int,
        ranks_filter: list[str] | None,
        divisions_filter: list[str] | None,
        batch_size: int,
        workers: int,
        embedding_model: str,
        milvus_batch_size: int,
        duration_seconds: float,
    ):
        """Record an insertion run to the metadata file."""
        timestamp = datetime.now().isoformat()
        ranks_str = ",".join(ranks_filter) if ranks_filter else "ALL"
        divisions_str = ",".join(divisions_filter) if divisions_filter else "ALL"

        with open(self.metadata_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    timestamp,
                    num_entities,
                    num_relationships,
                    ranks_str,
                    divisions_str,
                    batch_size,
                    workers,
                    embedding_model,
                    milvus_batch_size,
                    f"{duration_seconds:.2f}",
                ]
            )

        print(f"Recorded insertion metadata to {self.metadata_path}")


def _embedding_worker(
    worker_id: int,
    device_id: int,
    content_model_name: str,
    label_model_name: str,
    sub_batch_size: int,
    max_length_content: int,
    max_length_label: int,
    input_queue,
    output_queue,
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

            p = multiprocessing.Process(
                target=_embedding_worker,
                args=(
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
            p.start()
            self.workers.append(p)

    def embed_batch(self, batch_id: int, contents: list[str], labels: list[str]):
        """Submit a batch for dual embedding (content + label)."""
        self.input_queue.put((batch_id, contents, labels))

    def get_result(self, block: bool = True, timeout: float = None) -> tuple[int, np.ndarray, np.ndarray] | None:
        """Get completed embedding result.

        Returns:
            Tuple of (batch_id, content_embeddings, label_embeddings) or None
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
            p.join()

        print("All embedding workers closed")


class MilvusInserter:
    """Insert Taxonomy entities and relationships into Milvus."""

    def __init__(
        self,
        host: str = MILVUS_HOST,
        port: str = MILVUS_PORT,
        batch_size: int = MILVUS_BATCH_SIZE,
        drop_existing: bool = False,
    ):
        connections.connect(host=host, port=port)
        self.batch_size = batch_size
        self.entity_pending = []
        self.relationship_pending = []

        if drop_existing:
            self._drop_taxonomy_data()

        self._ensure_collections()

    def _drop_taxonomy_data(self):
        """Drop existing Taxonomy collections."""
        if utility.has_collection("entities_Taxonomy"):
            utility.drop_collection("entities_Taxonomy")
            print("Dropped existing Milvus collection: entities_Taxonomy")

        if utility.has_collection("relationships_Taxonomy_Taxonomy"):
            utility.drop_collection("relationships_Taxonomy_Taxonomy")
            print("Dropped existing Milvus collection: relationships_Taxonomy_Taxonomy")

    def _ensure_collections(self):
        """Ensure entities and relationships collections exist."""
        self.entities_collection_name = "entities_Taxonomy"
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

        self.relationships_collection_name = "relationships_Taxonomy_Taxonomy"
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

    def insert_entity_batch(
        self,
        records: list[TaxonomyRecord],
        contents: list[str],
        content_embeddings: np.ndarray,
        label_embeddings: np.ndarray,
    ):
        """Add Taxonomy entities to pending batch."""
        current_time = int(time.time())

        for record, content, content_emb, label_emb in zip(
            records, contents, content_embeddings, label_embeddings, strict=False
        ):
            entity_type = "Taxonomy"
            entity_name = f"TAX:{record.tax_id}"
            pk_id = self._compute_id(entity_name, prefix="ent-")

            label = record.scientific_name

            entity_record = {
                "id": pk_id,
                "vector": content_emb.tolist(),
                "label_embedding": label_emb.tolist(),
                "created_at": current_time,
                "entity_name": entity_name,
                "file_path": "taxonomy/nodes.dmp",
                "content": content,
                "label": label,
                "entity_type": entity_type,
                "entity_id": entity_name,
                "tax_id": record.tax_id,
                "scientific_name": record.scientific_name,
                "common_names": ",".join(record.common_names),
                "synonyms": ",".join(record.synonyms),
                "rank": record.rank,
                "division_id": record.division_id,
                "lineage": ",".join(record.lineage),
                "lineage_names": " > ".join(record.lineage_names),
                "source_id": "",
                "description": record.scientific_name,
            }
            self.entity_pending.append(entity_record)

            if len(self.entity_pending) >= self.batch_size:
                self._flush_entities()

    def insert_relationship_batch(
        self, parent_records: list[TaxonomyRecord], child_records: list[TaxonomyRecord], embeddings: np.ndarray
    ):
        """Add Taxonomy->Taxonomy relationships to pending batch."""
        current_time = int(time.time())

        for parent, child, embedding in zip(parent_records, child_records, embeddings, strict=False):
            src_id = f"TAX:{parent.tax_id}"
            tgt_id = f"TAX:{child.tax_id}"

            composite_key = f"{src_id}{GRAPH_FIELD_SEP}{tgt_id}"
            pk_id = self._compute_id(composite_key, prefix="rel-")

            content = f"{parent.scientific_name} ({parent.rank}) is parent of {child.scientific_name} ({child.rank})"

            rel_record = {
                "id": pk_id,
                "vector": embedding.tolist(),
                "created_at": current_time,
                "src_id": src_id,
                "tgt_id": tgt_id,
                "file_path": "taxonomy/nodes.dmp",
                "content": content,
                "description": "Taxonomic parent-child relationship",
                "keywords": f"taxonomy,hierarchy,{child.rank}",
                "weight": 0.9,
                "source_id": "",
                "parent_rank": parent.rank,
                "child_rank": child.rank,
            }
            self.relationship_pending.append(rel_record)

            if len(self.relationship_pending) >= self.batch_size:
                self._flush_relationships()

    def _flush_entities(self):
        """Flush pending entities to Milvus."""
        if not self.entity_pending:
            return
        self.entities_collection.insert(self.entity_pending)
        self.entity_pending = []

    def _flush_relationships(self):
        """Flush pending relationships to Milvus."""
        if not self.relationship_pending:
            return
        self.relationships_collection.insert(self.relationship_pending)
        self.relationship_pending = []

    def finalize_entities(self):
        """Flush entities, create indexes, and load collection."""
        self._flush_entities()

        print("Flushing entities to Milvus...")
        self.entities_collection.flush()
        print("Entities flushed to Milvus")

        print("Creating indexes for entity vector fields...")
        index_params = {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 256}}

        self.entities_collection.create_index("vector", index_params)
        print("Created index for entities content embeddings")

        self.entities_collection.create_index("label_embedding", index_params)
        print("Created index for entities label embeddings")

        self.entities_collection.load()
        print("Entities collection loaded and ready")

    def finalize_relationships(self):
        """Flush relationships, create indexes, and load collection."""
        self._flush_relationships()

        print("Flushing relationships to Milvus...")
        self.relationships_collection.flush()
        print("Relationships flushed to Milvus")

        print("Creating indexes for relationship vector fields...")
        index_params = {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 256}}

        self.relationships_collection.create_index("vector", index_params)
        print("Created index for relationships embeddings")

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


def process_taxonomy(args):
    """Main processing function."""
    start_time = time.time()

    print("=" * 60)
    print("Taxonomy Insertion")
    print("=" * 60)
    if args.drop_existing:
        print("WARNING: --drop-existing flag set - will delete all existing Taxonomy data!")
        print("=" * 60)
    print(f"Configuration file: {CONFIG_FILE}")
    print(f"Taxonomy directory: {args.taxonomy_dir}")
    print(f"Batch size: {args.batch_size}")
    if args.filter_ranks:
        print(f"Filter ranks: {args.filter_ranks}")
    if args.filter_divisions:
        print(f"Filter divisions: {args.filter_divisions}")
    if args.limit:
        print(f"Limit: {args.limit}")
    print(f"Workers: {WORKERS}")
    print()

    parser = TaxonomyParser(args.taxonomy_dir)
    records = parser.parse(filter_ranks=args.filter_ranks, filter_divisions=args.filter_divisions, limit=args.limit)

    if len(records) == 0:
        print("No taxonomy records to process. Exiting.")
        return

    print("\nBuilding parent-child relationships...")
    tax_id_to_record = {r.tax_id: r for r in records}
    relationships = []

    for record in records:
        if record.tax_id == record.parent_tax_id:
            continue

        parent_record = tax_id_to_record.get(record.parent_tax_id)
        if parent_record:
            relationships.append((parent_record, record))

    print(f"Found {len(relationships):,} parent-child relationships")
    print()

    embedder = ParallelEmbeddingGenerator()
    milvus = MilvusInserter(drop_existing=args.drop_existing)
    metadata_tracker = InsertionMetadataTracker()

    print(f"Step 4: Processing {len(records):,} Taxonomy entities...")
    print(f"Max pending batches: {MAX_PENDING_BATCHES}")
    print()

    batch_id = 0
    next_batch_idx = 0
    total_batches = (len(records) + args.batch_size - 1) // args.batch_size
    pending_batches = {}
    insertion_queue = []
    submitted_count = 0

    try:
        pbar_embed = tqdm(total=len(records), desc="Embedding entities", position=0, leave=True)
        pbar_insert = tqdm(total=len(records), desc="Inserting to Milvus", position=1, leave=True)

        while submitted_count < total_batches or pending_batches or insertion_queue:
            while submitted_count < total_batches and len(pending_batches) < MAX_PENDING_BATCHES:
                start_idx = next_batch_idx * args.batch_size
                end_idx = min(start_idx + args.batch_size, len(records))
                batch_records = records[start_idx:end_idx]

                contents = []
                labels = []
                for rec in batch_records:
                    common_str = ", ".join(rec.common_names) if rec.common_names else "N/A"
                    lineage_str = " > ".join(rec.lineage_names) if rec.lineage_names else "N/A"

                    content = f"TAX_ID: {rec.tax_id}\n"
                    content += f"Scientific Name: {rec.scientific_name}\n"
                    content += f"Common Names: {common_str}\n"
                    content += f"Rank: {rec.rank}\n"
                    content += f"Lineage: {lineage_str}"
                    contents.append(content)

                    labels.append(rec.scientific_name)

                embedder.embed_batch(batch_id, contents, labels)
                pending_batches[batch_id] = (batch_records, contents)

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
                        batch_records, batch_contents = pending_batches.pop(result_batch_id)

                        pbar_embed.update(len(batch_records))
                        insertion_queue.append((batch_records, batch_contents, content_embeddings, label_embeddings))

            if insertion_queue:
                batch_records, batch_contents, content_embeddings, label_embeddings = insertion_queue.pop(0)

                milvus.insert_entity_batch(batch_records, batch_contents, content_embeddings, label_embeddings)
                pbar_insert.update(len(batch_records))

        pbar_embed.close()
        pbar_insert.close()

        print()
        print("Finalizing entities in Milvus...")
        milvus.finalize_entities()

        print()
        print(f"Step 5: Processing {len(relationships):,} Taxonomy relationships...")
        print()

        batch_id = 0
        next_batch_idx = 0
        total_batches = (len(relationships) + args.batch_size - 1) // args.batch_size
        pending_batches = {}
        insertion_queue = []
        submitted_count = 0

        pbar_embed = tqdm(total=len(relationships), desc="Embedding relationships", position=0, leave=True)
        pbar_insert = tqdm(total=len(relationships), desc="Inserting to Milvus", position=1, leave=True)

        while submitted_count < total_batches or pending_batches or insertion_queue:
            while submitted_count < total_batches and len(pending_batches) < MAX_PENDING_BATCHES:
                start_idx = next_batch_idx * args.batch_size
                end_idx = min(start_idx + args.batch_size, len(relationships))
                batch_rels = relationships[start_idx:end_idx]

                contents = []
                labels = []
                for parent, child in batch_rels:
                    content = (
                        f"{parent.scientific_name} ({parent.rank}) is parent of {child.scientific_name} ({child.rank})"
                    )
                    contents.append(content)
                    labels.append(f"{parent.tax_id} -> {child.tax_id}")

                embedder.embed_batch(batch_id, contents, labels)
                pending_batches[batch_id] = batch_rels

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
                        batch_rels = pending_batches.pop(result_batch_id)

                        pbar_embed.update(len(batch_rels))
                        insertion_queue.append((batch_rels, content_embeddings))

            if insertion_queue:
                batch_rels, content_embeddings = insertion_queue.pop(0)

                parent_records = [p for p, c in batch_rels]
                child_records = [c for p, c in batch_rels]
                milvus.insert_relationship_batch(parent_records, child_records, content_embeddings)
                pbar_insert.update(len(batch_rels))

        pbar_embed.close()
        pbar_insert.close()

        print()
        print("Finalizing relationships in Milvus...")
        milvus.finalize_relationships()

        end_time = time.time()
        duration = end_time - start_time

        print()
        print("=" * 60)
        print(f"Inserted {len(records):,} Taxonomy entities")
        print(f"Inserted {len(relationships):,} Taxonomy relationships")
        print(f"Duration: {duration:.2f} seconds ({duration / 60:.2f} minutes)")
        print("=" * 60)

        metadata_tracker.record_insertion(
            num_entities=len(records),
            num_relationships=len(relationships),
            ranks_filter=args.filter_ranks,
            divisions_filter=args.filter_divisions,
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
        description="Insert NCBI Taxonomy into Milvus", epilog="Configuration loaded from config/db_init.conf"
    )
    parser.add_argument(
        "--taxonomy-dir",
        default=CONFIG.get("TAXONOMY_DIR", f"{DATA_DIR}/resources/taxonomy"),
        help="Path to taxonomy directory",
    )
    parser.add_argument("--filter-ranks", nargs="+", default=None, help="Filter by taxonomic rank(s)")
    parser.add_argument("--filter-divisions", nargs="+", default=None, help="Filter by division ID(s)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for processing")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of taxa to process")
    parser.add_argument(
        "--drop-existing", action="store_true", default=True, help="Drop existing Taxonomy collections (default: True)"
    )
    parser.add_argument(
        "--no-drop-existing", action="store_false", dest="drop_existing", help="Keep existing Taxonomy collections"
    )

    args = parser.parse_args()

    if args.filter_ranks is None:
        ranks_str = CONFIG.get("TAXONOMY_FILTER_RANKS", "")
        if ranks_str:
            args.filter_ranks = [r.strip() for r in ranks_str.split(",")]

    if args.filter_divisions is None:
        divs_str = CONFIG.get("TAXONOMY_FILTER_DIVISIONS", "")
        if divs_str:
            args.filter_divisions = [d.strip() for d in divs_str.split(",")]

    if args.batch_size is None:
        args.batch_size = int(CONFIG.get("TAXONOMY_BATCH_SIZE", "5000"))

    if args.limit is None:
        limit_val = int(CONFIG.get("TAXONOMY_LIMIT", "0"))
        if limit_val > 0:
            args.limit = limit_val

    if not os.path.exists(args.taxonomy_dir):
        print(f"Error: Taxonomy directory not found: {args.taxonomy_dir}")
        sys.exit(1)

    process_taxonomy(args)


if __name__ == "__main__":
    main()
