#!/usr/bin/env python3
"""
Insert UBERON Anatomy from uberon.owl into Milvus.

This script:
1. Parses UBERON Anatomy OWL file
2. Creates Anatomy entities in Milvus
3. Creates Anatomy->Anatomy hierarchical relationships in Milvus
4. Filters by organism subset and obsolete status

Configuration loaded from config/db_init.conf
"""

import argparse
import csv
import hashlib
import multiprocessing
import os
import time
from datetime import datetime
from multiprocessing import Pool, Queue, get_context
from pathlib import Path
from queue import Empty

import numpy as np
import torch
from config_loader import PROJECT_ROOT, load_config
from embedding_utils import masked_mean_pooling, normalize_embeddings
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from uberon_parser import UberonParser

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
UBERON_METADATA_FILE = CONFIG.get("UBERON_METADATA_FILE", "uberon.csv")

UBERON_FILE = CONFIG.get("UBERON_FILE", f"{DATA_DIR}/resources/ontologies/uberon.owl")

HF_TOKEN = CONFIG.get("HF_TOKEN", "")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN


class InsertionMetadataTracker:
    """Track metadata about UBERON anatomy insertions."""

    def __init__(self):
        self.metadata_dir = Path(METADATA_DIR)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.metadata_dir / UBERON_METADATA_FILE

        if not self.metadata_path.exists():
            with open(self.metadata_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "num_entities",
                        "num_relationships",
                        "subsets_filter",
                        "exclude_obsolete",
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
        subsets_filter: list[str],
        exclude_obsolete: bool,
        batch_size: int,
        workers: int,
        embedding_model: str,
        milvus_batch_size: int,
        duration_seconds: float,
    ):
        """Record insertion metadata."""
        timestamp = datetime.now().isoformat()
        subsets_str = ",".join(subsets_filter) if subsets_filter else "ALL"

        with open(self.metadata_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    timestamp,
                    num_entities,
                    num_relationships,
                    subsets_str,
                    str(exclude_obsolete),
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
    input_queue: Queue,
    output_queue: Queue,
):
    """Worker process for generating dual embeddings (content + label)."""
    try:
        device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        print(f"Worker {worker_id} using device: {device}")

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
            import traceback

            traceback.print_exc()
            continue


class ParallelEmbeddingGenerator:
    """Parallel embedding generator with multiple workers across GPUs."""

    def __init__(
        self,
        num_workers: int = WORKERS,
        sub_batch_size: int = EMBEDDING_SUB_BATCH_SIZE,
        content_model_name: str = EMBEDDING_MODEL,
        label_model_name: str = LABEL_EMBEDDING_MODEL,
        batch_size: int = 100,
        max_length_content: int = MAX_LENGTH_CONTENT,
        max_length_label: int = MAX_LENGTH_LABEL,
    ):
        """Initialize parallel embedding generator."""
        self.num_workers = num_workers
        self.sub_batch_size = sub_batch_size
        self.max_length_content = max_length_content
        self.max_length_label = max_length_label
        self.content_model_name = content_model_name
        self.label_model_name = label_model_name
        self.batch_size = batch_size

        cuda_devices = [int(d.strip()) for d in CUDA_DEVICES.split(",") if d.strip()]
        self.devices = cuda_devices if cuda_devices else [0]

        ctx = get_context("spawn")
        self.input_queue = ctx.Queue(maxsize=MAX_PENDING_BATCHES * 2)
        self.output_queue = ctx.Queue(maxsize=MAX_PENDING_BATCHES * 2)

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
        """Get completed embedding result (content embeddings, label embeddings)."""
        try:
            if block:
                return self.output_queue.get(timeout=timeout)
            else:
                return self.output_queue.get_nowait()
        except Empty:
            return None

    def close(self):
        """Shutdown workers."""
        for _ in range(self.num_workers):
            self.input_queue.put(None)

        for p in self.workers:
            p.close()
            p.join()


class MilvusInserter:
    """Handle Milvus collection creation and batch insertion."""

    def __init__(self, drop_existing: bool = False, batch_size: int = MILVUS_BATCH_SIZE):
        """Initialize Milvus connection and collections."""
        self.batch_size = batch_size

        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
        print(f"Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")

        self._drop_existing()

        self._ensure_collections()

        self.entity_pending = []
        self.relationship_pending = []

    def _drop_existing(self):
        """Drop existing UBERON Anatomy entities and relationships collections."""
        if utility.has_collection("entities_Anatomy"):
            utility.drop_collection("entities_Anatomy")
            print("Dropped existing Milvus collection: entities_Anatomy")

        if utility.has_collection("relationships_Anatomy_Anatomy"):
            utility.drop_collection("relationships_Anatomy_Anatomy")
            print("Dropped existing Milvus collection: relationships_Anatomy_Anatomy")

    def _ensure_collections(self):
        """Ensure Milvus collections exist."""
        self.entities_collection_name = "entities_Anatomy"
        if utility.has_collection(self.entities_collection_name):
            self.entities_collection = Collection(self.entities_collection_name)
            print(f"Using existing Milvus collection: {self.entities_collection_name}")
        else:
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
                FieldSchema(name="label_embedding", dtype=DataType.FLOAT_VECTOR, dim=LABEL_EMBEDDING_DIM),
                FieldSchema(name="created_at", dtype=DataType.INT64),
                FieldSchema(name="entity_name", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=32768),
            ]
            schema = CollectionSchema(fields, enable_dynamic_field=True)
            self.entities_collection = Collection(self.entities_collection_name, schema)
            print(
                f"Created Milvus collection: {self.entities_collection_name} (indexes will be created after insertion)"
            )

        self.relationships_collection_name = "relationships_Anatomy_Anatomy"
        if utility.has_collection(self.relationships_collection_name):
            self.relationships_collection = Collection(self.relationships_collection_name)
            print(f"Using existing Milvus collection: {self.relationships_collection_name}")
        else:
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
                FieldSchema(name="created_at", dtype=DataType.INT64),
                FieldSchema(name="src_id", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="tgt_id", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=32768),
            ]
            schema = CollectionSchema(fields, enable_dynamic_field=True)
            self.relationships_collection = Collection(self.relationships_collection_name, schema)
            print(
                f"Created Milvus collection: {self.relationships_collection_name} (indexes will be created after insertion)"
            )

    def _compute_id(self, composite_key: str, prefix: str) -> str:
        """Compute ID from composite key."""
        hash_value = hashlib.md5(composite_key.encode()).hexdigest()
        return f"{prefix}{hash_value}"

    def insert_entity_batch(
        self,
        uberon_ids: list[str],
        labels: list[str],
        definitions: list[str],
        synonyms_list: list[str],
        subsets_list: list[str],
        dbxrefs_list: list[str],
        content_embeddings: np.ndarray,
        label_embeddings: np.ndarray,
    ):
        """Add UBERON anatomy entities to pending batch."""
        current_time = int(time.time())

        for uberon_id, label, definition, synonyms, subsets, dbxrefs, content_emb, label_emb in zip(
            uberon_ids,
            labels,
            definitions,
            synonyms_list,
            subsets_list,
            dbxrefs_list,
            content_embeddings,
            label_embeddings,
            strict=False,
        ):
            entity_type = "Anatomy"
            pk_id = self._compute_id(uberon_id, prefix="ent-")

            content = f"{uberon_id}\n{label}"
            if definition:
                content += f"\n{definition}"
            if synonyms:
                content += f"\nSynonyms: {synonyms}"
            label_text = label

            record = {
                "id": pk_id,
                "vector": content_emb.tolist(),
                "label_embedding": label_emb.tolist(),
                "created_at": current_time,
                "entity_name": uberon_id,
                "file_path": "cell/uberon.owl",
                "content": content,
                "label": label_text,
                "source_id": "",
                "description": definition,
                "entity_type": entity_type,
                "entity_id": uberon_id,
                "uberon_id": uberon_id,
                "uberon_label": label,
                "uberon_definition": definition,
                "synonyms": synonyms,
                "subsets": subsets,
                "dbxrefs": dbxrefs,
            }
            self.entity_pending.append(record)

        if len(self.entity_pending) >= self.batch_size:
            self._flush_entities()

    def insert_relationship_batch(
        self,
        child_ids: list[str],
        parent_ids: list[str],
        child_labels: list[str],
        parent_labels: list[str],
        content_embeddings: np.ndarray,
    ):
        """Add Anatomy->Anatomy relationships to pending batch."""
        current_time = int(time.time())

        for child_id, parent_id, child_label, parent_label, content_emb in zip(
            child_ids, parent_ids, child_labels, parent_labels, content_embeddings, strict=False
        ):
            src_id = child_id
            tgt_id = parent_id
            composite_key = f"{src_id}{GRAPH_FIELD_SEP}{tgt_id}"
            pk_id = self._compute_id(composite_key, prefix="rel-")

            content = f"Anatomy {child_id} ({child_label}) is a subtype of {parent_id} ({parent_label})"

            record = {
                "id": pk_id,
                "vector": content_emb.tolist(),
                "created_at": current_time,
                "src_id": src_id,
                "tgt_id": tgt_id,
                "file_path": "cell/uberon.owl",
                "content": content,
                "source_id": "",
                "description": f"{child_label} is a type of {parent_label}",
                "relationship_type": "is_a",
                "child_uberon_id": child_id,
                "parent_uberon_id": parent_id,
                "child_label": child_label,
                "parent_label": parent_label,
            }
            self.relationship_pending.append(record)

        if len(self.relationship_pending) >= self.batch_size:
            self._flush_relationships()

    def _flush_entities(self):
        """Flush pending entity records to Milvus."""
        if self.entity_pending:
            self.entities_collection.insert(self.entity_pending)
            self.entity_pending = []

    def _flush_relationships(self):
        """Flush pending relationship records to Milvus."""
        if self.relationship_pending:
            self.relationships_collection.insert(self.relationship_pending)
            self.relationship_pending = []

    def finalize(self):
        """Flush remaining records and create indexes."""
        self._flush_entities()
        self._flush_relationships()

        print("\nCreating indexes...")

        if not self.entities_collection.has_index():
            index_params_vector = {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 200},
            }
            self.entities_collection.create_index(field_name="vector", index_params=index_params_vector)
            print(f"Created HNSW index on {self.entities_collection_name}.vector")

            index_params_label = {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 200},
            }
            self.entities_collection.create_index(field_name="label_embedding", index_params=index_params_label)
            print(f"Created HNSW index on {self.entities_collection_name}.label_embedding")

        if not self.relationships_collection.has_index():
            self.relationships_collection.create_index(field_name="vector", index_params=index_params_vector)
            print(f"Created HNSW index on {self.relationships_collection_name}.vector")

        self.entities_collection.load()
        self.relationships_collection.load()
        print("Loaded collections into memory")

    def close(self):
        """Close Milvus connection."""
        connections.disconnect("default")


def process_uberon(args):
    """Main processing function."""
    start_time = time.time()

    print("=" * 60)
    print("UBERON Anatomy Insertion")
    print("=" * 60)
    print("WARNING: Will delete all existing UBERON Anatomy data for fresh insertion!")
    print("=" * 60)
    print(f"Configuration file: {CONFIG_FILE}")
    print(f"OWL file: {args.owl_file}")
    print(f"Batch size: {args.batch_size}")
    if args.filter_subset:
        print(f"Filter subsets: {args.filter_subset}")
    if args.exclude_obsolete:
        print(f"Exclude obsolete: {args.exclude_obsolete}")
    if args.limit:
        print(f"Limit: {args.limit}")
    print(f"Workers: {WORKERS}")
    print()

    print("Step 1: Parsing UBERON Anatomy OWL file...")
    parser = UberonParser(args.owl_file)

    filter_subsets = args.filter_subset if args.filter_subset else None
    anatomy_records = parser.parse(
        filter_subsets=filter_subsets, exclude_obsolete=args.exclude_obsolete, limit=args.limit
    )

    hierarchies = parser.extract_hierarchies(anatomy_records)
    print()

    if len(anatomy_records) == 0:
        print("No anatomy terms to process. Exiting.")
        return

    print("Step 2: Initializing components...")
    embedder = ParallelEmbeddingGenerator(batch_size=args.batch_size)
    milvus = MilvusInserter(drop_existing=args.drop_existing)
    metadata_tracker = InsertionMetadataTracker()
    print()

    print(f"Step 3: Processing {len(anatomy_records):,} anatomy terms with parallel streams...")
    print(f"Max pending batches: {MAX_PENDING_BATCHES}")
    print()

    all_uberon_ids = [r.uberon_id for r in anatomy_records]
    all_labels = [r.label for r in anatomy_records]
    all_definitions = [r.definition for r in anatomy_records]
    all_synonyms = [",".join(r.synonyms) for r in anatomy_records]
    all_subsets = [",".join(r.subsets) for r in anatomy_records]
    all_dbxrefs = [",".join(r.dbxrefs) for r in anatomy_records]

    uberon_id_to_label = {r.uberon_id: r.label for r in anatomy_records}

    batch_id = 0
    next_batch_idx = 0
    total_batches = (len(anatomy_records) + args.batch_size - 1) // args.batch_size
    pending_batches = {}
    insertion_queue = []

    submitted_count = 0
    embedded_count = 0
    inserted_count = 0

    try:
        pbar_embed = tqdm(total=len(anatomy_records), desc="Embedding entities", position=0, leave=True)
        pbar_insert = tqdm(total=len(anatomy_records), desc="Inserting to Milvus", position=1, leave=True)

        while submitted_count < total_batches or pending_batches or insertion_queue:
            while submitted_count < total_batches and len(pending_batches) + len(insertion_queue) < MAX_PENDING_BATCHES:
                start_idx = next_batch_idx * args.batch_size
                end_idx = min(start_idx + args.batch_size, len(anatomy_records))

                batch_uberon_ids = all_uberon_ids[start_idx:end_idx]
                batch_labels = all_labels[start_idx:end_idx]
                batch_definitions = all_definitions[start_idx:end_idx]
                batch_synonyms = all_synonyms[start_idx:end_idx]

                contents = []
                labels = []
                for uberon_id, label, definition, _synonyms in zip(
                    batch_uberon_ids, batch_labels, batch_definitions, batch_synonyms, strict=False
                ):
                    content = f"{uberon_id}\n{label}"
                    if definition:
                        content += f"\n{definition}"
                    contents.append(content)
                    labels.append(label)

                embedder.embed_batch(batch_id, contents, labels)
                pending_batches[batch_id] = (
                    batch_uberon_ids,
                    batch_labels,
                    batch_definitions,
                    all_synonyms[start_idx:end_idx],
                    all_subsets[start_idx:end_idx],
                    all_dbxrefs[start_idx:end_idx],
                )

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
                        embedder.close()
                        milvus.close()
                        return

                    (
                        batch_uberon_ids,
                        batch_labels,
                        batch_definitions,
                        batch_synonyms,
                        batch_subsets,
                        batch_dbxrefs,
                    ) = pending_batches.pop(result_batch_id)

                    embedded_count += len(batch_uberon_ids)
                    pbar_embed.update(len(batch_uberon_ids))

                    insertion_queue.append(
                        (
                            batch_uberon_ids,
                            batch_labels,
                            batch_definitions,
                            batch_synonyms,
                            batch_subsets,
                            batch_dbxrefs,
                            content_embeddings,
                            label_embeddings,
                        )
                    )

            if insertion_queue:
                (
                    batch_uberon_ids,
                    batch_labels,
                    batch_definitions,
                    batch_synonyms,
                    batch_subsets,
                    batch_dbxrefs,
                    content_embeddings,
                    label_embeddings,
                ) = insertion_queue.pop(0)

                milvus.insert_entity_batch(
                    batch_uberon_ids,
                    batch_labels,
                    batch_definitions,
                    batch_synonyms,
                    batch_subsets,
                    batch_dbxrefs,
                    content_embeddings,
                    label_embeddings,
                )

                inserted_count += len(batch_uberon_ids)
                pbar_insert.update(len(batch_uberon_ids))

        pbar_embed.close()
        pbar_insert.close()
        print(f"\nInserted {inserted_count:,} anatomy entities")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Cleaning up...")
        embedder.close()
        milvus.close()
        return

    print(f"\nStep 4: Processing {len(hierarchies):,} hierarchical relationships...")

    valid_uberon_ids = set(all_uberon_ids)
    hierarchies = [
        (child, parent) for child, parent in hierarchies if child in valid_uberon_ids and parent in valid_uberon_ids
    ]
    print(f"Filtered to {len(hierarchies):,} relationships within dataset")

    if hierarchies:
        child_ids = [h[0] for h in hierarchies]
        parent_ids = [h[1] for h in hierarchies]
        child_labels = [uberon_id_to_label.get(c, c) for c in child_ids]
        parent_labels = [uberon_id_to_label.get(p, p) for p in parent_ids]

        batch_id = 0
        next_batch_idx = 0
        total_batches = (len(hierarchies) + args.batch_size - 1) // args.batch_size
        pending_batches = {}
        insertion_queue = []

        submitted_count = 0
        embedded_count = 0
        inserted_count = 0

        pbar_embed = tqdm(total=len(hierarchies), desc="Embedding relationships", position=0, leave=True)
        pbar_insert = tqdm(total=len(hierarchies), desc="Inserting to Milvus", position=1, leave=True)

        while submitted_count < total_batches or pending_batches or insertion_queue:
            while submitted_count < total_batches and len(pending_batches) + len(insertion_queue) < MAX_PENDING_BATCHES:
                start_idx = next_batch_idx * args.batch_size
                end_idx = min(start_idx + args.batch_size, len(hierarchies))

                batch_child_ids = child_ids[start_idx:end_idx]
                batch_parent_ids = parent_ids[start_idx:end_idx]
                batch_child_labels = child_labels[start_idx:end_idx]
                batch_parent_labels = parent_labels[start_idx:end_idx]

                contents = [
                    f"{child_label} is_a {parent_label}"
                    for child_label, parent_label in zip(batch_child_labels, batch_parent_labels, strict=False)
                ]
                labels = [
                    f"{child_id}->{parent_id}"
                    for child_id, parent_id in zip(batch_child_ids, batch_parent_ids, strict=False)
                ]

                embedder.embed_batch(batch_id, contents, labels)
                pending_batches[batch_id] = (batch_child_ids, batch_parent_ids, batch_child_labels, batch_parent_labels)

                submitted_count += 1
                next_batch_idx += 1
                batch_id += 1

            if pending_batches:
                result = embedder.get_result(block=False)
                if result is not None:
                    result_batch_id, content_embeddings, label_embeddings = result
                    (batch_child_ids, batch_parent_ids, batch_child_labels, batch_parent_labels) = pending_batches.pop(
                        result_batch_id
                    )

                    embedded_count += len(batch_child_ids)
                    pbar_embed.update(len(batch_child_ids))

                    insertion_queue.append(
                        (batch_child_ids, batch_parent_ids, batch_child_labels, batch_parent_labels, content_embeddings)
                    )

            if insertion_queue:
                (batch_child_ids, batch_parent_ids, batch_child_labels, batch_parent_labels, content_embeddings) = (
                    insertion_queue.pop(0)
                )

                milvus.insert_relationship_batch(
                    batch_child_ids, batch_parent_ids, batch_child_labels, batch_parent_labels, content_embeddings
                )

                inserted_count += len(batch_child_ids)
                pbar_insert.update(len(batch_child_ids))

        pbar_embed.close()
        pbar_insert.close()
        print(f"Inserted {inserted_count:,} hierarchical relationships")

    print("\nFinalizing insertion...")
    milvus.finalize()
    embedder.close()
    milvus.close()

    duration = time.time() - start_time
    metadata_tracker.record_insertion(
        num_entities=len(anatomy_records),
        num_relationships=len(hierarchies),
        subsets_filter=args.filter_subset if args.filter_subset else [],
        exclude_obsolete=args.exclude_obsolete,
        batch_size=args.batch_size,
        workers=WORKERS,
        embedding_model=EMBEDDING_MODEL,
        milvus_batch_size=MILVUS_BATCH_SIZE,
        duration_seconds=duration,
    )

    print(f"\nCompleted in {duration:.2f} seconds")
    print(f"Inserted {len(anatomy_records):,} anatomy entities")
    print(f"Inserted {len(hierarchies):,} hierarchical relationships")


def main():
    parser = argparse.ArgumentParser(
        description="Insert UBERON Anatomy into Milvus",
        epilog="Configuration loaded from config/db_init.conf. Command-line arguments override config.",
    )

    parser.add_argument("--config", default=str(CONFIG_FILE), help="Path to config file")
    parser.add_argument("--owl-file", default=UBERON_FILE, help="Path to uberon.owl file")
    parser.add_argument(
        "--filter-subset", nargs="+", help="Filter by organism subsets (e.g., human_subset mouse_subset)"
    )
    parser.add_argument(
        "--exclude-obsolete", action="store_true", default=True, help="Exclude obsolete anatomy terms (default: True)"
    )
    parser.add_argument("--include-obsolete", action="store_true", help="Include obsolete anatomy terms")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for processing")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of anatomy terms to process (0 = no limit)")
    parser.add_argument(
        "--drop-existing", action="store_true", help="Drop existing UBERON Anatomy data before insertion"
    )

    args = parser.parse_args()

    if args.include_obsolete:
        args.exclude_obsolete = False

    process_uberon(args)


if __name__ == "__main__":
    main()
