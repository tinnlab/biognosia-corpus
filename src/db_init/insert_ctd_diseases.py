#!/usr/bin/env python3
"""
Insert CTD Diseases into Milvus.

This script:
1. Parses CTD disease files
2. Creates Disease entities in Milvus
3. Creates Disease->Disease hierarchical relationships
4. Creates Gene->Disease relationships
5. Creates Disease->KEGG Pathway relationships
6. Creates Disease->Reactome Pathway relationships

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
from multiprocessing import Manager, Queue, get_context
from pathlib import Path
from queue import Empty

import numpy as np
import torch
from config_loader import PROJECT_ROOT, load_config
from ctd_disease_parser import CTDDiseaseParser
from embedding_utils import masked_mean_pooling
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
CTD_DISEASE_METADATA_FILE = CONFIG.get("CTD_DISEASE_METADATA_FILE", "ctd_diseases.csv")

CTD_DISEASE_FILE = CONFIG.get("CTD_DISEASE_FILE", f"{DATA_DIR}/resources/ctd/CTD_diseases.csv.gz")
CTD_DISEASE_GENE_FILE = CONFIG.get(
    "CTD_DISEASE_GENE_FILE", f"{DATA_DIR}/resources/ctd/CTD_curated_genes_diseases.csv.gz"
)
CTD_DISEASE_PATHWAY_FILE = CONFIG.get(
    "CTD_DISEASE_PATHWAY_FILE", f"{DATA_DIR}/resources/ctd/CTD_diseases_pathways.csv.gz"
)
CTD_DISEASE_BATCH_SIZE = int(CONFIG.get("CTD_DISEASE_BATCH_SIZE", "1000"))
CTD_DISEASE_LIMIT = int(CONFIG.get("CTD_DISEASE_LIMIT", "0"))

GENE_INFO_FILE = CONFIG.get("GENE_INFO_FILE", f"{DATA_DIR}/resources/genes/gene_info.gz")
GENE_FILTER_TAXID = [x.strip() for x in CONFIG.get("GENE_FILTER_TAXID", "").split(",") if x.strip()]

HF_TOKEN = CONFIG.get("HF_TOKEN", "")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN


class InsertionMetadataTracker:
    """Track metadata about CTD disease insertions."""

    def __init__(self):
        self.metadata_dir = Path(METADATA_DIR)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.metadata_dir / CTD_DISEASE_METADATA_FILE

        if not self.metadata_path.exists():
            with open(self.metadata_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "num_diseases",
                        "num_disease_hierarchies",
                        "num_gene_disease_rels",
                        "num_disease_kegg_pathway_rels",
                        "num_disease_reactome_pathway_rels",
                        "category_filter",
                        "batch_size",
                        "workers",
                        "embedding_model",
                        "milvus_batch_size",
                        "duration_seconds",
                    ]
                )

    def record_insertion(
        self,
        num_diseases: int,
        num_disease_hierarchies: int,
        num_gene_disease_rels: int,
        num_disease_kegg_pathway_rels: int,
        num_disease_reactome_pathway_rels: int,
        category_filter: list[str] | None,
        batch_size: int,
        workers: int,
        embedding_model: str,
        milvus_batch_size: int,
        duration_seconds: float,
    ):
        """Record insertion metadata."""
        timestamp = datetime.now().isoformat()
        category_str = ",".join(category_filter) if category_filter else "ALL"

        with open(self.metadata_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    timestamp,
                    num_diseases,
                    num_disease_hierarchies,
                    num_gene_disease_rels,
                    num_disease_kegg_pathway_rels,
                    num_disease_reactome_pathway_rels,
                    category_str,
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
        print(f"Worker {worker_id} failed to load models: {e}")
        output_queue.put(("WORKER_INIT_ERROR", str(e), None))
        return

    with torch.no_grad():
        while True:
            try:
                task = input_queue.get(timeout=1)
                if task is None:
                    break

                batch_id, contents, labels = task

                content_embeddings_list = []
                label_embeddings_list = []

                for i in range(0, len(contents), sub_batch_size):
                    sub_contents = contents[i : i + sub_batch_size]
                    sub_labels = labels[i : i + sub_batch_size]

                    content_inputs = content_tokenizer(
                        sub_contents, padding=True, truncation=True, return_tensors="pt", max_length=max_length_content
                    ).to(device)
                    content_outputs = content_model(**content_inputs)
                    content_emb = masked_mean_pooling(
                        content_outputs.last_hidden_state, content_inputs["attention_mask"]
                    )
                    content_emb = torch.nn.functional.normalize(content_emb, p=2, dim=1)
                    content_embeddings_list.append(content_emb.cpu().numpy())

                    label_inputs = label_tokenizer(
                        sub_labels, padding=True, truncation=True, return_tensors="pt", max_length=max_length_label
                    ).to(device)
                    label_outputs = label_model(**label_inputs)
                    label_emb = masked_mean_pooling(label_outputs.last_hidden_state, label_inputs["attention_mask"])
                    label_emb = torch.nn.functional.normalize(label_emb, p=2, dim=1)
                    label_embeddings_list.append(label_emb.cpu().numpy())

                content_embeddings = np.vstack(content_embeddings_list)
                label_embeddings = np.vstack(label_embeddings_list)

                output_queue.put((batch_id, content_embeddings, label_embeddings))

            except Empty:
                continue
            except Exception as e:
                print(f"Worker {worker_id} error: {e}")
                continue


class ParallelEmbeddingGenerator:
    """Generate embeddings in parallel across multiple GPUs."""

    def __init__(
        self,
        num_workers: int = WORKERS,
        sub_batch_size: int = EMBEDDING_SUB_BATCH_SIZE,
        content_model_name: str = EMBEDDING_MODEL,
        label_model_name: str = LABEL_EMBEDDING_MODEL,
        cuda_devices: str = CUDA_DEVICES,
        batch_size: int = 100,
        max_length_content: int = MAX_LENGTH_CONTENT,
        max_length_label: int = MAX_LENGTH_LABEL,
    ):
        self.num_workers = num_workers
        self.content_model_name = content_model_name
        self.label_model_name = label_model_name
        self.sub_batch_size = sub_batch_size
        self.max_length_content = max_length_content
        self.max_length_label = max_length_label
        self.batch_size = batch_size

        if cuda_devices and torch.cuda.is_available():
            self.devices = [int(d) for d in cuda_devices.split(",")]
        else:
            self.devices = [0]

        self.manager = Manager()
        self.input_queue = self.manager.Queue()
        self.output_queue = self.manager.Queue()

        ctx = get_context("spawn")
        self.workers = []
        for i in range(num_workers):
            device_id = self.devices[i % len(self.devices)]
            p = ctx.Process(
                target=_embedding_worker,
                args=(
                    i,
                    device_id,
                    content_model_name,
                    label_model_name,
                    sub_batch_size,
                    max_length_content,
                    max_length_label,
                    self.input_queue,
                    self.output_queue,
                ),
            )
            p.start()
            self.workers.append(p)

        print(f"Started {num_workers} embedding workers across {len(self.devices)} GPUs")

    def embed_batch(self, batch_id: int, contents: list[str], labels: list[str]):
        """Submit a batch for embedding."""
        self.input_queue.put((batch_id, contents, labels))

    def get_result(self, block: bool = True, timeout: float = 0.1):
        """Get result from output queue."""
        try:
            return self.output_queue.get(block=block, timeout=timeout)
        except Empty:
            return None

    def close(self):
        """Shutdown workers."""
        for _ in self.workers:
            self.input_queue.put(None)
        for w in self.workers:
            w.join()


class MilvusInserter:
    """Handle Milvus collection creation and batch insertion for CTD diseases."""

    def __init__(self, drop_existing: bool = False, batch_size: int = MILVUS_BATCH_SIZE):
        """Initialize Milvus connection and collections."""
        self.batch_size = batch_size

        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
        print(f"Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")

        self._drop_existing()

        self._ensure_collections()

        self.disease_entity_pending = []
        self.disease_hierarchy_pending = []
        self.gene_disease_pending = []
        self.disease_kegg_pathway_pending = []
        self.disease_reactome_pathway_pending = []

    def _drop_existing(self):
        """Drop existing CTD disease collections."""
        collections_to_drop = [
            "entities_Disease",
            "relationships_Disease_Disease",
            "relationships_Gene_Disease",
            "relationships_Disease_KEGG_Pathway",
            "relationships_Disease_Reactome_Pathway",
        ]

        for coll_name in collections_to_drop:
            if utility.has_collection(coll_name):
                utility.drop_collection(coll_name)
                print(f"Dropped existing Milvus collection: {coll_name}")

    def _ensure_collections(self):
        """Ensure Milvus collections exist."""
        self.disease_entities_collection_name = "entities_Disease"
        if utility.has_collection(self.disease_entities_collection_name):
            self.disease_entities_collection = Collection(self.disease_entities_collection_name)
            print(f"Using existing Milvus collection: {self.disease_entities_collection_name}")
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
            self.disease_entities_collection = Collection(self.disease_entities_collection_name, schema)
            print(f"Created Milvus collection: {self.disease_entities_collection_name}")

        self.disease_hierarchy_collection_name = "relationships_Disease_Disease"
        if utility.has_collection(self.disease_hierarchy_collection_name):
            self.disease_hierarchy_collection = Collection(self.disease_hierarchy_collection_name)
            print(f"Using existing Milvus collection: {self.disease_hierarchy_collection_name}")
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
            self.disease_hierarchy_collection = Collection(self.disease_hierarchy_collection_name, schema)
            print(f"Created Milvus collection: {self.disease_hierarchy_collection_name}")

        self.gene_disease_collection_name = "relationships_Gene_Disease"
        if utility.has_collection(self.gene_disease_collection_name):
            self.gene_disease_collection = Collection(self.gene_disease_collection_name)
            print(f"Using existing Milvus collection: {self.gene_disease_collection_name}")
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
            self.gene_disease_collection = Collection(self.gene_disease_collection_name, schema)
            print(f"Created Milvus collection: {self.gene_disease_collection_name}")

        self.disease_kegg_pathway_collection_name = "relationships_Disease_KEGG_Pathway"
        if utility.has_collection(self.disease_kegg_pathway_collection_name):
            self.disease_kegg_pathway_collection = Collection(self.disease_kegg_pathway_collection_name)
            print(f"Using existing Milvus collection: {self.disease_kegg_pathway_collection_name}")
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
            self.disease_kegg_pathway_collection = Collection(self.disease_kegg_pathway_collection_name, schema)
            print(f"Created Milvus collection: {self.disease_kegg_pathway_collection_name}")

        self.disease_reactome_pathway_collection_name = "relationships_Disease_Reactome_Pathway"
        if utility.has_collection(self.disease_reactome_pathway_collection_name):
            self.disease_reactome_pathway_collection = Collection(self.disease_reactome_pathway_collection_name)
            print(f"Using existing Milvus collection: {self.disease_reactome_pathway_collection_name}")
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
            self.disease_reactome_pathway_collection = Collection(self.disease_reactome_pathway_collection_name, schema)
            print(f"Created Milvus collection: {self.disease_reactome_pathway_collection_name}")

    def _compute_id(self, composite_key: str, prefix: str) -> str:
        """Compute ID from composite key."""
        hash_value = hashlib.md5(composite_key.encode()).hexdigest()
        return f"{prefix}{hash_value}"

    def insert_disease_entity_batch(
        self,
        disease_ids: list[str],
        disease_names: list[str],
        definitions: list[str],
        synonyms_list: list[str],
        slim_mappings_list: list[str],
        alt_ids_list: list[str],
        content_embeddings: np.ndarray,
        label_embeddings: np.ndarray,
    ):
        """Add disease entities to pending batch."""
        current_time = int(time.time())

        for disease_id, disease_name, definition, synonyms, slim_mappings, alt_ids, content_emb, label_emb in zip(
            disease_ids,
            disease_names,
            definitions,
            synonyms_list,
            slim_mappings_list,
            alt_ids_list,
            content_embeddings,
            label_embeddings,
            strict=False,
        ):
            entity_name = f"DISEASE:{disease_id}"
            pk_id = self._compute_id(entity_name, prefix="ent-")

            label = disease_name

            content = f"{disease_id}\n{disease_name}\n{definition}\nSynonyms: {synonyms}\nCategories: {slim_mappings}"

            record = {
                "id": pk_id,
                "vector": content_emb.tolist(),
                "label_embedding": label_emb.tolist(),
                "created_at": current_time,
                "entity_name": entity_name,
                "file_path": "ctd/CTD_diseases.csv.gz",
                "content": content,
                "description": content,
                "label": label,
                "disease_id": disease_id,
                "disease_name": disease_name,
                "definition": definition,
                "synonyms": synonyms,
                "alt_disease_ids": alt_ids,
                "slim_mappings": slim_mappings,
            }
            self.disease_entity_pending.append(record)

        if len(self.disease_entity_pending) >= self.batch_size:
            self._flush_disease_entities()

    def insert_disease_hierarchy_batch(
        self,
        child_ids: list[str],
        parent_ids: list[str],
        child_names: list[str],
        parent_names: list[str],
        content_embeddings: np.ndarray,
    ):
        """Add Disease->Disease hierarchical relationships to pending batch."""
        current_time = int(time.time())

        for child_id, parent_id, child_name, parent_name, content_emb in zip(
            child_ids, parent_ids, child_names, parent_names, content_embeddings, strict=False
        ):
            src_id = f"DISEASE:{child_id}"
            tgt_id = f"DISEASE:{parent_id}"
            composite_key = f"{src_id}{GRAPH_FIELD_SEP}{tgt_id}"
            pk_id = self._compute_id(composite_key, prefix="rel-")

            content = f"{child_name} is_a {parent_name}"

            record = {
                "id": pk_id,
                "vector": content_emb.tolist(),
                "created_at": current_time,
                "src_id": src_id,
                "tgt_id": tgt_id,
                "file_path": "ctd/CTD_diseases.csv.gz",
                "content": content,
                "description": f"{child_name} is a type of {parent_name}",
                "relationship_type": "is_a",
                "child_disease_id": child_id,
                "parent_disease_id": parent_id,
                "child_disease_name": child_name,
                "parent_disease_name": parent_name,
            }
            self.disease_hierarchy_pending.append(record)

        if len(self.disease_hierarchy_pending) >= self.batch_size:
            self._flush_disease_hierarchies()

    def insert_gene_disease_batch(
        self,
        gene_symbols: list[str],
        disease_ids: list[str],
        disease_names: list[str],
        evidence_list: list[str],
        omim_ids_list: list[str],
        pubmed_ids_list: list[str],
        content_embeddings: np.ndarray,
    ):
        """Add Gene->Disease relationships to pending batch."""
        current_time = int(time.time())

        for gene_symbol, disease_id, disease_name, evidence, omim_ids, pubmed_ids, content_emb in zip(
            gene_symbols,
            disease_ids,
            disease_names,
            evidence_list,
            omim_ids_list,
            pubmed_ids_list,
            content_embeddings,
            strict=False,
        ):
            src_id = f"GENE:{gene_symbol}"
            tgt_id = f"DISEASE:{disease_id}"
            composite_key = f"{src_id}{GRAPH_FIELD_SEP}{tgt_id}"
            pk_id = self._compute_id(composite_key, prefix="rel-")

            content = f"{gene_symbol} {evidence} {disease_name}"

            record = {
                "id": pk_id,
                "vector": content_emb.tolist(),
                "created_at": current_time,
                "src_id": src_id,
                "tgt_id": tgt_id,
                "file_path": "ctd/CTD_curated_genes_diseases.csv.gz",
                "content": content,
                "description": disease_name,
                "evidence": evidence,
                "omim_ids": omim_ids,
                "pubmed_ids": pubmed_ids,
                "gene_symbol": gene_symbol,
                "disease_id": disease_id,
                "disease_name": disease_name,
            }
            self.gene_disease_pending.append(record)

        if len(self.gene_disease_pending) >= self.batch_size:
            self._flush_gene_disease_relationships()

    def insert_disease_kegg_pathway_batch(
        self,
        disease_ids: list[str],
        pathway_ids: list[str],
        disease_names: list[str],
        pathway_names: list[str],
        inference_genes: list[str],
        content_embeddings: np.ndarray,
    ):
        """Add Disease->KEGG Pathway relationships to pending batch."""
        current_time = int(time.time())

        for disease_id, pathway_id, disease_name, pathway_name, inference_gene, content_emb in zip(
            disease_ids, pathway_ids, disease_names, pathway_names, inference_genes, content_embeddings, strict=False
        ):
            src_id = f"DISEASE:{disease_id}"
            tgt_id = pathway_id
            composite_key = f"{src_id}{GRAPH_FIELD_SEP}{tgt_id}"
            pk_id = self._compute_id(composite_key, prefix="rel-")

            content = f"{disease_name} associated_with {pathway_name}"

            record = {
                "id": pk_id,
                "vector": content_emb.tolist(),
                "created_at": current_time,
                "src_id": src_id,
                "tgt_id": tgt_id,
                "file_path": "ctd/CTD_diseases_pathways.csv.gz",
                "content": content,
                "description": pathway_name,
                "inference_gene_symbol": inference_gene,
                "disease_id": disease_id,
                "disease_name": disease_name,
                "pathway_id": pathway_id,
                "pathway_name": pathway_name,
            }
            self.disease_kegg_pathway_pending.append(record)

        if len(self.disease_kegg_pathway_pending) >= self.batch_size:
            self._flush_disease_kegg_pathway_relationships()

    def insert_disease_reactome_pathway_batch(
        self,
        disease_ids: list[str],
        pathway_ids: list[str],
        disease_names: list[str],
        pathway_names: list[str],
        inference_genes: list[str],
        content_embeddings: np.ndarray,
    ):
        """Add Disease->Reactome Pathway relationships to pending batch."""
        current_time = int(time.time())

        for disease_id, pathway_id, disease_name, pathway_name, inference_gene, content_emb in zip(
            disease_ids, pathway_ids, disease_names, pathway_names, inference_genes, content_embeddings, strict=False
        ):
            src_id = f"DISEASE:{disease_id}"
            tgt_id = pathway_id
            composite_key = f"{src_id}{GRAPH_FIELD_SEP}{tgt_id}"
            pk_id = self._compute_id(composite_key, prefix="rel-")

            content = f"{disease_name} associated_with {pathway_name}"

            record = {
                "id": pk_id,
                "vector": content_emb.tolist(),
                "created_at": current_time,
                "src_id": src_id,
                "tgt_id": tgt_id,
                "file_path": "ctd/CTD_diseases_pathways.csv.gz",
                "content": content,
                "description": pathway_name,
                "inference_gene_symbol": inference_gene,
                "disease_id": disease_id,
                "disease_name": disease_name,
                "pathway_id": pathway_id,
                "pathway_name": pathway_name,
            }
            self.disease_reactome_pathway_pending.append(record)

        if len(self.disease_reactome_pathway_pending) >= self.batch_size:
            self._flush_disease_reactome_pathway_relationships()

    def _flush_disease_entities(self):
        """Flush pending disease entity records to Milvus."""
        if self.disease_entity_pending:
            self.disease_entities_collection.insert(self.disease_entity_pending)
            self.disease_entity_pending = []

    def _flush_disease_hierarchies(self):
        """Flush pending disease hierarchy records to Milvus."""
        if self.disease_hierarchy_pending:
            self.disease_hierarchy_collection.insert(self.disease_hierarchy_pending)
            self.disease_hierarchy_pending = []

    def _flush_gene_disease_relationships(self):
        """Flush pending gene-disease relationship records to Milvus."""
        if self.gene_disease_pending:
            self.gene_disease_collection.insert(self.gene_disease_pending)
            self.gene_disease_pending = []

    def _flush_disease_kegg_pathway_relationships(self):
        """Flush pending disease-KEGG pathway relationship records to Milvus."""
        if self.disease_kegg_pathway_pending:
            self.disease_kegg_pathway_collection.insert(self.disease_kegg_pathway_pending)
            self.disease_kegg_pathway_pending = []

    def _flush_disease_reactome_pathway_relationships(self):
        """Flush pending disease-Reactome pathway relationship records to Milvus."""
        if self.disease_reactome_pathway_pending:
            self.disease_reactome_pathway_collection.insert(self.disease_reactome_pathway_pending)
            self.disease_reactome_pathway_pending = []

    def finalize(self):
        """Flush all pending records and create indexes."""
        print("\nFlushing remaining records...")
        self._flush_disease_entities()
        self._flush_disease_hierarchies()
        self._flush_gene_disease_relationships()
        self._flush_disease_kegg_pathway_relationships()
        self._flush_disease_reactome_pathway_relationships()

        print("\nCreating indexes...")
        index_params_vector = {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200},
        }
        index_params_label = {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 200}}

        if not self.disease_entities_collection.has_index():
            self.disease_entities_collection.create_index(field_name="vector", index_params=index_params_vector)
            print(f"Created HNSW index on {self.disease_entities_collection_name}.vector")
            self.disease_entities_collection.create_index(field_name="label_embedding", index_params=index_params_label)
            print(f"Created HNSW index on {self.disease_entities_collection_name}.label_embedding")

        for coll, coll_name in [
            (self.disease_hierarchy_collection, self.disease_hierarchy_collection_name),
            (self.gene_disease_collection, self.gene_disease_collection_name),
            (self.disease_kegg_pathway_collection, self.disease_kegg_pathway_collection_name),
            (self.disease_reactome_pathway_collection, self.disease_reactome_pathway_collection_name),
        ]:
            if not coll.has_index():
                coll.create_index(field_name="vector", index_params=index_params_vector)
                print(f"Created HNSW index on {coll_name}.vector")

        self.disease_entities_collection.load()
        self.disease_hierarchy_collection.load()
        self.gene_disease_collection.load()
        self.disease_kegg_pathway_collection.load()
        self.disease_reactome_pathway_collection.load()
        print("Loaded all collections into memory")

    def close(self):
        """Close Milvus connection."""
        connections.disconnect("default")


def process_ctd_diseases(args):
    """Main processing function."""
    start_time = time.time()

    print("=" * 60)
    print("CTD Disease Insertion")
    print("=" * 60)
    print("WARNING: Will delete all existing CTD Disease data for fresh insertion!")
    print("=" * 60)
    print(f"Configuration file: {CONFIG_FILE}")
    print(f"Diseases file: {args.diseases_file}")
    print(f"Gene-Disease file: {args.gene_disease_file}")
    print(f"Disease-Pathway file: {args.disease_pathway_file}")
    print(f"Batch size: {args.batch_size}")
    if args.filter_disease_category:
        print(f"Filter categories: {args.filter_disease_category}")
    if args.limit:
        print(f"Limit: {args.limit}")
    print(f"Workers: {WORKERS}")
    print()

    print("Step 1: Parsing CTD disease files...")
    parser = CTDDiseaseParser()

    disease_records = parser.parse_diseases(
        args.diseases_file, filter_category=args.filter_disease_category, limit=args.limit
    )

    disease_hierarchies = parser.extract_disease_hierarchies(disease_records)
    print()

    if len(disease_records) == 0:
        print("No diseases to process. Exiting.")
        return

    print("Step 2: Initializing components...")
    embedder = ParallelEmbeddingGenerator(batch_size=args.batch_size)
    milvus = MilvusInserter(drop_existing=args.drop_existing)
    metadata_tracker = InsertionMetadataTracker()

    if not args.skip_gene_disease:
        print("Loading gene mapper for validation...")
        gene_mapper = GeneIDMapper(GENE_INFO_FILE, filter_taxid=GENE_FILTER_TAXID if GENE_FILTER_TAXID else None)
        gene_mapper.build_mapping()
    print()

    print(f"Step 3: Processing {len(disease_records):,} disease entities...")

    disease_id_to_name = {rec.disease_id: rec.disease_name for rec in disease_records}
    valid_disease_ids = set(disease_id_to_name.keys())

    batch_id = 0
    next_batch_idx = 0
    total_batches = (len(disease_records) + args.batch_size - 1) // args.batch_size
    pending_batches = {}
    insertion_queue = []

    submitted_count = 0
    embedded_count = 0
    inserted_count = 0

    pbar_embed = tqdm(total=len(disease_records), desc="Embedding diseases", position=0, leave=True)
    pbar_insert = tqdm(total=len(disease_records), desc="Inserting to Milvus", position=1, leave=True)

    while submitted_count < total_batches or pending_batches or insertion_queue:
        while submitted_count < total_batches and len(pending_batches) + len(insertion_queue) < MAX_PENDING_BATCHES:
            start_idx = next_batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, len(disease_records))
            batch_diseases = disease_records[start_idx:end_idx]

            contents = []
            labels = []
            for rec in batch_diseases:
                synonyms_str = ", ".join(rec.synonyms) if rec.synonyms else ""
                categories_str = ", ".join(rec.slim_mappings) if rec.slim_mappings else ""

                content = f"{rec.disease_id}\n{rec.disease_name}\n{rec.definition}\nSynonyms: {synonyms_str}\nCategories: {categories_str}"
                label = rec.disease_name

                contents.append(content)
                labels.append(label)

            embedder.embed_batch(batch_id, contents, labels)
            pending_batches[batch_id] = batch_diseases

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
                    batch_diseases = pending_batches.pop(result_batch_id)

                    embedded_count += len(batch_diseases)
                    pbar_embed.update(len(batch_diseases))

                    insertion_queue.append((batch_diseases, content_embeddings, label_embeddings))

        if insertion_queue:
            batch_diseases, content_embeddings, label_embeddings = insertion_queue.pop(0)

            disease_ids = [rec.disease_id for rec in batch_diseases]
            disease_names = [rec.disease_name for rec in batch_diseases]
            definitions = [rec.definition for rec in batch_diseases]
            synonyms_list = [", ".join(rec.synonyms) if rec.synonyms else "" for rec in batch_diseases]
            slim_mappings_list = [", ".join(rec.slim_mappings) if rec.slim_mappings else "" for rec in batch_diseases]
            alt_ids_list = [", ".join(rec.alt_disease_ids) if rec.alt_disease_ids else "" for rec in batch_diseases]

            milvus.insert_disease_entity_batch(
                disease_ids,
                disease_names,
                definitions,
                synonyms_list,
                slim_mappings_list,
                alt_ids_list,
                content_embeddings,
                label_embeddings,
            )

            inserted_count += len(batch_diseases)
            pbar_insert.update(len(batch_diseases))

    pbar_embed.close()
    pbar_insert.close()
    print(f"Inserted {inserted_count:,} disease entities")

    num_disease_hierarchies = 0
    if disease_hierarchies:
        print(f"\nStep 4: Processing {len(disease_hierarchies):,} disease hierarchy relationships...")

        child_ids = [h[0] for h in disease_hierarchies]
        parent_ids = [h[1] for h in disease_hierarchies]
        child_names = [disease_id_to_name.get(c, c) for c in child_ids]
        parent_names = [disease_id_to_name.get(p, p) for p in parent_ids]

        batch_id = 0
        next_batch_idx = 0
        total_batches = (len(disease_hierarchies) + args.batch_size - 1) // args.batch_size
        pending_batches = {}
        insertion_queue = []

        submitted_count = 0
        embedded_count = 0
        inserted_count = 0

        pbar_embed = tqdm(total=len(disease_hierarchies), desc="Embedding hierarchies", position=0, leave=True)
        pbar_insert = tqdm(total=len(disease_hierarchies), desc="Inserting to Milvus", position=1, leave=True)

        while submitted_count < total_batches or pending_batches or insertion_queue:
            while submitted_count < total_batches and len(pending_batches) + len(insertion_queue) < MAX_PENDING_BATCHES:
                start_idx = next_batch_idx * args.batch_size
                end_idx = min(start_idx + args.batch_size, len(disease_hierarchies))

                batch_child_ids = child_ids[start_idx:end_idx]
                batch_parent_ids = parent_ids[start_idx:end_idx]
                batch_child_names = child_names[start_idx:end_idx]
                batch_parent_names = parent_names[start_idx:end_idx]

                contents = [
                    f"{child_name} is_a {parent_name}"
                    for child_name, parent_name in zip(batch_child_names, batch_parent_names, strict=False)
                ]
                labels = contents

                embedder.embed_batch(batch_id, contents, labels)
                pending_batches[batch_id] = (batch_child_ids, batch_parent_ids, batch_child_names, batch_parent_names)

                submitted_count += 1
                next_batch_idx += 1
                batch_id += 1

            if pending_batches:
                result = embedder.get_result(block=False)
                if result is not None:
                    result_batch_id, content_embeddings, label_embeddings = result
                    (batch_child_ids, batch_parent_ids, batch_child_names, batch_parent_names) = pending_batches.pop(
                        result_batch_id
                    )

                    embedded_count += len(batch_child_ids)
                    pbar_embed.update(len(batch_child_ids))

                    insertion_queue.append(
                        (batch_child_ids, batch_parent_ids, batch_child_names, batch_parent_names, content_embeddings)
                    )

            if insertion_queue:
                (batch_child_ids, batch_parent_ids, batch_child_names, batch_parent_names, content_embeddings) = (
                    insertion_queue.pop(0)
                )

                milvus.insert_disease_hierarchy_batch(
                    batch_child_ids, batch_parent_ids, batch_child_names, batch_parent_names, content_embeddings
                )

                inserted_count += len(batch_child_ids)
                pbar_insert.update(len(batch_child_ids))

        pbar_embed.close()
        pbar_insert.close()
        num_disease_hierarchies = inserted_count
        print(f"Inserted {inserted_count:,} disease hierarchy relationships")

    num_gene_disease_rels = 0
    if not args.skip_gene_disease:
        print("\nStep 5: Processing Gene-Disease relationships...")

        gene_disease_rels = []
        for rec in parser.parse_gene_disease_relationships(args.gene_disease_file):
            if rec.disease_id not in valid_disease_ids:
                continue

            gene_symbol = gene_mapper.get_symbol(rec.gene_id)
            if not gene_symbol:
                continue

            gene_disease_rels.append(rec)

        print(f"Found {len(gene_disease_rels):,} valid Gene-Disease relationships")

        if gene_disease_rels:
            batch_id = 0
            next_batch_idx = 0
            total_batches = (len(gene_disease_rels) + args.batch_size - 1) // args.batch_size
            pending_batches = {}
            insertion_queue = []

            submitted_count = 0
            embedded_count = 0
            inserted_count = 0

            pbar_embed = tqdm(total=len(gene_disease_rels), desc="Embedding gene-disease", position=0, leave=True)
            pbar_insert = tqdm(total=len(gene_disease_rels), desc="Inserting to Milvus", position=1, leave=True)

            while submitted_count < total_batches or pending_batches or insertion_queue:
                while (
                    submitted_count < total_batches
                    and len(pending_batches) + len(insertion_queue) < MAX_PENDING_BATCHES
                ):
                    start_idx = next_batch_idx * args.batch_size
                    end_idx = min(start_idx + args.batch_size, len(gene_disease_rels))
                    batch_rels = gene_disease_rels[start_idx:end_idx]

                    contents = []
                    labels = []
                    for rec in batch_rels:
                        content = f"{rec.gene_symbol} {rec.direct_evidence} {rec.disease_name}"
                        contents.append(content)
                        labels.append(content)

                    embedder.embed_batch(batch_id, contents, labels)
                    pending_batches[batch_id] = batch_rels

                    submitted_count += 1
                    next_batch_idx += 1
                    batch_id += 1

                if pending_batches:
                    result = embedder.get_result(block=False)
                    if result is not None:
                        result_batch_id, content_embeddings, label_embeddings = result
                        batch_rels = pending_batches.pop(result_batch_id)

                        embedded_count += len(batch_rels)
                        pbar_embed.update(len(batch_rels))

                        insertion_queue.append((batch_rels, content_embeddings))

                if insertion_queue:
                    batch_rels, content_embeddings = insertion_queue.pop(0)

                    gene_symbols = [rec.gene_symbol for rec in batch_rels]
                    disease_ids = [rec.disease_id for rec in batch_rels]
                    disease_names = [rec.disease_name for rec in batch_rels]
                    evidence_list = [rec.direct_evidence for rec in batch_rels]
                    omim_ids_list = [rec.omim_ids for rec in batch_rels]
                    pubmed_ids_list = [rec.pubmed_ids for rec in batch_rels]

                    milvus.insert_gene_disease_batch(
                        gene_symbols,
                        disease_ids,
                        disease_names,
                        evidence_list,
                        omim_ids_list,
                        pubmed_ids_list,
                        content_embeddings,
                    )

                    inserted_count += len(batch_rels)
                    pbar_insert.update(len(batch_rels))

            pbar_embed.close()
            pbar_insert.close()
            num_gene_disease_rels = inserted_count
            print(f"Inserted {inserted_count:,} Gene-Disease relationships")

    num_disease_kegg_pathway_rels = 0
    if not args.skip_kegg_pathway:
        print("\nStep 6: Processing Disease-KEGG Pathway relationships...")

        kegg_pathway_rels = []
        for rec in parser.parse_disease_pathway_relationships(args.disease_pathway_file):
            if rec.disease_id not in valid_disease_ids:
                continue

            if not rec.pathway_id.startswith("KEGG:"):
                continue

            kegg_pathway_rels.append(rec)

        print(f"Found {len(kegg_pathway_rels):,} Disease-KEGG Pathway relationships")

        if kegg_pathway_rels:
            batch_id = 0
            next_batch_idx = 0
            total_batches = (len(kegg_pathway_rels) + args.batch_size - 1) // args.batch_size
            pending_batches = {}
            insertion_queue = []

            submitted_count = 0
            embedded_count = 0
            inserted_count = 0

            pbar_embed = tqdm(total=len(kegg_pathway_rels), desc="Embedding KEGG pathway", position=0, leave=True)
            pbar_insert = tqdm(total=len(kegg_pathway_rels), desc="Inserting to Milvus", position=1, leave=True)

            while submitted_count < total_batches or pending_batches or insertion_queue:
                while (
                    submitted_count < total_batches
                    and len(pending_batches) + len(insertion_queue) < MAX_PENDING_BATCHES
                ):
                    start_idx = next_batch_idx * args.batch_size
                    end_idx = min(start_idx + args.batch_size, len(kegg_pathway_rels))
                    batch_rels = kegg_pathway_rels[start_idx:end_idx]

                    contents = []
                    labels = []
                    for rec in batch_rels:
                        content = f"{rec.disease_name} associated_with {rec.pathway_name}"
                        contents.append(content)
                        labels.append(content)

                    embedder.embed_batch(batch_id, contents, labels)
                    pending_batches[batch_id] = batch_rels

                    submitted_count += 1
                    next_batch_idx += 1
                    batch_id += 1

                if pending_batches:
                    result = embedder.get_result(block=False)
                    if result is not None:
                        result_batch_id, content_embeddings, label_embeddings = result
                        batch_rels = pending_batches.pop(result_batch_id)

                        embedded_count += len(batch_rels)
                        pbar_embed.update(len(batch_rels))

                        insertion_queue.append((batch_rels, content_embeddings))

                if insertion_queue:
                    batch_rels, content_embeddings = insertion_queue.pop(0)

                    disease_ids = [rec.disease_id for rec in batch_rels]
                    pathway_ids = [rec.pathway_id for rec in batch_rels]
                    disease_names = [rec.disease_name for rec in batch_rels]
                    pathway_names = [rec.pathway_name for rec in batch_rels]
                    inference_genes = [rec.inference_gene_symbol for rec in batch_rels]

                    milvus.insert_disease_kegg_pathway_batch(
                        disease_ids, pathway_ids, disease_names, pathway_names, inference_genes, content_embeddings
                    )

                    inserted_count += len(batch_rels)
                    pbar_insert.update(len(batch_rels))

            pbar_embed.close()
            pbar_insert.close()
            num_disease_kegg_pathway_rels = inserted_count
            print(f"Inserted {inserted_count:,} Disease-KEGG Pathway relationships")

    num_disease_reactome_pathway_rels = 0
    if not args.skip_reactome_pathway:
        print("\nStep 7: Processing Disease-Reactome Pathway relationships...")

        reactome_pathway_rels = []
        for rec in parser.parse_disease_pathway_relationships(args.disease_pathway_file):
            if rec.disease_id not in valid_disease_ids:
                continue

            if not rec.pathway_id.startswith("REACT:"):
                continue

            reactome_pathway_rels.append(rec)

        print(f"Found {len(reactome_pathway_rels):,} Disease-Reactome Pathway relationships")

        if reactome_pathway_rels:
            batch_id = 0
            next_batch_idx = 0
            total_batches = (len(reactome_pathway_rels) + args.batch_size - 1) // args.batch_size
            pending_batches = {}
            insertion_queue = []

            submitted_count = 0
            embedded_count = 0
            inserted_count = 0

            pbar_embed = tqdm(
                total=len(reactome_pathway_rels), desc="Embedding Reactome pathway", position=0, leave=True
            )
            pbar_insert = tqdm(total=len(reactome_pathway_rels), desc="Inserting to Milvus", position=1, leave=True)

            while submitted_count < total_batches or pending_batches or insertion_queue:
                while (
                    submitted_count < total_batches
                    and len(pending_batches) + len(insertion_queue) < MAX_PENDING_BATCHES
                ):
                    start_idx = next_batch_idx * args.batch_size
                    end_idx = min(start_idx + args.batch_size, len(reactome_pathway_rels))
                    batch_rels = reactome_pathway_rels[start_idx:end_idx]

                    contents = []
                    labels = []
                    for rec in batch_rels:
                        content = f"{rec.disease_name} associated_with {rec.pathway_name}"
                        contents.append(content)
                        labels.append(content)

                    embedder.embed_batch(batch_id, contents, labels)
                    pending_batches[batch_id] = batch_rels

                    submitted_count += 1
                    next_batch_idx += 1
                    batch_id += 1

                if pending_batches:
                    result = embedder.get_result(block=False)
                    if result is not None:
                        result_batch_id, content_embeddings, label_embeddings = result
                        batch_rels = pending_batches.pop(result_batch_id)

                        embedded_count += len(batch_rels)
                        pbar_embed.update(len(batch_rels))

                        insertion_queue.append((batch_rels, content_embeddings))

                if insertion_queue:
                    batch_rels, content_embeddings = insertion_queue.pop(0)

                    disease_ids = [rec.disease_id for rec in batch_rels]
                    pathway_ids = [rec.pathway_id for rec in batch_rels]
                    disease_names = [rec.disease_name for rec in batch_rels]
                    pathway_names = [rec.pathway_name for rec in batch_rels]
                    inference_genes = [rec.inference_gene_symbol for rec in batch_rels]

                    milvus.insert_disease_reactome_pathway_batch(
                        disease_ids, pathway_ids, disease_names, pathway_names, inference_genes, content_embeddings
                    )

                    inserted_count += len(batch_rels)
                    pbar_insert.update(len(batch_rels))

            pbar_embed.close()
            pbar_insert.close()
            num_disease_reactome_pathway_rels = inserted_count
            print(f"Inserted {inserted_count:,} Disease-Reactome Pathway relationships")

    print("\nFinalizing insertion...")
    milvus.finalize()
    embedder.close()
    milvus.close()

    duration = time.time() - start_time
    metadata_tracker.record_insertion(
        num_diseases=len(disease_records),
        num_disease_hierarchies=num_disease_hierarchies,
        num_gene_disease_rels=num_gene_disease_rels,
        num_disease_kegg_pathway_rels=num_disease_kegg_pathway_rels,
        num_disease_reactome_pathway_rels=num_disease_reactome_pathway_rels,
        category_filter=args.filter_disease_category,
        batch_size=args.batch_size,
        workers=WORKERS,
        embedding_model=EMBEDDING_MODEL,
        milvus_batch_size=MILVUS_BATCH_SIZE,
        duration_seconds=duration,
    )

    print(f"\nCompleted in {duration:.2f} seconds")
    print(f"Inserted {len(disease_records):,} disease entities")
    print(f"Inserted {num_disease_hierarchies:,} disease hierarchy relationships")
    print(f"Inserted {num_gene_disease_rels:,} Gene-Disease relationships")
    print(f"Inserted {num_disease_kegg_pathway_rels:,} Disease-KEGG Pathway relationships")
    print(f"Inserted {num_disease_reactome_pathway_rels:,} Disease-Reactome Pathway relationships")


def main():
    parser = argparse.ArgumentParser(
        description="Insert CTD Diseases into Milvus",
        epilog="Configuration loaded from config/db_init.conf. Command-line arguments override config.",
    )

    parser.add_argument("--config", default=str(CONFIG_FILE), help="Path to config file")
    parser.add_argument("--diseases-file", default=CTD_DISEASE_FILE, help="Path to CTD_diseases.csv.gz")
    parser.add_argument(
        "--gene-disease-file", default=CTD_DISEASE_GENE_FILE, help="Path to CTD_curated_genes_diseases.csv.gz"
    )
    parser.add_argument(
        "--disease-pathway-file", default=CTD_DISEASE_PATHWAY_FILE, help="Path to CTD_diseases_pathways.csv.gz"
    )
    parser.add_argument("--filter-disease-category", nargs="+", help="Filter by disease categories")
    parser.add_argument("--batch-size", type=int, default=CTD_DISEASE_BATCH_SIZE, help="Batch size for processing")
    parser.add_argument("--limit", type=int, default=CTD_DISEASE_LIMIT, help="Limit diseases (0 = no limit)")
    parser.add_argument("--skip-gene-disease", action="store_true", help="Skip gene-disease relationships")
    parser.add_argument("--skip-kegg-pathway", action="store_true", help="Skip disease-KEGG pathway relationships")
    parser.add_argument(
        "--skip-reactome-pathway", action="store_true", help="Skip disease-Reactome pathway relationships"
    )
    parser.add_argument("--drop-existing", action="store_true", help="Drop existing collections (always true)")

    args = parser.parse_args()

    try:
        process_ctd_diseases(args)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
