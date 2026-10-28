#!/usr/bin/env python3
"""
Insert CTD Chemicals into Milvus.

This script:
1. Parses CTD chemical files
2. Creates Chemical entities in Milvus
3. Creates Chemical->Chemical hierarchical relationships
4. Creates Chemical->Gene interaction relationships
5. Creates Chemical->Disease relationships
6. Creates Chemical->KEGG Pathway enrichment relationships
7. Creates Chemical->Reactome Pathway enrichment relationships
8. Creates Chemical->GO term enrichment relationships

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
from ctd_chemical_parser import CTDChemicalParser
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
CTD_CHEMICAL_METADATA_FILE = CONFIG.get("CTD_CHEMICAL_METADATA_FILE", "ctd_chemicals.csv")

CTD_CHEMICAL_FILE = CONFIG.get("CTD_CHEMICAL_FILE", f"{DATA_DIR}/resources/ctd/CTD_chemicals.csv.gz")
CTD_CHEMICAL_GENE_FILE = CONFIG.get("CTD_CHEMICAL_GENE_FILE", f"{DATA_DIR}/resources/ctd/CTD_chem_gene_ixns.csv.gz")
CTD_CHEMICAL_DISEASE_FILE = CONFIG.get(
    "CTD_CHEMICAL_DISEASE_FILE", f"{DATA_DIR}/resources/ctd/CTD_curated_chemicals_diseases.csv.gz"
)
CTD_CHEMICAL_PATHWAY_FILE = CONFIG.get(
    "CTD_CHEMICAL_PATHWAY_FILE", f"{DATA_DIR}/resources/ctd/CTD_chem_pathways_enriched.csv.gz"
)
CTD_CHEMICAL_GO_FILE = CONFIG.get("CTD_CHEMICAL_GO_FILE", f"{DATA_DIR}/resources/ctd/CTD_chem_go_enriched.csv.gz")
CTD_CHEMICAL_BATCH_SIZE = int(CONFIG.get("CTD_CHEMICAL_BATCH_SIZE", "1000"))
CTD_CHEMICAL_LIMIT = int(CONFIG.get("CTD_CHEMICAL_LIMIT", "0"))

CTD_CHEMICAL_FILTER_ORGANISM = [
    x.strip() for x in CONFIG.get("CTD_CHEMICAL_FILTER_ORGANISM", "").split(",") if x.strip()
]
CTD_CHEMICAL_FILTER_GO_ONTOLOGY = [
    x.strip() for x in CONFIG.get("CTD_CHEMICAL_FILTER_GO_ONTOLOGY", "").split(",") if x.strip()
]

GENE_INFO_FILE = CONFIG.get("GENE_INFO_FILE", f"{DATA_DIR}/resources/genes/gene_info.gz")
GENE_FILTER_TAXID = [x.strip() for x in CONFIG.get("GENE_FILTER_TAXID", "").split(",") if x.strip()]

HF_TOKEN = CONFIG.get("HF_TOKEN", "")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN


class InsertionMetadataTracker:
    """Track metadata about CTD chemical insertions."""

    def __init__(self):
        self.metadata_dir = Path(METADATA_DIR)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.metadata_dir / CTD_CHEMICAL_METADATA_FILE

        if not self.metadata_path.exists():
            with open(self.metadata_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "num_chemicals",
                        "num_chemical_hierarchies",
                        "num_chemical_gene_rels",
                        "num_chemical_disease_rels",
                        "num_chemical_kegg_pathway_rels",
                        "num_chemical_reactome_pathway_rels",
                        "num_chemical_go_rels",
                        "organism_filter",
                        "go_ontology_filter",
                        "batch_size",
                        "workers",
                        "embedding_model",
                        "milvus_batch_size",
                        "duration_seconds",
                    ]
                )

    def record_insertion(
        self,
        num_chemicals: int,
        num_chemical_hierarchies: int,
        num_chemical_gene_rels: int,
        num_chemical_disease_rels: int,
        num_chemical_kegg_pathway_rels: int,
        num_chemical_reactome_pathway_rels: int,
        num_chemical_go_rels: int,
        organism_filter: list[str] | None,
        go_ontology_filter: list[str] | None,
        batch_size: int,
        workers: int,
        embedding_model: str,
        milvus_batch_size: int,
        duration_seconds: float,
    ):
        """Record insertion metadata."""
        timestamp = datetime.now().isoformat()
        organism_str = ",".join(organism_filter) if organism_filter else "ALL"
        go_ontology_str = ",".join(go_ontology_filter) if go_ontology_filter else "ALL"

        with open(self.metadata_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    timestamp,
                    num_chemicals,
                    num_chemical_hierarchies,
                    num_chemical_gene_rels,
                    num_chemical_disease_rels,
                    num_chemical_kegg_pathway_rels,
                    num_chemical_reactome_pathway_rels,
                    num_chemical_go_rels,
                    organism_str,
                    go_ontology_str,
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
            except (BrokenPipeError, EOFError, ConnectionResetError) as e:
                print(f"Worker {worker_id} connection lost ({type(e).__name__}), shutting down")
                break
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
    """Handle Milvus collection creation and batch insertion for CTD chemicals."""

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
        self.chemical_disease_pending = []
        self.disease_kegg_pathway_pending = []
        self.disease_reactome_pathway_pending = []
        self.chemical_go_pending = []

    def _drop_existing(self):
        """Drop existing CTD chemical collections."""
        collections_to_drop = [
            "entities_Chemical",
            "relationships_Chemical_Chemical",
            "relationships_Chemical_Gene",
            "relationships_Chemical_Disease",
            "relationships_Chemical_KEGG_Pathway",
            "relationships_Chemical_Reactome_Pathway",
            "relationships_Chemical_GO",
        ]

        for coll_name in collections_to_drop:
            if utility.has_collection(coll_name):
                utility.drop_collection(coll_name)
                print(f"Dropped existing Milvus collection: {coll_name}")

    def _ensure_collections(self):
        """Ensure Milvus collections exist."""
        self.disease_entities_collection_name = "entities_Chemical"
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

        self.disease_hierarchy_collection_name = "relationships_Chemical_Chemical"
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

        self.gene_disease_collection_name = "relationships_Chemical_Gene"
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

        self.disease_kegg_pathway_collection_name = "relationships_Chemical_KEGG_Pathway"
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

        self.disease_reactome_pathway_collection_name = "relationships_Chemical_Reactome_Pathway"
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

        self.chemical_disease_collection_name = "relationships_Chemical_Disease"
        if utility.has_collection(self.chemical_disease_collection_name):
            self.chemical_disease_collection = Collection(self.chemical_disease_collection_name)
            print(f"Using existing Milvus collection: {self.chemical_disease_collection_name}")
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
            self.chemical_disease_collection = Collection(self.chemical_disease_collection_name, schema)
            print(f"Created Milvus collection: {self.chemical_disease_collection_name}")

        self.chemical_go_collection_name = "relationships_Chemical_GO"
        if utility.has_collection(self.chemical_go_collection_name):
            self.chemical_go_collection = Collection(self.chemical_go_collection_name)
            print(f"Using existing Milvus collection: {self.chemical_go_collection_name}")
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
            self.chemical_go_collection = Collection(self.chemical_go_collection_name, schema)
            print(f"Created Milvus collection: {self.chemical_go_collection_name}")

    def _compute_id(self, composite_key: str, prefix: str) -> str:
        """Compute ID from composite key."""
        hash_value = hashlib.md5(composite_key.encode()).hexdigest()
        return f"{prefix}{hash_value}"

    def insert_disease_entity_batch(
        self,
        chemical_ids: list[str],
        chemical_names: list[str],
        definitions: list[str],
        synonyms_list: list[str],
        tree_numbers_list: list[str],
        cas_rn_list: list[str],
        content_embeddings: np.ndarray,
        label_embeddings: np.ndarray,
    ):
        """Add chemical entities to pending batch."""
        current_time = int(time.time())

        for chemical_id, chemical_name, definition, synonyms, tree_numbers, cas_rn, content_emb, label_emb in zip(
            chemical_ids,
            chemical_names,
            definitions,
            synonyms_list,
            tree_numbers_list,
            cas_rn_list,
            content_embeddings,
            label_embeddings,
            strict=False,
        ):
            entity_name = f"CHEM:{chemical_id}"
            pk_id = self._compute_id(entity_name, prefix="ent-")

            label = chemical_name

            record = {
                "id": pk_id,
                "vector": content_emb.tolist(),
                "label_embedding": label_emb.tolist(),
                "created_at": current_time,
                "entity_name": entity_name,
                "file_path": "ctd/CTD_chemicals.csv.gz",
                "content": f"{chemical_id}\n{chemical_name}\n{definition}\nSynonyms: {synonyms}",
                "label": label,
                "chemical_id": chemical_id,
                "chemical_name": chemical_name,
                "definition": definition,
                "synonyms": synonyms,
                "cas_rn": cas_rn,
                "tree_numbers": tree_numbers,
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
            src_id = f"CHEM:{child_id}"
            tgt_id = f"CHEM:{parent_id}"
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
                "child_chemical_id": child_id,
                "parent_chemical_id": parent_id,
                "child_chemical_name": child_name,
                "parent_chemical_name": parent_name,
            }
            self.disease_hierarchy_pending.append(record)

        if len(self.disease_hierarchy_pending) >= self.batch_size:
            self._flush_disease_hierarchies()

    def insert_gene_disease_batch(
        self,
        gene_symbols: list[str],
        chemical_ids: list[str],
        chemical_names: list[str],
        interaction_list: list[str],
        interaction_actions_list: list[str],
        pubmed_ids_list: list[str],
        content_embeddings: np.ndarray,
    ):
        """Add Chemical->Gene interaction relationships to pending batch."""
        current_time = int(time.time())

        for gene_symbol, chemical_id, chemical_name, interaction, interaction_actions, pubmed_ids, content_emb in zip(
            gene_symbols,
            chemical_ids,
            chemical_names,
            interaction_list,
            interaction_actions_list,
            pubmed_ids_list,
            content_embeddings,
            strict=False,
        ):
            src_id = f"CHEM:{chemical_id}"
            tgt_id = f"GENE:{gene_symbol}"
            composite_key = f"{src_id}{GRAPH_FIELD_SEP}{tgt_id}"
            pk_id = self._compute_id(composite_key, prefix="rel-")

            content = f"{gene_symbol} interacts_with {chemical_name}: {interaction}"

            record = {
                "id": pk_id,
                "vector": content_emb.tolist(),
                "created_at": current_time,
                "src_id": src_id,
                "tgt_id": tgt_id,
                "file_path": "ctd/CTD_chem_gene_ixns.csv.gz",
                "content": content,
                "description": interaction,
                "interaction": interaction,
                "interaction_actions": interaction_actions,
                "pubmed_ids": pubmed_ids,
                "gene_symbol": gene_symbol,
                "chemical_id": chemical_id,
                "chemical_name": chemical_name,
            }
            self.gene_disease_pending.append(record)

        if len(self.gene_disease_pending) >= self.batch_size:
            self._flush_gene_disease_relationships()

    def insert_disease_kegg_pathway_batch(
        self,
        chemical_ids: list[str],
        pathway_ids: list[str],
        chemical_names: list[str],
        pathway_names: list[str],
        p_values: list[str],
        corrected_p_values: list[str],
        target_match_qtys: list[str],
        target_total_qtys: list[str],
        content_embeddings: np.ndarray,
    ):
        """Add Chemical->KEGG Pathway enrichment relationships to pending batch."""
        current_time = int(time.time())

        for (
            chemical_id,
            pathway_id,
            chemical_name,
            pathway_name,
            p_value,
            corrected_p_value,
            match_qty,
            total_qty,
            content_emb,
        ) in zip(
            chemical_ids,
            pathway_ids,
            chemical_names,
            pathway_names,
            p_values,
            corrected_p_values,
            target_match_qtys,
            target_total_qtys,
            content_embeddings,
            strict=False,
        ):
            src_id = f"CHEM:{chemical_id}"
            tgt_id = pathway_id
            composite_key = f"{src_id}{GRAPH_FIELD_SEP}{tgt_id}"
            pk_id = self._compute_id(composite_key, prefix="rel-")

            content = f"{chemical_name} enriched_in {pathway_name} (p-value: {p_value}, corrected: {corrected_p_value})"

            record = {
                "id": pk_id,
                "vector": content_emb.tolist(),
                "created_at": current_time,
                "src_id": src_id,
                "tgt_id": tgt_id,
                "file_path": "ctd/CTD_chem_pathways_enriched.csv.gz",
                "content": content,
                "description": pathway_name,
                "p_value": p_value,
                "corrected_p_value": corrected_p_value,
                "target_match_qty": match_qty,
                "target_total_qty": total_qty,
                "chemical_id": chemical_id,
                "chemical_name": chemical_name,
                "pathway_id": pathway_id,
                "pathway_name": pathway_name,
            }
            self.disease_kegg_pathway_pending.append(record)

        if len(self.disease_kegg_pathway_pending) >= self.batch_size:
            self._flush_disease_kegg_pathway_relationships()

    def insert_disease_reactome_pathway_batch(
        self,
        chemical_ids: list[str],
        pathway_ids: list[str],
        chemical_names: list[str],
        pathway_names: list[str],
        p_values: list[str],
        corrected_p_values: list[str],
        target_match_qtys: list[str],
        target_total_qtys: list[str],
        content_embeddings: np.ndarray,
    ):
        """Add Chemical->Reactome Pathway enrichment relationships to pending batch."""
        current_time = int(time.time())

        for (
            chemical_id,
            pathway_id,
            chemical_name,
            pathway_name,
            p_value,
            corrected_p_value,
            match_qty,
            total_qty,
            content_emb,
        ) in zip(
            chemical_ids,
            pathway_ids,
            chemical_names,
            pathway_names,
            p_values,
            corrected_p_values,
            target_match_qtys,
            target_total_qtys,
            content_embeddings,
            strict=False,
        ):
            src_id = f"CHEM:{chemical_id}"
            tgt_id = pathway_id
            composite_key = f"{src_id}{GRAPH_FIELD_SEP}{tgt_id}"
            pk_id = self._compute_id(composite_key, prefix="rel-")

            content = f"{chemical_name} enriched_in {pathway_name} (p-value: {p_value}, corrected: {corrected_p_value})"

            record = {
                "id": pk_id,
                "vector": content_emb.tolist(),
                "created_at": current_time,
                "src_id": src_id,
                "tgt_id": tgt_id,
                "file_path": "ctd/CTD_chem_pathways_enriched.csv.gz",
                "content": content,
                "description": pathway_name,
                "p_value": p_value,
                "corrected_p_value": corrected_p_value,
                "target_match_qty": match_qty,
                "target_total_qty": total_qty,
                "chemical_id": chemical_id,
                "chemical_name": chemical_name,
                "pathway_id": pathway_id,
                "pathway_name": pathway_name,
            }
            self.disease_reactome_pathway_pending.append(record)

        if len(self.disease_reactome_pathway_pending) >= self.batch_size:
            self._flush_disease_reactome_pathway_relationships()

    def insert_chemical_disease_batch(
        self,
        chemical_ids: list[str],
        disease_ids: list[str],
        chemical_names: list[str],
        disease_names: list[str],
        direct_evidence_list: list[str],
        pubmed_ids_list: list[str],
        content_embeddings: np.ndarray,
    ):
        """Add Chemical->Disease relationships to pending batch."""
        current_time = int(time.time())

        for chemical_id, disease_id, chemical_name, disease_name, direct_evidence, pubmed_ids, content_emb in zip(
            chemical_ids,
            disease_ids,
            chemical_names,
            disease_names,
            direct_evidence_list,
            pubmed_ids_list,
            content_embeddings,
            strict=False,
        ):
            src_id = f"CHEM:{chemical_id}"
            tgt_id = f"DISEASE:{disease_id}"
            composite_key = f"{src_id}{GRAPH_FIELD_SEP}{tgt_id}"
            pk_id = self._compute_id(composite_key, prefix="rel-")

            content = f"{chemical_name} associated_with {disease_name} ({direct_evidence})"

            record = {
                "id": pk_id,
                "vector": content_emb.tolist(),
                "created_at": current_time,
                "src_id": src_id,
                "tgt_id": tgt_id,
                "file_path": "ctd/CTD_curated_chemicals_diseases.csv.gz",
                "content": content,
                "description": disease_name,
                "direct_evidence": direct_evidence,
                "pubmed_ids": pubmed_ids,
                "chemical_id": chemical_id,
                "chemical_name": chemical_name,
                "disease_id": disease_id,
                "disease_name": disease_name,
            }
            self.chemical_disease_pending.append(record)

        if len(self.chemical_disease_pending) >= self.batch_size:
            self._flush_chemical_disease_relationships()

    def insert_chemical_go_batch(
        self,
        chemical_ids: list[str],
        go_term_ids: list[str],
        chemical_names: list[str],
        go_term_names: list[str],
        ontologies: list[str],
        highest_go_levels: list[str],
        p_values: list[str],
        corrected_p_values: list[str],
        target_match_qtys: list[str],
        target_total_qtys: list[str],
        content_embeddings: np.ndarray,
    ):
        """Add Chemical->GO enrichment relationships to pending batch."""
        current_time = int(time.time())

        for (
            chemical_id,
            go_term_id,
            chemical_name,
            go_term_name,
            ontology,
            go_level,
            p_value,
            corrected_p_value,
            match_qty,
            total_qty,
            content_emb,
        ) in zip(
            chemical_ids,
            go_term_ids,
            chemical_names,
            go_term_names,
            ontologies,
            highest_go_levels,
            p_values,
            corrected_p_values,
            target_match_qtys,
            target_total_qtys,
            content_embeddings,
            strict=False,
        ):
            src_id = f"CHEM:{chemical_id}"
            tgt_id = go_term_id
            composite_key = f"{src_id}{GRAPH_FIELD_SEP}{tgt_id}"
            pk_id = self._compute_id(composite_key, prefix="rel-")

            content = f"{chemical_name} enriched_in {go_term_name} ({ontology}, p-value: {p_value}, corrected: {corrected_p_value})"

            record = {
                "id": pk_id,
                "vector": content_emb.tolist(),
                "created_at": current_time,
                "src_id": src_id,
                "tgt_id": tgt_id,
                "file_path": "ctd/CTD_chem_go_enriched.csv.gz",
                "content": content,
                "description": go_term_name,
                "ontology": ontology,
                "highest_go_level": go_level,
                "p_value": p_value,
                "corrected_p_value": corrected_p_value,
                "target_match_qty": match_qty,
                "target_total_qty": total_qty,
                "chemical_id": chemical_id,
                "chemical_name": chemical_name,
                "go_term_id": go_term_id,
                "go_term_name": go_term_name,
            }
            self.chemical_go_pending.append(record)

        if len(self.chemical_go_pending) >= self.batch_size:
            self._flush_chemical_go_relationships()

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

    def _flush_chemical_disease_relationships(self):
        """Flush pending chemical-disease relationship records to Milvus."""
        if self.chemical_disease_pending:
            self.chemical_disease_collection.insert(self.chemical_disease_pending)
            self.chemical_disease_pending = []

    def _flush_chemical_go_relationships(self):
        """Flush pending chemical-GO relationship records to Milvus."""
        if self.chemical_go_pending:
            self.chemical_go_collection.insert(self.chemical_go_pending)
            self.chemical_go_pending = []

    def finalize(self):
        """Flush all pending records and create indexes."""
        print("\nFlushing remaining records...")
        self._flush_disease_entities()
        self._flush_disease_hierarchies()
        self._flush_gene_disease_relationships()
        self._flush_chemical_disease_relationships()
        self._flush_disease_kegg_pathway_relationships()
        self._flush_disease_reactome_pathway_relationships()
        self._flush_chemical_go_relationships()

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
            (self.chemical_disease_collection, self.chemical_disease_collection_name),
            (self.disease_kegg_pathway_collection, self.disease_kegg_pathway_collection_name),
            (self.disease_reactome_pathway_collection, self.disease_reactome_pathway_collection_name),
            (self.chemical_go_collection, self.chemical_go_collection_name),
        ]:
            if not coll.has_index():
                coll.create_index(field_name="vector", index_params=index_params_vector)
                print(f"Created HNSW index on {coll_name}.vector")

        self.disease_entities_collection.load()
        self.disease_hierarchy_collection.load()
        self.gene_disease_collection.load()
        self.chemical_disease_collection.load()
        self.disease_kegg_pathway_collection.load()
        self.disease_reactome_pathway_collection.load()
        self.chemical_go_collection.load()
        print("Loaded all collections into memory")

    def close(self):
        """Close Milvus connection."""
        connections.disconnect("default")


def process_ctd_diseases(args):
    """Main processing function."""
    start_time = time.time()

    print("=" * 60)
    print("CTD Chemical Insertion")
    print("=" * 60)
    print("WARNING: Will delete all existing CTD Chemical data for fresh insertion!")
    print("=" * 60)
    print(f"Configuration file: {CONFIG_FILE}")
    print(f"Chemicals file: {args.chemicals_file}")
    print(f"Chemical-Gene interactions file: {args.gene_interaction_file}")
    print(f"Chemical-Disease file: {args.disease_file}")
    print(f"Chemical-Pathway file: {args.pathway_file}")
    print(f"Chemical-GO file: {args.go_file}")
    print(f"Batch size: {args.batch_size}")
    if hasattr(args, "filter_organism") and args.filter_organism:
        print(f"Filter organisms: {args.filter_organism}")
    if hasattr(args, "filter_go_ontology") and args.filter_go_ontology:
        print(f"Filter GO ontologies: {args.filter_go_ontology}")
    if args.limit:
        print(f"Limit: {args.limit}")
    print(f"Workers: {WORKERS}")
    print()

    print("Step 1: Parsing CTD chemical files...")
    parser = CTDChemicalParser()

    chemical_records = parser.parse_chemicals(args.chemicals_file, limit=args.limit)

    chemical_hierarchies = parser.extract_chemical_hierarchies(chemical_records)
    print()

    if len(chemical_records) == 0:
        print("No chemicals to process. Exiting.")
        return

    print("Step 2: Initializing components...")
    embedder = ParallelEmbeddingGenerator(batch_size=args.batch_size)
    milvus = MilvusInserter(drop_existing=args.drop_existing)
    metadata_tracker = InsertionMetadataTracker()

    if not args.skip_gene_interaction:
        print("Loading gene mapper for gene ID to symbol mapping...")
        gene_mapper = GeneIDMapper(GENE_INFO_FILE, filter_taxid=GENE_FILTER_TAXID if GENE_FILTER_TAXID else None)
        gene_mapper.build_mapping()
    print()

    print(f"Step 3: Processing {len(chemical_records):,} chemical entities...")

    chemical_id_to_name = {rec.chemical_id: rec.chemical_name for rec in chemical_records}
    valid_chemical_ids = set(chemical_id_to_name.keys())

    batch_id = 0
    next_batch_idx = 0
    total_batches = (len(chemical_records) + args.batch_size - 1) // args.batch_size
    pending_batches = {}
    insertion_queue = []

    submitted_count = 0
    embedded_count = 0
    inserted_count = 0

    pbar_embed = tqdm(total=len(chemical_records), desc="Embedding diseases", position=0, leave=True)
    pbar_insert = tqdm(total=len(chemical_records), desc="Inserting to Milvus", position=1, leave=True)

    while submitted_count < total_batches or pending_batches or insertion_queue:
        while submitted_count < total_batches and len(pending_batches) + len(insertion_queue) < MAX_PENDING_BATCHES:
            start_idx = next_batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, len(chemical_records))
            batch_diseases = chemical_records[start_idx:end_idx]

            contents = []
            labels = []
            for rec in batch_diseases:
                synonyms_str = ", ".join(rec.synonyms) if rec.synonyms else ""

                content = f"{rec.chemical_id}\n{rec.chemical_name}\n{rec.definition}\nSynonyms: {synonyms_str}"
                label = rec.chemical_name

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

            chemical_ids = [rec.chemical_id for rec in batch_diseases]
            chemical_names = [rec.chemical_name for rec in batch_diseases]
            definitions = [rec.definition for rec in batch_diseases]
            synonyms_list = [", ".join(rec.synonyms) if rec.synonyms else "" for rec in batch_diseases]
            tree_numbers_list = [", ".join(rec.tree_numbers) if rec.tree_numbers else "" for rec in batch_diseases]
            cas_rn_list = [rec.cas_rn for rec in batch_diseases]

            milvus.insert_disease_entity_batch(
                chemical_ids,
                chemical_names,
                definitions,
                synonyms_list,
                tree_numbers_list,
                cas_rn_list,
                content_embeddings,
                label_embeddings,
            )

            inserted_count += len(batch_diseases)
            pbar_insert.update(len(batch_diseases))

    pbar_embed.close()
    pbar_insert.close()
    print(f"Inserted {inserted_count:,} chemical entities")

    num_chemical_hierarchies = 0
    if chemical_hierarchies:
        print(f"\nStep 4: Processing {len(chemical_hierarchies):,} chemical hierarchy relationships...")

        child_ids = [h[0] for h in chemical_hierarchies]
        parent_ids = [h[1] for h in chemical_hierarchies]
        child_names = [chemical_id_to_name.get(c, c) for c in child_ids]
        parent_names = [chemical_id_to_name.get(p, p) for p in parent_ids]

        batch_id = 0
        next_batch_idx = 0
        total_batches = (len(chemical_hierarchies) + args.batch_size - 1) // args.batch_size
        pending_batches = {}
        insertion_queue = []

        submitted_count = 0
        embedded_count = 0
        inserted_count = 0

        pbar_embed = tqdm(total=len(chemical_hierarchies), desc="Embedding hierarchies", position=0, leave=True)
        pbar_insert = tqdm(total=len(chemical_hierarchies), desc="Inserting to Milvus", position=1, leave=True)

        while submitted_count < total_batches or pending_batches or insertion_queue:
            while submitted_count < total_batches and len(pending_batches) + len(insertion_queue) < MAX_PENDING_BATCHES:
                start_idx = next_batch_idx * args.batch_size
                end_idx = min(start_idx + args.batch_size, len(chemical_hierarchies))

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
        num_chemical_hierarchies = inserted_count
        print(f"Inserted {inserted_count:,} chemical hierarchy relationships")

    num_chemical_gene_rels = 0
    if not args.skip_gene_interaction:
        print("\nStep 5: Processing Chemical-Gene interaction relationships...")

        chemical_gene_rels = []
        for rec in parser.parse_chemical_gene_interactions(
            args.gene_interaction_file, filter_organism=args.filter_organism if args.filter_organism else None
        ):
            if rec.chemical_id not in valid_chemical_ids:
                continue

            gene_symbol = gene_mapper.get_symbol(rec.gene_id)
            if not gene_symbol:
                continue

            chemical_gene_rels.append(rec)

        print(f"Found {len(chemical_gene_rels):,} valid Chemical-Gene interaction relationships")

        if chemical_gene_rels:
            batch_id = 0
            next_batch_idx = 0
            total_batches = (len(chemical_gene_rels) + args.batch_size - 1) // args.batch_size
            pending_batches = {}
            insertion_queue = []

            submitted_count = 0
            embedded_count = 0
            inserted_count = 0

            pbar_embed = tqdm(total=len(chemical_gene_rels), desc="Embedding chemical-gene", position=0, leave=True)
            pbar_insert = tqdm(total=len(chemical_gene_rels), desc="Inserting to Milvus", position=1, leave=True)

            while submitted_count < total_batches or pending_batches or insertion_queue:
                while (
                    submitted_count < total_batches
                    and len(pending_batches) + len(insertion_queue) < MAX_PENDING_BATCHES
                ):
                    start_idx = next_batch_idx * args.batch_size
                    end_idx = min(start_idx + args.batch_size, len(chemical_gene_rels))
                    batch_rels = chemical_gene_rels[start_idx:end_idx]

                    contents = []
                    labels = []
                    for rec in batch_rels:
                        content = f"{rec.gene_symbol} interacts_with {rec.chemical_name}: {rec.interaction}"
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
                    chemical_ids = [rec.chemical_id for rec in batch_rels]
                    chemical_names = [rec.chemical_name for rec in batch_rels]
                    interaction_list = [rec.interaction for rec in batch_rels]
                    interaction_actions_list = [rec.interaction_actions for rec in batch_rels]
                    pubmed_ids_list = [rec.pubmed_ids for rec in batch_rels]

                    milvus.insert_gene_disease_batch(
                        gene_symbols,
                        chemical_ids,
                        chemical_names,
                        interaction_list,
                        interaction_actions_list,
                        pubmed_ids_list,
                        content_embeddings,
                    )

                    inserted_count += len(batch_rels)
                    pbar_insert.update(len(batch_rels))

            pbar_embed.close()
            pbar_insert.close()
            num_chemical_gene_rels = inserted_count
            print(f"Inserted {inserted_count:,} Chemical-Gene interaction relationships")

    num_chemical_kegg_pathway_rels = 0
    if not args.skip_kegg_pathway:
        print("\nStep 6: Processing Chemical-KEGG Pathway enrichment relationships...")

        kegg_pathway_rels = []
        for rec in parser.parse_chemical_pathway_enrichments(args.pathway_file):
            if rec.chemical_id not in valid_chemical_ids:
                continue

            if not rec.pathway_id.startswith("KEGG:"):
                continue

            kegg_pathway_rels.append(rec)

        print(f"Found {len(kegg_pathway_rels):,} Chemical-KEGG Pathway enrichment relationships")

        if kegg_pathway_rels:
            batch_id = 0
            next_batch_idx = 0
            total_batches = (len(kegg_pathway_rels) + args.batch_size - 1) // args.batch_size
            pending_batches = {}
            insertion_queue = []

            submitted_count = 0
            embedded_count = 0
            inserted_count = 0

            pbar_embed = tqdm(total=len(kegg_pathway_rels), desc="Embedding KEGG enrichments", position=0, leave=True)
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
                        content = f"{rec.chemical_name} enriched_in {rec.pathway_name} (p-value: {rec.p_value}, corrected: {rec.corrected_p_value})"
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

                    chemical_ids = [rec.chemical_id for rec in batch_rels]
                    pathway_ids = [rec.pathway_id for rec in batch_rels]
                    chemical_names = [rec.chemical_name for rec in batch_rels]
                    pathway_names = [rec.pathway_name for rec in batch_rels]
                    p_values = [rec.p_value for rec in batch_rels]
                    corrected_p_values = [rec.corrected_p_value for rec in batch_rels]
                    target_match_qtys = [rec.target_match_qty for rec in batch_rels]
                    target_total_qtys = [rec.target_total_qty for rec in batch_rels]

                    milvus.insert_disease_kegg_pathway_batch(
                        chemical_ids,
                        pathway_ids,
                        chemical_names,
                        pathway_names,
                        p_values,
                        corrected_p_values,
                        target_match_qtys,
                        target_total_qtys,
                        content_embeddings,
                    )

                    inserted_count += len(batch_rels)
                    pbar_insert.update(len(batch_rels))

            pbar_embed.close()
            pbar_insert.close()
            num_chemical_kegg_pathway_rels = inserted_count
            print(f"Inserted {inserted_count:,} Chemical-KEGG Pathway enrichment relationships")

    num_chemical_reactome_pathway_rels = 0
    if not args.skip_reactome_pathway:
        print("\nStep 7: Processing Chemical-Reactome Pathway enrichment relationships...")

        reactome_pathway_rels = []
        for rec in parser.parse_chemical_pathway_enrichments(args.pathway_file):
            if rec.chemical_id not in valid_chemical_ids:
                continue

            if not rec.pathway_id.startswith("REACT:"):
                continue

            reactome_pathway_rels.append(rec)

        print(f"Found {len(reactome_pathway_rels):,} Chemical-Reactome Pathway enrichment relationships")

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
                total=len(reactome_pathway_rels), desc="Embedding Reactome enrichments", position=0, leave=True
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
                        content = f"{rec.chemical_name} enriched_in {rec.pathway_name} (p-value: {rec.p_value}, corrected: {rec.corrected_p_value})"
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

                    chemical_ids = [rec.chemical_id for rec in batch_rels]
                    pathway_ids = [rec.pathway_id for rec in batch_rels]
                    chemical_names = [rec.chemical_name for rec in batch_rels]
                    pathway_names = [rec.pathway_name for rec in batch_rels]
                    p_values = [rec.p_value for rec in batch_rels]
                    corrected_p_values = [rec.corrected_p_value for rec in batch_rels]
                    target_match_qtys = [rec.target_match_qty for rec in batch_rels]
                    target_total_qtys = [rec.target_total_qty for rec in batch_rels]

                    milvus.insert_disease_reactome_pathway_batch(
                        chemical_ids,
                        pathway_ids,
                        chemical_names,
                        pathway_names,
                        p_values,
                        corrected_p_values,
                        target_match_qtys,
                        target_total_qtys,
                        content_embeddings,
                    )

                    inserted_count += len(batch_rels)
                    pbar_insert.update(len(batch_rels))

            pbar_embed.close()
            pbar_insert.close()
            num_chemical_reactome_pathway_rels = inserted_count
            print(f"Inserted {inserted_count:,} Chemical-Reactome Pathway enrichment relationships")

    num_chemical_disease_rels = 0
    if not args.skip_disease:
        print("\nStep 8: Processing Chemical-Disease relationships...")

        chemical_disease_rels = []
        for rec in parser.parse_chemical_disease_relationships(args.disease_file):
            if rec.chemical_id not in valid_chemical_ids:
                continue

            chemical_disease_rels.append(rec)

        print(f"Found {len(chemical_disease_rels):,} Chemical-Disease relationships")

        if chemical_disease_rels:
            batch_id = 0
            next_batch_idx = 0
            total_batches = (len(chemical_disease_rels) + args.batch_size - 1) // args.batch_size
            pending_batches = {}
            insertion_queue = []

            submitted_count = 0
            embedded_count = 0
            inserted_count = 0

            pbar_embed = tqdm(total=len(chemical_disease_rels), desc="Embedding chem-disease", position=0, leave=True)
            pbar_insert = tqdm(total=len(chemical_disease_rels), desc="Inserting to Milvus", position=1, leave=True)

            while submitted_count < total_batches or pending_batches or insertion_queue:
                while (
                    submitted_count < total_batches
                    and len(pending_batches) + len(insertion_queue) < MAX_PENDING_BATCHES
                ):
                    start_idx = next_batch_idx * args.batch_size
                    end_idx = min(start_idx + args.batch_size, len(chemical_disease_rels))
                    batch_rels = chemical_disease_rels[start_idx:end_idx]

                    contents = []
                    labels = []
                    for rec in batch_rels:
                        content = f"{rec.chemical_name} associated_with {rec.disease_name} ({rec.direct_evidence})"
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

                    chemical_ids = [rec.chemical_id for rec in batch_rels]
                    disease_ids = [rec.disease_id for rec in batch_rels]
                    chemical_names = [rec.chemical_name for rec in batch_rels]
                    disease_names = [rec.disease_name for rec in batch_rels]
                    direct_evidence_list = [rec.direct_evidence for rec in batch_rels]
                    pubmed_ids_list = [rec.pubmed_ids for rec in batch_rels]

                    milvus.insert_chemical_disease_batch(
                        chemical_ids,
                        disease_ids,
                        chemical_names,
                        disease_names,
                        direct_evidence_list,
                        pubmed_ids_list,
                        content_embeddings,
                    )

                    inserted_count += len(batch_rels)
                    pbar_insert.update(len(batch_rels))

            pbar_embed.close()
            pbar_insert.close()
            num_chemical_disease_rels = inserted_count
            print(f"Inserted {inserted_count:,} Chemical-Disease relationships")

    num_chemical_go_rels = 0
    if not args.skip_go:
        print("\nStep 9: Processing Chemical-GO enrichment relationships...")

        chemical_go_rels = []
        for rec in parser.parse_chemical_go_enrichments(
            args.go_file, filter_ontology=args.filter_go_ontology if args.filter_go_ontology else None
        ):
            if rec.chemical_id not in valid_chemical_ids:
                continue

            chemical_go_rels.append(rec)

        print(f"Found {len(chemical_go_rels):,} Chemical-GO enrichment relationships")

        if chemical_go_rels:
            batch_id = 0
            next_batch_idx = 0
            total_batches = (len(chemical_go_rels) + args.batch_size - 1) // args.batch_size
            pending_batches = {}
            insertion_queue = []

            submitted_count = 0
            embedded_count = 0
            inserted_count = 0

            pbar_embed = tqdm(total=len(chemical_go_rels), desc="Embedding GO enrichments", position=0, leave=True)
            pbar_insert = tqdm(total=len(chemical_go_rels), desc="Inserting to Milvus", position=1, leave=True)

            while submitted_count < total_batches or pending_batches or insertion_queue:
                while (
                    submitted_count < total_batches
                    and len(pending_batches) + len(insertion_queue) < MAX_PENDING_BATCHES
                ):
                    start_idx = next_batch_idx * args.batch_size
                    end_idx = min(start_idx + args.batch_size, len(chemical_go_rels))
                    batch_rels = chemical_go_rels[start_idx:end_idx]

                    contents = []
                    labels = []
                    for rec in batch_rels:
                        content = f"{rec.chemical_name} enriched_in {rec.go_term_name} ({rec.ontology}, p-value: {rec.p_value}, corrected: {rec.corrected_p_value})"
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

                    chemical_ids = [rec.chemical_id for rec in batch_rels]
                    go_term_ids = [rec.go_term_id for rec in batch_rels]
                    chemical_names = [rec.chemical_name for rec in batch_rels]
                    go_term_names = [rec.go_term_name for rec in batch_rels]
                    ontologies = [rec.ontology for rec in batch_rels]
                    highest_go_levels = [rec.highest_go_level for rec in batch_rels]
                    p_values = [rec.p_value for rec in batch_rels]
                    corrected_p_values = [rec.corrected_p_value for rec in batch_rels]
                    target_match_qtys = [rec.target_match_qty for rec in batch_rels]
                    target_total_qtys = [rec.target_total_qty for rec in batch_rels]

                    milvus.insert_chemical_go_batch(
                        chemical_ids,
                        go_term_ids,
                        chemical_names,
                        go_term_names,
                        ontologies,
                        highest_go_levels,
                        p_values,
                        corrected_p_values,
                        target_match_qtys,
                        target_total_qtys,
                        content_embeddings,
                    )

                    inserted_count += len(batch_rels)
                    pbar_insert.update(len(batch_rels))

            pbar_embed.close()
            pbar_insert.close()
            num_chemical_go_rels = inserted_count
            print(f"Inserted {inserted_count:,} Chemical-GO enrichment relationships")

    print("\nFinalizing insertion...")
    milvus.finalize()
    embedder.close()
    milvus.close()

    duration = time.time() - start_time
    metadata_tracker.record_insertion(
        num_chemicals=len(chemical_records),
        num_chemical_hierarchies=num_chemical_hierarchies,
        num_chemical_gene_rels=num_chemical_gene_rels,
        num_chemical_disease_rels=num_chemical_disease_rels,
        num_chemical_kegg_pathway_rels=num_chemical_kegg_pathway_rels,
        num_chemical_reactome_pathway_rels=num_chemical_reactome_pathway_rels,
        num_chemical_go_rels=num_chemical_go_rels,
        organism_filter=args.filter_organism if hasattr(args, "filter_organism") else None,
        go_ontology_filter=args.filter_go_ontology if hasattr(args, "filter_go_ontology") else None,
        batch_size=args.batch_size,
        workers=WORKERS,
        embedding_model=EMBEDDING_MODEL,
        milvus_batch_size=MILVUS_BATCH_SIZE,
        duration_seconds=duration,
    )

    print(f"\nCompleted in {duration:.2f} seconds")
    print(f"Inserted {len(chemical_records):,} chemical entities")
    print(f"Inserted {num_chemical_hierarchies:,} chemical hierarchy relationships")
    print(f"Inserted {num_chemical_gene_rels:,} Chemical-Gene interaction relationships")
    print(f"Inserted {num_chemical_disease_rels:,} Chemical-Disease relationships")
    print(f"Inserted {num_chemical_kegg_pathway_rels:,} Chemical-KEGG Pathway enrichment relationships")
    print(f"Inserted {num_chemical_reactome_pathway_rels:,} Chemical-Reactome Pathway enrichment relationships")
    print(f"Inserted {num_chemical_go_rels:,} Chemical-GO enrichment relationships")


def main():
    parser = argparse.ArgumentParser(
        description="Insert CTD Chemicals into Milvus",
        epilog="Configuration loaded from config/db_init.conf. Command-line arguments override config.",
    )

    parser.add_argument("--config", default=str(CONFIG_FILE), help="Path to config file")
    parser.add_argument("--chemicals-file", default=CTD_CHEMICAL_FILE, help="Path to CTD_chemicals.csv.gz")
    parser.add_argument(
        "--gene-interaction-file", default=CTD_CHEMICAL_GENE_FILE, help="Path to CTD_chem_gene_ixns.csv.gz"
    )
    parser.add_argument(
        "--disease-file", default=CTD_CHEMICAL_DISEASE_FILE, help="Path to CTD_curated_chemicals_diseases.csv.gz"
    )
    parser.add_argument(
        "--pathway-file", default=CTD_CHEMICAL_PATHWAY_FILE, help="Path to CTD_chem_pathways_enriched.csv.gz"
    )
    parser.add_argument("--go-file", default=CTD_CHEMICAL_GO_FILE, help="Path to CTD_chem_go_enriched.csv.gz")
    parser.add_argument(
        "--filter-organism", nargs="+", help="Filter chemical-gene interactions by organism ID (e.g., 9606 for human)"
    )
    parser.add_argument(
        "--filter-go-ontology",
        nargs="+",
        help="Filter GO enrichments by ontology (e.g., 'Biological Process', 'Molecular Function')",
    )
    parser.add_argument("--batch-size", type=int, default=CTD_CHEMICAL_BATCH_SIZE, help="Batch size for processing")
    parser.add_argument("--limit", type=int, default=CTD_CHEMICAL_LIMIT, help="Limit chemicals (0 = no limit)")
    parser.add_argument(
        "--skip-gene-interaction", action="store_true", help="Skip chemical-gene interaction relationships"
    )
    parser.add_argument("--skip-disease", action="store_true", help="Skip chemical-disease relationships")
    parser.add_argument("--skip-kegg-pathway", action="store_true", help="Skip chemical-KEGG pathway enrichments")
    parser.add_argument(
        "--skip-reactome-pathway", action="store_true", help="Skip chemical-Reactome pathway enrichments"
    )
    parser.add_argument("--skip-go", action="store_true", help="Skip chemical-GO enrichments")
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
