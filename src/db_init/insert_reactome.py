#!/usr/bin/env python3
"""
Insert Reactome Pathways into Milvus.

This script:
1. Parses Reactome pathway files (ReactomePathways.txt, ReactomePathways.gmt)
2. Creates Reactome_Pathway entities in Milvus
3. Creates Gene->Reactome_Pathway relationships in Milvus
4. Filters by organism

Configuration loaded from config/db_init.conf
"""

import argparse
import csv
import hashlib
import json
import multiprocessing
import os
import time
from datetime import datetime
from multiprocessing import Manager
from pathlib import Path
from queue import Empty

import numpy as np
import torch
from config_loader import PROJECT_ROOT, load_config
from embedding_utils import masked_mean_pooling, normalize_embeddings
from insert_gene2go import GeneIDMapper
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
from reactome_parser import ReactomeParser, ReactomePathwayRecord
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

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
WORKSPACE = CONFIG.get("WORKSPACE", "lightrag")
GRAPH_FIELD_SEP = CONFIG.get("GRAPH_FIELD_SEP", "<SEP>")
METADATA_DIR = Path(CONFIG.get("METADATA_DIR", f"{DATA_DIR}/db-init"))
REACTOME_METADATA_FILE = CONFIG.get("REACTOME_METADATA_FILE", "reactome_pathways.csv")


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
    """
    Worker process for generating dual embeddings on a specific GPU.
    """
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
    """Handles insertion of Reactome pathways and relationships into Milvus."""

    def __init__(
        self,
        host: str = MILVUS_HOST,
        port: str = MILVUS_PORT,
        batch_size: int = MILVUS_BATCH_SIZE,
        drop_existing: bool = True,
        skip_pathways: bool = False,
        skip_relationships: bool = False,
    ):
        self.host = host
        self.port = port
        self.batch_size = batch_size
        self.drop_existing = drop_existing
        self.skip_pathways = skip_pathways
        self.skip_relationships = skip_relationships

        connections.connect(host=host, port=port)

        if drop_existing:
            self._drop_reactome_data()

        self._ensure_collections()

        if not skip_pathways:
            self.pathway_collection = Collection("entities_Reactome_Pathway")
        if not skip_relationships:
            self.relationship_collection = Collection("relationships_Gene_Reactome_Pathway")

    @staticmethod
    def _compute_id(content: str, prefix: str = "") -> str:
        """Compute a prefixed MD5 hash ID."""
        return prefix + hashlib.md5(content.encode()).hexdigest()

    def _drop_reactome_data(self):
        """Drop existing Reactome collections."""
        if not self.skip_pathways and utility.has_collection("entities_Reactome_Pathway"):
            utility.drop_collection("entities_Reactome_Pathway")
            print("Dropped existing entities_Reactome_Pathway collection")

        if not self.skip_relationships and utility.has_collection("relationships_Gene_Reactome_Pathway"):
            utility.drop_collection("relationships_Gene_Reactome_Pathway")
            print("Dropped existing relationships_Gene_Reactome_Pathway collection")

    def _ensure_collections(self):
        """Create collections if they don't exist."""
        if not self.skip_pathways and not utility.has_collection("entities_Reactome_Pathway"):
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
                FieldSchema(name="label_embedding", dtype=DataType.FLOAT_VECTOR, dim=LABEL_EMBEDDING_DIM),
                FieldSchema(name="created_at", dtype=DataType.INT64),
                FieldSchema(name="entity_name", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=32768),
            ]
            schema = CollectionSchema(fields=fields, enable_dynamic_field=True, description="Reactome Pathway entities")
            Collection(name="entities_Reactome_Pathway", schema=schema)
            print("Created Milvus collection: entities_Reactome_Pathway")

        if not self.skip_relationships and not utility.has_collection("relationships_Gene_Reactome_Pathway"):
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
                FieldSchema(name="label_embedding", dtype=DataType.FLOAT_VECTOR, dim=LABEL_EMBEDDING_DIM),
                FieldSchema(name="created_at", dtype=DataType.INT64),
                FieldSchema(name="src_id", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="tgt_id", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=32768),
            ]
            schema = CollectionSchema(
                fields=fields, enable_dynamic_field=True, description="Gene to Reactome Pathway relationships"
            )
            Collection(name="relationships_Gene_Reactome_Pathway", schema=schema)
            print("Created Milvus collection: relationships_Gene_Reactome_Pathway")

    def insert_pathway_batch(
        self,
        pathway_ids: list[str],
        pathway_names: list[str],
        organisms: list[str],
        genes_lists: list[list[str]],
        content_embeddings: np.ndarray,
        label_embeddings: np.ndarray,
    ):
        """Insert a batch of pathway entities."""
        timestamp = int(time.time() * 1000)

        entities = []
        for i, (pathway_id, pathway_name, organism, genes) in enumerate(
            zip(pathway_ids, pathway_names, organisms, genes_lists, strict=False)
        ):
            entity_name = f"REACT:{pathway_id}"
            entity_id = self._compute_id(entity_name, prefix="ent-")

            genes_str = ", ".join(genes[:1000]) if genes else ""
            content = genes_str

            species_code = pathway_id.split("-")[1] if "-" in pathway_id else "UNKNOWN"

            full_description = f"{pathway_name} ({organism})"

            entity = {
                "id": entity_id,
                "vector": content_embeddings[i].tolist(),
                "label_embedding": label_embeddings[i].tolist(),
                "created_at": timestamp,
                "entity_name": entity_name,
                "file_path": "",
                "content": content,
                "label": pathway_name,
                "description": full_description,
                "entity_type": "Reactome_Pathway",
                "entity_id": pathway_id,
                "pathway_id": pathway_id,
                "pathway_name": pathway_name,
                "organism": organism,
                "species_code": species_code,
                "genes": json.dumps(genes),
            }
            entities.append(entity)

        self.pathway_collection.insert(entities)

    def insert_relationship_batch(
        self,
        gene_symbols: list[str],
        pathway_ids: list[str],
        pathway_names: list[str],
        content_embeddings: np.ndarray,
        label_embeddings: np.ndarray,
    ):
        """Insert a batch of gene-pathway relationships."""
        timestamp = int(time.time() * 1000)

        relationships = []
        for i, (gene, pathway_id, pathway_name) in enumerate(
            zip(gene_symbols, pathway_ids, pathway_names, strict=False)
        ):
            src_id_key = f"GENE:{gene}"
            tgt_id_key = f"REACT:{pathway_id}"
            composite_key = f"{src_id_key}{GRAPH_FIELD_SEP}{tgt_id_key}"
            rel_id = self._compute_id(composite_key, prefix="rel-")

            content = f"{gene} participates_in {pathway_name} {pathway_id}"

            relationship = {
                "id": rel_id,
                "vector": content_embeddings[i].tolist(),
                "label_embedding": label_embeddings[i].tolist(),
                "created_at": timestamp,
                "src_id": f"GENE:{gene}",
                "tgt_id": f"REACT:{pathway_id}",
                "file_path": "",
                "content": content,
                "description": content,
                "keywords": f"{gene};participates_in;{pathway_name}",
                "weight": 1.0,
                "source_id": "",
                "gene_symbol": gene,
                "pathway_id": pathway_id,
                "pathway_name": pathway_name,
            }
            relationships.append(relationship)

        self.relationship_collection.insert(relationships)

    def finalize_pathways(self):
        """Flush, create indexes, and load pathway collection."""
        if self.skip_pathways:
            return

        print("\nFinalizing pathways in Milvus...")
        print("Flushing pathways to Milvus...")
        self.pathway_collection.flush()
        print("Pathways flushed to Milvus")

        print("Creating indexes for pathway vector fields...")
        index_params = {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 200}}
        self.pathway_collection.create_index(field_name="vector", index_params=index_params)
        print("Created index for pathways content embeddings")

        self.pathway_collection.create_index(field_name="label_embedding", index_params=index_params)
        print("Created index for pathways label embeddings")

        self.pathway_collection.load()
        print("Pathways collection loaded and ready")

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

        self.relationship_collection.create_index(field_name="label_embedding", index_params=index_params)
        print("Created index for relationships label embeddings")

        self.relationship_collection.load()
        print("Relationships collection loaded and ready")


def process_pathway_entities(
    pathways: dict[str, ReactomePathwayRecord], embedder: ParallelEmbeddingGenerator, inserter: MilvusInserter, args
):
    """Process Reactome pathway entities with three parallel streams."""
    if args.skip_pathways:
        return

    pathway_records = list(pathways.values())
    if args.limit > 0:
        pathway_records = pathway_records[: args.limit]

    total_pathways = len(pathway_records)
    print(f"Step 4: Processing {total_pathways:,} Reactome pathway entities...")
    print(f"Max pending batches: {args.max_pending_batches}\n")

    pending_batches = {}
    next_batch_id = 0
    batch_idx = 0

    embed_pbar = tqdm(total=total_pathways, desc="Embedding pathways", position=0)
    insert_pbar = tqdm(total=total_pathways, desc="Inserting to Milvus", position=1)

    batches = []
    for i in range(0, total_pathways, args.batch_size):
        batches.append(pathway_records[i : i + args.batch_size])

    while batch_idx < len(batches) or pending_batches:
        while batch_idx < len(batches) and len(pending_batches) < args.max_pending_batches:
            batch = batches[batch_idx]
            batch_id = next_batch_id
            next_batch_id += 1
            batch_idx += 1

            pathway_ids = [p.pathway_id for p in batch]
            pathway_names = [p.pathway_name for p in batch]
            organisms = [p.organism for p in batch]
            genes_lists = [p.genes for p in batch]

            contents = []
            labels = []
            for p in batch:
                genes_str = ", ".join(p.genes) if p.genes else ""
                contents.append(genes_str)
                labels.append(p.pathway_name)

            embedder.embed_batch(batch_id, contents, labels)
            pending_batches[batch_id] = (pathway_ids, pathway_names, organisms, genes_lists)

        collected_any = False
        for bid in list(pending_batches.keys()):
            result = embedder.get_result(bid, timeout=0.001)
            if result is not None:
                result_bid, content_emb, label_emb = result
                if content_emb is not None:
                    pathway_ids, pathway_names, organisms, genes_lists = pending_batches.pop(result_bid)

                    inserter.insert_pathway_batch(
                        pathway_ids, pathway_names, organisms, genes_lists, content_emb, label_emb
                    )

                    embed_pbar.update(len(pathway_ids))
                    insert_pbar.update(len(pathway_ids))
                    collected_any = True

        if not collected_any:
            time.sleep(0.01)

    embed_pbar.close()
    insert_pbar.close()

    inserter.finalize_pathways()


def process_relationships(
    relationships: list[tuple[str, str]],
    pathways: dict[str, ReactomePathwayRecord],
    embedder: ParallelEmbeddingGenerator,
    inserter: MilvusInserter,
    args,
):
    """Process Gene-Reactome_Pathway relationships with three streams."""
    if args.skip_relationships:
        return

    if args.limit > 0:
        relationships = relationships[: args.limit * 100]

    total_relationships = len(relationships)
    print(f"\nStep 5: Processing {total_relationships:,} Gene-Reactome_Pathway relationships...")

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

            gene_symbols = [gene for gene, _ in batch]
            pathway_ids = [pathway_id for _, pathway_id in batch]
            pathway_names = [pathways[pathway_id].pathway_name for _, pathway_id in batch]

            contents = []
            labels = []
            for gene, pathway_id in batch:
                pathway_name = pathways[pathway_id].pathway_name
                content = f"{gene} participates_in {pathway_name} {pathway_id}"
                contents.append(content)
                labels.append(f"{gene}->{pathway_id}")

            embedder.embed_batch(batch_id, contents, labels)
            pending_batches[batch_id] = (gene_symbols, pathway_ids, pathway_names)

        collected_any = False
        for bid in list(pending_batches.keys()):
            result = embedder.get_result(bid, timeout=0.001)
            if result is not None:
                result_bid, content_emb, label_emb = result
                if content_emb is not None:
                    gene_symbols, pathway_ids, pathway_names = pending_batches.pop(result_bid)

                    inserter.insert_relationship_batch(gene_symbols, pathway_ids, pathway_names, content_emb, label_emb)

                    embed_pbar.update(len(gene_symbols))
                    insert_pbar.update(len(gene_symbols))
                    collected_any = True

        if not collected_any:
            time.sleep(0.01)

    embed_pbar.close()
    insert_pbar.close()

    inserter.finalize_relationships()


def record_metadata(args, num_pathways: int, num_relationships: int, duration: float, organisms: list[str]):
    """Record insertion metadata to CSV file."""
    metadata_file = METADATA_DIR / REACTOME_METADATA_FILE

    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    file_exists = metadata_file.exists()

    with open(metadata_file, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(
                [
                    "timestamp",
                    "num_pathways",
                    "num_relationships",
                    "organisms",
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
                num_pathways,
                num_relationships,
                ",".join(organisms) if organisms else "ALL",
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
    parser = argparse.ArgumentParser(description="Insert Reactome pathways into Milvus")
    parser.add_argument(
        "--reactome-dir",
        type=str,
        default=CONFIG.get("REACTOME_DIR", f"{DATA_DIR}/resources/reactome"),
        help="Directory containing Reactome files",
    )
    parser.add_argument(
        "--gene-info-file",
        type=str,
        default=CONFIG.get("GENE_INFO_FILE", f"{DATA_DIR}/resources/genes/gene_info.gz"),
        help="Path to gene_info.gz file",
    )
    parser.add_argument(
        "--filter-organisms",
        type=str,
        default=CONFIG.get("REACTOME_FILTER_ORGANISMS", "Homo sapiens"),
        help="Filter by organism(s) (comma-separated, empty for all)",
    )
    parser.add_argument(
        "--filter-taxid",
        type=str,
        default=CONFIG.get("GENE_FILTER_TAXID", "9606"),
        help="Filter genes by taxonomy ID (for gene mapping)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=int(CONFIG.get("REACTOME_BATCH_SIZE", "1000")), help="Processing batch size"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=int(CONFIG.get("REACTOME_LIMIT", "0")),
        help="Limit number of pathways (0 = no limit)",
    )
    parser.add_argument("--workers", type=int, default=WORKERS, help="Number of parallel embedding workers")
    parser.add_argument("--max-pending-batches", type=int, default=MAX_PENDING_BATCHES, help="Maximum pending batches")
    parser.add_argument(
        "--drop-existing", action="store_true", default=True, help="Drop existing Reactome collections (default: True)"
    )
    parser.add_argument(
        "--no-drop-existing", action="store_false", dest="drop_existing", help="Keep existing Reactome collections"
    )
    parser.add_argument("--skip-pathways", action="store_true", default=False, help="Skip pathway entities insertion")
    parser.add_argument("--skip-relationships", action="store_true", default=False, help="Skip relationships insertion")

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Reactome Pathway Insertion")
    print("=" * 60)
    if args.drop_existing and not args.skip_pathways:
        print("WARNING: --drop-existing flag set - will delete all existing Reactome data!")
    print("=" * 60)
    print(f"Configuration file: {CONFIG_FILE}")
    print(f"Reactome directory: {args.reactome_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Limit: {args.limit}")
    print(f"Workers: {args.workers}")

    start_time = time.time()

    print("\nStep 1: Parsing Reactome files...")
    parser = ReactomeParser(args.reactome_dir)
    all_pathways = parser.merge_pathway_data()
    print(f"Parsed {len(all_pathways):,} total pathways")

    print("\nStep 2: Filtering by organism...")
    organism_list = [o.strip() for o in args.filter_organisms.split(",") if o.strip()]
    if organism_list:
        pathways = parser.filter_by_organism(all_pathways, organism_list)
        print(f"Filtered to {len(pathways):,} pathways for: {', '.join(organism_list)}")
    else:
        pathways = all_pathways
        print(f"Using all {len(pathways):,} pathways (no organism filter)")

    print("\nStep 3: Loading valid gene symbols...")
    gene_mapper = GeneIDMapper(
        gene_info_file=args.gene_info_file, filter_taxid=[args.filter_taxid] if args.filter_taxid else []
    )
    gene_mapper.build_mapping()
    valid_genes = set(gene_mapper.mapping.values())
    print(f"Loaded {len(valid_genes):,} valid gene symbols")

    relationships = parser.build_gene_pathway_relationships(pathways, valid_genes)

    inserter = MilvusInserter(
        host=MILVUS_HOST,
        port=MILVUS_PORT,
        batch_size=MILVUS_BATCH_SIZE,
        drop_existing=args.drop_existing,
        skip_pathways=args.skip_pathways,
        skip_relationships=args.skip_relationships,
    )

    embedder = ParallelEmbeddingGenerator(
        workers=args.workers,
        content_model=EMBEDDING_MODEL,
        label_model=LABEL_EMBEDDING_MODEL,
        sub_batch_size=EMBEDDING_SUB_BATCH_SIZE,
    )

    process_pathway_entities(pathways, embedder, inserter, args)

    process_relationships(relationships, pathways, embedder, inserter, args)

    embedder.close()

    duration = time.time() - start_time
    num_pathways = len(pathways) if args.limit == 0 else min(len(pathways), args.limit)
    num_relationships = len(relationships) if args.limit == 0 else min(len(relationships), args.limit * 100)
    record_metadata(args, num_pathways, num_relationships, duration, organism_list)

    print("\n" + "=" * 60)
    print(f"Inserted {num_pathways:,} Reactome pathway entities")
    print(f"Inserted {num_relationships:,} Gene-Reactome_Pathway relationships")
    print(f"Duration: {duration:.2f} seconds ({duration / 60:.2f} minutes)")
    print("=" * 60)


if __name__ == "__main__":
    main()
