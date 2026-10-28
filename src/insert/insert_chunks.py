#!/usr/bin/env python3
"""
Chunk insertion with parallel embedding generation.

Writes each chunk to three stores:
- Milvus: BGE-M3 content embedding in the <workspace>_chunks collection
- Redis: chunk text and document status under <workspace>_text_chunks and <workspace>_doc_status
- Elasticsearch: chunk text in the lightrag_chunks index for keyword search

Input:
- Chunk directories: <DATA_DIR>/chunk/<paper_id>/1.txt, 2.txt, ...
- Chunk metadata: <DATA_DIR>/chunk-stats/chunk_stats.csv
"""

import argparse
import csv
import hashlib
import json
import logging
import multiprocessing
import sys
import time
from datetime import datetime
from multiprocessing import Manager
from pathlib import Path
from queue import Empty

import numpy as np
import redis
import torch
from elasticsearch import Elasticsearch
from elasticsearch import helpers as es_helpers
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

multiprocessing.set_start_method("spawn", force=True)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.db_init.config_loader import PROJECT_ROOT, load_config  # noqa: E402
from src.db_init.embedding_utils import masked_mean_pooling, normalize_embeddings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logging.getLogger("elastic_transport.transport").setLevel(logging.WARNING)
logging.getLogger("elasticsearch").setLevel(logging.WARNING)


CONFIG_FILE = PROJECT_ROOT / "config" / "pipeline.conf"
CONFIG = load_config(str(CONFIG_FILE))

CHUNK_INDEX = "lightrag_chunks"
CHUNK_EMBEDDING_WORKERS = int(CONFIG.get("CHUNK_EMBEDDING_WORKERS", "20"))
EMBEDDING_SUB_BATCH_SIZE = int(CONFIG.get("EMBEDDING_SUB_BATCH_SIZE", "512"))
CHUNKS_PER_BATCH = int(CONFIG.get("CHUNKS_PER_BATCH", "500"))
MILVUS_INSERT_BATCH_PAPERS = int(CONFIG.get("MILVUS_INSERT_BATCH_PAPERS", "50"))
MAX_PENDING_BATCHES = int(CONFIG.get("MAX_PENDING_BATCHES", "10"))
CUDA_DEVICES = CONFIG.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """Compute a prefixed MD5 hash ID."""
    safe_bytes = content.encode("utf-8", errors="replace")
    return prefix + hashlib.md5(safe_bytes).hexdigest()


def _embedding_worker(worker_id: int, device_id: int, model_name: str, sub_batch_size: int, input_queue, output_queue):
    """Worker process for generating BGE-M3 embeddings on a specific GPU."""
    import traceback

    try:
        device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name, use_safetensors=True)
            model.to(device)
            model.eval()
            logger.info(f"Worker {worker_id} loaded {model_name} on {device}")
        except Exception as e:
            logger.error(f"Worker {worker_id} failed to load model: {e}")
            output_queue.put(("WORKER_INIT_ERROR", str(e)))
            return

        while True:
            try:
                item = input_queue.get(timeout=1)
                if item is None:
                    logger.info(f"Worker {worker_id} received poison pill, exiting")
                    break

                batch_id, texts = item

                all_embeddings = []
                for i in range(0, len(texts), sub_batch_size):
                    sub_texts = texts[i : i + sub_batch_size]

                    inputs = tokenizer(sub_texts, return_tensors="pt", padding=True, truncation=True, max_length=768)

                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = model(**inputs)

                        embeddings = masked_mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
                        embeddings = normalize_embeddings(embeddings)
                        embeddings = embeddings.cpu().numpy()
                        all_embeddings.append(embeddings)

                final_embeddings = np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]
                output_queue.put((batch_id, final_embeddings))

            except Empty:
                continue
            except Exception as e:
                logger.error(
                    f"Worker {worker_id} error processing batch {batch_id if 'batch_id' in locals() else 'unknown'}: {e}"
                )
                logger.error(f"Worker {worker_id} traceback: {traceback.format_exc()}")
                if "batch_id" in locals():
                    output_queue.put((batch_id, None))

    except Exception as e:
        logger.error(f"Worker {worker_id} FATAL error: {e}")
        logger.error(f"Worker {worker_id} FATAL traceback: {traceback.format_exc()}")


class ParallelEmbeddingGenerator:
    """Generate embeddings in parallel across multiple GPUs."""

    def __init__(self, num_workers: int, cuda_devices: str, model_name: str, sub_batch_size: int):
        self.num_workers = num_workers
        self.model_name = model_name
        self.sub_batch_size = sub_batch_size

        if cuda_devices and torch.cuda.is_available():
            self.devices = [int(d) for d in cuda_devices.split(",")]
        else:
            self.devices = [0]

        self.manager = Manager()
        self.input_queue = self.manager.Queue()
        self.output_queue = self.manager.Queue()
        self.workers = []

        logger.info(f"Initializing {num_workers} embedding workers across {len(self.devices)} GPUs...")
        logger.info(f"Model: {model_name}")
        self._start_workers()

    def _start_workers(self):
        """Start worker processes."""
        for worker_id in range(self.num_workers):
            device_id = self.devices[worker_id % len(self.devices)]
            p = multiprocessing.Process(
                target=_embedding_worker,
                args=(worker_id, device_id, self.model_name, self.sub_batch_size, self.input_queue, self.output_queue),
            )
            p.start()
            self.workers.append(p)

    def embed_batch(self, batch_id: int, texts: list[str]):
        """Submit a batch for embedding."""
        self.input_queue.put((batch_id, texts))

    def get_result(self, block: bool = True, timeout: float = None) -> tuple[int, np.ndarray] | None:
        """Get completed embedding result."""
        try:
            return self.output_queue.get(block=block, timeout=timeout)
        except Empty:
            return None

    def close(self):
        """Shutdown workers."""
        for _ in range(self.num_workers):
            self.input_queue.put(None)

        for p in self.workers:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
                p.join()


class MilvusChunkManager:
    """Manage the Milvus chunks collection."""

    def __init__(self, host: str, port: str, workspace: str = "lightrag"):
        self.host = host
        self.port = port
        self.workspace = workspace
        self.collection_name = f"{workspace}_chunks"
        self.embedding_dim = 1024

        connections.connect(host=host, port=port)
        logger.info(f"Connected to Milvus at {host}:{port}")

    def ensure_collection(self, drop_existing: bool = False):
        """Ensure chunks collection exists with correct schema."""
        if drop_existing and utility.has_collection(self.collection_name):
            logger.info(f"Dropping existing collection: {self.collection_name}")
            utility.drop_collection(self.collection_name)

        if not utility.has_collection(self.collection_name):
            logger.info(f"Creating collection: {self.collection_name}")

            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
                FieldSchema(name="created_at", dtype=DataType.INT64),
                FieldSchema(name="full_doc_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=32768),
            ]

            schema = CollectionSchema(fields=fields, description="Chunk embeddings", enable_dynamic_field=True)

            collection = Collection(name=self.collection_name, schema=schema)

            vector_index_params = {
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 16, "efConstruction": 256},
            }
            collection.create_index("vector", vector_index_params)
            logger.info("Created HNSW index on vector field")

            scalar_index_params = {"index_type": "STL_SORT"}
            collection.create_index("created_at", scalar_index_params)
            logger.info("Created index on created_at field")

            logger.info(f"Created collection: {self.collection_name}")
        else:
            logger.info(f"Collection already exists: {self.collection_name}")

        collection = Collection(self.collection_name)
        collection.load()
        logger.info(f"Collection loaded: {self.collection_name}")

    def insert_chunks(self, chunks_data: list[dict]) -> int:
        """
        Insert chunks to Milvus using INSERT (not upsert).

        Uses INSERT for speed since we track insertions in log file.
        """
        if not chunks_data:
            return 0

        collection = Collection(self.collection_name)
        collection.insert(chunks_data)
        collection.flush()
        return len(chunks_data)

    def delete_chunks_after_timestamp(self, timestamp: int) -> int:
        """
        Delete all chunks with created_at > timestamp.

        Args:
            timestamp: Unix timestamp - delete chunks created after this

        Returns:
            Always returns 0 (actual count not available with expression-based delete)
        """
        collection = Collection(self.collection_name)

        expr = f"created_at > {timestamp}"

        try:
            collection.delete(expr=expr)
            collection.flush()
            logger.info(f"Deleted chunks with created_at > {timestamp}")
            return 0
        except Exception as e:
            logger.warning(f"Could not delete chunks: {e}")
            return 0

    def close(self):
        """Disconnect from Milvus."""
        connections.disconnect("default")


class RedisChunkManager:
    """Manage Redis chunk storage."""

    def __init__(self, host: str = "localhost", port: int = 6379, workspace: str = "lightrag"):
        self.workspace = workspace
        self.text_chunks_namespace = f"{workspace}_text_chunks"
        self.doc_status_namespace = f"{workspace}_doc_status"

        self.redis = redis.Redis(host=host, port=port, decode_responses=True)
        self.redis.ping()
        logger.info(f"Connected to Redis at {host}:{port}")

    def insert_chunks(self, chunks_data: list[dict]) -> int:
        """
        Insert chunks to Redis text_chunks storage.

        Uses upsert (SET) - if key exists, it will be overwritten.
        Uses timestamp from chunk data (created_at) for consistency.
        """
        if not chunks_data:
            return 0

        pipe = self.redis.pipeline()
        for chunk in chunks_data:
            chunk_id = chunk["id"]
            key = f"{self.text_chunks_namespace}:{chunk_id}"

            timestamp = chunk.get("created_at", int(time.time()))

            value = {
                "_id": chunk_id,
                "content": chunk["content"],
                "full_doc_id": chunk["full_doc_id"],
                "tokens": chunk.get("tokens", 0),
                "chunk_order_index": chunk.get("chunk_order_index", 0),
                "file_path": chunk["file_path"],
                "create_time": timestamp,
                "update_time": timestamp,
                "llm_cache_list": [],
            }

            pipe.set(key, json.dumps(value))

        pipe.execute()
        return len(chunks_data)

    def update_doc_status(self, doc_id: str, file_path: str, chunk_ids: list[str], timestamp: int):
        """
        Update document status in Redis.

        Args:
            timestamp: Consistent timestamp to use (same as chunk insertion)
        """

        key = f"{self.doc_status_namespace}:{doc_id}"
        value = {
            "status": "success",
            "file_path": file_path,
            "chunks_list": chunk_ids,
            "metadata": {},
            "error_msg": None,
            "track_id": doc_id,
            "created_at": timestamp,
            "updated_at": timestamp,
        }

        self.redis.set(key, json.dumps(value))

    def clear_workspace_data(self):
        """
        Clear all Redis data for this workspace.

        Deletes all keys matching:
        - {workspace}_text_chunks:*
        - {workspace}_doc_status:*
        """
        logger.info(f"Clearing Redis data for workspace: {self.workspace}")

        text_chunks_pattern = f"{self.text_chunks_namespace}:*"
        cursor = 0
        deleted_chunks = 0
        while True:
            cursor, keys = self.redis.scan(cursor, match=text_chunks_pattern, count=1000)
            if keys:
                deleted_chunks += self.redis.delete(*keys)
            if cursor == 0:
                break

        doc_status_pattern = f"{self.doc_status_namespace}:*"
        cursor = 0
        deleted_docs = 0
        while True:
            cursor, keys = self.redis.scan(cursor, match=doc_status_pattern, count=1000)
            if keys:
                deleted_docs += self.redis.delete(*keys)
            if cursor == 0:
                break

        logger.info(f"Deleted {deleted_chunks} text chunks and {deleted_docs} doc status entries from Redis")

    def close(self):
        """Close Redis connection."""
        self.redis.close()


class ElasticsearchChunkManager:
    """Manage Elasticsearch chunk indexing."""

    def __init__(self, host: str = "localhost", port: int = 9200, workspace: str = "lightrag"):
        self.workspace = workspace
        self.index_name = CHUNK_INDEX
        self.es = Elasticsearch([f"http://{host}:{port}"], request_timeout=30, max_retries=3, retry_on_timeout=True)

        if not self.es.ping():
            raise ConnectionError(f"Cannot connect to Elasticsearch at {host}:{port}")

        logger.info(f"Connected to Elasticsearch at {host}:{port}")

    def ensure_index(self):
        """Create Elasticsearch index with biomedical analyzer if it doesn't exist."""
        if self.es.indices.exists(index=self.index_name):
            logger.info(f"Elasticsearch index '{self.index_name}' already exists")
            return

        logger.info(f"Creating Elasticsearch index: {self.index_name}")

        mapping = {
            "settings": {
                "number_of_shards": 3,
                "number_of_replicas": 1,
                "analysis": {
                    "analyzer": {
                        "biomedical_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "asciifolding", "biomedical_stop", "english_stemmer"],
                        }
                    },
                    "filter": {
                        "biomedical_stop": {
                            "type": "stop",
                            "stopwords": [
                                "a",
                                "and",
                                "are",
                                "as",
                                "at",
                                "be",
                                "but",
                                "by",
                                "for",
                                "if",
                                "in",
                                "into",
                                "is",
                                "it",
                                "no",
                                "not",
                                "of",
                                "on",
                                "or",
                                "such",
                                "that",
                                "the",
                                "their",
                                "then",
                                "there",
                                "these",
                                "they",
                                "this",
                                "to",
                                "was",
                                "will",
                                "with",
                            ],
                        },
                        "english_stemmer": {"type": "stemmer", "language": "english"},
                    },
                },
            },
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "content": {
                        "type": "text",
                        "analyzer": "biomedical_analyzer",
                        "fields": {"exact": {"type": "keyword"}, "english": {"type": "text", "analyzer": "english"}},
                    },
                    "full_doc_id": {"type": "keyword"},
                    "chunk_order_index": {"type": "integer"},
                    "created_at": {"type": "date"},
                }
            },
        }

        self.es.indices.create(index=self.index_name, body=mapping)
        logger.info(f"Created Elasticsearch index: {self.index_name}")

    def insert_chunks(self, chunks_data: list[dict]) -> tuple[int, int]:
        """Bulk insert chunks to Elasticsearch. Returns (success_count, failed_count)."""
        if not chunks_data:
            return 0, 0

        actions = []
        for chunk in chunks_data:
            action = {
                "_index": self.index_name,
                "_id": chunk["id"],
                "_source": {
                    "chunk_id": chunk["id"],
                    "content": chunk["content"],
                    "full_doc_id": chunk["full_doc_id"],
                    "chunk_order_index": chunk.get("chunk_order_index", 0),
                    "created_at": datetime.utcnow().isoformat(),
                },
            }
            actions.append(action)

        try:
            success, failed = es_helpers.bulk(self.es, actions, raise_on_error=False, stats_only=True)
            return success, failed
        except Exception as e:
            logger.error(f"Elasticsearch bulk insert failed: {e}")
            return 0, len(chunks_data)

    def close(self):
        """Close Elasticsearch connection."""
        self.es.close()


def check_chunks_exist_in_milvus(collection: Collection, chunk_ids: list[str]) -> dict[str, bool]:
    """Check if chunks exist in Milvus. Returns dict mapping chunk_id -> exists."""
    if not chunk_ids:
        return {}

    try:
        expr = f"id in {chunk_ids}"
        results = collection.query(expr=expr, output_fields=["id"], limit=len(chunk_ids))
        existing_ids = {r["id"] for r in results}
        return {chunk_id: (chunk_id in existing_ids) for chunk_id in chunk_ids}
    except Exception as e:
        logger.warning(f"Milvus existence check failed: {e}, assuming none exist")
        return dict.fromkeys(chunk_ids, False)


def check_chunks_exist_in_redis(redis_client, chunk_ids: list[str], workspace: str) -> dict[str, bool]:
    """Check if chunks exist in Redis. Returns dict mapping chunk_id -> exists."""
    if not chunk_ids:
        return {}

    try:
        keys = [f"{workspace}_text_chunks:{chunk_id}" for chunk_id in chunk_ids]
        pipe = redis_client.pipeline()
        for key in keys:
            pipe.exists(key)
        results = pipe.execute()
        return {chunk_id: bool(exists) for chunk_id, exists in zip(chunk_ids, results, strict=False)}
    except Exception as e:
        logger.warning(f"Redis existence check failed: {e}, assuming none exist")
        return dict.fromkeys(chunk_ids, False)


def check_chunks_exist_in_elasticsearch(es_client, chunk_ids: list[str], index_name: str) -> dict[str, bool]:
    """Check if chunks exist in Elasticsearch. Returns dict mapping chunk_id -> exists."""
    if not chunk_ids:
        return {}

    try:
        if not es_client.indices.exists(index=index_name):
            return dict.fromkeys(chunk_ids, False)

        response = es_client.mget(index=index_name, body={"ids": chunk_ids})
        return {doc["_id"]: doc["found"] for doc in response["docs"]}
    except Exception as e:
        logger.warning(f"ES existence check failed: {e}, assuming none exist")
        return dict.fromkeys(chunk_ids, False)


def load_chunk_stats(chunk_stats_path: Path) -> dict[str, list[dict]]:
    """Load chunk statistics from CSV."""
    chunks_by_paper = {}

    with open(chunk_stats_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            paper_id = row["id"]
            if paper_id not in chunks_by_paper:
                chunks_by_paper[paper_id] = []

            chunks_by_paper[paper_id].append(
                {
                    "chunk_number": int(row["chunk"]),
                    "num_tokens": int(row["num_tokens"]),
                }
            )

    return chunks_by_paper


def load_paper_chunks(paper_id: str, chunks_dir: Path, chunk_info: list[dict]) -> tuple[str, list[str], list[dict]]:
    """
    Load all chunks for a paper.

    Chunk numbers are read directly from filenames (1.txt, 2.txt, ...),
    NOT generated by code. This ensures chunk IDs match the actual filesystem.
    """
    paper_dir = chunks_dir / paper_id

    if not paper_dir.exists():
        return paper_id, [], []

    chunk_texts = []
    chunk_metadata = []

    chunk_files = sorted(paper_dir.glob("*.txt"), key=lambda f: int(f.stem))

    token_lookup = {c["chunk_number"]: c["num_tokens"] for c in chunk_info}

    for chunk_file in chunk_files:
        chunk_number = int(chunk_file.stem)

        content = chunk_file.read_text(encoding="utf-8")
        chunk_texts.append(content)

        num_tokens = token_lookup.get(chunk_number, len(content.split()))

        chunk_metadata.append({"chunk_number": chunk_number, "num_tokens": num_tokens})

    return paper_id, chunk_texts, chunk_metadata


def save_progress(log_file: Path, paper_id: str, doc_id: str, num_chunks: int, timestamp: int):
    """
    Save progress to log file.

    Use the same timestamp that was used for chunk insertion
    to maintain consistency between log and database.
    """
    file_exists = log_file.exists()

    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["paper_id", "doc_id", "num_chunks", "timestamp"])
        writer.writerow([paper_id, doc_id, num_chunks, timestamp])


def load_processed_papers(log_file: Path) -> tuple[set, int | None]:
    """
    Load set of already processed paper IDs and max timestamp.

    Returns:
        Tuple of (processed_papers_set, max_timestamp)
        max_timestamp is None if log doesn't exist or is empty
    """
    if not log_file.exists():
        return set(), None

    processed = set()
    max_timestamp = None

    with open(log_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed.add(row["paper_id"])

            try:
                ts = int(row["timestamp"])
                if max_timestamp is None or ts > max_timestamp:
                    max_timestamp = ts
            except (ValueError, KeyError):
                continue

    return processed, max_timestamp


def main():
    parser = argparse.ArgumentParser(
        description="Insert chunks into Milvus, Redis and Elasticsearch with parallel embedding"
    )
    parser.add_argument("--input", required=True, help="Input chunks directory")
    parser.add_argument("--chunk-stats", required=True, help="Path to chunk_stats.csv")
    parser.add_argument("--milvus-host", default="localhost", help="Milvus host")
    parser.add_argument("--milvus-port", default="19530", help="Milvus port")
    parser.add_argument("--redis-host", default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument("--elasticsearch-host", default="localhost", help="Elasticsearch host")
    parser.add_argument("--elasticsearch-port", type=int, default=9200, help="Elasticsearch port")
    parser.add_argument("--workspace", default="lightrag", help="Workspace prefix of the collections and keys")
    parser.add_argument("--batch-size", type=int, default=1000, help="Paper batch size")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of papers to process")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N papers")
    parser.add_argument("--drop-existing", action="store_true", help="Drop existing collections")
    parser.add_argument("--clear-redis", action="store_true", help="Clear Redis data for workspace")
    parser.add_argument("--es-only-mode", action="store_true", help="ES-only: skip Milvus/Redis if exists")
    parser.add_argument("--skip-existing", action="store_true", default=True, help="Check existence before inserting")
    parser.add_argument("--force-reinsert", action="store_true", help="Force re-insert even if exists")

    args = parser.parse_args()

    es_only_mode = args.es_only_mode
    skip_existing = args.skip_existing and not args.force_reinsert

    chunks_dir = Path(args.input)
    chunk_stats_path = Path(args.chunk_stats)

    if not chunks_dir.exists():
        logger.error(f"Chunks directory not found: {chunks_dir}")
        sys.exit(1)

    if not chunk_stats_path.exists():
        logger.error(f"Chunk stats file not found: {chunk_stats_path}")
        sys.exit(1)

    log_file = chunks_dir.parent / "chunks-inserted.csv"
    processed_papers, max_log_timestamp = load_processed_papers(log_file)

    logger.info("=" * 60)
    logger.info("Chunk insertion")
    logger.info("=" * 60)
    logger.info(f"Chunks directory: {chunks_dir}")
    logger.info(f"Milvus: {args.milvus_host}:{args.milvus_port}")
    logger.info(f"Redis: {args.redis_host}:{args.redis_port}")
    logger.info(f"Workspace: {args.workspace}")
    logger.info(f"Already processed: {len(processed_papers)} papers")
    if max_log_timestamp:
        logger.info(f"Max log timestamp: {max_log_timestamp} ({time.ctime(max_log_timestamp)})")
    logger.info(f"Workers: {CHUNK_EMBEDDING_WORKERS}")
    logger.info(f"Chunks per batch: {CHUNKS_PER_BATCH} (groups multiple papers)")
    logger.info(f"Sub-batch size: {EMBEDDING_SUB_BATCH_SIZE}")
    logger.info(f"Milvus insert batch: {MILVUS_INSERT_BATCH_PAPERS} papers")
    logger.info(f"Max pending batches: {MAX_PENDING_BATCHES}")

    logger.info("")
    embedder = ParallelEmbeddingGenerator(
        num_workers=CHUNK_EMBEDDING_WORKERS,
        cuda_devices=CUDA_DEVICES,
        model_name="BAAI/bge-m3",
        sub_batch_size=EMBEDDING_SUB_BATCH_SIZE,
    )

    logger.info("")
    milvus_manager = MilvusChunkManager(args.milvus_host, args.milvus_port, args.workspace)
    milvus_manager.ensure_collection(drop_existing=args.drop_existing)

    if args.drop_existing and log_file.exists():
        log_file.unlink()
        logger.info(f"Deleted log file: {log_file}")

    if not args.drop_existing and max_log_timestamp is not None:
        logger.info("")
        logger.info("Removing chunks newer than the last logged insertion...")
        milvus_manager.delete_chunks_after_timestamp(max_log_timestamp)

    logger.info("")
    redis_manager = RedisChunkManager(args.redis_host, args.redis_port, args.workspace)

    if args.clear_redis:
        logger.info("")
        redis_manager.clear_workspace_data()

    logger.info("")
    es_manager = None
    try:
        es_manager = ElasticsearchChunkManager(args.elasticsearch_host, args.elasticsearch_port, args.workspace)
        es_manager.ensure_index()
    except Exception as e:
        logger.warning(f"Elasticsearch connection failed: {e}")
        logger.warning("Continuing without Elasticsearch - ES indexing will be skipped")
        es_manager = None

    logger.info("")
    logger.info("Loading chunk statistics...")
    chunks_by_paper = load_chunk_stats(chunk_stats_path)
    logger.info(f"Loaded stats for {len(chunks_by_paper)} papers")

    paper_ids = sorted(chunks_by_paper.keys())

    if args.offset > 0:
        paper_ids = paper_ids[args.offset :]

    if args.limit > 0:
        paper_ids = paper_ids[: args.limit]

    paper_ids = [pid for pid in paper_ids if pid not in processed_papers]
    logger.info(f"Papers to process: {len(paper_ids)}")

    total_chunks_to_process = sum(len(chunks_by_paper[pid]) for pid in paper_ids)
    logger.info(f"Chunks to process: {total_chunks_to_process}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("Processing Papers (Parallel Embedding)")
    logger.info("=" * 60)

    total_chunks = 0
    successful_papers = 0
    failed_batches = 0
    preparation_errors = 0
    redis_errors = 0

    try:
        batch_id = 0
        pending_batches = {}
        insertion_queue = []

        pbar_papers = tqdm(total=len(paper_ids), desc="Papers", position=0)
        pbar_chunks = tqdm(total=total_chunks_to_process, desc="Chunks", position=1)

        next_paper_idx = 0

        logger.info("Checking worker initialization...")
        time.sleep(2)
        init_errors = []
        while True:
            result = embedder.get_result(block=False)
            if result is None:
                break
            if result[0] == "WORKER_INIT_ERROR":
                init_errors.append(result[1])

        if init_errors:
            raise Exception("Worker initialization failed:\n" + "\n".join(init_errors))
        logger.info("All workers initialized successfully\n")

        while next_paper_idx < len(paper_ids) or pending_batches or insertion_queue:
            while next_paper_idx < len(paper_ids) and len(pending_batches) < MAX_PENDING_BATCHES:
                batch_papers = []
                batch_all_texts = []
                batch_all_metadata = []
                batch_chunk_counts = []

                while next_paper_idx < len(paper_ids) and len(batch_all_texts) < CHUNKS_PER_BATCH:
                    paper_id = paper_ids[next_paper_idx]
                    chunk_info = chunks_by_paper[paper_id]

                    _, chunk_texts, chunk_metadata = load_paper_chunks(paper_id, chunks_dir, chunk_info)

                    if not chunk_texts:
                        next_paper_idx += 1
                        continue

                    batch_papers.append(paper_id)
                    batch_all_texts.extend(chunk_texts)
                    batch_all_metadata.extend(chunk_metadata)
                    batch_chunk_counts.append(len(chunk_texts))

                    next_paper_idx += 1

                    if len(batch_all_texts) >= CHUNKS_PER_BATCH:
                        break

                if batch_all_texts:
                    embedder.embed_batch(batch_id, batch_all_texts)
                    pending_batches[batch_id] = (batch_papers, batch_all_texts, batch_all_metadata, batch_chunk_counts)
                    batch_id += 1

            if pending_batches:
                result = embedder.get_result(block=False)
                if result is not None:
                    result_batch_id, embeddings = result

                    if embeddings is not None and result_batch_id in pending_batches:
                        batch_papers, batch_all_texts, batch_all_metadata, batch_chunk_counts = pending_batches.pop(
                            result_batch_id
                        )

                        embedding_offset = 0
                        for _i, (paper_id, num_chunks) in enumerate(
                            zip(batch_papers, batch_chunk_counts, strict=False)
                        ):
                            paper_texts = batch_all_texts[embedding_offset : embedding_offset + num_chunks]
                            paper_metadata = batch_all_metadata[embedding_offset : embedding_offset + num_chunks]
                            paper_embeddings = embeddings[embedding_offset : embedding_offset + num_chunks]

                            insertion_queue.append((paper_id, paper_texts, paper_metadata, paper_embeddings))
                            embedding_offset += num_chunks
                    elif embeddings is None and result_batch_id in pending_batches:
                        batch_papers, _, _, _ = pending_batches.pop(result_batch_id)
                        failed_batches += 1
                        logger.error(
                            f"Embedding failed for batch {result_batch_id} "
                            f"({len(batch_papers)} papers) - papers not marked complete"
                        )

            should_flush = insertion_queue and (
                len(insertion_queue) >= MILVUS_INSERT_BATCH_PAPERS
                or (next_paper_idx >= len(paper_ids) and not pending_batches)
            )

            if should_flush:
                batch_size = min(MILVUS_INSERT_BATCH_PAPERS, len(insertion_queue))
                all_chunks_data = []
                paper_records = []

                for _ in range(batch_size):
                    paper_id, chunk_texts, chunk_metadata, embeddings = insertion_queue.pop(0)

                    try:
                        doc_id = paper_id
                        insertion_timestamp = int(time.time())

                        chunks_data = []
                        chunk_ids = []

                        for content, metadata, embedding in zip(chunk_texts, chunk_metadata, embeddings, strict=False):
                            chunk_number = metadata["chunk_number"]
                            chunk_id = f"chunk-{doc_id}-{chunk_number:06d}"
                            chunk_ids.append(chunk_id)

                            file_path = f"papers/{paper_id}.txt"

                            chunk_data = {
                                "id": chunk_id,
                                "vector": embedding.tolist(),
                                "created_at": insertion_timestamp,
                                "full_doc_id": doc_id,
                                "file_path": file_path,
                                "content": content,
                                "tokens": metadata["num_tokens"],
                                "chunk_order_index": chunk_number,
                            }

                            chunks_data.append(chunk_data)

                        all_chunks_data.extend(chunks_data)
                        paper_records.append((paper_id, doc_id, chunks_data, chunk_ids, insertion_timestamp, file_path))

                    except Exception as e:
                        logger.error(f"Error preparing {paper_id}: {e}")
                        preparation_errors += 1

                if all_chunks_data and skip_existing:
                    all_chunk_ids = [c["id"] for c in all_chunks_data]
                    milvus_existence = check_chunks_exist_in_milvus(
                        Collection(milvus_manager.collection_name), all_chunk_ids
                    )
                    redis_existence = check_chunks_exist_in_redis(redis_manager.redis, all_chunk_ids, args.workspace)
                    es_existence = (
                        check_chunks_exist_in_elasticsearch(
                            es_manager.es if es_manager else None, all_chunk_ids, CHUNK_INDEX
                        )
                        if es_manager
                        else {}
                    )

                    chunks_for_milvus = [c for c in all_chunks_data if not milvus_existence.get(c["id"], False)]
                    chunks_for_redis = [c for c in all_chunks_data if not redis_existence.get(c["id"], False)]
                    chunks_for_es = [c for c in all_chunks_data if not es_existence.get(c["id"], False)]
                else:
                    chunks_for_milvus = all_chunks_data
                    chunks_for_redis = all_chunks_data
                    chunks_for_es = all_chunks_data

                if all_chunks_data:
                    if chunks_for_milvus and not es_only_mode:
                        milvus_manager.insert_chunks(chunks_for_milvus)
                        logger.debug(f"Milvus: inserted {len(chunks_for_milvus)}/{len(all_chunks_data)} chunks")
                    elif skip_existing:
                        logger.debug(f"Milvus: all {len(all_chunks_data)} chunks already exist, skipped")

                    for paper_id, doc_id, chunks_data, chunk_ids, insertion_timestamp, file_path in paper_records:
                        try:
                            paper_chunks_for_redis = [
                                c for c in chunks_data if c["id"] in [x["id"] for x in chunks_for_redis]
                            ]

                            if paper_chunks_for_redis and not es_only_mode:
                                redis_manager.insert_chunks(paper_chunks_for_redis)
                                redis_manager.update_doc_status(doc_id, file_path, chunk_ids, insertion_timestamp)
                                logger.debug(
                                    f"Redis: inserted {len(paper_chunks_for_redis)}/{len(chunks_data)} chunks for {paper_id}"
                                )

                            if es_manager:
                                paper_chunks_for_es = [
                                    c for c in chunks_data if c["id"] in [x["id"] for x in chunks_for_es]
                                ]
                                if paper_chunks_for_es:
                                    es_success, es_failed = es_manager.insert_chunks(paper_chunks_for_es)
                                    if es_success:
                                        logger.debug(f"ES: indexed {es_success} chunks for {paper_id}")
                                    if es_failed:
                                        logger.warning(f"ES: failed {es_failed} chunks for {paper_id}")

                            save_progress(log_file, paper_id, doc_id, len(chunks_data), insertion_timestamp)

                            total_chunks += len(chunks_data)
                            successful_papers += 1
                            pbar_papers.update(1)
                            pbar_chunks.update(len(chunks_data))

                        except Exception as e:
                            logger.error(f"Error updating Redis/ES for {paper_id}: {e}")
                            redis_errors += 1

        pbar_papers.close()
        pbar_chunks.close()

    finally:
        embedder.close()
        milvus_manager.close()
        redis_manager.close()
        if es_manager:
            es_manager.close()

    logger.info("")
    logger.info("=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info(f"Papers processed: {successful_papers}/{len(paper_ids)}")
    logger.info(f"Total chunks inserted: {total_chunks}")
    if failed_batches > 0:
        logger.warning(f"Failed embedding batches: {failed_batches}")
    if preparation_errors > 0:
        logger.warning(f"Paper preparation errors: {preparation_errors}")
    if redis_errors > 0:
        logger.warning(f"Redis update errors: {redis_errors}")
    logger.info(f"Progress log: {log_file}")
    logger.info("")

    if failed_batches > 0 or preparation_errors > 0 or redis_errors > 0:
        logger.warning("Insertion completed with errors! Check logs above for details.")
    else:
        logger.info("Insertion complete!")


if __name__ == "__main__":
    main()
