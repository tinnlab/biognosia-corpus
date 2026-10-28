"""
N-gram entity matching with long-lived worker processes.

Workers:
1. Chunk loading workers (CPU): read chunks and generate n-grams
2. Embedding workers (GPU): compute MedCPT-Query-Encoder embeddings
3. Query workers: one set per entities_* collection, searching label_embedding

The Genes and Gene_Synonyms collections are searched together and reduced to the
single best gene hit per n-gram. Search results for each n-gram are cached in Redis.
"""

import argparse
import csv
import json
import os
import re
import signal
import sys
import time
from dataclasses import dataclass
from multiprocessing import Process, Queue
from pathlib import Path
from queue import Empty

import numpy as np
import redis
from tqdm import tqdm

from .ngram_entity_matcher import (
    EntityMatch,
    EntitySearcher,
    NGram,
    NGramEmbedder,
    NGramGenerator,
    merge_gene_matches,
)


@dataclass
class ChunkInfo:
    """Information about a chunk to be processed."""

    paper_id: str
    chunk_id: str
    chunk_path: Path
    text: str
    ngrams: list[NGram]


@dataclass
class MatchResult:
    """Result of n-gram matching for a single chunk."""

    paper_id: str
    chunk_id: str
    num_ngrams_generated: int
    num_ngrams_matched: int
    num_unique_entities: int
    avg_similarity_label: float
    avg_similarity_name: float
    processing_time: float
    success: bool
    error_message: str = ""


SENTINEL = None


def normalize_entity_name(entity_name: str) -> str:
    """
    Normalize entity name for consistency.

    For GENE: entities, uppercase the gene symbol portion.
    Example: "GENE:rabS" -> "GENE:RABS"

    Args:
        entity_name: Entity name to normalize

    Returns:
        Normalized entity name
    """
    if entity_name.startswith("GENE:"):
        prefix = "GENE:"
        symbol = entity_name[len(prefix) :]
        return prefix + symbol.upper()
    return entity_name


def cache_load_worker(worker_id: int, task_queue: Queue, result_queue: Queue):
    """
    Worker that loads cache from TSV result files.

    Args:
        worker_id: Worker identifier
        task_queue: Queue to receive TSV file paths
        result_queue: Queue to send cache data
    """
    import csv

    while True:
        task = task_queue.get()

        if task is SENTINEL:
            break

        try:
            tsv_path = task
            cache_data = {}

            with open(tsv_path, encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")

                for row in reader:
                    extracted_text = row["extracted_text"].strip()
                    entity_name = row["entity_name"].strip()
                    entity_label = row["entity_label"].strip()
                    entity_type = row["entity_type"].strip()
                    similarity_score_label = float(row["similarity_score_label"])

                    if extracted_text not in cache_data:
                        cache_data[extracted_text] = [
                            {
                                "entity_name": entity_name,
                                "entity_label": entity_label,
                                "entity_type": entity_type,
                                "entity_content": "",
                                "similarity_score_label": similarity_score_label,
                            }
                        ]

            result_queue.put((tsv_path, cache_data, None))

        except Exception as e:
            result_queue.put((None, None, str(e)))


def chunk_load_worker(worker_id: int, task_queue: Queue, result_queue: Queue, k_values: list[int]):
    """
    Long-lived worker that loads chunks and generates n-grams.

    Args:
        worker_id: Worker identifier
        task_queue: Queue to receive chunk paths
        result_queue: Queue to send (ChunkInfo, unique_ngrams) results
        k_values: N-gram sizes
    """
    generator = NGramGenerator(k_values)

    while True:
        chunk_path = task_queue.get()

        if chunk_path is SENTINEL:
            break

        try:
            chunk_id = chunk_path.stem
            paper_id = chunk_path.parent.name

            with open(chunk_path, encoding="utf-8") as f:
                text = f.read()

            ngrams = generator.generate(text)

            chunk_info = ChunkInfo(
                paper_id=paper_id, chunk_id=chunk_id, chunk_path=chunk_path, text=text, ngrams=ngrams
            )

            unique_ngrams = {ng.text for ng in ngrams}

            result_queue.put((chunk_info, unique_ngrams))

        except Exception as e:
            result_queue.put((None, str(e)))


def embedding_worker(worker_id: int, task_queue: Queue, result_queue: Queue, device: str, batch_size: int):
    """
    Long-lived worker that computes embeddings on a specific GPU.

    Args:
        worker_id: Worker identifier
        task_queue: Queue to receive text batches
        result_queue: Queue to send embedding results
        device: CUDA device (e.g., "cuda:0")
        batch_size: Batch size for embedding
    """
    embedder = NGramEmbedder(model_name="ncbi/MedCPT-Query-Encoder", device=device, batch_size=batch_size)

    while True:
        task = task_queue.get()

        if task is SENTINEL:
            break

        try:
            texts, task_id = task

            embeddings = embedder.embed_batch(texts)

            result_queue.put((task_id, embeddings, None))

        except Exception as e:
            result_queue.put((None, None, str(e)))


def query_worker(
    worker_id: int,
    collection_name: str,
    task_queue: Queue,
    result_queue: Queue,
    milvus_host: str,
    milvus_port: str,
    workspace: str,
    min_similarity: float,
):
    """
    Long-lived worker that queries a specific entity collection.

    Each worker is assigned to ONE entity table and queries it repeatedly.
    Handles internal sub-batching if embeddings exceed Milvus limit.

    Args:
        worker_id: Worker identifier
        collection_name: Name of entity collection to query
        task_queue: Queue to receive embedding batches
        result_queue: Queue to send query results
        milvus_host: Milvus host
        milvus_port: Milvus port
        workspace: Workspace prefix
        min_similarity: Minimum similarity threshold
    """
    time.sleep(worker_id * 0.5)

    searcher = EntitySearcher(milvus_host=milvus_host, milvus_port=milvus_port, workspace=workspace)

    max_milvus_batch = 15000

    while True:
        task = task_queue.get()

        if task is SENTINEL:
            break

        try:
            embeddings, task_id = task

            if len(embeddings) <= max_milvus_batch:
                results = searcher.search_single_collection(collection_name, embeddings, top_k=1, min_similarity=0.0)
            else:
                results = [[] for _ in range(len(embeddings))]

                for batch_start in range(0, len(embeddings), max_milvus_batch):
                    batch_end = min(batch_start + max_milvus_batch, len(embeddings))
                    sub_batch = embeddings[batch_start:batch_end]

                    sub_results = searcher.search_single_collection(
                        collection_name, sub_batch, top_k=1, min_similarity=0.0
                    )

                    for local_idx, matches in enumerate(sub_results):
                        results[batch_start + local_idx].extend(matches)

            result_queue.put((collection_name, results, task_id, None))

        except Exception as e:
            result_queue.put((collection_name, None, None, str(e)))


class ParallelNGramMatcher:
    """
    Parallel n-gram entity matcher with long-lived worker processes.
    """

    def __init__(
        self,
        k_values: list[int],
        min_similarity: float,
        milvus_host: str,
        milvus_port: str,
        workspace: str,
        output_dir: Path,
        embedding_batch_size: int = 128,
        chunk_load_workers: int = 16,
        embedding_workers: int = 8,
        query_workers_per_collection: int = 2,
        redis_host: str = "localhost",
        redis_port: int = 6380,
        redis_batch_size: int = 10000,
    ):
        """
        Initialize parallel matcher and create all worker processes.

        Args:
            k_values: N-gram sizes
            min_similarity: Similarity threshold applied when matches are written
            milvus_host: Milvus host
            milvus_port: Milvus port
            workspace: Workspace prefix
            output_dir: Directory for output TSV files
            embedding_batch_size: Batch size for computing embeddings
            chunk_load_workers: Number of parallel workers for loading chunks and cache
            embedding_workers: Number of parallel GPU workers for embedding
            query_workers_per_collection: Number of parallel query workers per collection
            redis_host: Redis host for caching
            redis_port: Redis port for caching
            redis_batch_size: Batch size for Redis cache operations (read/write)
        """
        self.k_values = k_values
        self.min_similarity = min_similarity
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.workspace = workspace
        self.output_dir = output_dir
        self.embedding_batch_size = embedding_batch_size
        self.chunk_load_workers = chunk_load_workers
        self.embedding_workers = embedding_workers
        self.query_workers_per_collection = query_workers_per_collection
        self.redis_batch_size = redis_batch_size

        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=0,
            decode_responses=False,
            socket_connect_timeout=5,
            socket_keepalive=True,
        )

        self.positive_cache_key = f"{workspace}:ngram:positive"
        self.negative_cache_key = f"{workspace}:ngram:negative"

        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")
        self.gpu_ids = [int(d.strip()) for d in cuda_devices.split(",") if d.strip()]

        temp_searcher = EntitySearcher(milvus_host, milvus_port, workspace)
        self.collection_names = temp_searcher.get_collection_names()
        del temp_searcher

        print("\nParallel Architecture:")
        print(f"  Chunk load workers: {chunk_load_workers}")
        print(f"  Embedding workers: {embedding_workers} (across {len(self.gpu_ids)} GPUs)")
        print(f"  Entity collections: {len(self.collection_names)}")
        print(f"  Query workers per collection: {query_workers_per_collection}")
        print(f"  Total query workers: {len(self.collection_names) * query_workers_per_collection}")
        print(f"  Output directory: {output_dir}")
        print(f"  Redis cache: {redis_host}:{redis_port}")

        self._check_redis_cache()

        self._create_workers()

    def _check_redis_cache(self):
        """Check Redis cache status and report statistics."""
        try:
            self.redis_client.ping()
            positive_count = self.redis_client.hlen(self.positive_cache_key)
            negative_count = self.redis_client.scard(self.negative_cache_key)
            total_cached = positive_count + negative_count

            print("\n  Redis Cache Status:")
            print(f"    Positive cache: {positive_count} n-grams with matches")
            print(f"    Negative cache: {negative_count} n-grams without matches")
            print(f"    Total cached: {total_cached} n-grams")

        except redis.ConnectionError as e:
            print(f"\n  Warning: Failed to connect to Redis: {e}")
            conn_kwargs = self.redis_client.connection_pool.connection_kwargs
            print(
                f"  Cache will start empty. Please ensure Redis is running at "
                f"{conn_kwargs['host']}:{conn_kwargs['port']}"
            )
        except Exception as e:
            print(f"\n  Warning: Error checking Redis cache: {e}")

    def _create_workers(self):
        """Create all worker processes and queues."""

        print(f"\nCreating {self.chunk_load_workers} chunk loading workers...")
        self.chunk_task_queue = Queue(maxsize=self.chunk_load_workers * 2)
        self.chunk_result_queue = Queue()
        self.chunk_workers = []

        for i in range(self.chunk_load_workers):
            worker = Process(
                target=chunk_load_worker, args=(i, self.chunk_task_queue, self.chunk_result_queue, self.k_values)
            )
            worker.start()
            self.chunk_workers.append(worker)

        print(f"Creating {self.embedding_workers} embedding workers...")
        self.embedding_task_queue = Queue(maxsize=self.embedding_workers * 2)
        self.embedding_result_queue = Queue()
        self.embedding_worker_processes = []

        for i in range(self.embedding_workers):
            device = f"cuda:{self.gpu_ids[i % len(self.gpu_ids)]}"
            worker = Process(
                target=embedding_worker,
                args=(i, self.embedding_task_queue, self.embedding_result_queue, device, self.embedding_batch_size),
            )
            worker.start()
            self.embedding_worker_processes.append(worker)

        print(f"Creating {len(self.collection_names) * self.query_workers_per_collection} query workers...")
        self.query_task_queues = {}
        self.query_result_queue = Queue()
        self.query_worker_processes = []

        for collection_name in self.collection_names:
            self.query_task_queues[collection_name] = []

            for _worker_idx in range(self.query_workers_per_collection):
                task_queue = Queue(maxsize=5)
                self.query_task_queues[collection_name].append(task_queue)

                worker = Process(
                    target=query_worker,
                    args=(
                        len(self.query_worker_processes),
                        collection_name,
                        task_queue,
                        self.query_result_queue,
                        self.milvus_host,
                        self.milvus_port,
                        self.workspace,
                        self.min_similarity,
                    ),
                )
                worker.start()
                self.query_worker_processes.append(worker)

        print("All workers created and ready!\n")

    def _compute_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        vec1 = vec1.flatten()
        vec2 = vec2.flatten()

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def process_batch(self, chunk_paths: list[Path]) -> tuple[list[ChunkInfo], dict[str, dict], dict[str, list[dict]]]:
        """
        Process a batch of chunks using long-lived workers with caching.

        Steps:
        1. Send chunk paths to chunk workers
        2. Collect chunk infos and unique n-grams
        3. Check cache for existing results
        4. Send uncached n-grams to embedding workers
        5. Collect embeddings
        6. Send embeddings to all query workers (each queries its table)
        7. Collect query results and merge with cache

        Returns:
            Tuple of (chunk_infos, ngram_matches, full_ngram_results)
            - ngram_matches: all n-grams with their best entity matches
            - full_ngram_results: ALL query results for caching
        """
        batch_start_time = time.time()

        step1_start = time.time()
        print(f"  [Step 1] Sending {len(chunk_paths)} chunks to {self.chunk_load_workers} workers...")

        for chunk_path in chunk_paths:
            self.chunk_task_queue.put(chunk_path)

        chunk_infos = []
        unique_ngrams = set()

        for _ in range(len(chunk_paths)):
            try:
                chunk_info, chunk_unique_ngrams = self.chunk_result_queue.get(timeout=300)
            except Empty as exc:
                print("  [ERROR] Timeout waiting for chunk worker results after 5 minutes")
                raise RuntimeError("Chunk worker timeout - workers may have crashed") from exc

            if chunk_info is not None:
                chunk_infos.append(chunk_info)
                unique_ngrams.update(chunk_unique_ngrams)

        step1_time = time.time() - step1_start
        print(f"  [Step 1] Loaded {len(chunk_infos)} chunks, {len(unique_ngrams)} unique n-grams ({step1_time:.2f}s)")

        if not unique_ngrams:
            return chunk_infos, {}, {}

        cache_check_start = time.time()
        cached_ngrams = {}
        uncached_ngrams = []

        ngram_list = list(unique_ngrams)

        print(f"  [Cache] Checking {len(ngram_list)} n-grams in Redis cache (batch size: {self.redis_batch_size})...")

        for batch_start in range(0, len(ngram_list), self.redis_batch_size):
            batch_end = min(batch_start + self.redis_batch_size, len(ngram_list))
            batch_ngrams = ngram_list[batch_start:batch_end]

            positive_results = self.redis_client.hmget(self.positive_cache_key, batch_ngrams)

            try:
                negative_results = self.redis_client.smismember(self.negative_cache_key, batch_ngrams)
            except redis.ResponseError:
                pipe = self.redis_client.pipeline()
                for ngram in batch_ngrams:
                    pipe.sismember(self.negative_cache_key, ngram)
                negative_results = pipe.execute()

            for i, ngram_text in enumerate(batch_ngrams):
                positive_value = positive_results[i]
                is_negative = negative_results[i]

                if positive_value is not None:
                    cached_ngrams[ngram_text] = json.loads(positive_value.decode("utf-8"))
                elif is_negative:
                    cached_ngrams[ngram_text] = []
                else:
                    uncached_ngrams.append(ngram_text)

        cache_check_time = time.time() - cache_check_start

        total_ngrams = len(unique_ngrams)
        cache_hit_rate = (len(cached_ngrams) / total_ngrams * 100) if total_ngrams > 0 else 0
        positive_hits = sum(1 for v in cached_ngrams.values() if v)
        negative_hits = len(cached_ngrams) - positive_hits

        print(f"  [Cache] {len(cached_ngrams)}/{total_ngrams} n-grams cached ({cache_hit_rate:.1f}% hit rate)")
        print(f"          {positive_hits} with matches, {negative_hits} without matches")
        print(f"  [Cache] {len(uncached_ngrams)} n-grams need database queries ({cache_check_time:.2f}s)")

        full_ngram_results = dict(cached_ngrams)

        if not uncached_ngrams:
            print("  [Cache] All n-grams found in cache, skipping database queries!")
            step2_time = 0.0
            step3_time = 0.0
        else:
            uncached_ngram_list = sorted(uncached_ngrams)

            step2_start = time.time()
            print(
                f"  [Step 2] Embedding {len(uncached_ngram_list)} UNCACHED n-grams "
                f"with {self.embedding_workers} workers..."
            )

            texts_per_worker = (len(uncached_ngram_list) + self.embedding_workers - 1) // self.embedding_workers

            for worker_idx in range(self.embedding_workers):
                start_idx = worker_idx * texts_per_worker
                end_idx = min(start_idx + texts_per_worker, len(uncached_ngram_list))

                if start_idx >= len(uncached_ngram_list):
                    break

                texts = uncached_ngram_list[start_idx:end_idx]
                self.embedding_task_queue.put((texts, worker_idx))

            embedding_results = {}
            num_workers_used = min(
                self.embedding_workers, (len(uncached_ngram_list) + texts_per_worker - 1) // texts_per_worker
            )

            for _ in range(num_workers_used):
                try:
                    task_id, embeddings, error = self.embedding_result_queue.get(timeout=600)
                except Empty as exc:
                    print("  [ERROR] Timeout waiting for embedding worker results after 10 minutes")
                    raise RuntimeError("Embedding worker timeout - workers may have crashed") from exc

                if error:
                    raise RuntimeError(f"Embedding worker error: {error}")
                embedding_results[task_id] = embeddings

            uncached_embeddings = np.concatenate([embedding_results[i] for i in range(num_workers_used)], axis=0)

            step2_time = time.time() - step2_start
            print(f"  [Step 2] Embedded {len(uncached_embeddings)} n-grams ({step2_time:.2f}s)")

            step3_start = time.time()
            print(
                f"  [Step 3] Querying {len(self.collection_names)} collections with "
                f"{self.query_workers_per_collection} workers each..."
            )
            print(f"           Each worker handles a slice of {len(uncached_embeddings)} n-grams")

            collection_results = {name: [[] for _ in range(len(uncached_embeddings))] for name in self.collection_names}

            embeddings_per_worker = (
                len(uncached_embeddings) + self.query_workers_per_collection - 1
            ) // self.query_workers_per_collection

            task_counter = 0
            worker_assignments = {}

            for collection_name in self.collection_names:
                for worker_idx in range(self.query_workers_per_collection):
                    start_idx = worker_idx * embeddings_per_worker
                    end_idx = min(start_idx + embeddings_per_worker, len(uncached_embeddings))

                    if start_idx >= len(uncached_embeddings):
                        break

                    embeddings_slice = uncached_embeddings[start_idx:end_idx]

                    self.query_task_queues[collection_name][worker_idx].put((embeddings_slice, task_counter))

                    worker_assignments[task_counter] = (collection_name, start_idx, end_idx)
                    task_counter += 1

            print(
                f"           Sent {task_counter} tasks "
                f"({len(self.collection_names)} collections x {self.query_workers_per_collection} workers)"
            )

            for _ in range(task_counter):
                try:
                    collection_name, results, task_id, error = self.query_result_queue.get(timeout=600)
                except Empty as exc:
                    print("  [ERROR] Timeout waiting for query worker results after 10 minutes")
                    raise RuntimeError(
                        "Query worker timeout - workers may have crashed or Milvus connection failed"
                    ) from exc

                if error:
                    print(f"Warning: Query worker error for {collection_name}: {error}")
                    continue

                _, start_idx, end_idx = worker_assignments[task_id]

                for local_idx, matches in enumerate(results):
                    global_idx = start_idx + local_idx
                    if global_idx < len(uncached_embeddings):
                        collection_results[collection_name][global_idx].extend(matches)

            search_results = [[] for _ in range(len(uncached_embeddings))]

            for results in collection_results.values():
                for ngram_idx, matches in enumerate(results):
                    search_results[ngram_idx].extend(matches)

            search_results = [merge_gene_matches(matches) for matches in search_results]

            step3_time = time.time() - step3_start
            print(f"  [Step 3] Queried all collections ({step3_time:.2f}s)")

            cache_update_start = time.time()

            positive_updates = {}
            negative_ngrams = []

            for ngram_text, results in zip(uncached_ngram_list, search_results, strict=False):
                full_ngram_results[ngram_text] = results
                if results:
                    positive_updates[ngram_text] = json.dumps(results, ensure_ascii=False)
                else:
                    negative_ngrams.append(ngram_text)

            if positive_updates:
                print(
                    f"  [Cache] Updating {len(positive_updates)} positive cache entries "
                    f"(batch size: {self.redis_batch_size})..."
                )
                positive_items = list(positive_updates.items())

                for batch_start in range(0, len(positive_items), self.redis_batch_size):
                    batch_end = min(batch_start + self.redis_batch_size, len(positive_items))
                    batch_dict = dict(positive_items[batch_start:batch_end])
                    self.redis_client.hset(self.positive_cache_key, mapping=batch_dict)

            if negative_ngrams:
                print(
                    f"  [Cache] Updating {len(negative_ngrams)} negative cache entries "
                    f"(batch size: {self.redis_batch_size})..."
                )

                for batch_start in range(0, len(negative_ngrams), self.redis_batch_size):
                    batch_end = min(batch_start + self.redis_batch_size, len(negative_ngrams))
                    batch = negative_ngrams[batch_start:batch_end]
                    self.redis_client.sadd(self.negative_cache_key, *batch)

            cache_update_time = time.time() - cache_update_start
            print(f"  [Cache] Updated Redis cache in {cache_update_time:.2f}s")

        step4_start = time.time()
        print("  [Step 4] Building result lookup table...")

        ngram_matches = {}
        ngrams_with_matches = 0
        ngrams_without_matches = 0

        for ngram_text, results in full_ngram_results.items():
            if not results:
                ngram_matches[ngram_text] = [
                    {
                        "entity_name": "",
                        "entity_label": "",
                        "entity_type": "",
                        "entity_content": "",
                        "similarity_score_label": 0.0,
                    }
                ]
                ngrams_without_matches += 1
            else:
                matches_by_type = {}
                for match in results:
                    entity_type = match["entity_type"]
                    if entity_type not in matches_by_type:
                        matches_by_type[entity_type] = []
                    matches_by_type[entity_type].append(match)

                top_matches = []
                for type_matches in matches_by_type.values():
                    sorted_matches = sorted(type_matches, key=lambda x: x["similarity_score_label"], reverse=True)
                    for match in sorted_matches[:5]:
                        if match["similarity_score_label"] >= self.min_similarity:
                            normalized_entity_name = normalize_entity_name(match["entity_name"])
                            top_matches.append(
                                {
                                    "entity_name": normalized_entity_name,
                                    "entity_label": match["entity_label"],
                                    "entity_type": match["entity_type"],
                                    "entity_content": match["entity_content"] or "",
                                    "similarity_score_label": match["similarity_score_label"],
                                }
                            )

                if top_matches:
                    ngram_matches[ngram_text] = top_matches
                    ngrams_with_matches += 1
                else:
                    ngram_matches[ngram_text] = [
                        {
                            "entity_name": "",
                            "entity_label": "",
                            "entity_type": "",
                            "entity_content": "",
                            "similarity_score_label": 0.0,
                        }
                    ]
                    ngrams_without_matches += 1

        step4_time = time.time() - step4_start
        print(f"  [Step 4] Built lookup table for {len(ngram_matches)} n-grams ({step4_time:.2f}s)")
        print(f"           {ngrams_with_matches} with entity matches, {ngrams_without_matches} without matches")

        batch_total_time = time.time() - batch_start_time
        print(f"  [TIMING] Total batch processing time: {batch_total_time:.2f}s")

        return chunk_infos, ngram_matches, full_ngram_results

    def shutdown(self):
        """Shutdown all worker processes. Redis cache persists automatically."""
        print("\nShutting down workers...")

        try:
            positive_count = self.redis_client.hlen(self.positive_cache_key)
            negative_count = self.redis_client.scard(self.negative_cache_key)
            total = positive_count + negative_count
            print(
                f"  Redis cache persisted: {positive_count} positive + {negative_count} negative = "
                f"{total} total n-grams"
            )
        except Exception as e:
            print(f"  Warning: Could not get Redis cache status: {e}")

        for _ in range(self.chunk_load_workers):
            self.chunk_task_queue.put(SENTINEL)

        for worker in self.chunk_workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()

        for _ in range(self.embedding_workers):
            self.embedding_task_queue.put(SENTINEL)

        for worker in self.embedding_worker_processes:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()

        for collection_name in self.collection_names:
            for task_queue in self.query_task_queues[collection_name]:
                task_queue.put(SENTINEL)

        for worker in self.query_worker_processes:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()

        try:
            self.redis_client.close()
        except Exception:
            pass

        print("All workers shutdown complete!")


def process_paper_batch(
    chunk_paths: list[Path],
    output_dir: Path,
    error_output_dir: Path,
    stats_output_dir: Path,
    matcher: ParallelNGramMatcher,
    force: bool = False,
) -> list[MatchResult]:
    """Process a batch of papers using the matcher."""

    if not force:
        chunks_to_process = []
        for chunk_path in chunk_paths:
            chunk_id = chunk_path.stem
            paper_id = chunk_path.parent.name
            output_path = output_dir / paper_id / f"{chunk_id}.tsv"

            if not output_path.exists():
                chunks_to_process.append(chunk_path)

        chunk_paths = chunks_to_process

    if not chunk_paths:
        return []

    try:
        chunk_infos, ngram_matches, full_ngram_results = matcher.process_batch(chunk_paths)
    except Exception as e:
        print(f"Error processing batch: {e}")
        import traceback

        traceback.print_exc()

        results = []
        for chunk_path in chunk_paths:
            chunk_id = chunk_path.stem
            paper_id = chunk_path.parent.name
            results.append(
                MatchResult(
                    paper_id=paper_id,
                    chunk_id=chunk_id,
                    num_ngrams_generated=0,
                    num_ngrams_matched=0,
                    num_unique_entities=0,
                    avg_similarity_label=0.0,
                    avg_similarity_name=0.0,
                    processing_time=0.0,
                    success=False,
                    error_message=str(e),
                )
            )
        return results

    results = []

    for chunk_info in chunk_infos:
        start_time = time.time()

        try:
            matches_with_entities = []
            matches_without_entities = []

            for ngram in chunk_info.ngrams:
                if ngram.text in ngram_matches:
                    match_infos = ngram_matches[ngram.text]
                    for match_info in match_infos:
                        match = EntityMatch(
                            extracted_text=ngram.text,
                            start=ngram.start,
                            end=ngram.end,
                            entity_name=match_info["entity_name"],
                            entity_label=match_info["entity_label"],
                            entity_type=match_info["entity_type"],
                            entity_content=match_info["entity_content"],
                            similarity_score_label=match_info["similarity_score_label"],
                            similarity_score_content=0.0,
                        )

                        if match_info["entity_name"]:
                            matches_with_entities.append(match)
                        else:
                            matches_without_entities.append(match)

            matches = matches_with_entities + matches_without_entities

            matches = sorted(matches, key=lambda m: (-m.similarity_score_label, m.start))

            num_unique_entities = len({m.entity_name for m in matches if m.entity_name})
            matches_with_entities = [m for m in matches if m.entity_name]
            avg_similarity_label = (
                sum(m.similarity_score_label for m in matches_with_entities) / len(matches_with_entities)
                if matches_with_entities
                else 0.0
            )

            output_path = output_dir / chunk_info.paper_id / f"{chunk_info.chunk_id}.tsv"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write("extracted_text\tstart\tend\tentity_name\tentity_label\tentity_type\tsimilarity_score_label\n")

                for match in matches:
                    sanitized_text = match.extracted_text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
                    sanitized_text = re.sub(r"\s+", " ", sanitized_text).strip()

                    f.write(
                        f"{sanitized_text}\t{match.start}\t{match.end}\t"
                        f"{match.entity_name}\t{match.entity_label}\t"
                        f"{match.entity_type}\t"
                        f"{match.similarity_score_label:.4f}\n"
                    )

            error_path = error_output_dir / chunk_info.paper_id / f"{chunk_info.chunk_id}.txt"
            if error_path.exists():
                error_path.unlink()

            processing_time = time.time() - start_time

            results.append(
                MatchResult(
                    paper_id=chunk_info.paper_id,
                    chunk_id=chunk_info.chunk_id,
                    num_ngrams_generated=len(chunk_info.ngrams),
                    num_ngrams_matched=len(matches),
                    num_unique_entities=num_unique_entities,
                    avg_similarity_label=avg_similarity_label,
                    avg_similarity_name=0.0,
                    processing_time=processing_time,
                    success=True,
                )
            )

        except Exception as e:
            processing_time = time.time() - start_time

            error_path = error_output_dir / chunk_info.paper_id / f"{chunk_info.chunk_id}.txt"
            error_path.parent.mkdir(parents=True, exist_ok=True)
            with open(error_path, "w", encoding="utf-8") as f:
                f.write(f"Error: {e}\n")

            results.append(
                MatchResult(
                    paper_id=chunk_info.paper_id,
                    chunk_id=chunk_info.chunk_id,
                    num_ngrams_generated=0,
                    num_ngrams_matched=0,
                    num_unique_entities=0,
                    avg_similarity_label=0.0,
                    avg_similarity_name=0.0,
                    processing_time=processing_time,
                    success=False,
                    error_message=str(e),
                )
            )

    return results


def discover_papers_and_chunks(input_dir: Path, limit: int | None = None, offset: int | None = None):
    """Map each paper id to its sorted list of chunk files."""
    paper_dirs = sorted(d for d in input_dir.iterdir() if d.is_dir())
    if offset:
        paper_dirs = paper_dirs[offset:]
    if limit:
        paper_dirs = paper_dirs[:limit]

    papers = {}
    for paper_dir in paper_dirs:
        chunks = sorted(paper_dir.glob("*.txt"))
        if chunks:
            papers[paper_dir.name] = chunks
    return papers


def write_stats_to_csv(results: list[MatchResult], stats_file: Path):
    """Append matching results to the statistics CSV file."""
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    file_exists = stats_file.exists()

    with open(stats_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "paper_id",
                    "chunk_id",
                    "num_ngrams_generated",
                    "num_ngrams_matched",
                    "num_unique_entities",
                    "avg_similarity_label",
                    "avg_similarity_name",
                    "processing_time_sec",
                    "status",
                    "error_message",
                ]
            )
        for result in results:
            writer.writerow(
                [
                    result.paper_id,
                    result.chunk_id,
                    result.num_ngrams_generated,
                    result.num_ngrams_matched,
                    result.num_unique_entities,
                    f"{result.avg_similarity_label:.4f}",
                    f"{result.avg_similarity_name:.4f}",
                    f"{result.processing_time:.4f}",
                    "success" if result.success else "error",
                    result.error_message,
                ]
            )


_matcher = None


def signal_handler(signum, frame):
    """Handle termination signals to cleanup workers properly."""
    global _matcher
    print(f"\n\nReceived signal {signum}, shutting down gracefully...")
    if _matcher:
        try:
            _matcher.shutdown()
        except Exception as e:
            print(f"Error during shutdown: {e}")
    sys.exit(1)


def main():
    """Main entry point with long-lived worker architecture."""
    global _matcher

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(description="Match entities with long-lived parallel workers")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--error-output", type=Path, required=True)
    parser.add_argument("--stats-output", type=Path, required=True)
    parser.add_argument("--paper-batch", type=int, default=100)
    parser.add_argument("--k-values", type=str, default="1,2,3,4,5")
    parser.add_argument("--min-similarity", type=float, default=0.85)
    parser.add_argument("--milvus-host", type=str, default="localhost")
    parser.add_argument("--milvus-port", type=str, default="19530")
    parser.add_argument("--workspace", type=str, default="lightrag")
    parser.add_argument("--redis-host", type=str, default="localhost")
    parser.add_argument("--redis-port", type=int, default=6380)
    parser.add_argument("--redis-batch-size", type=int, default=10000)
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--chunk-load-workers", type=int, default=16)
    parser.add_argument("--embedding-workers", type=int, default=8)
    parser.add_argument("--query-workers-per-collection", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--paper-ids",
        type=str,
        default=None,
        help="Comma-separated list of specific paper IDs to process",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--clear-cache", action="store_true", help="Clear Redis cache before starting entity extraction"
    )

    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    args.error_output.mkdir(parents=True, exist_ok=True)
    args.stats_output.mkdir(parents=True, exist_ok=True)

    k_values = [int(k.strip()) for k in args.k_values.split(",")]

    print("Parallel N-gram Matching Configuration:")
    print(f"  K-values: {k_values}")
    print(f"  Minimum similarity: {args.min_similarity}")
    print(f"  Paper batch size: {args.paper_batch}")
    print(f"  Chunk load workers: {args.chunk_load_workers}")
    print(f"  Embedding workers: {args.embedding_workers}")
    print(f"  Query workers per collection: {args.query_workers_per_collection}")

    print(f"\nDiscovering papers in {args.input}...")
    limit = args.limit if args.limit > 0 else None
    offset = args.offset if args.offset > 0 else None
    papers = discover_papers_and_chunks(args.input, limit=limit, offset=offset)

    if args.paper_ids:
        paper_ids_set = {pid.strip() for pid in args.paper_ids.split(",") if pid.strip()}
        print(f"Filtering to specific paper IDs: {sorted(paper_ids_set)}")
        papers = {pid: chunks for pid, chunks in papers.items() if pid in paper_ids_set}

    total_chunks = sum(len(chunks) for chunks in papers.values())
    print(f"Found {len(papers)} papers with {total_chunks} total chunks")

    if not papers:
        print("No papers found!")
        return

    if args.clear_cache:
        print("\n" + "=" * 80)
        print("Clearing Redis cache...")
        print("=" * 80)
        try:
            cache_client = redis.Redis(
                host=args.redis_host, port=args.redis_port, db=0, decode_responses=False, socket_connect_timeout=5
            )
            cache_client.ping()

            positive_key = f"{args.workspace}:ngram:positive"
            negative_key = f"{args.workspace}:ngram:negative"

            positive_count = cache_client.hlen(positive_key)
            negative_count = cache_client.scard(negative_key)

            cache_client.delete(positive_key)
            cache_client.delete(negative_key)

            cache_client.close()

            print(f"Cleared {positive_count} positive cache entries")
            print(f"Cleared {negative_count} negative cache entries")
            print(f"Total cleared: {positive_count + negative_count} n-gram entries")
        except redis.ConnectionError as e:
            print(f"Failed to connect to Redis: {e}")
            print("  Cache not cleared")
        except Exception as e:
            print(f"Error clearing cache: {e}")
        print("=" * 80 + "\n")

    matcher = ParallelNGramMatcher(
        k_values=k_values,
        min_similarity=args.min_similarity,
        milvus_host=args.milvus_host,
        milvus_port=args.milvus_port,
        workspace=args.workspace,
        output_dir=args.output,
        embedding_batch_size=args.embedding_batch_size,
        chunk_load_workers=args.chunk_load_workers,
        embedding_workers=args.embedding_workers,
        query_workers_per_collection=args.query_workers_per_collection,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_batch_size=args.redis_batch_size,
    )
    _matcher = matcher

    try:
        stats_file = args.stats_output / "matching_stats.csv"
        paper_ids = sorted(papers.keys())
        num_batches = (len(paper_ids) + args.paper_batch - 1) // args.paper_batch

        print(f"\nProcessing {len(paper_ids)} papers in {num_batches} batches...")
        start_time = time.time()

        total_successful = 0
        total_failed = 0

        with tqdm(total=total_chunks, desc="Matching entities", unit="chunk") as pbar:
            for batch_idx in range(num_batches):
                batch_start = batch_idx * args.paper_batch
                batch_end = min(batch_start + args.paper_batch, len(paper_ids))
                batch_paper_ids = paper_ids[batch_start:batch_end]

                batch_chunks = []
                for paper_id in batch_paper_ids:
                    batch_chunks.extend(papers[paper_id])

                print(
                    f"\nBatch {batch_idx + 1}/{num_batches}: {len(batch_paper_ids)} papers, {len(batch_chunks)} chunks"
                )

                results = process_paper_batch(
                    batch_chunks, args.output, args.error_output, args.stats_output, matcher, force=args.force
                )

                if results:
                    write_stats_to_csv(results, stats_file)

                    for result in results:
                        if result.success:
                            total_successful += 1
                        else:
                            total_failed += 1

                    pbar.update(len(results))

                    elapsed = time.time() - start_time
                    rate = (total_successful + total_failed) / elapsed if elapsed > 0 else 0
                    pbar.set_postfix({"success": total_successful, "failed": total_failed, "rate": f"{rate:.2f}/s"})

        elapsed_time = time.time() - start_time

        print("\n" + "=" * 80)
        print("Entity matching complete!")
        print(f"Total time: {elapsed_time:.2f} seconds")
        print(f"Successful: {total_successful}")
        print(f"Failed: {total_failed}")
        if (total_successful + total_failed) > 0:
            print(f"Processing rate: {(total_successful + total_failed) / elapsed_time:.2f} chunks/sec")
        print("=" * 80)

    finally:
        matcher.shutdown()


if __name__ == "__main__":
    main()
