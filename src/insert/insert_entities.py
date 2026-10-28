#!/usr/bin/env python3
"""
Entity insertion.

Links the entities matched in each chunk to the reference entities in Milvus.
New entities are written to the <workspace>_entities collection in Milvus.
The entity-to-chunk and chunk-to-entity mappings are written to Redis.
"""

import argparse
import csv
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import redis
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

PREFIX_TO_COLLECTION = {
    "GENE:": "entities_Genes",
    "DISEASE:": "entities_Disease",
    "CHEM:": "entities_Chemical",
    "ANATOMY:": "entities_Anatomy",
    "CL:": "entities_Cell_Ontology",
    "TAX:": "entities_Taxonomy",
    "GO:": "entities_Gene_Ontology",
    "REACTOME:": "entities_Reactome_Pathway",
    "KEGG:": "entities_KEGG_Pathway",
}

REFERENCE_COLLECTIONS = {
    "genes": "entities_Genes",
    "disease": "entities_Disease",
    "chemical": "entities_Chemical",
    "anatomy": "entities_Anatomy",
    "cell": "entities_Cell_Ontology",
    "taxonomy": "entities_Taxonomy",
    "go": "entities_Gene_Ontology",
    "reactome": "entities_Reactome_Pathway",
    "kegg": "entities_KEGG_Pathway",
}

VALID_ENTITY_PREFIXES = [
    "GENE:",
    "DISEASE:",
    "CHEM:",
    "ANATOMY:",
    "CL:",
    "TAX:",
    "GO:",
    "REACTOME:",
    "KEGG:",
]


def get_partition_for_entity(entity_name: str) -> str:
    """
    Determine partition name based on entity name prefix.

    Partition name = prefix without colon.
    Example: "GENE:BRCA1" -> partition "GENE"

    Args:
        entity_name: Entity name with prefix

    Returns:
        Partition name (prefix without colon)

    Raises:
        ValueError: If entity has unknown/unrecognized prefix
    """
    for prefix in VALID_ENTITY_PREFIXES:
        if entity_name.startswith(prefix):
            return prefix.rstrip(":")

    raise ValueError(
        f"Unknown entity type for entity: {entity_name}\n"
        f"Expected one of: {VALID_ENTITY_PREFIXES}\n"
        f"This indicates a problem in the data pipeline."
    )


def ensure_partition_exists(collection, partition_name: str) -> None:
    """
    Ensure partition exists, create if it doesn't.

    Args:
        collection: Milvus collection
        partition_name: Name of partition to ensure exists
    """
    if not collection.has_partition(partition_name):
        collection.create_partition(partition_name)
        print(f"  Created partition: {partition_name}")


def validate_entity_types(entities: list[dict]) -> None:
    """
    Validate all entities have recognized type prefixes.

    Args:
        entities: List of entity dicts with 'entity_name' field

    Raises:
        ValueError: If any entity has unknown prefix
    """
    unknown_entities = []

    for entity in entities:
        entity_name = entity.get("entity_name", "")
        try:
            get_partition_for_entity(entity_name)
        except ValueError:
            unknown_entities.append(entity_name)

    if unknown_entities:
        examples = unknown_entities[:10]
        total = len(unknown_entities)

        raise ValueError(
            f"Found {total} entities with unknown type prefixes:\n"
            f"Examples: {examples}\n"
            f"Expected prefixes: {VALID_ENTITY_PREFIXES}\n"
            f"Please check the entity extraction pipeline."
        )


class RedisEntitySourceManager:
    """Manage entity source_id storage in Redis."""

    def __init__(self, host: str = "localhost", port: int = 6379, workspace: str = "lightrag"):
        self.workspace = workspace
        self.entity_sources_namespace = f"{workspace}_entity_sources"
        self.redis = redis.Redis(
            host=host,
            port=port,
            decode_responses=True,
            socket_keepalive=True,
            socket_timeout=60,
            socket_connect_timeout=10,
        )
        self.redis.ping()
        print(f"Connected to Redis at {host}:{port}")

    def upsert_entity_sources(self, entities_data: list[dict], batch_size: int = 1000) -> int:
        """
        Upsert entity source_ids to Redis using SETs (forward and reverse indices).

        Creates two indices:
        - Forward: {workspace}_entity_sources:{entity_id} -> SET of chunk IDs
        - Reverse: {workspace}_chunk_entities:{chunk_id} -> SET of entity names

        Args:
            entities_data: List of dicts with keys:
                - id: entity ID (e.g., ent-a1b2c3d4...)
                - entity_name: entity name (e.g., GENE:BRCA1)
                - source_id: chunk references string (e.g., "chunk-A-0<SEP>chunk-B-1")
            batch_size: Number of entities to process per Redis pipeline batch

        Returns:
            Number of entities updated
        """
        if not entities_data:
            return 0

        total_processed = 0
        for batch_start in range(0, len(entities_data), batch_size):
            batch = entities_data[batch_start : batch_start + batch_size]

            pipe_write = self.redis.pipeline()

            for entity in batch:
                entity_id = entity["id"]
                entity_name = entity["entity_name"]
                source_id = entity["source_id"]

                chunk_ids = source_id.split("<SEP>") if source_id else []

                forward_key = f"{self.entity_sources_namespace}:{entity_id}"
                if chunk_ids:
                    pipe_write.sadd(forward_key, *chunk_ids)

                for chunk_id in chunk_ids:
                    if chunk_id:
                        reverse_key = f"{self.workspace}_chunk_entities:{chunk_id}"
                        pipe_write.sadd(reverse_key, entity_name)

            pipe_write.execute()
            total_processed += len(batch)

            if (batch_start // batch_size + 1) % 10 == 0:
                print(f"  Redis: Processed {total_processed}/{len(entities_data)} entities")

        return total_processed

    def get_entity_source(self, entity_id: str) -> list[str]:
        """Get entity source chunk IDs from Redis SET."""
        key = f"{self.entity_sources_namespace}:{entity_id}"
        return list(self.redis.smembers(key))

    def get_entity_sources_batch(self, entity_ids: list[str]) -> dict[str, list[str]]:
        """Get multiple entity source chunk IDs in batch."""
        pipe = self.redis.pipeline()
        keys = [f"{self.entity_sources_namespace}:{eid}" for eid in entity_ids]

        for key in keys:
            pipe.smembers(key)

        results = pipe.execute()

        entity_sources = {}
        for entity_id, chunk_set in zip(entity_ids, results, strict=False):
            if chunk_set:
                entity_sources[entity_id] = list(chunk_set)

        return entity_sources

    def clear_workspace_data(self):
        """Clear all entity source data (forward and reverse indices) for this workspace."""
        print(f"Clearing entity source data for workspace: {self.workspace}")

        forward_pattern = f"{self.entity_sources_namespace}:*"
        cursor = 0
        forward_deleted = 0

        while True:
            cursor, keys = self.redis.scan(cursor, match=forward_pattern, count=1000)
            if keys:
                forward_deleted += self.redis.delete(*keys)
            if cursor == 0:
                break

        print(f"Deleted {forward_deleted} forward index entries from Redis")

        reverse_pattern = f"{self.workspace}_chunk_entities:*"
        cursor = 0
        reverse_deleted = 0

        while True:
            cursor, keys = self.redis.scan(cursor, match=reverse_pattern, count=1000)
            if keys:
                reverse_deleted += self.redis.delete(*keys)
            if cursor == 0:
                break

        print(f"Deleted {reverse_deleted} reverse index entries from Redis")

    def close(self):
        """Close Redis connection."""
        self.redis.close()


def load_processed_papers(log_file):
    """Load set of already processed paper IDs."""
    if not log_file.exists():
        return set()

    processed = set()
    with open(log_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed.add(row["paper_id"])

    return processed


def save_progress(log_file, paper_id, num_entities, timestamp):
    """Append paper processing record to log."""
    file_exists = log_file.exists()

    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["paper_id", "num_entities", "timestamp"])
        writer.writerow([paper_id, num_entities, timestamp])


def get_unprocessed_papers(entities_dir, processed_papers, limit=0, paper_ids=None):
    """Get list of unprocessed paper directories.

    Args:
        entities_dir: Directory containing paper subdirectories
        processed_papers: Set of already-processed paper IDs
        limit: Maximum number of papers to return (0 = no limit)
        paper_ids: Optional list of specific paper IDs to process

    Returns:
        List of paper directory paths
    """
    paper_dirs = sorted([d for d in entities_dir.iterdir() if d.is_dir()])

    if paper_ids:
        paper_dirs = [d for d in paper_dirs if d.name in paper_ids]

    paper_dirs = [d for d in paper_dirs if d.name not in processed_papers]

    if limit > 0:
        paper_dirs = paper_dirs[:limit]

    return paper_dirs


def normalize_entity_name(entity_name: str) -> str:
    """
    Normalize entity name for consistent lookups.

    For GENE: entities, uppercase the gene symbol portion.
    Example: "GENE:Med12" -> "GENE:MED12"

    Args:
        entity_name: Entity name from TSV (e.g., "GENE:Med12")

    Returns:
        Normalized entity name (e.g., "GENE:MED12")
    """
    if entity_name.startswith("GENE:"):
        prefix = "GENE:"
        symbol = entity_name[len(prefix) :]
        return prefix + symbol.upper()
    return entity_name


def filter_entities_batch(paper_dirs, similarity_threshold):
    """Collect matched entities above the similarity threshold for a batch of papers."""
    entity_records = []
    papers_to_process = []

    for paper_dir in paper_dirs:
        paper_id = paper_dir.name
        papers_to_process.append(paper_id)

        for tsv_file in paper_dir.glob("*.tsv"):
            chunk_id = tsv_file.stem

            df = pd.read_csv(tsv_file, sep="\t")

            df = df[df["similarity_score_label"] >= similarity_threshold]

            df = df[df["entity_name"].notna() & (df["entity_name"] != "")]

            for _, row in df.iterrows():
                entity_name = normalize_entity_name(row["entity_name"])
                entity_records.append(
                    {
                        "entity_name": entity_name,
                        "similarity": row["similarity_score_label"],
                        "paper_id": paper_id,
                        "chunk_id": chunk_id,
                    }
                )

    return entity_records, papers_to_process


def deduplicate_entities(entity_records):
    """Group matches by entity name, keeping the maximum similarity and all sources."""
    grouped = defaultdict(lambda: {"similarities": [], "sources": []})

    for record in entity_records:
        entity_name = record["entity_name"]
        grouped[entity_name]["similarities"].append(record["similarity"])
        grouped[entity_name]["sources"].append((record["paper_id"], record["chunk_id"]))

    deduplicated = {}
    for entity_name, data in grouped.items():
        deduplicated[entity_name] = {
            "max_similarity": max(data["similarities"]),
            "occurrence_count": len(data["similarities"]),
            "sources": data["sources"],
        }

    print(f"Deduplicated to {len(deduplicated)} unique entities (from {len(entity_records)} occurrences)")
    return deduplicated


def load_entity_statistics(stats_file):
    """Load entity statistics from TSV file."""
    if not stats_file or not stats_file.exists():
        print(f"Warning: Entity statistics file not found: {stats_file}")
        return {}

    stats = {}
    try:
        df = pd.read_csv(stats_file, sep="\t")
        for _, row in df.iterrows():
            entity_name = row["entity_name"]
            stats[entity_name] = {
                "pct_paper": float(row["pct_paper"]),
                "pct_chunk": float(row["pct_chunk"]),
                "count": int(row["count"]),
                "n_paper": int(row["n_paper"]),
                "n_chunk": int(row["n_chunk"]),
            }

        print(f"Loaded statistics for {len(stats)} entities from {stats_file}")
        return stats
    except Exception as e:
        print(f"Error loading entity statistics: {e}")
        return {}


def filter_by_entity_statistics(deduplicated, entity_stats, max_pct_paper=None, max_pct_chunk=None, verbose=True):
    if not entity_stats:
        if verbose:
            print("No entity statistics provided, skipping statistics filter")
        return deduplicated

    if max_pct_paper is None and max_pct_chunk is None:
        if verbose:
            print("No statistics thresholds provided, skipping statistics filter")
        return deduplicated

    filtered = {}
    filtered_out_by_paper = 0
    filtered_out_by_chunk = 0
    not_in_stats = 0

    for entity_name, data in deduplicated.items():
        if entity_name not in entity_stats:
            not_in_stats += 1
            continue

        stats = entity_stats[entity_name]

        if max_pct_paper is not None and stats["pct_paper"] >= max_pct_paper:
            filtered_out_by_paper += 1
            continue

        if max_pct_chunk is not None and stats["pct_chunk"] >= max_pct_chunk:
            filtered_out_by_chunk += 1
            continue

        filtered[entity_name] = data

    if verbose:
        print("Statistics filter:")
        print(f"  Input: {len(deduplicated)} entities")
        print(f"  Filtered out by pct_paper >= {max_pct_paper}: {filtered_out_by_paper}")
        print(f"  Filtered out by pct_chunk >= {max_pct_chunk}: {filtered_out_by_chunk}")
        print(f"  Filtered out (not in statistics): {not_in_stats}")
        print(f"  Output: {len(filtered)} entities")

    return filtered


def process_worker_batch(args):
    """
    Worker that filters and deduplicates matched entities for a set of papers.

    Args:
        args: Tuple of (worker_id, paper_dirs, similarity_threshold, entity_stats,
                       max_pct_paper, max_pct_chunk)

    Returns:
        Dict with keys:
            - worker_id: Worker ID
            - filtered_entities: Dict of filtered entities (entity_name -> data)
            - papers_processed: List of paper IDs processed
            - stats: Dict with step statistics
    """
    (worker_id, paper_dirs, similarity_threshold, entity_stats, max_pct_paper, max_pct_chunk) = args

    print(f"  Worker {worker_id}: Starting with {len(paper_dirs)} papers")

    entity_records, papers_to_process = filter_entities_batch(paper_dirs, similarity_threshold)

    if not entity_records:
        print(f"  Worker {worker_id}: No entities found")
        return {
            "worker_id": worker_id,
            "filtered_entities": {},
            "papers_processed": [],
            "stats": {"entity_records": 0, "deduplicated": 0, "filtered": 0},
        }

    deduplicated = deduplicate_entities(entity_records)

    filtered = filter_by_entity_statistics(deduplicated, entity_stats, max_pct_paper, max_pct_chunk, verbose=False)

    print(f"  Worker {worker_id}: Completed - {len(entity_records)} records -> {len(filtered)} filtered entities")

    return {
        "worker_id": worker_id,
        "filtered_entities": filtered,
        "papers_processed": papers_to_process,
        "stats": {"entity_records": len(entity_records), "deduplicated": len(deduplicated), "filtered": len(filtered)},
    }


def query_single_collection(collection_name, names, batch_size):
    """
    Query a single collection for entities.

    Args:
        collection_name: Name of collection to query
        names: List of entity names to look up
        batch_size: Batch size for queries

    Returns:
        Dict of entity_name -> entity_data

    Raises:
        Exception: If query fails
    """
    results = {}

    try:
        collection = Collection(collection_name)
        collection.load()

        for i in range(0, len(names), batch_size):
            batch = names[i : i + batch_size]
            expr = " or ".join([f'entity_name == "{n}"' for n in batch])

            batch_results = collection.query(expr=expr, output_fields=["*"], limit=len(batch))

            for entity_data in batch_results:
                results[entity_data["entity_name"]] = entity_data

    except Exception as e:
        raise Exception(f"Error querying {collection_name}: {e}") from e

    return results


def lookup_reference_entities(entity_names, milvus_host, milvus_port, batch_size=100):
    """
    Look up entities in the reference collections (parallel queries).

    Raises:
        Exception: If database connection or query fails
    """
    try:
        connections.connect(host=milvus_host, port=milvus_port)
    except Exception as e:
        raise Exception(f"Failed to connect to Milvus at {milvus_host}:{milvus_port}: {e}") from e

    by_collection = defaultdict(list)
    for entity_name in entity_names:
        for prefix, collection_name in PREFIX_TO_COLLECTION.items():
            if entity_name.startswith(prefix):
                by_collection[collection_name].append(entity_name)
                break

    results = {}
    errors = []

    with ThreadPoolExecutor(max_workers=len(by_collection)) as executor:
        future_to_collection = {
            executor.submit(query_single_collection, collection_name, names, batch_size): collection_name
            for collection_name, names in by_collection.items()
        }

        for future in future_to_collection:
            collection_name = future_to_collection[future]
            try:
                collection_results = future.result()
                results.update(collection_results)
            except Exception as e:
                error_msg = str(e)
                print(f"  ERROR: {error_msg}")
                errors.append(error_msg)

    if errors:
        raise Exception("Failed to query reference collections:\n" + "\n".join(errors))

    print(f"Found {len(results)}/{len(entity_names)} entities in reference collections")
    return results


def query_existing_entities(collection_name, entity_names, batch_size=100):
    """
    Query existing entities in the entities collection.

    Raises:
        Exception: If database query fails
    """
    if not utility.has_collection(collection_name):
        print("Collection does not exist yet, all entities are new")
        return {}

    try:
        collection = Collection(collection_name)
        collection.load()
    except Exception as e:
        raise Exception(f"Failed to load collection {collection_name}: {e}") from e

    existing = {}
    errors = []

    for i in range(0, len(entity_names), batch_size):
        batch = entity_names[i : i + batch_size]
        expr = " or ".join([f'entity_name == "{n}"' for n in batch])

        try:
            batch_results = collection.query(expr=expr, output_fields=["entity_name", "id"], limit=len(batch))

            for entity_data in batch_results:
                existing[entity_data["entity_name"]] = entity_data

        except Exception as e:
            error_msg = f"Error querying existing entities (batch {i // batch_size + 1}): {e}"
            print(f"  ERROR: {error_msg}")
            errors.append(error_msg)

    if errors:
        raise Exception("Failed to query existing entities:\n" + "\n".join(errors))

    print(f"Found {len(existing)} existing entities in the entities collection")
    return existing


def build_source_id(sources):
    """Build source_id from chunk sources.

    TSV files are named 1.tsv, 2.tsv, ... (1-based)
    Chunk IDs are chunk-xxx-000001, chunk-xxx-000002, ... (1-based with 6-digit padding)
    Direct mapping: N.tsv -> chunk-xxx-{N:06d}

    Deduplicates sources to prevent duplicate chunk IDs when the same entity
    appears in multiple TSV files for the same paper.
    """
    unique_sources = list(set(sources))
    chunk_ids = [f"chunk-{paper_id}-{int(chunk_id):06d}" for paper_id, chunk_id in unique_sources]
    return "<SEP>".join(sorted(chunk_ids))


def merge_source_ids(existing_source_id, new_source_id):
    """Merge existing and new source_id, removing duplicates."""
    existing_chunks = set(existing_source_id.split("<SEP>")) if existing_source_id else set()
    new_chunks = set(new_source_id.split("<SEP>"))

    all_chunks = existing_chunks | new_chunks
    all_chunks = sorted(all_chunks)

    return "<SEP>".join(all_chunks)


def merge_worker_results(worker_results):
    """
    Merge results from multiple workers and deduplicate again.

    Args:
        worker_results: List of dicts from process_worker_batch

    Returns:
        Tuple of (merged_entities, all_papers_processed, combined_stats)
    """
    all_entities = defaultdict(lambda: {"similarities": [], "sources": []})
    all_papers = []
    total_stats = {"entity_records": 0, "deduplicated": 0, "filtered": 0}

    for result in worker_results:
        all_papers.extend(result["papers_processed"])

        for key in total_stats:
            total_stats[key] += result["stats"][key]

        for entity_name, data in result["filtered_entities"].items():
            all_entities[entity_name]["similarities"].append(data["max_similarity"])
            all_entities[entity_name]["sources"].extend(data["sources"])

    merged = {}
    for entity_name, data in all_entities.items():
        unique_sources = list(set(data["sources"]))
        merged[entity_name] = {
            "max_similarity": max(data["similarities"]),
            "occurrence_count": len(unique_sources),
            "sources": unique_sources,
        }

    print(f"  Worker merge: Combined {len(worker_results)} workers -> {len(merged)} unique entities")
    print(f"    Total entity occurrences: {total_stats['entity_records']}")
    print(f"    After deduplication: {total_stats['deduplicated']}")
    print(f"    After filtering: {total_stats['filtered']}")
    print(f"    Final after global dedup: {len(merged)}")

    return merged, all_papers, total_stats


def prepare_entities_for_upsert(reference_entities, ner_data, existing_entities):
    """Prepare new entities for Milvus and source_id entries for Redis."""
    entities_to_insert = []
    entities_source_data = []
    new_count = 0
    skipped_count = 0
    current_time = int(time.time())

    insertion_order = 0
    for entity_name, reference_data in reference_entities.items():
        new_source_id = build_source_id(ner_data[entity_name]["sources"])

        if entity_name in existing_entities:
            skipped_count += 1

            entities_source_data.append(
                {"id": existing_entities[entity_name]["id"], "entity_name": entity_name, "source_id": new_source_id}
            )
        else:
            entity = {k: v for k, v in reference_data.items() if k != "label_embedding"}

            created_at_hires = current_time + (insertion_order / 1000000.0)
            insertion_order += 1
            entity["created_at"] = current_time
            entity["created_at_hires"] = created_at_hires

            entities_to_insert.append(entity)
            new_count += 1

            entities_source_data.append({"id": entity["id"], "entity_name": entity_name, "source_id": new_source_id})

    print(f"Prepared {new_count} new entities for Milvus, skipped {skipped_count} existing entities")
    print(f"        Prepared {len(entities_source_data)} source_id entries for Redis")
    return entities_to_insert, entities_source_data


def insert_entities_to_stores(
    collection_name,
    entities,
    entities_source_data,
    redis_manager,
    drop_existing=False,
    batch_size=1000,
    redis_batch_size=1000,
):
    """Insert new entities into Milvus and upsert all source_ids into Redis."""

    if drop_existing:
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print(f"Dropped existing collection: {collection_name}")

        redis_manager.clear_workspace_data()

    if not utility.has_collection(collection_name):
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
            FieldSchema(name="created_at", dtype=DataType.INT64),
            FieldSchema(name="created_at_hires", dtype=DataType.DOUBLE),
            FieldSchema(name="entity_name", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=32768),
        ]

        schema = CollectionSchema(fields, enable_dynamic_field=True, description="Entities")

        collection = Collection(collection_name, schema)

        index_params = {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 256}}
        collection.create_index("vector", index_params)

        collection.create_index("created_at_hires", {"index_type": "STL_SORT"})

        print(f"Created collection: {collection_name}")

    collection = Collection(collection_name)

    if entities:
        validate_entity_types(entities)

        by_partition = defaultdict(list)
        for entity in entities:
            entity_name = entity["entity_name"]
            partition_name = get_partition_for_entity(entity_name)
            by_partition[partition_name].append(entity)

        total_inserted = 0
        for partition_name, partition_entities in by_partition.items():
            ensure_partition_exists(collection, partition_name)

            for i in range(0, len(partition_entities), batch_size):
                batch = partition_entities[i : i + batch_size]
                collection.insert(batch, partition_name=partition_name)
                total_inserted += len(batch)
                print(f"  Inserted batch to partition '{partition_name}': {len(batch)} entities")

        collection.flush()
        collection.load()
        print(f"Inserted {total_inserted} new entities to Milvus across {len(by_partition)} partitions")
    else:
        print("No new entities to insert into Milvus")

    num_upserted = redis_manager.upsert_entity_sources(entities_source_data, batch_size=redis_batch_size)
    print(f"Upserted {num_upserted} entity source_ids to Redis")


def copy_reference_entities(
    source_collection_name: str, target_collection_name: str, batch_size: int, milvus_host: str, milvus_port: str
) -> int:
    """
    Copy entities from a reference collection into the entities collection.

    Args:
        source_collection_name: Source collection (e.g., 'entities_Genes')
        target_collection_name: Target collection (e.g., '<workspace>_entities')
        batch_size: Batch size for processing
        milvus_host: Milvus host
        milvus_port: Milvus port

    Returns:
        Number of entities copied
    """
    connections.connect(host=milvus_host, port=milvus_port)

    if not utility.has_collection(source_collection_name):
        print(f"Source collection {source_collection_name} does not exist, skipping")
        return 0

    source_collection = Collection(source_collection_name)
    source_collection.load()

    if not utility.has_collection(target_collection_name):
        print(f"Target collection {target_collection_name} does not exist")
        print("  Creating target collection...")
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
            FieldSchema(name="created_at", dtype=DataType.INT64),
            FieldSchema(name="created_at_hires", dtype=DataType.DOUBLE),
            FieldSchema(name="entity_name", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=32768),
        ]

        schema = CollectionSchema(fields, enable_dynamic_field=True, description="Entities")

        target_collection = Collection(target_collection_name, schema)

        index_params = {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 256}}
        target_collection.create_index("vector", index_params)

        target_collection.create_index("created_at_hires", {"index_type": "STL_SORT"})

        print(f"  Created target collection: {target_collection_name}")
    else:
        target_collection = Collection(target_collection_name)

    target_collection.load()

    print(f"  Querying existing entities in {target_collection_name}...")
    existing_entities = set()

    offset = 0
    limit = 1000
    while True:
        try:
            results = target_collection.query(
                expr="entity_name != ''", output_fields=["entity_name"], limit=limit, offset=offset
            )
            if not results:
                break
            existing_entities.update(r["entity_name"] for r in results)
            offset += limit
            if len(results) < limit:
                break
        except Exception:
            break

    print(f"  Found {len(existing_entities)} existing entities in target collection")

    print(f"  Fetching entities from {source_collection_name}...")
    all_entities = []
    offset = 0

    while True:
        try:
            results = source_collection.query(
                expr="entity_name != ''", output_fields=["*"], limit=batch_size, offset=offset
            )

            if not results:
                break

            new_entities = [r for r in results if r["entity_name"] not in existing_entities]

            all_entities.extend(new_entities)
            offset += batch_size

            if len(results) < batch_size:
                break
        except Exception:
            break

    if not all_entities:
        print(f"  No new entities to copy from {source_collection_name}")
        return 0

    print(f"  Copying {len(all_entities)} new entities to {target_collection_name}...")

    current_time = int(time.time())
    entities_to_insert = []
    insertion_order = 0

    for entity in all_entities:
        entity_copy = {k: v for k, v in entity.items() if k != "label_embedding"}

        if "created_at" not in entity_copy:
            entity_copy["created_at"] = current_time
        if "created_at_hires" not in entity_copy:
            entity_copy["created_at_hires"] = current_time + (insertion_order / 1000000.0)
            insertion_order += 1

        entities_to_insert.append(entity_copy)

    validate_entity_types(entities_to_insert)

    by_partition = defaultdict(list)
    for entity in entities_to_insert:
        partition_name = get_partition_for_entity(entity["entity_name"])
        by_partition[partition_name].append(entity)

    inserted_count = 0
    for partition_name, partition_entities in by_partition.items():
        ensure_partition_exists(target_collection, partition_name)

        for i in range(0, len(partition_entities), batch_size):
            batch = partition_entities[i : i + batch_size]
            target_collection.insert(batch, partition_name=partition_name)
            inserted_count += len(batch)
            print(f"    Inserted batch to partition '{partition_name}': {len(batch)} entities")

    target_collection.flush()
    print(f"  Copied {inserted_count} entities from {source_collection_name} to {len(by_partition)} partitions")

    return inserted_count


def main():
    parser = argparse.ArgumentParser(description="Insert matched entities into Milvus and Redis via reference lookup")
    parser.add_argument("--input", required=True, help="N-gram entity matching output directory")
    parser.add_argument("--entity-stats", type=Path, default=None, help="Path to the entity statistics TSV file")
    parser.add_argument("--milvus-host", default="localhost")
    parser.add_argument("--milvus-port", default="19530")
    parser.add_argument("--redis-host", default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument("--workspace", default="lightrag")
    parser.add_argument(
        "--similarity-threshold", type=float, required=True, help="Minimum similarity_score_label threshold (0.0-1.0)"
    )
    parser.add_argument("--max-pct-paper", type=float, default=None, help="Maximum pct_paper")
    parser.add_argument("--max-pct-chunk", type=float, default=None, help="Maximum pct_chunk")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--papers-per-batch", type=int, default=200, help="Number of papers to process per batch ")
    parser.add_argument("--redis-batch-size", type=int, default=500, help="Number of entities per Redis pipeline batch")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--paper-ids",
        type=str,
        default=None,
        help="Comma-separated list of specific paper IDs to process",
    )
    parser.add_argument("--drop-existing", action="store_true")

    parser.add_argument("--insert-genes", type=str, default="false")
    parser.add_argument("--insert-disease", type=str, default="false")
    parser.add_argument("--insert-chemical", type=str, default="false")
    parser.add_argument("--insert-anatomy", type=str, default="false")
    parser.add_argument("--insert-cell", type=str, default="false")
    parser.add_argument("--insert-taxonomy", type=str, default="false")
    parser.add_argument("--insert-go", type=str, default="false")
    parser.add_argument("--insert-reactome", type=str, default="false")
    parser.add_argument("--insert-kegg", type=str, default="false")
    parser.add_argument("--reference-batch-size", type=int, default=1000)

    args = parser.parse_args()

    entities_dir = Path(args.input)
    collection_name = f"{args.workspace}_entities"
    log_file = entities_dir.parent / "entities-inserted.csv"

    def parse_bool(value):
        if isinstance(value, bool):
            return value
        return value.lower() == "true"

    insert_flags = {
        "genes": parse_bool(args.insert_genes),
        "disease": parse_bool(args.insert_disease),
        "chemical": parse_bool(args.insert_chemical),
        "anatomy": parse_bool(args.insert_anatomy),
        "cell": parse_bool(args.insert_cell),
        "taxonomy": parse_bool(args.insert_taxonomy),
        "go": parse_bool(args.insert_go),
        "reactome": parse_bool(args.insert_reactome),
        "kegg": parse_bool(args.insert_kegg),
    }

    reference_insertion_requested = any(insert_flags.values())

    if not reference_insertion_requested:
        print("=" * 60)
        print("Entity insertion")
        print("=" * 60)
        print(f"Input: {entities_dir}")
        print(f"Collection: {collection_name}")
        print(f"Progress log: {log_file}")
        print(f"Similarity threshold: {args.similarity_threshold}")
        if args.limit > 0:
            print(f"Paper limit: {args.limit}")
        print()

        print("Entity Statistics Filtering:")
        if args.entity_stats:
            print(f"  Stats file: {args.entity_stats}")
            if args.max_pct_paper is not None:
                print(f"  Max pct_paper: {args.max_pct_paper}%")
            if args.max_pct_chunk is not None:
                print(f"  Max pct_chunk: {args.max_pct_chunk}%")
        else:
            print("  Disabled (no statistics file provided)")
        print()

        redis_manager = RedisEntitySourceManager(args.redis_host, args.redis_port, args.workspace)
        print()

        processed_papers = load_processed_papers(log_file)
        print(f"Loaded progress log ({len(processed_papers)} papers already processed)")

        entity_stats = {}
        if args.entity_stats:
            entity_stats = load_entity_statistics(args.entity_stats)

        paper_ids_set = None
        if args.paper_ids:
            paper_ids_set = {pid.strip() for pid in args.paper_ids.split(",") if pid.strip()}
            print(f"Filtering to specific paper IDs: {sorted(paper_ids_set)}")

        all_unprocessed_papers = get_unprocessed_papers(entities_dir, processed_papers, args.limit, paper_ids_set)
        total_papers = len(all_unprocessed_papers)

        if total_papers == 0:
            print("No new papers to process!")
        else:
            print(f"\nFound {total_papers} unprocessed papers")
            print(f"Processing in batches of {args.papers_per_batch} papers\n")

            total_entities_inserted = 0
            total_sources_upserted = 0
            total_papers_processed = 0

            pipeline_start_time = time.time()
            batch_times = []

            for batch_idx in range(0, total_papers, args.papers_per_batch):
                batch_start_time = time.time()
                batch_papers = all_unprocessed_papers[batch_idx : batch_idx + args.papers_per_batch]
                batch_num = (batch_idx // args.papers_per_batch) + 1
                total_batches = (total_papers + args.papers_per_batch - 1) // args.papers_per_batch

                print("=" * 60)
                print(f"BATCH {batch_num}/{total_batches}: Processing {len(batch_papers)} papers")
                print("=" * 60)

                steps_1_4_start = time.time()

                if args.workers > 1:
                    total_batch_papers = len(batch_papers)
                    base_papers_per_worker = total_batch_papers // args.workers
                    remainder = total_batch_papers % args.workers

                    worker_paper_chunks = []
                    current_idx = 0

                    for worker_id in range(args.workers):
                        papers_for_this_worker = base_papers_per_worker + (1 if worker_id < remainder else 0)
                        worker_papers = batch_papers[current_idx : current_idx + papers_for_this_worker]
                        current_idx += papers_for_this_worker

                        if worker_papers:
                            worker_paper_chunks.append(worker_papers)

                    print(f"  Starting {len(worker_paper_chunks)} workers...")
                    print(f"  Papers per worker: {[len(chunk) for chunk in worker_paper_chunks]}")

                    worker_args = [
                        (
                            worker_id,
                            worker_papers,
                            args.similarity_threshold,
                            entity_stats,
                            args.max_pct_paper,
                            args.max_pct_chunk,
                        )
                        for worker_id, worker_papers in enumerate(worker_paper_chunks)
                    ]

                    with Pool(processes=len(worker_paper_chunks)) as pool:
                        worker_results = pool.map(process_worker_batch, worker_args)

                    filtered, papers_to_process, _ = merge_worker_results(worker_results)

                    if not filtered:
                        print("  No entities after worker processing, skipping...\n")
                        continue

                    print(f"  Looking up {len(filtered)} unique entities in reference DB...")
                    reference_entities = lookup_reference_entities(
                        list(filtered.keys()), args.milvus_host, args.milvus_port, args.batch_size
                    )
                else:
                    entity_records, papers_to_process = filter_entities_batch(batch_papers, args.similarity_threshold)

                    if not entity_records:
                        print("  No entities found in this batch, skipping...\n")
                        continue

                    print(f"  Extracted {len(entity_records)} entity occurrences")

                    deduplicated = deduplicate_entities(entity_records)

                    if not deduplicated:
                        print("  No entities after deduplication, skipping...\n")
                        continue

                    filtered = filter_by_entity_statistics(
                        deduplicated, entity_stats, args.max_pct_paper, args.max_pct_chunk
                    )

                    if not filtered:
                        print("  No entities after statistics filtering, skipping...\n")
                        continue

                    reference_entities = lookup_reference_entities(
                        list(filtered.keys()), args.milvus_host, args.milvus_port, args.batch_size
                    )

                steps_1_4_time = time.time() - steps_1_4_start

                if not reference_entities:
                    print("  No reference entities found, skipping...\n")
                    continue

                print(f"  Matching and lookup completed in {steps_1_4_time:.1f}s")

                step5_start = time.time()
                if args.drop_existing and batch_idx == 0:
                    print("  Skipping existing entity query (--drop-existing)")
                    existing_entities = {}
                else:
                    existing_entities = query_existing_entities(
                        collection_name, list(reference_entities.keys()), args.batch_size
                    )
                step5_time = time.time() - step5_start

                step6_start = time.time()
                entities_to_insert, entities_source_data = prepare_entities_for_upsert(
                    reference_entities, filtered, existing_entities
                )
                step6_time = time.time() - step6_start

                step7_start = time.time()
                insert_entities_to_stores(
                    collection_name,
                    entities_to_insert,
                    entities_source_data,
                    redis_manager,
                    args.drop_existing and batch_idx == 0,
                    args.batch_size,
                    args.redis_batch_size,
                )
                step7_time = time.time() - step7_start

                entity_counts_by_paper = defaultdict(int)
                for entity_name in reference_entities.keys():
                    sources = filtered[entity_name]["sources"]
                    for paper_id, _ in sources:
                        entity_counts_by_paper[paper_id] += 1

                timestamp = int(time.time())
                for paper_id in papers_to_process:
                    num_entities = entity_counts_by_paper.get(paper_id, 0)
                    save_progress(log_file, paper_id, num_entities, timestamp)

                total_entities_inserted += len(entities_to_insert)
                total_sources_upserted += len(entities_source_data)
                total_papers_processed += len(papers_to_process)

                batch_time = time.time() - batch_start_time
                batch_times.append(batch_time)
                avg_batch_time = sum(batch_times) / len(batch_times)
                remaining_batches = total_batches - batch_num
                eta_seconds = avg_batch_time * remaining_batches
                eta_minutes = eta_seconds / 60
                eta_hours = eta_minutes / 60

                print(f"\n  Batch {batch_num} complete:")
                print(f"    New entities: {len(entities_to_insert)}")
                print(f"    Source IDs: {len(entities_source_data)}")
                print(f"    Papers: {len(papers_to_process)}")
                print(f"    Batch time: {batch_time:.1f}s")
                step_times_msg = (
                    f"    Stage times: lookup={steps_1_4_time:.1f}s, existing={step5_time:.1f}s, "
                    f"prepare={step6_time:.1f}s, insert={step7_time:.1f}s"
                )
                print(step_times_msg)
                if remaining_batches > 0:
                    if eta_hours >= 1:
                        eta_msg = (
                            f"    ETA: {eta_hours:.1f}h "
                            f"({remaining_batches} batches remaining, avg {avg_batch_time:.1f}s/batch)"
                        )
                        print(eta_msg)
                    else:
                        eta_msg = (
                            f"    ETA: {eta_minutes:.1f}m "
                            f"({remaining_batches} batches remaining, avg {avg_batch_time:.1f}s/batch)"
                        )
                        print(eta_msg)
                print()

            total_time = time.time() - pipeline_start_time
            total_minutes = total_time / 60
            total_hours = total_minutes / 60

            print("=" * 60)
            print("ALL BATCHES COMPLETE!")
            print(f"  Total new entities inserted: {total_entities_inserted}")
            print(f"  Total source IDs upserted: {total_sources_upserted}")
            print(f"  Total papers processed: {total_papers_processed}")
            if total_hours >= 1:
                print(f"  Total time: {total_hours:.1f}h ({total_time:.1f}s)")
            else:
                print(f"  Total time: {total_minutes:.1f}m ({total_time:.1f}s)")
            if batch_times:
                print(f"  Average batch time: {sum(batch_times) / len(batch_times):.1f}s")
            print(f"  Progress saved to: {log_file}")
            print("=" * 60)

    if reference_insertion_requested:
        print()
        print("=" * 60)
        print("Reference entity insertion")
        print("=" * 60)
        print(f"Target collection: {collection_name}")
        print(f"Batch size: {args.reference_batch_size}")
        print()

        total_copied = 0

        for entity_type, should_insert in insert_flags.items():
            if should_insert:
                source_collection = REFERENCE_COLLECTIONS[entity_type]
                print(f"Processing {entity_type} ({source_collection})...")

                copied = copy_reference_entities(
                    source_collection_name=source_collection,
                    target_collection_name=collection_name,
                    batch_size=args.reference_batch_size,
                    milvus_host=args.milvus_host,
                    milvus_port=args.milvus_port,
                )

                total_copied += copied
                print()

        print("=" * 60)
        print("Reference entity insertion complete!")
        print(f"Total entities copied: {total_copied}")
        print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
