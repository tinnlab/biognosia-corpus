#!/usr/bin/env python3
"""
Relationship insertion into Milvus and Neo4j.

For each paper, the script pairs the entities matched in its chunks, looks up the
pairs that have a documented relationship in the relationships_* reference
collections, and counts one co-occurrence per paper as the edge weight.
Relationships are written to the <workspace>_relationships collection in Milvus
and to Neo4j as nodes and edges.
"""

import argparse
import csv
import hashlib
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

GRAPH_FIELD_SEP = "<SEP>"


def _compute_id(content: str, prefix: str = "") -> str:
    """Compute a prefixed MD5 hash ID."""
    return prefix + hashlib.md5(content.encode()).hexdigest()


def discover_relationship_schemas():
    """
    Discover all relationship collections and parse their entity type mappings.

    Returns:
        Dict mapping collection_name -> (src_prefix, tgt_prefix)

    Example:
        "relationships_Gene_Disease" -> ("GENE:", "DISEASE:")
        "relationships_Chemical_Gene" -> ("CHEM:", "GENE:")
    """
    all_collections = utility.list_collections()
    rel_collections = [c for c in all_collections if c.startswith("relationships_")]

    schemas = {}

    type_to_prefix = {
        "Gene": "GENE:",
        "Disease": "DISEASE:",
        "Chemical": "CHEM:",
        "Cell": "CL:",
        "Anatomy": "ANATOMY:",
        "Taxonomy": "TAX:",
        "Gene_Ontology": "GO:",
        "GO": "GO:",
        "Reactome_Pathway": "REACTOME:",
        "Reactome": "REACTOME:",
        "KEGG_Pathway": "KEGG:",
        "KEGG": "KEGG:",
    }

    print("\nDiscovered Relationship Schemas:")
    print("=" * 60)

    for collection_name in sorted(rel_collections):
        parts = collection_name.replace("relationships_", "").split("_")

        if len(parts) == 2:
            type_a, type_b = parts
            src_prefix = type_to_prefix.get(type_a)
            tgt_prefix = type_to_prefix.get(type_b)
        elif len(parts) == 3:
            type_combined = "_".join(parts[1:])
            if type_combined in type_to_prefix:
                type_a = parts[0]
                src_prefix = type_to_prefix.get(type_a)
                tgt_prefix = type_to_prefix.get(type_combined)
            else:
                type_combined = "_".join(parts[:2])
                if type_combined in type_to_prefix:
                    src_prefix = type_to_prefix.get(type_combined)
                    tgt_prefix = type_to_prefix.get(parts[2])
                else:
                    type_a = parts[0]
                    type_b = "_".join(parts[1:])
                    src_prefix = type_to_prefix.get(type_a)
                    tgt_prefix = type_to_prefix.get(type_b)
        else:
            print(f"  Cannot parse: {collection_name}")
            continue

        if src_prefix and tgt_prefix:
            schemas[collection_name] = (src_prefix, tgt_prefix)
            print(f"  {collection_name:<45} {src_prefix:<15} -> {tgt_prefix}")
        else:
            print(f"  Unknown types in: {collection_name}")

    print("=" * 60)
    print(f"Total: {len(schemas)} relationship schemas")
    print()

    return schemas


def ensure_relationship_indexes(rel_schemas):
    """Ensure indexes on src_id and tgt_id for all relationship collections."""
    print("Ensuring indexes on relationship collections...")

    for collection_name in rel_schemas.keys():
        try:
            collection = Collection(collection_name)
            collection.load()

            indexes = collection.indexes
            indexed_fields = {idx.field_name for idx in indexes}

            if "src_id" not in indexed_fields:
                collection.create_index(field_name="src_id", index_params={"index_type": "TRIE"})
                print(f"  Created src_id index on {collection_name}")

            if "tgt_id" not in indexed_fields:
                collection.create_index(field_name="tgt_id", index_params={"index_type": "TRIE"})
                print(f"  Created tgt_id index on {collection_name}")

        except Exception as e:
            print(f"  Error on {collection_name}: {e}")

    print("  Index check complete\n")


def load_processed_papers(log_file):
    """Load set of already-processed paper IDs."""
    if not log_file.exists():
        return set()

    processed = set()
    with open(log_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed.add(row["paper_id"])

    return processed


def save_paper_progress(log_file, paper_id, num_relationships):
    """Append paper progress to log."""
    file_exists = log_file.exists()

    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["paper_id", "num_relationships", "timestamp"])
        writer.writerow([paper_id, num_relationships, int(time.time())])


def get_unprocessed_papers(entities_dir, processed_papers, limit=0):
    """Get list of unprocessed paper directories."""
    paper_dirs = sorted([d for d in entities_dir.iterdir() if d.is_dir()])

    unprocessed = [d for d in paper_dirs if d.name not in processed_papers]

    if limit > 0:
        unprocessed = unprocessed[:limit]

    return unprocessed


def load_entity_statistics(stats_file):
    """Load entity statistics from a TSV file."""
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


def load_entities_from_paper(
    paper_dir, similarity_threshold, entity_stats=None, max_pct_paper=None, max_pct_chunk=None
):
    """Load the entities of a single paper that pass the thresholds, grouped by type prefix."""
    entities_by_type = defaultdict(set)

    for tsv_file in paper_dir.glob("*.tsv"):
        try:
            df = pd.read_csv(tsv_file, sep="\t")
            df = df[df["similarity_score_label"] >= similarity_threshold]

            for entity_name in df["entity_name"].unique():
                if entity_stats:
                    if entity_name not in entity_stats:
                        continue

                    stats = entity_stats[entity_name]

                    if max_pct_paper is not None and stats["pct_paper"] >= max_pct_paper:
                        continue

                    if max_pct_chunk is not None and stats["pct_chunk"] >= max_pct_chunk:
                        continue

                for prefix in ["GENE:", "DISEASE:", "CHEM:", "CL:", "ANATOMY:", "TAX:", "GO:", "REACTOME:", "KEGG:"]:
                    if entity_name.startswith(prefix):
                        entities_by_type[prefix].add(entity_name)
                        break
        except Exception:
            continue

    return entities_by_type


def filter_indexed_entities(entity_names, entities_collection):
    """Return the subset of entity names present in the entities collection."""
    if not entity_names:
        return set()

    collection = Collection(entities_collection)
    collection.load()

    existing = set()
    entity_list = list(entity_names)
    batch_size = 100

    for i in range(0, len(entity_list), batch_size):
        batch = entity_list[i : i + batch_size]
        expr = " or ".join([f'entity_name == "{e}"' for e in batch])

        try:
            results = collection.query(expr=expr, output_fields=["entity_name"], limit=min(15000, len(batch)))
            existing.update(r["entity_name"] for r in results)
        except Exception:
            pass

    return existing


def generate_possible_relationships(entities_by_type, rel_schemas, indexed_entities):
    """
    Generate all possible relationship IDs based on entity types in paper.

    Returns:
        Dict mapping collection_name -> List[relationship_id]
    """
    possible_rels = defaultdict(list)

    for collection_name, (src_prefix, tgt_prefix) in rel_schemas.items():
        src_entities = entities_by_type.get(src_prefix, set())
        tgt_entities = entities_by_type.get(tgt_prefix, set())

        src_entities = src_entities.intersection(indexed_entities)
        tgt_entities = tgt_entities.intersection(indexed_entities)

        for src in src_entities:
            for tgt in tgt_entities:
                if src != tgt:
                    composite_key = f"{src}{GRAPH_FIELD_SEP}{tgt}"
                    rel_id = _compute_id(composite_key, prefix="rel-")
                    possible_rels[collection_name].append((rel_id, src, tgt))

    return possible_rels


def query_relationships_by_ids(collection_name, rel_id_tuples):
    """
    Query relationships by ID (primary key) - very fast!

    Args:
        collection_name: Relationship collection
        rel_id_tuples: List of (rel_id, src_id, tgt_id)

    Returns:
        List of relationship records that exist
    """
    if not rel_id_tuples:
        return []

    collection = Collection(collection_name)
    collection.load()

    relationships = []
    batch_size = 100

    for i in range(0, len(rel_id_tuples), batch_size):
        batch = rel_id_tuples[i : i + batch_size]
        ids = [rel_id for rel_id, _, _ in batch]

        expr = " or ".join([f'id == "{rid}"' for rid in ids])

        try:
            results = collection.query(expr=expr, output_fields=["*"], limit=len(ids))
            relationships.extend(results)
        except Exception as e:
            print(f"    Error querying {collection_name}: {e}")

    return relationships


def query_collection_parallel_worker(collection_name, rel_id_tuples):
    """Worker function for parallel querying of a single collection."""
    return collection_name, query_relationships_by_ids(collection_name, rel_id_tuples)


def query_relationships_parallel(possible_rels, num_workers):
    """
    Query relationships from multiple collections in parallel.

    Args:
        possible_rels: Dict mapping collection_name -> List[(rel_id, src, tgt)]
        num_workers: Number of parallel query workers

    Returns:
        List of all found relationships

    Raises:
        Exception: If any worker encounters an error (e.g., database disconnection)
    """
    all_relationships = []
    errors = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for collection_name, rel_id_tuples in possible_rels.items():
            if rel_id_tuples:
                future = executor.submit(query_collection_parallel_worker, collection_name, rel_id_tuples)
                futures.append(future)

        for future in as_completed(futures):
            try:
                collection_name, relationships = future.result()
                all_relationships.extend(relationships)
            except Exception as e:
                error_msg = f"Error in parallel query worker: {e}"
                print(f"    ERROR: {error_msg}")
                errors.append(error_msg)

    if errors:
        raise Exception("Worker errors occurred:\n" + "\n".join(errors))

    return all_relationships


def generate_ids_for_single_paper(
    paper_dir,
    similarity_threshold,
    rel_schemas,
    entities_collection,
    entity_stats=None,
    max_pct_paper=None,
    max_pct_chunk=None,
):
    """
    Generate relationship IDs for a single paper.

    Returns:
        Tuple of (paper_id, possible_rels_dict)
        where possible_rels_dict maps collection_name -> List[(rel_id, src, tgt)]
    """
    paper_id = paper_dir.name

    entities_by_type = load_entities_from_paper(
        paper_dir, similarity_threshold, entity_stats, max_pct_paper, max_pct_chunk
    )

    total_entities = sum(len(ents) for ents in entities_by_type.values())
    if total_entities == 0:
        return paper_id, {}

    all_entities = set()
    for ents in entities_by_type.values():
        all_entities.update(ents)

    indexed_entities = filter_indexed_entities(all_entities, entities_collection)

    if not indexed_entities:
        return paper_id, {}

    possible_rels = generate_possible_relationships(entities_by_type, rel_schemas, indexed_entities)

    return paper_id, possible_rels


def generate_ids_for_paper_batch(
    paper_dirs,
    similarity_threshold,
    rel_schemas,
    entities_collection,
    num_workers,
    entity_stats=None,
    max_pct_paper=None,
    max_pct_chunk=None,
):
    """
    Generate relationship IDs for multiple papers in parallel.

    Returns:
        Dict mapping collection_name -> List[(rel_id, src, tgt, paper_id)]

    Raises:
        Exception: If any worker encounters an error (e.g., database disconnection)
    """
    aggregated_rels = defaultdict(list)
    errors = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for paper_dir in paper_dirs:
            future = executor.submit(
                generate_ids_for_single_paper,
                paper_dir,
                similarity_threshold,
                rel_schemas,
                entities_collection,
                entity_stats,
                max_pct_paper,
                max_pct_chunk,
            )
            futures.append(future)

        for future in as_completed(futures):
            try:
                paper_id, possible_rels = future.result()
                for collection_name, rel_id_tuples in possible_rels.items():
                    for rel_id, src, tgt in rel_id_tuples:
                        aggregated_rels[collection_name].append((rel_id, src, tgt, paper_id))
            except Exception as e:
                error_msg = f"Error generating IDs for paper: {e}"
                print(f"    ERROR: {error_msg}")
                errors.append(error_msg)

    if errors:
        raise Exception("Worker errors occurred:\n" + "\n".join(errors))

    return aggregated_rels


def accumulate_weights(weight_accumulator, relationships, paper_id):
    """Accumulate relationship weights (count papers where relationship appears)."""
    for rel in relationships:
        rel_id = rel["id"]
        if rel_id not in weight_accumulator:
            weight_accumulator[rel_id] = {"relationship": rel, "weight": 0, "papers": set()}
        weight_accumulator[rel_id]["weight"] += 1
        weight_accumulator[rel_id]["papers"].add(paper_id)


def prepare_relationships_for_upsert(weight_accumulator):
    """Prepare relationships with weights for Milvus."""
    prepared = []
    current_time = int(time.time())

    for rel_data in weight_accumulator.values():
        rel = dict(rel_data["relationship"])
        rel["created_at"] = current_time
        rel["weight"] = float(rel_data["weight"])
        prepared.append(rel)

    return prepared


def upsert_relationships_to_milvus(collection_name, relationships, drop_existing=False, batch_size=1000):
    """Upsert relationships to Milvus with weight accumulation."""
    if drop_existing and utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print("  Dropped existing collection")

    if not utility.has_collection(collection_name):
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
            FieldSchema(name="created_at", dtype=DataType.INT64),
            FieldSchema(name="src_id", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="tgt_id", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=32768),
        ]

        schema = CollectionSchema(fields, enable_dynamic_field=True, description="Relationships")
        collection = Collection(collection_name, schema)

        collection.create_index(
            "vector", {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 256}}
        )
        collection.create_index("src_id", {"index_type": "TRIE"})
        collection.create_index("tgt_id", {"index_type": "TRIE"})

        print("  Created collection with indexes")

    collection = Collection(collection_name)
    collection.load()

    rel_ids = [rel["id"] for rel in relationships]
    existing_weights = {}

    if rel_ids:
        query_batch_size = 100
        for i in range(0, len(rel_ids), query_batch_size):
            batch_ids = rel_ids[i : i + query_batch_size]
            expr = " or ".join([f'id == "{rid}"' for rid in batch_ids])

            try:
                results = collection.query(expr=expr, output_fields=["id", "weight"], limit=len(batch_ids))
                for r in results:
                    existing_weights[r["id"]] = r.get("weight", 0.0)
            except Exception:
                pass

    for rel in relationships:
        if rel["id"] in existing_weights:
            rel["weight"] = rel["weight"] + existing_weights[rel["id"]]

    for i in range(0, len(relationships), batch_size):
        batch = relationships[i : i + batch_size]
        collection.upsert(batch)

    collection.flush()
    print(f"  Upserted {len(relationships)} relationships to Milvus (weights accumulated)")


def clear_neo4j_workspace(neo4j_driver, workspace):
    """Clear all Neo4j nodes and relationships."""
    with neo4j_driver.session() as session:
        session.run("MATCH ()-[r]->() DELETE r")
        session.run("MATCH (n) DELETE n")


def ensure_neo4j_indexes(neo4j_driver, workspace):
    """Create a unique constraint on entity_id for the workspace label."""
    with neo4j_driver.session() as session:
        try:
            session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:`{workspace}`) REQUIRE n.entity_id IS UNIQUE")
        except Exception:
            pass


def get_entity_data_from_milvus(entities_collection, entity_names):
    """Query entity data from Milvus including IDs."""
    if not entity_names:
        return {}

    collection = Collection(entities_collection)
    collection.load()

    entities = {}
    entity_list = list(entity_names)
    batch_size = 100

    for i in range(0, len(entity_list), batch_size):
        batch = entity_list[i : i + batch_size]
        expr = " or ".join([f'entity_name == "{e}"' for e in batch])

        try:
            results = collection.query(
                expr=expr, output_fields=["id", "entity_name", "entity_type", "content"], limit=min(15000, len(batch))
            )

            for entity in results:
                entities[entity["entity_name"]] = entity
        except Exception:
            pass

    return entities


def get_neo4j_label_from_entity_name(entity_name):
    """Extract Neo4j label from entity name prefix."""
    prefix_to_label = {
        "GENE:": "Gene",
        "DISEASE:": "Disease",
        "CHEM:": "Chemical",
        "CL:": "Cell",
        "ANATOMY:": "Anatomy",
        "TAX:": "Taxonomy",
        "GO:": "GeneOntology",
        "REACTOME:": "ReactomePathway",
        "KEGG:": "KEGGPathway",
    }

    for prefix, label in prefix_to_label.items():
        if entity_name.startswith(prefix):
            return label

    return "Unknown"


def get_entity_type_from_name(entity_name):
    """Extract entity type from entity name prefix (e.g., GENE:BRCA1 -> GENE)."""
    if ":" in entity_name:
        return entity_name.split(":", 1)[0]
    return "Unknown"


def upsert_entities_to_neo4j(neo4j_driver, workspace, entities, batch_size=1000):
    """Upsert entities to Neo4j with entity_id = entity_name."""
    entity_list = list(entities.values())

    entities_by_label = defaultdict(list)
    for entity in entity_list:
        label = get_neo4j_label_from_entity_name(entity["entity_name"])
        entities_by_label[label].append(entity)

    with neo4j_driver.session() as session:
        for label, label_entities in entities_by_label.items():
            for i in range(0, len(label_entities), batch_size):
                batch = label_entities[i : i + batch_size]

                query = f"""
                UNWIND $entities AS entity
                MERGE (n:`{workspace}` {{entity_id: entity.entity_id}})
                SET n.entity_type = entity.entity_type,
                    n.description = entity.description,
                    n:`{label}`
                """

                entity_data = [
                    {
                        "entity_id": e["entity_name"],
                        "entity_type": get_entity_type_from_name(e["entity_name"]),
                        "description": e.get("content", ""),
                    }
                    for e in batch
                ]

                session.run(query, entities=entity_data)

    print(f"  Upserted {len(entity_list)} entities to Neo4j ({len(entities_by_label)} type labels)")


def upsert_relationships_to_neo4j(neo4j_driver, workspace, relationships, batch_size=1000):
    """Upsert relationships to Neo4j between nodes keyed by entity_id."""

    rel_data = []
    for rel in relationships:
        rel_data.append(
            {
                "src_entity_id": rel["src_id"],
                "tgt_entity_id": rel["tgt_id"],
                "weight": rel["weight"],
                "description": rel.get("content", ""),
            }
        )

    with neo4j_driver.session() as session:
        for i in range(0, len(rel_data), batch_size):
            batch = rel_data[i : i + batch_size]

            query = f"""
            UNWIND $relationships AS rel
            MATCH (source:`{workspace}` {{entity_id: rel.src_entity_id}})
            MATCH (target:`{workspace}` {{entity_id: rel.tgt_entity_id}})
            MERGE (source)-[r:DIRECTED]-(target)
            ON CREATE SET r.weight = rel.weight, r.description = rel.description
            ON MATCH SET r.weight = r.weight + rel.weight, r.description = rel.description
            """

            session.run(query, relationships=batch)

    print(f"  Upserted {len(relationships)} relationships to Neo4j")


def main():
    parser = argparse.ArgumentParser(description="Insert relationships into Milvus and Neo4j")
    parser.add_argument("--milvus-host", default="localhost")
    parser.add_argument("--milvus-port", default="19530")
    parser.add_argument("--neo4j-uri", default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--neo4j-username", default=os.environ.get("NEO4J_USERNAME", "neo4j"))
    parser.add_argument(
        "--neo4j-password",
        default=os.environ.get("NEO4J_PASSWORD"),
        help="Neo4j password (defaults to the NEO4J_PASSWORD environment variable)",
    )
    parser.add_argument("--workspace", default="lightrag")
    parser.add_argument(
        "--paper-batch-size", type=int, default=100, help="Number of papers to process before upserting"
    )
    parser.add_argument("--relationship-batch-size", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers for ID generation")
    parser.add_argument(
        "--query-workers", type=int, default=4, help="Parallel workers for querying relationship collections"
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--entities-ngram-dir", required=True)
    parser.add_argument("--similarity-threshold", type=float, default=0.85)
    parser.add_argument(
        "--entity-stats",
        type=Path,
        default=None,
        help="Path to entity statistics TSV file (entity_statistics_GLOBAL.tsv)",
    )
    parser.add_argument(
        "--max-pct-paper",
        type=float,
        default=None,
        help="Maximum pct_paper threshold - filter out entities appearing in >= this %% of papers",
    )
    parser.add_argument(
        "--max-pct-chunk",
        type=float,
        default=None,
        help="Maximum pct_chunk threshold - filter out entities appearing in >= this %% of chunks",
    )
    parser.add_argument("--neo4j-only", action="store_true", help="Skip Milvus insertion and write only to Neo4j")
    parser.add_argument("--drop-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=0)

    args = parser.parse_args()
    if not args.neo4j_password:
        parser.error("--neo4j-password or the NEO4J_PASSWORD environment variable is required")

    data_dir = Path(args.data_dir)
    entities_dir = Path(args.entities_ngram_dir)
    entities_collection = f"{args.workspace}_entities"
    relationships_collection = f"{args.workspace}_relationships"
    log_file = data_dir / "relationships-inserted.csv"

    print("=" * 60)
    if args.neo4j_only:
        print("Relationship insertion (Neo4j only)")
    else:
        print("Relationship insertion")
    print("=" * 60)
    print(f"Workspace: {args.workspace}")
    if args.neo4j_only:
        print(f"Milvus: {args.milvus_host}:{args.milvus_port} (SKIPPED)")
    else:
        print(f"Milvus: {args.milvus_host}:{args.milvus_port}")
    print(f"Neo4j: {args.neo4j_uri}")
    print(f"Paper batch size: {args.paper_batch_size}")
    print(f"ID generation workers: {args.workers}")
    print(f"Entities directory: {entities_dir}")
    print(f"Similarity threshold: {args.similarity_threshold}")

    print(f"Query workers: {args.query_workers}")

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

    connections.connect(host=args.milvus_host, port=args.milvus_port)
    neo4j_driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_username, args.neo4j_password))

    if not utility.has_collection(entities_collection):
        print(f"ERROR: Entities collection {entities_collection} does not exist!")
        return 1

    rel_schemas = discover_relationship_schemas()

    if not rel_schemas:
        print("ERROR: No relationship collections found!")
        return 1

    ensure_relationship_indexes(rel_schemas)

    if args.drop_existing:
        print("Dropping existing data...")
        if not args.neo4j_only:
            if utility.has_collection(relationships_collection):
                utility.drop_collection(relationships_collection)
                print(f"  Dropped Milvus collection: {relationships_collection}")
        else:
            print("  Skipped Milvus drop (neo4j-only mode)")

        clear_neo4j_workspace(neo4j_driver, args.workspace)
        print("  Cleared Neo4j workspace")

        ensure_neo4j_indexes(neo4j_driver, args.workspace)
        print("  Created Neo4j indexes")

        if log_file.exists():
            log_file.unlink()
            print(f"  Deleted log file: {log_file}")
        print()

    entity_stats = {}
    if args.entity_stats:
        entity_stats = load_entity_statistics(args.entity_stats)
        print()

    processed_papers = load_processed_papers(log_file)
    print(f"Already processed: {len(processed_papers)} papers\n")

    unprocessed_papers = get_unprocessed_papers(entities_dir, processed_papers, args.limit)

    if args.limit > 0:
        print(f"Unprocessed papers: {len(unprocessed_papers)} (limited to {args.limit})")
    else:
        print(f"Unprocessed papers: {len(unprocessed_papers)}")

    if not unprocessed_papers:
        print("No papers to process!")
        return 0

    print()

    weight_accumulator = {}
    total_papers = 0
    total_relationships = 0

    start_time = time.time()
    total_batches = (len(unprocessed_papers) + args.paper_batch_size - 1) // args.paper_batch_size

    batch_idx = 0
    try:
        for batch_start in range(0, len(unprocessed_papers), args.paper_batch_size):
            batch_end = min(batch_start + args.paper_batch_size, len(unprocessed_papers))
            paper_batch = unprocessed_papers[batch_start:batch_end]
            batch_idx += 1

            print(f"\n{'=' * 60}")
            print(
                f"Batch {batch_idx}/{total_batches}: "
                f"Processing papers {batch_start + 1}-{batch_end}/{len(unprocessed_papers)}"
            )
            print(f"{'=' * 60}")

            try:
                print(f"  [1/3] Generating relationship IDs ({args.workers} workers)...")
                id_gen_start = time.time()
                aggregated_rels = generate_ids_for_paper_batch(
                    paper_batch,
                    args.similarity_threshold,
                    rel_schemas,
                    entities_collection,
                    args.workers,
                    entity_stats,
                    args.max_pct_paper,
                    args.max_pct_chunk,
                )
                id_gen_time = time.time() - id_gen_start

                total_possible = sum(len(rels) for rels in aggregated_rels.values())
                print(f"    Generated {total_possible:,} possible relationships ({id_gen_time:.1f}s)")

                if not aggregated_rels:
                    print("    No relationships found for this batch")
                    for paper_dir in paper_batch:
                        save_paper_progress(log_file, paper_dir.name, 0)
                    total_papers += len(paper_batch)
                    continue

                print(f"  [2/3] Querying Milvus ({args.query_workers} workers)...")

                query_start = time.time()

                query_input = {}
                rel_to_papers = defaultdict(set)
                for collection_name, rel_tuples in aggregated_rels.items():
                    query_input[collection_name] = []
                    for rel_id, src, tgt, paper_id in rel_tuples:
                        query_input[collection_name].append((rel_id, src, tgt))
                        rel_to_papers[rel_id].add(paper_id)

                all_relationships = query_relationships_parallel(query_input, args.query_workers)
                query_time = time.time() - query_start

                print(f"    Found {len(all_relationships):,} existing relationships ({query_time:.1f}s)")

                if not all_relationships:
                    print("    No relationships found for this batch")
                    for paper_dir in paper_batch:
                        save_paper_progress(log_file, paper_dir.name, 0)
                    total_papers += len(paper_batch)
                    continue

                print("  [3/3] Accumulating weights...")
                for rel in all_relationships:
                    rel_id = rel["id"]
                    papers = rel_to_papers[rel_id]
                    for paper_id in papers:
                        accumulate_weights(weight_accumulator, [rel], paper_id)

                if weight_accumulator:
                    if args.neo4j_only:
                        print(f"\n  Upserting to Neo4j: {len(weight_accumulator)} unique relationships")
                    else:
                        print(f"\n  Upserting: {len(weight_accumulator)} unique relationships")

                    prepared_rels = prepare_relationships_for_upsert(weight_accumulator)

                    if not args.neo4j_only:
                        upsert_relationships_to_milvus(
                            relationships_collection,
                            prepared_rels,
                            drop_existing=False,
                            batch_size=args.relationship_batch_size,
                        )
                    else:
                        print("  Skipped Milvus upsert (neo4j-only mode)")

                    unique_entities = set()
                    for rel in prepared_rels:
                        unique_entities.add(rel["src_id"])
                        unique_entities.add(rel["tgt_id"])

                    entity_data = get_entity_data_from_milvus(entities_collection, unique_entities)

                    upsert_entities_to_neo4j(neo4j_driver, args.workspace, entity_data, args.relationship_batch_size)
                    upsert_relationships_to_neo4j(
                        neo4j_driver, args.workspace, prepared_rels, args.relationship_batch_size
                    )

                    total_relationships += len(prepared_rels)

                for paper_dir in paper_batch:
                    save_paper_progress(log_file, paper_dir.name, len(weight_accumulator))

                total_papers += len(paper_batch)

                weight_accumulator = {}

                elapsed_time = time.time() - start_time
                papers_per_second = total_papers / elapsed_time if elapsed_time > 0 else 0
                remaining_papers = len(unprocessed_papers) - total_papers
                eta_seconds = remaining_papers / papers_per_second if papers_per_second > 0 else 0

                eta_minutes = int(eta_seconds // 60)
                eta_secs = int(eta_seconds % 60)

                print(f"  Batch complete ({elapsed_time:.1f}s elapsed, {papers_per_second:.1f} papers/s)")
                if remaining_papers > 0:
                    print(f"  ETA: {eta_minutes}m {eta_secs}s ({remaining_papers} papers remaining)")

            except Exception as e:
                print(f"\n{'=' * 60}")
                print(f"FATAL ERROR in batch {batch_idx}:")
                print(f"  {e}")
                print(f"{'=' * 60}")
                print(f"\nProcessing stopped at batch {batch_idx}/{total_batches}")
                print(f"Papers processed successfully: {total_papers}/{len(unprocessed_papers)}")
                print(f"Papers NOT marked as completed: {len(paper_batch)}")
                print("\nTo resume, re-run the script. Already-processed papers will be skipped.")
                raise

    except KeyboardInterrupt:
        print(f"\n{'=' * 60}")
        print("Interrupted by user (Ctrl+C)")
        print(f"Papers processed successfully: {total_papers}/{len(unprocessed_papers)}")
        print(f"{'=' * 60}")
        raise

    finally:
        neo4j_driver.close()

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("ALL PAPERS COMPLETE!")
    print(f"  Total papers: {total_papers}")
    print(f"  Total relationships: {total_relationships}")
    print(f"  Time: {elapsed:.1f}s ({elapsed / 60:.1f}m)")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
