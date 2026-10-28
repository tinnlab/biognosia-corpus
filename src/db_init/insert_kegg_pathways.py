#!/usr/bin/env python3
"""
Insert KEGG pathways and gene-pathway relationships into Milvus.

This script:
1. Fetches KEGG data from REST API with caching
2. Builds gene symbol mappings (reuses GeneIDMapper)
3. Processes pathways with dual embeddings (BGE-M3 + MedCPT)
4. Processes gene-pathway relationships with dual embeddings
5. Uses parallel streams for embedding and insertion

Collections created:
- entities_KEGG_Pathway: KEGG pathway entities
- relationships_Gene_KEGG_Pathway: Gene -> KEGG_Pathway relationships
"""

import argparse
import hashlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config_loader import PROJECT_ROOT, load_config
from gene_id_mapper import GeneIDMapper
from insert_gene2go import ParallelEmbeddingGenerator

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
WORKERS = int(CONFIG.get("WORKERS", "20"))
CUDA_DEVICES = CONFIG.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")
MAX_PENDING_BATCHES = int(CONFIG.get("MAX_PENDING_BATCHES", "10"))

KEGG_CACHE_DAYS = int(CONFIG.get("KEGG_CACHE_DAYS", "30"))
KEGG_CACHE_DIR = Path(CONFIG.get("KEGG_CACHE_DIR", f"{DATA_DIR}/db-init/kegg-cache"))
KEGG_LIMIT = int(CONFIG.get("KEGG_LIMIT", "0"))

MILVUS_HOST = CONFIG.get("MILVUS_HOST", "localhost")
MILVUS_PORT = CONFIG.get("MILVUS_PORT", "19530")
MILVUS_BATCH_SIZE = int(CONFIG.get("MILVUS_BATCH_SIZE", "5000"))

METADATA_DIR = Path(CONFIG.get("METADATA_DIR", f"{DATA_DIR}/db-init"))
METADATA_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class PathwayDetails:
    """KEGG pathway details."""

    pathway_id: str
    name: str
    description: str = ""
    pathway_class: str = ""
    organism: str = ""


def clean_pathway_name(name: str) -> str:
    """
    Remove organism suffix from pathway name.

    Example: "Mannose type O-glycan biosynthesis - Homo sapiens (human)"
             -> "Mannose type O-glycan biosynthesis"
    """
    if " - " in name:
        parts = name.rsplit(" - ", 1)
        return parts[0].strip()
    return name.strip()


class KEGGCache:
    """Manages KEGG API requests with file-based caching."""

    BASE_URL = "https://rest.kegg.jp"

    def __init__(self, cache_dir: Path = KEGG_CACHE_DIR, cache_days: int = KEGG_CACHE_DAYS):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_days = cache_days
        self.last_request_time = 0

    def _rate_limit(self):
        """Enforce 2 second delay between API requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < 2.0:
            time.sleep(2.0 - elapsed)
        self.last_request_time = time.time()

    def _is_cache_valid(self, cache_file: Path) -> bool:
        """Check if cache file exists and is recent."""
        if not cache_file.exists():
            return False
        age_days = (time.time() - cache_file.stat().st_mtime) / (24 * 3600)
        return age_days < self.cache_days

    def _fetch_url(self, url: str, max_retries: int = 3) -> str:
        """Fetch URL with rate limiting and retry logic."""
        self._rate_limit()
        print(f"  Fetching: {url}")

        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 429:
                    print("Rate limited (429), waiting 5 seconds...")
                    time.sleep(5)
                    continue
                response.raise_for_status()
                return response.text
            except requests.RequestException as e:
                wait_time = 2**attempt
                print(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"  Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise

    def _fetch_or_cache(self, url: str, cache_file: Path) -> str:
        """Fetch from URL or load from cache."""
        if self._is_cache_valid(cache_file):
            return cache_file.read_text()

        data = self._fetch_url(url)

        cache_file.write_text(data)
        return data

    def get_organisms(self) -> list[tuple[str, str, str, str]]:
        """Get list of organisms from KEGG.

        Returns:
            List of (t_number, org_code, species_name, taxonomy)
        """
        cache_file = self.cache_dir / "organisms.tsv"
        url = f"{self.BASE_URL}/list/organism"

        data = self._fetch_or_cache(url, cache_file)

        organisms = []
        for line in data.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) >= 4:
                organisms.append((parts[0], parts[1], parts[2], parts[3]))

        return organisms

    def get_pathway_list(self, org: str) -> dict[str, str]:
        """Get pathway list for organism.

        Args:
            org: Organism code (e.g., 'hsa' for human)

        Returns:
            Dict mapping pathway_id -> pathway_name
        """
        cache_file = self.cache_dir / f"pathways_list_{org}.tsv"
        url = f"{self.BASE_URL}/list/pathway/{org}"

        data = self._fetch_or_cache(url, cache_file)

        pathways = {}
        for line in data.strip().split("\n"):
            parts = line.split("\t", 1)
            if len(parts) >= 2:
                pathway_id = parts[0].replace("path:", "")
                pathway_name = parts[1]
                pathways[pathway_id] = pathway_name

        return pathways

    def _parse_pathway_details(self, flat_file: str, pathway_id: str) -> PathwayDetails:
        """Parse KEGG flat file format for pathway details."""
        name = ""
        description = ""
        pathway_class = ""
        organism = ""

        current_section = None
        description_lines = []

        for line in flat_file.split("\n"):
            if not line.strip():
                continue

            if line[0] != " " and not line.startswith("///"):
                parts = line.split(None, 1)
                if len(parts) >= 1:
                    current_section = parts[0]
                    content = parts[1] if len(parts) > 1 else ""

                    if current_section == "NAME":
                        name = content
                    elif current_section == "DESCRIPTION":
                        description_lines = [content]
                    elif current_section == "CLASS":
                        pathway_class = content
                    elif current_section == "ORGANISM":
                        organism = content
            else:
                if current_section == "DESCRIPTION":
                    description_lines.append(line.strip())

        description = " ".join(description_lines) if description_lines else ""

        if not description and name:
            description = name

        return PathwayDetails(
            pathway_id=pathway_id, name=name, description=description, pathway_class=pathway_class, organism=organism
        )

    def get_pathway_details(self, pathway_ids: list[str]) -> dict[str, PathwayDetails]:
        """Get detailed information for pathways.

        Args:
            pathway_ids: List of pathway IDs

        Returns:
            Dict mapping pathway_id -> PathwayDetails
        """
        details = {}

        for i in range(0, len(pathway_ids), 10):
            batch = pathway_ids[i : i + 10]

            all_cached = all(self._is_cache_valid(self.cache_dir / f"pathway_details_{pid}.txt") for pid in batch)

            if all_cached:
                for pathway_id in batch:
                    cache_file = self.cache_dir / f"pathway_details_{pathway_id}.txt"
                    flat_file = cache_file.read_text()
                    details[pathway_id] = self._parse_pathway_details(flat_file, pathway_id)
            else:
                batch_query = "+".join(batch)
                url = f"{self.BASE_URL}/get/{batch_query}"

                try:
                    data = self._fetch_url(url)

                    entries = data.split("///")
                    for entry in entries:
                        if not entry.strip():
                            continue

                        for line in entry.split("\n"):
                            if line.startswith("ENTRY"):
                                parts = line.split()
                                if len(parts) >= 2:
                                    pathway_id = parts[1]
                                    cache_file = self.cache_dir / f"pathway_details_{pathway_id}.txt"
                                    cache_file.write_text(entry)
                                    details[pathway_id] = self._parse_pathway_details(entry, pathway_id)
                                    break

                except Exception as e:
                    print(f"Failed to fetch batch {batch_query}: {e}")
                    for pathway_id in batch:
                        cache_file = self.cache_dir / f"pathway_details_{pathway_id}.txt"
                        if self._is_cache_valid(cache_file):
                            flat_file = cache_file.read_text()
                            details[pathway_id] = self._parse_pathway_details(flat_file, pathway_id)
                        else:
                            try:
                                url = f"{self.BASE_URL}/get/{pathway_id}"
                                data = self._fetch_url(url)
                                cache_file.write_text(data)
                                details[pathway_id] = self._parse_pathway_details(data, pathway_id)
                            except Exception as e2:
                                print(f"Failed to fetch {pathway_id}: {e2}")

        return details

    def get_gene_mapping(self, org: str) -> dict[str, str]:
        """Get Entrez ID <-> KEGG gene ID mapping.

        Args:
            org: Organism code (e.g., 'hsa')

        Returns:
            Dict mapping entrez_id -> kegg_gene_id
        """
        cache_file = self.cache_dir / f"gene_mapping_{org}.tsv"
        url = f"{self.BASE_URL}/conv/{org}/ncbi-geneid"

        data = self._fetch_or_cache(url, cache_file)

        mapping = {}
        for line in data.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) >= 2:
                kegg_id = parts[0].replace(f"{org}:", "")
                entrez_id = parts[1].replace("ncbi-geneid:", "")
                mapping[entrez_id] = kegg_id

        return mapping

    def get_gene_pathway_links(self, org: str) -> list[tuple[str, str]]:
        """Get gene-pathway links for organism.

        Args:
            org: Organism code (e.g., 'hsa')

        Returns:
            List of (kegg_gene_id, pathway_id) tuples
        """
        cache_file = self.cache_dir / f"gene_pathway_links_{org}.tsv"
        url = f"{self.BASE_URL}/link/pathway/{org}"

        data = self._fetch_or_cache(url, cache_file)

        links = []
        for line in data.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) >= 2:
                kegg_id = parts[0].replace(f"{org}:", "")
                pathway_id = parts[1].replace("path:", "")
                links.append((kegg_id, pathway_id))

        return links


class MilvusInserter:
    """Insert KEGG pathways and relationships into Milvus."""

    def __init__(
        self,
        host: str = MILVUS_HOST,
        port: str = MILVUS_PORT,
        batch_size: int = MILVUS_BATCH_SIZE,
        drop_existing: bool = False,
        skip_entities: bool = False,
        skip_relationships: bool = False,
    ):
        connections.connect(host=host, port=port)
        self.batch_size = batch_size
        self.skip_entities = skip_entities
        self.skip_relationships = skip_relationships
        self.pathway_pending = []
        self.relationship_pending = []
        self.pathway_total_inserted = 0
        self.pathway_total_time = 0.0
        self.rel_total_inserted = 0
        self.rel_total_time = 0.0

        self._drop_kegg_data(drop_existing, skip_entities, skip_relationships)

        self._ensure_collections()

    def _drop_kegg_data(self, drop_existing: bool, skip_entities: bool, skip_relationships: bool):
        """Drop existing KEGG pathway and relationships collections based on flags.

        Args:
            drop_existing: If True, drop collections
            skip_entities: If True, don't drop entities collection even if drop_existing is True
            skip_relationships: If True, don't drop relationships collection even if drop_existing is True
        """
        if drop_existing:
            if not skip_entities and utility.has_collection("entities_KEGG_Pathway"):
                utility.drop_collection("entities_KEGG_Pathway")
                print("Dropped existing Milvus collection: entities_KEGG_Pathway")

            if not skip_relationships and utility.has_collection("relationships_Gene_KEGG_Pathway"):
                utility.drop_collection("relationships_Gene_KEGG_Pathway")
                print("Dropped existing Milvus collection: relationships_Gene_KEGG_Pathway")

    def _ensure_collections(self):
        """Ensure pathway and relationships collections exist based on skip flags."""
        if not self.skip_entities:
            self.pathways_collection_name = "entities_KEGG_Pathway"
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
                FieldSchema(name="label_embedding", dtype=DataType.FLOAT_VECTOR, dim=LABEL_EMBEDDING_DIM),
                FieldSchema(name="created_at", dtype=DataType.INT64),
                FieldSchema(name="entity_name", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=32768),
            ]
            schema = CollectionSchema(fields, enable_dynamic_field=True)
            self.pathways_collection = Collection(self.pathways_collection_name, schema)
            print(f"Created Milvus collection: {self.pathways_collection_name}")
            print("  (indexes will be created after insertion)")
        else:
            self.pathways_collection_name = None
            self.pathways_collection = None
            print("Skipping entities collection creation (--skip-entities)")

        if not self.skip_relationships:
            self.relationships_collection_name = "relationships_Gene_KEGG_Pathway"
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
        else:
            self.relationships_collection_name = None
            self.relationships_collection = None
            print("Skipping relationships collection creation (--skip-relationships)")

    def insert_pathway_batch(
        self,
        pathways: list[PathwayDetails],
        content_embeddings: np.ndarray,
        label_embeddings: np.ndarray,
        pathway_genes: dict[str, list[str]] = None,
    ):
        """Add KEGG pathways to pending batch."""
        current_time = int(time.time())
        if pathway_genes is None:
            pathway_genes = {}

        for pathway, content_emb, label_emb in zip(pathways, content_embeddings, label_embeddings, strict=False):
            entity_type = "KEGG_Pathway"
            entity_id = f"KEGG:{pathway.pathway_id}"
            pk_id = self._compute_id(entity_id, prefix="ent-")

            genes = pathway_genes.get(pathway.pathway_id, [])
            if genes:
                content = ", ".join(genes[:1000])
            else:
                content = ""
            label = clean_pathway_name(pathway.name)

            if pathway.description:
                full_description = f"{pathway.name}: {pathway.description}"
            else:
                full_description = pathway.name

            record = {
                "id": pk_id,
                "vector": content_emb.tolist(),
                "label_embedding": label_emb.tolist(),
                "created_at": current_time,
                "entity_name": entity_id,
                "file_path": "kegg_rest_api",
                "content": content,
                "label": label,
                "description": full_description,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "pathway_id": pathway.pathway_id,
                "pathway_name": pathway.name,
                "pathway_class": pathway.pathway_class,
                "organism": pathway.organism,
            }
            self.pathway_pending.append(record)

            if len(self.pathway_pending) >= self.batch_size:
                self._flush_pathways()

    def insert_relationship_batch(
        self,
        gene_symbols: list[str],
        pathway_ids: list[str],
        pathways_dict: dict[str, PathwayDetails],
        content_embeddings: np.ndarray,
    ):
        """Add Gene->KEGG_Pathway relationships to pending batch."""
        current_time = int(time.time())

        for gene_symbol, pathway_id, content_emb in zip(gene_symbols, pathway_ids, content_embeddings, strict=False):
            src_id = f"GENE:{gene_symbol}"
            tgt_id = f"KEGG:{pathway_id}"

            composite_key = f"{src_id}{GRAPH_FIELD_SEP}{tgt_id}"
            pk_id = self._compute_id(composite_key, prefix="rel-")

            pathway = pathways_dict.get(pathway_id)
            pathway_name = pathway.name if pathway else pathway_id

            content = f"{gene_symbol} participates_in {pathway_name} {pathway_id}"

            rel_record = {
                "id": pk_id,
                "vector": content_emb.tolist(),
                "created_at": current_time,
                "src_id": src_id,
                "tgt_id": tgt_id,
                "file_path": "kegg_rest_api",
                "content": content,
                "description": pathway_name,
                "keywords": "participates_in",
                "weight": 0.8,
                "source_id": "",
                "gene_symbol": gene_symbol,
                "pathway_id": pathway_id,
            }
            self.relationship_pending.append(rel_record)

            if len(self.relationship_pending) >= self.batch_size:
                self._flush_relationships()

    def _flush_pathways(self):
        """Flush pending pathways to Milvus using insert."""
        if not self.pathway_pending or self.skip_entities or self.pathways_collection is None:
            return

        flush_start = time.time()
        self.pathways_collection.insert(self.pathway_pending)

        flush_duration = time.time() - flush_start
        batch_count = len(self.pathway_pending)
        self.pathway_total_inserted += batch_count
        self.pathway_total_time += flush_duration

        self.pathway_pending = []

    def _flush_relationships(self):
        """Flush pending relationships to Milvus using insert."""
        if not self.relationship_pending or self.skip_relationships or self.relationships_collection is None:
            return

        flush_start = time.time()
        self.relationships_collection.insert(self.relationship_pending)

        flush_duration = time.time() - flush_start
        batch_count = len(self.relationship_pending)
        self.rel_total_inserted += batch_count
        self.rel_total_time += flush_duration

        self.relationship_pending = []

    def finalize_pathways(self):
        """Flush pathways data, create indexes, and load collection."""
        if self.skip_entities or self.pathways_collection is None:
            print("Skipping pathway finalization (--skip-entities)")
            return

        self._flush_pathways()

        print("Flushing pathways to Milvus...")
        self.pathways_collection.flush()
        print("Pathways flushed to Milvus")

        print("Creating indexes for pathway vector fields...")
        index_params = {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 256}}

        self.pathways_collection.create_index("vector", index_params)
        print("Created index for pathway content embeddings (vector)")

        self.pathways_collection.create_index("label_embedding", index_params)
        print("Created index for pathway label embeddings (label_embedding)")

        self.pathways_collection.load()
        print("Pathways collection loaded and ready")

    def finalize_relationships(self):
        """Flush relationships data, create indexes, and load collection."""
        if self.skip_relationships or self.relationships_collection is None:
            print("Skipping relationship finalization (--skip-relationships)")
            return

        self._flush_relationships()

        print("Flushing relationships to Milvus...")
        self.relationships_collection.flush()
        print("Relationships flushed to Milvus")

        print("Creating indexes for relationship vector fields...")
        index_params = {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 256}}

        self.relationships_collection.create_index("vector", index_params)
        print("Created index for relationship embeddings (vector)")

        self.relationships_collection.load()
        print("Relationships collection loaded and ready")

    def close(self):
        """Flush and disconnect."""
        self._flush_pathways()
        self._flush_relationships()
        connections.disconnect("default")

    @staticmethod
    def _compute_id(content: str, prefix: str = "") -> str:
        """Compute a prefixed MD5 hash ID."""
        return prefix + hashlib.md5(content.encode()).hexdigest()


def process_kegg_pathways(args):
    """Main processing function."""
    start_time = time.time()

    print("=" * 60)
    print("KEGG Pathway Insertion")
    print("=" * 60)
    print(f"Configuration file: {CONFIG_FILE}")
    print(f"Organisms: {args.organisms if args.organisms else 'ALL (will be fetched)'}")
    print(f"Batch size: {args.batch_size}")
    print(f"Workers: {WORKERS}")
    print()

    print("Step 1: Initializing KEGG cache...")
    kegg_cache = KEGGCache()

    if not args.organisms:
        print("  No organisms specified, fetching all available organisms from KEGG...")
        all_organisms = kegg_cache.get_organisms()
        args.organisms = [org_code for _, org_code, _, _ in all_organisms]
        print(f"  Found {len(args.organisms)} organisms to process")
    print()

    print("Step 2: Loading gene symbol mapping...")
    gene_mapper = GeneIDMapper(args.gene_info_file, filter_taxid=args.filter_taxid)
    gene_mapper.build_mapping()
    entrez_to_symbol = gene_mapper.mapping
    print(f"Loaded {len(entrez_to_symbol):,} gene symbols")
    print()

    all_pathways = {}
    all_relationships = []

    for org in args.organisms:
        print(f"Processing organism: {org}")

        try:
            print("  Fetching pathway list...")
            pathway_list = kegg_cache.get_pathway_list(org)
            print(f"  Found {len(pathway_list):,} pathways")

            print("  Fetching pathway details...")
            pathway_ids = list(pathway_list.keys())

            if KEGG_LIMIT > 0:
                pathway_ids = pathway_ids[:KEGG_LIMIT]
                print(f"  Limiting to first {KEGG_LIMIT} pathways")

            pathway_details = kegg_cache.get_pathway_details(pathway_ids)
            print(f"  Retrieved {len(pathway_details):,} pathway details")

            all_pathways.update(pathway_details)

            print("  Fetching gene mappings...")
            entrez_to_kegg = kegg_cache.get_gene_mapping(org)
            print(f"  Found {len(entrez_to_kegg):,} gene mappings")

            print("  Fetching gene-pathway links...")
            gene_pathway_links = kegg_cache.get_gene_pathway_links(org)
            print(f"  Found {len(gene_pathway_links):,} gene-pathway links")

            print("  Building relationships...")

            pathway_ids_set = set(pathway_ids)

            skipped_no_entrez = 0
            skipped_no_symbol = 0
            skipped_not_in_pathway_set = 0
            for kegg_gene_id, pathway_id in gene_pathway_links:
                if pathway_id not in pathway_ids_set:
                    skipped_not_in_pathway_set += 1
                    continue

                org_gene_id = f"{org}:{kegg_gene_id}"
                ncbi_gene_id = entrez_to_kegg.get(org_gene_id)

                if ncbi_gene_id is None:
                    skipped_no_entrez += 1
                    continue

                if ncbi_gene_id.startswith("ncbi-geneid:"):
                    entrez_id = ncbi_gene_id.split(":", 1)[1]
                else:
                    entrez_id = ncbi_gene_id

                gene_symbol = entrez_to_symbol.get(entrez_id)
                if gene_symbol is None:
                    skipped_no_symbol += 1
                    continue

                all_relationships.append((gene_symbol, pathway_id))

            print(f"  Built {len(all_relationships):,} relationships")
            if skipped_not_in_pathway_set > 0:
                print(f"  Skipped {skipped_not_in_pathway_set:,} (pathway not in limited set)")
            if skipped_no_entrez > 0:
                print(f"  Skipped {skipped_no_entrez:,} (no entrez mapping)")
            if skipped_no_symbol > 0:
                print(f"  Skipped {skipped_no_symbol:,} (gene not in filtered set)")
            print()

        except Exception as e:
            print(f"  Error processing organism {org}: {e}")
            print(f"  Skipping organism {org} and continuing with next organism...")
            print()
            continue

    print(f"Total: {len(all_pathways):,} KEGG pathways")
    print(f"Total: {len(all_relationships):,} Gene->Pathway relationships")
    print()

    if len(all_pathways) == 0:
        print("No pathways to process. Exiting.")
        return

    print("Building pathway -> genes mapping...")
    pathway_genes: dict[str, list[str]] = {}
    for gene_symbol, pathway_id in all_relationships:
        if pathway_id not in pathway_genes:
            pathway_genes[pathway_id] = set()
        pathway_genes[pathway_id].add(gene_symbol.upper())

    for pathway_id in pathway_genes:
        pathway_genes[pathway_id] = sorted(pathway_genes[pathway_id])

    print(f"Mapped genes to {len(pathway_genes):,} pathways")
    print()

    print("Step 4: Initializing embedding workers and Milvus...")
    embedder = ParallelEmbeddingGenerator()
    milvus = MilvusInserter(
        drop_existing=args.drop_existing, skip_entities=args.skip_entities, skip_relationships=args.skip_relationships
    )
    print()

    if not args.skip_entities:
        print(f"Step 5: Processing {len(all_pathways):,} KEGG pathways")
        print("        with independent parallel streams...")
        buffer_size = MAX_PENDING_BATCHES * args.batch_size
        print(f"Max pending batches: {MAX_PENDING_BATCHES} (buffer size: {buffer_size:,} pathways)")
        print()

        pathway_list = list(all_pathways.values())

        batch_id = 0
        next_batch_idx = 0
        total_batches = (len(pathway_list) + args.batch_size - 1) // args.batch_size
        pending_batches = {}
        insertion_queue = []
        submitted_count = 0
        embedded_count = 0
        inserted_count = 0

        pbar_embed = tqdm(total=len(pathway_list), desc="Embedding pathways", position=0, leave=True)
        pbar_insert = tqdm(total=len(pathway_list), desc="Inserting to Milvus", position=1, leave=True)
    else:
        print("Step 5: Skipping pathway entities processing (--skip-entities)")
        print()

    try:
        if not args.skip_entities:
            while submitted_count < total_batches or pending_batches or insertion_queue:
                while (
                    submitted_count < total_batches
                    and len(pending_batches) + len(insertion_queue) < MAX_PENDING_BATCHES
                ):
                    start_idx = next_batch_idx * args.batch_size
                    end_idx = min(start_idx + args.batch_size, len(pathway_list))
                    batch_pathways = pathway_list[start_idx:end_idx]

                    contents = []
                    labels = []
                    for pathway in batch_pathways:
                        genes = pathway_genes.get(pathway.pathway_id, [])
                        if genes:
                            content = ", ".join(genes)
                        else:
                            content = ""

                        contents.append(content)
                        labels.append(clean_pathway_name(pathway.name))

                    embedder.embed_batch(batch_id, contents, labels)
                    pending_batches[batch_id] = batch_pathways

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
                            batch_pathways = pending_batches.pop(result_batch_id)

                            embedded_count += len(batch_pathways)
                            pbar_embed.update(len(batch_pathways))

                            insertion_queue.append((batch_pathways, content_embeddings, label_embeddings))

                if insertion_queue:
                    batch_pathways, content_embeddings, label_embeddings = insertion_queue.pop(0)

                    milvus.insert_pathway_batch(batch_pathways, content_embeddings, label_embeddings, pathway_genes)

                    inserted_count += len(batch_pathways)
                    pbar_insert.update(len(batch_pathways))

            pbar_embed.close()
            pbar_insert.close()

            print()
            print("Finalizing KEGG pathways in Milvus...")
            milvus.finalize_pathways()

        if not args.skip_relationships:
            print()
            print(f"Step 6: Processing {len(all_relationships):,} Gene->Pathway relationships")
            print("        with independent parallel streams...")
            buffer_size = MAX_PENDING_BATCHES * args.batch_size
            print(f"Max pending batches: {MAX_PENDING_BATCHES} (buffer size: {buffer_size:,} relationships)")
            print()

            batch_id = 0
            next_batch_idx = 0
            total_batches = (len(all_relationships) + args.batch_size - 1) // args.batch_size
            pending_batches = {}
            insertion_queue = []
            submitted_count = 0
            embedded_count = 0
            inserted_count = 0

            pbar_embed = tqdm(total=len(all_relationships), desc="Embedding relationships", position=0, leave=True)
            pbar_insert = tqdm(total=len(all_relationships), desc="Inserting to Milvus", position=1, leave=True)

            while submitted_count < total_batches or pending_batches or insertion_queue:
                while (
                    submitted_count < total_batches
                    and len(pending_batches) + len(insertion_queue) < MAX_PENDING_BATCHES
                ):
                    start_idx = next_batch_idx * args.batch_size
                    end_idx = min(start_idx + args.batch_size, len(all_relationships))
                    batch_rels = all_relationships[start_idx:end_idx]
                    batch_gene_symbols = [r[0] for r in batch_rels]
                    batch_pathway_ids = [r[1] for r in batch_rels]

                    contents = []
                    labels = []
                    for gene_symbol, pathway_id in batch_rels:
                        pathway = all_pathways.get(pathway_id)
                        pathway_name = pathway.name if pathway else pathway_id
                        content = f"{gene_symbol} participates_in {pathway_name} {pathway_id}"
                        contents.append(content)
                        labels.append(f"{gene_symbol}->{pathway_id}")

                    embedder.embed_batch(batch_id, contents, labels)
                    pending_batches[batch_id] = (batch_gene_symbols, batch_pathway_ids)

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
                            batch_gene_symbols, batch_pathway_ids = pending_batches.pop(result_batch_id)

                            embedded_count += len(batch_gene_symbols)
                            pbar_embed.update(len(batch_gene_symbols))

                            insertion_queue.append((batch_gene_symbols, batch_pathway_ids, content_embeddings))

                if insertion_queue:
                    batch_gene_symbols, batch_pathway_ids, content_embeddings = insertion_queue.pop(0)

                    milvus.insert_relationship_batch(
                        batch_gene_symbols, batch_pathway_ids, all_pathways, content_embeddings
                    )

                    inserted_count += len(batch_gene_symbols)
                    pbar_insert.update(len(batch_gene_symbols))

            pbar_embed.close()
            pbar_insert.close()

            print()
            print("Finalizing relationships in Milvus...")
            milvus.finalize_relationships()
        else:
            print()
            print("Step 6: Skipping gene-pathway relationships processing (--skip-relationships)")
            print()

        end_time = time.time()
        duration = end_time - start_time

        print()
        print("=" * 60)
        if not args.skip_entities:
            print(f"Inserted {len(all_pathways):,} KEGG pathways")
        else:
            print(f"Skipped {len(all_pathways):,} KEGG pathways (--skip-entities)")

        if not args.skip_relationships:
            print(f"Inserted {len(all_relationships):,} Gene->Pathway relationships")
        else:
            print(f"Skipped {len(all_relationships):,} Gene->Pathway relationships (--skip-relationships)")

        print(f"Duration: {duration:.2f} seconds ({duration / 60:.2f} minutes)")
        print("=" * 60)

    finally:
        embedder.close()
        milvus.close()


def main():
    parser = argparse.ArgumentParser(
        description="Insert KEGG pathways into Milvus",
        epilog="Configuration is loaded from config/db_init.conf. Command-line arguments override config.",
    )

    parser.add_argument("--config", default=str(CONFIG_FILE), help="Path to config file")
    parser.add_argument("--organisms", default="hsa", help="Comma-separated organism codes (default: hsa)")
    parser.add_argument(
        "--gene-info-file",
        default=CONFIG.get("GENE_INFO_FILE", f"{DATA_DIR}/resources/genes/gene_info.gz"),
        help="Path to gene_info.gz file",
    )
    parser.add_argument("--filter-taxid", nargs="+", help="Filter by taxonomy ID(s) (e.g., 9606 for human)")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=MILVUS_BATCH_SIZE,
        help=f"Batch size for processing (default: {MILVUS_BATCH_SIZE})",
    )
    parser.add_argument(
        "--workers", type=int, default=WORKERS, help=f"Number of embedding workers (default: {WORKERS})"
    )
    parser.add_argument("--drop-existing", action="store_true", help="Drop existing KEGG data before insertion")
    parser.add_argument("--skip-entities", action="store_true", help="Skip inserting pathway entities")
    parser.add_argument("--skip-relationships", action="store_true", help="Skip inserting gene-pathway relationships")

    args = parser.parse_args()

    args.organisms = [org.strip() for org in args.organisms.split(",") if org.strip()]

    process_kegg_pathways(args)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)

    main()
