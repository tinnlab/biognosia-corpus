#!/usr/bin/env python3
"""
Parse MSigDB SQLite database.

This module provides functions to:
1. Query gene sets with metadata
2. Query gene-gene_set memberships
3. Filter by collection and species
4. Map gene symbols
"""

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MSigDBGeneSet:
    """MSigDB gene set with metadata."""

    db_id: int
    standard_name: str
    collection_name: str
    systematic_name: str
    description_brief: str
    description_full: str
    external_url: str
    source_species: str
    num_genes: int
    genes: list[str]
    publication_pmid: str = ""
    tags: str = ""
    collection_full_name: str = ""


@dataclass
class CollectionInfo:
    """Collection metadata."""

    collection_name: str
    full_name: str
    description: str
    num_gene_sets: int


class MSigDBParser:
    """Parse MSigDB SQLite database."""

    def __init__(self, db_path: str):
        """Initialize parser with database path."""
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def get_collections_info(self) -> dict[str, CollectionInfo]:
        """Get information about all collections."""
        query = """
        SELECT
            c.collection_name,
            c.full_name,
            c.description,
            COUNT(gs.id) as num_gene_sets
        FROM collection c
        LEFT JOIN gene_set gs ON c.collection_name = gs.collection_name
        GROUP BY c.collection_name, c.full_name, c.description
        ORDER BY num_gene_sets DESC
        """

        cursor = self.conn.execute(query)
        collections = {}

        for row in cursor:
            info = CollectionInfo(
                collection_name=row["collection_name"],
                full_name=row["full_name"],
                description=row["description"],
                num_gene_sets=row["num_gene_sets"],
            )
            collections[info.collection_name] = info

        return collections

    def get_gene_sets(
        self, collections: list[str] | None = None, species_code: str = "HS", limit: int = 0
    ) -> dict[str, MSigDBGeneSet]:
        """
        Get gene sets with metadata.

        Args:
            collections: List of collection names to filter (e.g., ["H", "C2:CP:REACTOME"])
                        If None, return all collections
            species_code: Species code ("HS" or "MM")
            limit: Maximum number of gene sets to return (0 = no limit)

        Returns:
            Dict mapping standard_name -> MSigDBGeneSet
        """
        query = """
        SELECT
            gs.id,
            gs.standard_name,
            gs.collection_name,
            gs.tags,
            gsd.systematic_name,
            gsd.description_brief,
            gsd.description_full,
            gsd.external_details_URL,
            gsd.source_species_code,
            p.PMID,
            c.full_name as collection_full_name
        FROM gene_set gs
        LEFT JOIN gene_set_details gsd ON gs.id = gsd.gene_set_id
        LEFT JOIN publication p ON gsd.publication_id = p.id
        LEFT JOIN collection c ON gs.collection_name = c.collection_name
        WHERE 1=1
        """

        params = []

        if species_code:
            query += " AND gsd.source_species_code = ?"
            params.append(species_code)

        if collections:
            placeholders = ",".join("?" * len(collections))
            query += f" AND gs.collection_name IN ({placeholders})"
            params.extend(collections)

        query += " ORDER BY gs.standard_name"

        if limit > 0:
            query += " LIMIT ?"
            params.append(limit)

        cursor = self.conn.execute(query, params)
        gene_sets = {}

        for row in cursor:
            gene_set = MSigDBGeneSet(
                db_id=row["id"],
                standard_name=row["standard_name"],
                collection_name=row["collection_name"],
                systematic_name=row["systematic_name"] or "",
                description_brief=row["description_brief"] or "",
                description_full=row["description_full"] or "",
                external_url=row["external_details_URL"] or "",
                source_species=row["source_species_code"] or species_code,
                num_genes=0,
                genes=[],
                publication_pmid=row["PMID"] or "",
                tags=row["tags"] or "",
                collection_full_name=row["collection_full_name"] or "",
            )
            gene_sets[gene_set.standard_name] = gene_set

        print(f"Loaded {len(gene_sets):,} gene sets from database")
        return gene_sets

    def get_gene_set_members(self, gene_set_ids: list[int]) -> dict[int, list[str]]:
        """
        Get gene members for gene sets.

        Args:
            gene_set_ids: List of gene set database IDs

        Returns:
            Dict mapping gene_set_id -> list of gene symbols
        """
        if not gene_set_ids:
            return {}

        placeholders = ",".join("?" * len(gene_set_ids))
        query = f"""
        SELECT
            gsgs.gene_set_id,
            gs_sym.symbol
        FROM gene_set_gene_symbol gsgs
        JOIN gene_symbol gs_sym ON gsgs.gene_symbol_id = gs_sym.id
        WHERE gsgs.gene_set_id IN ({placeholders})
        ORDER BY gsgs.gene_set_id, gs_sym.symbol
        """

        cursor = self.conn.execute(query, gene_set_ids)

        gene_members = {}
        for row in cursor:
            gene_set_id = row["gene_set_id"]
            symbol = row["symbol"]

            if gene_set_id not in gene_members:
                gene_members[gene_set_id] = []

            gene_members[gene_set_id].append(symbol)

        print(f"Loaded gene members for {len(gene_members):,} gene sets")
        return gene_members

    def populate_gene_members(self, gene_sets: dict[str, MSigDBGeneSet]) -> None:
        """
        Populate gene members for gene sets in-place.

        Args:
            gene_sets: Dict of gene sets (will be modified)
        """
        gene_set_ids = [gs.db_id for gs in gene_sets.values()]

        gene_members = self.get_gene_set_members(gene_set_ids)

        for gene_set in gene_sets.values():
            genes = gene_members.get(gene_set.db_id, [])
            gene_set.genes = genes
            gene_set.num_genes = len(genes)

        print(f"Populated gene members for {len(gene_sets):,} gene sets")

    def filter_valid_genes(self, gene_sets: dict[str, MSigDBGeneSet], valid_genes: set[str]) -> dict[str, int]:
        """
        Filter gene sets to only include valid genes.

        Args:
            gene_sets: Dict of gene sets (will be modified in-place)
            valid_genes: Set of valid gene symbols

        Returns:
            Dict with statistics: {"total_genes": int, "valid_genes": int, "filtered_genes": int}
        """
        total_genes = 0
        valid_count = 0
        filtered_count = 0

        for gene_set in gene_sets.values():
            original_genes = gene_set.genes
            total_genes += len(original_genes)

            valid_set_genes = [g for g in original_genes if g in valid_genes]
            valid_count += len(valid_set_genes)
            filtered_count += len(original_genes) - len(valid_set_genes)

            gene_set.genes = valid_set_genes
            gene_set.num_genes = len(valid_set_genes)

        stats = {"total_genes": total_genes, "valid_genes": valid_count, "filtered_genes": filtered_count}

        print("Gene filtering stats:")
        print(f"  Total genes: {total_genes:,}")
        print(f"  Valid genes: {valid_count:,}")
        print(f"  Filtered out: {filtered_count:,}")

        return stats
