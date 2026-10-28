#!/usr/bin/env python3
"""
Gene ID Mapper - Reusable module for Entrez GeneID to Gene Symbol mapping.

This module provides a cached mapping between Entrez GeneIDs and Gene Symbols,
with filtering capabilities for taxonomy IDs and gene validation rules.

The mapping is built from gene_info.gz and cached to JSON files for fast reloading.
"""

import gzip
import json
from pathlib import Path


class GeneIDMapper:
    """Build and maintain Entrez GeneID -> Gene Symbol mapping with caching."""

    def __init__(self, gene_info_file: str, filter_taxid: list[str] | None = None, cache_dir: str = "data/db-init"):
        """
        Initialize GeneIDMapper.

        Args:
            gene_info_file: Path to gene_info.gz file
            filter_taxid: Optional list of taxonomy IDs to filter (e.g., ["3702"] for Arabidopsis)
            cache_dir: Directory to store/load cache files
        """
        self.gene_info_file = gene_info_file
        self.filter_taxid = set(filter_taxid) if filter_taxid else None
        self.mapping: dict[str, str] = {}
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self) -> Path:
        """Get cache file path based on filter settings."""
        if self.filter_taxid:
            taxid_str = "_".join(sorted(self.filter_taxid))
            cache_name = f"gene_id_mapping_{taxid_str}.json"
        else:
            cache_name = "gene_id_mapping_all.json"
        return self.cache_dir / cache_name

    def _is_valid_gene(self, symbol: str, description: str = "", chromosome: str = "") -> bool:
        """
        Check if gene is valid according to filtering rules.

        Rules:
        - Skip NEWENTRY placeholders or empty symbols
        - Skip LOC + number patterns (e.g., LOC110610980)
        - Skip symbols with underscores
        - Skip genes with "uncharacterized protein" in description
        - Skip genes with unknown/missing chromosome

        Args:
            symbol: Gene symbol
            description: Gene description (optional)
            chromosome: Chromosome location (optional)

        Returns:
            True if gene passes all validation rules
        """
        if symbol == "NEWENTRY" or symbol == "-" or not symbol:
            return False

        if symbol.startswith("LOC") and symbol[3:].isdigit():
            return False

        if "_" in symbol:
            return False

        if description and "uncharacterized protein" in description.lower():
            return False

        if not chromosome or chromosome == "-" or chromosome.lower() == "unknown":
            return False

        return True

    def build_mapping(self) -> int:
        """
        Build the GeneID -> Symbol mapping from gene_info.gz or load from cache.

        If a cache file exists for the current filter settings, loads from cache.
        Otherwise, builds the mapping from gene_info.gz and saves to cache.

        Returns:
            Number of genes in the mapping
        """
        cache_path = self._get_cache_path()

        if cache_path.exists():
            print(f"Loading GeneID -> Symbol mapping from cache: {cache_path.name}")
            with open(cache_path) as f:
                self.mapping = json.load(f)
            print(f"Loaded mapping for {len(self.mapping):,} genes from cache")
            return len(self.mapping)

        print("Building GeneID -> Symbol mapping from gene_info.gz...")
        count = 0

        with gzip.open(self.gene_info_file, "rt") as f:
            next(f)

            for line in f:
                fields = line.strip().split("\t")
                if len(fields) < 9:
                    continue

                tax_id = fields[0]
                gene_id = fields[1]
                symbol = fields[2]
                chromosome = fields[6]
                description = fields[8]

                if not self._is_valid_gene(symbol, description, chromosome):
                    continue

                if self.filter_taxid and tax_id not in self.filter_taxid:
                    continue

                self.mapping[gene_id] = symbol
                count += 1

        print(f"Built mapping for {count:,} genes")

        print(f"Saving mapping to cache: {cache_path.name}")
        with open(cache_path, "w") as f:
            json.dump(self.mapping, f)
        print("Cache saved")

        return count

    def get_symbol(self, gene_id: str) -> str | None:
        """
        Get gene symbol for a given Entrez GeneID.

        Args:
            gene_id: Entrez GeneID (as string)

        Returns:
            Gene symbol if found, None otherwise
        """
        return self.mapping.get(gene_id)

    def get_all_gene_ids(self) -> set[str]:
        """
        Get all Entrez GeneIDs in the mapping.

        Returns:
            Set of all gene IDs
        """
        return set(self.mapping.keys())

    def get_all_symbols(self) -> set[str]:
        """
        Get all gene symbols in the mapping.

        Returns:
            Set of all gene symbols
        """
        return set(self.mapping.values())

    def __len__(self) -> int:
        """Return number of genes in mapping."""
        return len(self.mapping)

    def __contains__(self, gene_id: str) -> bool:
        """Check if gene_id exists in mapping."""
        return gene_id in self.mapping
