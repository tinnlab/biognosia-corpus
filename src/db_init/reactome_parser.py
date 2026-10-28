"""
Reactome Pathway Parser

Parses Reactome pathway data from:
- ReactomePathways.txt (pathway metadata)
- ReactomePathways.gmt (gene-pathway associations)
"""

import os
import re
from dataclasses import dataclass


@dataclass
class ReactomePathwayRecord:
    """Represents a Reactome pathway with metadata and associated genes"""

    pathway_id: str
    pathway_name: str
    organism: str
    species_code: str
    genes: list[str]


class ReactomeParser:
    """Parser for Reactome pathway files"""

    def __init__(self, reactome_dir: str):
        """
        Initialize parser with Reactome data directory

        Args:
            reactome_dir: Directory containing ReactomePathways.txt and ReactomePathways.gmt
        """
        self.reactome_dir = reactome_dir
        self.pathways_txt = os.path.join(reactome_dir, "ReactomePathways.txt")
        self.pathways_gmt = os.path.join(reactome_dir, "ReactomePathways.gmt")

        if not os.path.exists(self.pathways_txt):
            raise FileNotFoundError(f"ReactomePathways.txt not found: {self.pathways_txt}")
        if not os.path.exists(self.pathways_gmt):
            raise FileNotFoundError(f"ReactomePathways.gmt not found: {self.pathways_gmt}")

    def _extract_species_code(self, pathway_id: str) -> str:
        """
        Extract species code from pathway ID

        Args:
            pathway_id: Pathway ID like "R-HSA-164843"

        Returns:
            Species code like "HSA"
        """
        match = re.match(r"R-([A-Z]+)-\d+", pathway_id)
        if match:
            return match.group(1)
        return "UNKNOWN"

    def parse_pathways_txt(self) -> dict[str, ReactomePathwayRecord]:
        """
        Parse ReactomePathways.txt for pathway metadata

        Returns:
            Dict mapping pathway_id -> ReactomePathwayRecord (with empty genes list)
        """
        pathways = {}

        with open(self.pathways_txt, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) != 3:
                    continue

                pathway_id, pathway_name, organism = parts
                species_code = self._extract_species_code(pathway_id)

                pathways[pathway_id] = ReactomePathwayRecord(
                    pathway_id=pathway_id,
                    pathway_name=pathway_name,
                    organism=organism,
                    species_code=species_code,
                    genes=[],
                )

        return pathways

    def _filter_gene_symbol(self, symbol: str) -> bool:
        """
        Check if gene symbol should be included

        Args:
            symbol: Gene symbol to check

        Returns:
            True if symbol should be included, False otherwise
        """
        symbol = symbol.upper()

        if not symbol:
            return False

        if symbol.isdigit():
            return False

        generic_names = {"GAG", "POL", "REV", "VIF", "VPR", "VPU", "NEF", "TAT", "ENV"}
        if symbol in generic_names:
            return False

        if symbol.startswith("GAG-"):
            return False

        return True

    def parse_pathways_gmt(self) -> dict[str, list[str]]:
        """
        Parse ReactomePathways.gmt for gene-pathway associations

        Returns:
            Dict mapping pathway_id -> [gene_symbols]
        """
        pathway_genes = {}

        with open(self.pathways_gmt, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) < 3:
                    continue

                parts[0]
                pathway_id = parts[1]
                gene_symbols = parts[2:]

                filtered_genes = []
                for symbol in gene_symbols:
                    symbol_upper = symbol.upper()
                    if self._filter_gene_symbol(symbol_upper):
                        filtered_genes.append(symbol_upper)

                pathway_genes[pathway_id] = sorted(set(filtered_genes))

        return pathway_genes

    def merge_pathway_data(self) -> dict[str, ReactomePathwayRecord]:
        """
        Merge pathway metadata from txt and gene associations from GMT

        Returns:
            Dict mapping pathway_id -> ReactomePathwayRecord (with genes filled)
        """
        pathways = self.parse_pathways_txt()

        pathway_genes = self.parse_pathways_gmt()

        for pathway_id, genes in pathway_genes.items():
            if pathway_id in pathways:
                pathways[pathway_id].genes = genes

        return pathways

    def filter_by_organism(
        self, records: dict[str, ReactomePathwayRecord], organisms: list[str]
    ) -> dict[str, ReactomePathwayRecord]:
        """
        Filter pathways by organism scientific names

        Args:
            records: All pathway records
            organisms: List of organism scientific names (e.g., ["Homo sapiens", "Mus musculus"])

        Returns:
            Filtered pathway records
        """
        if not organisms:
            return records

        filtered = {}
        for pathway_id, record in records.items():
            if record.organism in organisms:
                filtered[pathway_id] = record

        return filtered

    def build_gene_pathway_relationships(
        self, records: dict[str, ReactomePathwayRecord], gene_symbols: set[str]
    ) -> list[tuple[str, str]]:
        """
        Extract gene-pathway relationships, filtered by valid gene symbols

        Args:
            records: Pathway records with genes
            gene_symbols: Set of valid gene symbols from gene database

        Returns:
            List of (gene_symbol, pathway_id) tuples
        """
        relationships = []
        total_associations = 0
        skipped_not_in_db = 0

        for pathway_id, record in records.items():
            for gene in record.genes:
                total_associations += 1

                if gene in gene_symbols:
                    relationships.append((gene, pathway_id))
                else:
                    skipped_not_in_db += 1

        print(f"\nBuilt {len(relationships):,} Gene-Reactome_Pathway relationships")
        print(f"  Total associations in Reactome: {total_associations:,}")
        print(f"  Skipped (gene not in database): {skipped_not_in_db:,}")

        return relationships
