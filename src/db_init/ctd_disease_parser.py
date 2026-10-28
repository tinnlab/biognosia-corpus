#!/usr/bin/env python3
"""
CTD Disease Parser

Parses CTD (Comparative Toxicogenomics Database) disease files to extract:
- Disease entities
- Disease hierarchies
- Gene-Disease relationships
- Disease-Pathway relationships
"""

import csv
import gzip
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CTDDiseaseRecord:
    """Represents a disease from CTD."""

    disease_id: str
    disease_name: str
    alt_disease_ids: list[str]
    definition: str
    parent_ids: list[str]
    tree_numbers: list[str]
    synonyms: list[str]
    slim_mappings: list[str]


@dataclass
class GeneDiseaseRecord:
    """Represents a Gene-Disease association."""

    gene_symbol: str
    gene_id: str
    disease_name: str
    disease_id: str
    direct_evidence: str
    omim_ids: str
    pubmed_ids: str


@dataclass
class DiseasePathwayRecord:
    """Represents a Disease-Pathway association."""

    disease_name: str
    disease_id: str
    pathway_name: str
    pathway_id: str
    inference_gene_symbol: str


class CTDDiseaseParser:
    """Parse CTD disease files."""

    def __init__(self):
        pass

    def _parse_pipe_delimited(self, field: str) -> list[str]:
        """Parse pipe-delimited field into list."""
        if not field or field == "":
            return []
        return [item.strip() for item in field.split("|") if item.strip()]

    def parse_diseases(
        self, diseases_file: str, filter_category: list[str] | None = None, limit: int = 0
    ) -> list[CTDDiseaseRecord]:
        """Parse CTD_diseases.csv.gz and return disease records.

        Args:
            diseases_file: Path to CTD_diseases.csv.gz
            filter_category: Optional list of disease categories to filter by
            limit: Maximum number of diseases to return (0 = no limit)

        Returns:
            List of CTDDiseaseRecord objects
        """
        diseases_path = Path(diseases_file)
        if not diseases_path.exists():
            raise FileNotFoundError(f"Disease file not found: {diseases_file}")

        records = []
        filter_set = set(filter_category) if filter_category else None

        print(f"Parsing CTD diseases from {diseases_file}...")

        with gzip.open(diseases_file, "rt", encoding="utf-8") as f:
            reader = csv.reader(f)

            for row in reader:
                if not row or row[0].startswith("#"):
                    continue
                if len(row) < 9:
                    continue

                disease_name = row[0].strip()
                disease_id = row[1].strip()
                alt_disease_ids = self._parse_pipe_delimited(row[2])
                definition = row[3].strip()
                parent_ids = self._parse_pipe_delimited(row[4])
                tree_numbers = self._parse_pipe_delimited(row[5])
                synonyms = self._parse_pipe_delimited(row[7])
                slim_mappings = self._parse_pipe_delimited(row[8])

                if filter_set:
                    if not any(cat in filter_set for cat in slim_mappings):
                        continue

                record = CTDDiseaseRecord(
                    disease_id=disease_id,
                    disease_name=disease_name,
                    alt_disease_ids=alt_disease_ids,
                    definition=definition,
                    parent_ids=parent_ids,
                    tree_numbers=tree_numbers,
                    synonyms=synonyms,
                    slim_mappings=slim_mappings,
                )

                records.append(record)

                if limit > 0 and len(records) >= limit:
                    break

        print(f"Parsed {len(records):,} diseases")
        return records

    def parse_gene_disease_relationships(self, gene_disease_file: str) -> Generator[GeneDiseaseRecord, None, None]:
        """Parse CTD_curated_genes_diseases.csv.gz.

        Args:
            gene_disease_file: Path to CTD_curated_genes_diseases.csv.gz

        Yields:
            GeneDiseaseRecord objects
        """
        gene_disease_path = Path(gene_disease_file)
        if not gene_disease_path.exists():
            raise FileNotFoundError(f"Gene-Disease file not found: {gene_disease_file}")

        with gzip.open(gene_disease_file, "rt", encoding="utf-8") as f:
            reader = csv.reader(f)

            for row in reader:
                if not row or row[0].startswith("#"):
                    continue
                if len(row) < 7:
                    continue

                gene_symbol = row[0].strip()
                gene_id = row[1].strip()
                disease_name = row[2].strip()
                disease_id = row[3].strip()
                direct_evidence = row[4].strip()
                omim_ids = row[5].strip()
                pubmed_ids = row[6].strip()

                record = GeneDiseaseRecord(
                    gene_symbol=gene_symbol,
                    gene_id=gene_id,
                    disease_name=disease_name,
                    disease_id=disease_id,
                    direct_evidence=direct_evidence,
                    omim_ids=omim_ids,
                    pubmed_ids=pubmed_ids,
                )

                yield record

    def parse_disease_pathway_relationships(
        self, disease_pathway_file: str
    ) -> Generator[DiseasePathwayRecord, None, None]:
        """Parse CTD_diseases_pathways.csv.gz.

        Args:
            disease_pathway_file: Path to CTD_diseases_pathways.csv.gz

        Yields:
            DiseasePathwayRecord objects
        """
        disease_pathway_path = Path(disease_pathway_file)
        if not disease_pathway_path.exists():
            raise FileNotFoundError(f"Disease-Pathway file not found: {disease_pathway_file}")

        with gzip.open(disease_pathway_file, "rt", encoding="utf-8") as f:
            reader = csv.reader(f)

            for row in reader:
                if not row or row[0].startswith("#"):
                    continue
                if len(row) < 5:
                    continue

                disease_name = row[0].strip()
                disease_id = row[1].strip()
                pathway_name = row[2].strip()
                pathway_id = row[3].strip()
                inference_gene_symbol = row[4].strip()

                record = DiseasePathwayRecord(
                    disease_name=disease_name,
                    disease_id=disease_id,
                    pathway_name=pathway_name,
                    pathway_id=pathway_id,
                    inference_gene_symbol=inference_gene_symbol,
                )

                yield record

    def extract_disease_hierarchies(self, records: list[CTDDiseaseRecord]) -> list[tuple[str, str]]:
        """Extract Disease->Disease parent-child relationships.

        Args:
            records: List of parsed disease records

        Returns:
            List of (child_disease_id, parent_disease_id) tuples
        """
        hierarchies = []

        valid_disease_ids = {record.disease_id for record in records}

        for record in records:
            for parent_id in record.parent_ids:
                if parent_id in valid_disease_ids:
                    hierarchies.append((record.disease_id, parent_id))

        print(f"Extracted {len(hierarchies):,} disease hierarchical relationships")
        return hierarchies
