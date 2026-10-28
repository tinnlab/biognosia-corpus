#!/usr/bin/env python3
"""
CTD Chemical Parser

Parses CTD (Comparative Toxicogenomics Database) chemical files to extract:
- Chemical entities
- Chemical hierarchies
- Chemical-Gene interactions
- Chemical-Disease relationships
- Chemical-Pathway enrichments (KEGG + Reactome)
- Chemical-GO enrichments
"""

import csv
import gzip
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CTDChemicalRecord:
    """Represents a chemical from CTD."""

    chemical_id: str
    chemical_name: str
    cas_rn: str
    definition: str
    parent_ids: list[str]
    tree_numbers: list[str]
    synonyms: list[str]


@dataclass
class ChemicalGeneInteraction:
    """Represents a Chemical-Gene interaction."""

    chemical_id: str
    chemical_name: str
    cas_rn: str
    gene_symbol: str
    gene_id: str
    gene_forms: str
    organism: str
    organism_id: str
    interaction: str
    interaction_actions: str
    pubmed_ids: str


@dataclass
class ChemicalDiseaseRecord:
    """Represents a Chemical-Disease association."""

    chemical_id: str
    chemical_name: str
    cas_rn: str
    disease_id: str
    disease_name: str
    direct_evidence: str
    pubmed_ids: str


@dataclass
class ChemicalPathwayRecord:
    """Represents a Chemical-Pathway enrichment."""

    chemical_id: str
    chemical_name: str
    cas_rn: str
    pathway_name: str
    pathway_id: str
    p_value: str
    corrected_p_value: str
    target_match_qty: str
    target_total_qty: str


@dataclass
class ChemicalGORecord:
    """Represents a Chemical-GO term enrichment."""

    chemical_id: str
    chemical_name: str
    cas_rn: str
    ontology: str
    go_term_name: str
    go_term_id: str
    highest_go_level: str
    p_value: str
    corrected_p_value: str
    target_match_qty: str
    target_total_qty: str


class CTDChemicalParser:
    """Parse CTD chemical files."""

    def __init__(self):
        pass

    def _parse_pipe_delimited(self, field: str) -> list[str]:
        """Parse pipe-delimited field into list."""
        if not field or field == "":
            return []
        return [item.strip() for item in field.split("|") if item.strip()]

    def parse_chemicals(self, chemicals_file: str, limit: int = 0) -> list[CTDChemicalRecord]:
        """Parse CTD_chemicals.csv.gz and return chemical records.

        Args:
            chemicals_file: Path to CTD_chemicals.csv.gz
            limit: Maximum number of chemicals to return (0 = no limit)

        Returns:
            List of CTDChemicalRecord objects
        """
        chemicals_path = Path(chemicals_file)
        if not chemicals_path.exists():
            raise FileNotFoundError(f"Chemical file not found: {chemicals_file}")

        records = []

        print(f"Parsing CTD chemicals from {chemicals_file}...")

        with gzip.open(chemicals_file, "rt", encoding="utf-8") as f:
            reader = csv.reader(f)

            for row in reader:
                if not row or row[0].startswith("#"):
                    continue
                if len(row) < 8:
                    continue

                chemical_name = row[0].strip()
                chemical_id = row[1].strip()
                cas_rn = row[2].strip()
                definition = row[3].strip()
                parent_ids = self._parse_pipe_delimited(row[4])
                tree_numbers = self._parse_pipe_delimited(row[5])
                synonyms = self._parse_pipe_delimited(row[7])

                record = CTDChemicalRecord(
                    chemical_id=chemical_id,
                    chemical_name=chemical_name,
                    cas_rn=cas_rn,
                    definition=definition,
                    parent_ids=parent_ids,
                    tree_numbers=tree_numbers,
                    synonyms=synonyms,
                )

                records.append(record)

                if limit > 0 and len(records) >= limit:
                    break

        print(f"Parsed {len(records):,} chemicals")
        return records

    def parse_chemical_gene_interactions(
        self, chem_gene_file: str, filter_organism: list[str] | None = None
    ) -> Generator[ChemicalGeneInteraction, None, None]:
        """Parse CTD_chem_gene_ixns.csv.gz.

        Args:
            chem_gene_file: Path to CTD_chem_gene_ixns.csv.gz
            filter_organism: Optional list of organism IDs to filter (e.g., ["9606"] for human)

        Yields:
            ChemicalGeneInteraction objects
        """
        chem_gene_path = Path(chem_gene_file)
        if not chem_gene_path.exists():
            raise FileNotFoundError(f"Chemical-Gene file not found: {chem_gene_file}")

        filter_set = set(filter_organism) if filter_organism else None

        with gzip.open(chem_gene_file, "rt", encoding="utf-8") as f:
            reader = csv.reader(f)

            for row in reader:
                if not row or row[0].startswith("#"):
                    continue
                if len(row) < 11:
                    continue

                chemical_name = row[0].strip()
                chemical_id = row[1].strip()
                cas_rn = row[2].strip()
                gene_symbol = row[3].strip()
                gene_id = row[4].strip()
                gene_forms = row[5].strip()
                organism = row[6].strip()
                organism_id = row[7].strip()
                interaction = row[8].strip()
                interaction_actions = row[9].strip()
                pubmed_ids = row[10].strip()

                if chemical_id and not chemical_id.startswith("MESH:"):
                    chemical_id = f"MESH:{chemical_id}"

                if filter_set and organism_id not in filter_set:
                    continue

                record = ChemicalGeneInteraction(
                    chemical_id=chemical_id,
                    chemical_name=chemical_name,
                    cas_rn=cas_rn,
                    gene_symbol=gene_symbol,
                    gene_id=gene_id,
                    gene_forms=gene_forms,
                    organism=organism,
                    organism_id=organism_id,
                    interaction=interaction,
                    interaction_actions=interaction_actions,
                    pubmed_ids=pubmed_ids,
                )

                yield record

    def parse_chemical_disease_relationships(
        self, chem_disease_file: str
    ) -> Generator[ChemicalDiseaseRecord, None, None]:
        """Parse CTD_curated_chemicals_diseases.csv.gz.

        Args:
            chem_disease_file: Path to CTD_curated_chemicals_diseases.csv.gz

        Yields:
            ChemicalDiseaseRecord objects
        """
        chem_disease_path = Path(chem_disease_file)
        if not chem_disease_path.exists():
            raise FileNotFoundError(f"Chemical-Disease file not found: {chem_disease_file}")

        with gzip.open(chem_disease_file, "rt", encoding="utf-8") as f:
            reader = csv.reader(f)

            for row in reader:
                if not row or row[0].startswith("#"):
                    continue
                if len(row) < 7:
                    continue

                chemical_name = row[0].strip()
                chemical_id = row[1].strip()
                cas_rn = row[2].strip()
                disease_name = row[3].strip()
                disease_id = row[4].strip()
                direct_evidence = row[5].strip()
                pubmed_ids = row[6].strip()

                if chemical_id and not chemical_id.startswith("MESH:"):
                    chemical_id = f"MESH:{chemical_id}"
                if disease_id and not disease_id.startswith("MESH:"):
                    disease_id = f"MESH:{disease_id}"

                record = ChemicalDiseaseRecord(
                    chemical_id=chemical_id,
                    chemical_name=chemical_name,
                    cas_rn=cas_rn,
                    disease_id=disease_id,
                    disease_name=disease_name,
                    direct_evidence=direct_evidence,
                    pubmed_ids=pubmed_ids,
                )

                yield record

    def parse_chemical_pathway_enrichments(
        self, chem_pathway_file: str
    ) -> Generator[ChemicalPathwayRecord, None, None]:
        """Parse CTD_chem_pathways_enriched.csv.gz.

        Contains both KEGG and Reactome pathways - filter by PathwayID prefix.

        Args:
            chem_pathway_file: Path to CTD_chem_pathways_enriched.csv.gz

        Yields:
            ChemicalPathwayRecord objects
        """
        chem_pathway_path = Path(chem_pathway_file)
        if not chem_pathway_path.exists():
            raise FileNotFoundError(f"Chemical-Pathway file not found: {chem_pathway_file}")

        with gzip.open(chem_pathway_file, "rt", encoding="utf-8") as f:
            reader = csv.reader(f)

            for row in reader:
                if not row or row[0].startswith("#"):
                    continue
                if len(row) < 9:
                    continue

                chemical_name = row[0].strip()
                chemical_id = row[1].strip()
                cas_rn = row[2].strip()
                pathway_name = row[3].strip()
                pathway_id = row[4].strip()
                p_value = row[5].strip()
                corrected_p_value = row[6].strip()
                target_match_qty = row[7].strip()
                target_total_qty = row[8].strip()

                if chemical_id and not chemical_id.startswith("MESH:"):
                    chemical_id = f"MESH:{chemical_id}"

                record = ChemicalPathwayRecord(
                    chemical_id=chemical_id,
                    chemical_name=chemical_name,
                    cas_rn=cas_rn,
                    pathway_name=pathway_name,
                    pathway_id=pathway_id,
                    p_value=p_value,
                    corrected_p_value=corrected_p_value,
                    target_match_qty=target_match_qty,
                    target_total_qty=target_total_qty,
                )

                yield record

    def parse_chemical_go_enrichments(
        self, chem_go_file: str, filter_ontology: list[str] | None = None
    ) -> Generator[ChemicalGORecord, None, None]:
        """Parse CTD_chem_go_enriched.csv.gz.

        Args:
            chem_go_file: Path to CTD_chem_go_enriched.csv.gz
            filter_ontology: Optional list of ontology types to filter
                           (e.g., ["Biological Process", "Molecular Function"])

        Yields:
            ChemicalGORecord objects
        """
        chem_go_path = Path(chem_go_file)
        if not chem_go_path.exists():
            raise FileNotFoundError(f"Chemical-GO file not found: {chem_go_file}")

        filter_set = set(filter_ontology) if filter_ontology else None

        with gzip.open(chem_go_file, "rt", encoding="utf-8") as f:
            reader = csv.reader(f)

            for row in reader:
                if not row or row[0].startswith("#"):
                    continue
                if len(row) < 11:
                    continue

                chemical_name = row[0].strip()
                chemical_id = row[1].strip()
                cas_rn = row[2].strip()
                ontology = row[3].strip()
                go_term_name = row[4].strip()
                go_term_id = row[5].strip()
                highest_go_level = row[6].strip()
                p_value = row[7].strip()
                corrected_p_value = row[8].strip()
                target_match_qty = row[9].strip()
                target_total_qty = row[10].strip()

                if chemical_id and not chemical_id.startswith("MESH:"):
                    chemical_id = f"MESH:{chemical_id}"

                if filter_set and ontology not in filter_set:
                    continue

                record = ChemicalGORecord(
                    chemical_id=chemical_id,
                    chemical_name=chemical_name,
                    cas_rn=cas_rn,
                    ontology=ontology,
                    go_term_name=go_term_name,
                    go_term_id=go_term_id,
                    highest_go_level=highest_go_level,
                    p_value=p_value,
                    corrected_p_value=corrected_p_value,
                    target_match_qty=target_match_qty,
                    target_total_qty=target_total_qty,
                )

                yield record

    def extract_chemical_hierarchies(self, records: list[CTDChemicalRecord]) -> list[tuple[str, str]]:
        """Extract Chemical->Chemical parent-child relationships.

        Args:
            records: List of parsed chemical records

        Returns:
            List of (child_chemical_id, parent_chemical_id) tuples
        """
        hierarchies = []

        valid_chemical_ids = {record.chemical_id for record in records}

        for record in records:
            for parent_id in record.parent_ids:
                if parent_id in valid_chemical_ids:
                    hierarchies.append((record.chemical_id, parent_id))

        print(f"Extracted {len(hierarchies):,} chemical hierarchical relationships")
        return hierarchies
