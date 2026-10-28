#!/usr/bin/env python3
"""
NCBI Taxonomy Parser

Parses NCBI Taxonomy dump files (.dmp format) to extract taxonomic information.
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TaxonomyRecord:
    """Represents a taxonomic node from NCBI Taxonomy."""

    tax_id: str
    scientific_name: str
    common_names: list[str]
    synonyms: list[str]
    parent_tax_id: str
    rank: str
    division_id: str
    lineage: list[str]
    lineage_names: list[str]


class TaxonomyParser:
    """Parse NCBI Taxonomy dump files."""

    def __init__(self, taxonomy_dir: str):
        """Initialize parser with taxonomy directory path."""
        self.taxonomy_dir = Path(taxonomy_dir)
        if not self.taxonomy_dir.exists():
            raise FileNotFoundError(f"Taxonomy directory not found: {taxonomy_dir}")

        self.nodes_file = self.taxonomy_dir / "nodes.dmp"
        self.names_file = self.taxonomy_dir / "names.dmp"
        self.merged_file = self.taxonomy_dir / "merged.dmp"

        if not self.nodes_file.exists():
            raise FileNotFoundError(f"nodes.dmp not found: {self.nodes_file}")
        if not self.names_file.exists():
            raise FileNotFoundError(f"names.dmp not found: {self.names_file}")

    def _parse_nodes(self) -> dict[str, tuple[str, str, str]]:
        """Parse nodes.dmp file.

        Returns:
            Dict mapping tax_id -> (parent_tax_id, rank, division_id)
        """
        print("Parsing nodes.dmp...")
        nodes = {}
        skipped = 0

        with open(self.nodes_file) as f:
            for line_num, line in enumerate(f, 1):
                try:
                    fields = line.strip().split("\t|\t")
                    if len(fields) < 3:
                        skipped += 1
                        continue

                    tax_id = fields[0].strip()
                    parent_tax_id = fields[1].strip()
                    rank = fields[2].strip()
                    division_id = fields[4].strip() if len(fields) > 4 else "0"

                    nodes[tax_id] = (parent_tax_id, rank, division_id)

                except Exception as e:
                    print(f"Warning: Skipping malformed line {line_num} in nodes.dmp: {e}")
                    skipped += 1
                    continue

        print(f"Parsed {len(nodes):,} nodes ({skipped} skipped)")
        return nodes

    def _parse_names(self) -> dict[str, dict[str, any]]:
        """Parse names.dmp file.

        Returns:
            Dict mapping tax_id -> {scientific_name, common_names, synonyms}
        """
        print("Parsing names.dmp...")
        names_data = defaultdict(lambda: {"scientific_name": None, "common_names": [], "synonyms": []})
        skipped = 0

        with open(self.names_file) as f:
            for line_num, line in enumerate(f, 1):
                try:
                    fields = line.strip().split("\t|\t")
                    if len(fields) < 4:
                        skipped += 1
                        continue

                    tax_id = fields[0].strip()
                    name_txt = fields[1].strip()
                    name_class = fields[3].strip().rstrip("\t|")

                    if name_class == "scientific name":
                        names_data[tax_id]["scientific_name"] = name_txt
                    elif name_class in ("common name", "genbank common name"):
                        if name_txt not in names_data[tax_id]["common_names"]:
                            names_data[tax_id]["common_names"].append(name_txt)
                    elif name_class in ("synonym", "equivalent name"):
                        if name_txt not in names_data[tax_id]["synonyms"]:
                            names_data[tax_id]["synonyms"].append(name_txt)

                except Exception as e:
                    print(f"Warning: Skipping malformed line {line_num} in names.dmp: {e}")
                    skipped += 1
                    continue

        print(f"Parsed names for {len(names_data):,} taxa ({skipped} skipped)")
        return dict(names_data)

    def _parse_merged(self) -> dict[str, str]:
        """Parse merged.dmp file.

        Returns:
            Dict mapping old_tax_id -> new_tax_id
        """
        if not self.merged_file.exists():
            return {}

        print("Parsing merged.dmp...")
        merged = {}

        with open(self.merged_file) as f:
            for line in f:
                try:
                    fields = line.strip().split("\t|\t")
                    if len(fields) >= 2:
                        old_tax_id = fields[0].strip()
                        new_tax_id = fields[1].strip().rstrip("\t|")
                        merged[old_tax_id] = new_tax_id
                except Exception:
                    continue

        print(f"Parsed {len(merged):,} merged tax IDs")
        return merged

    def _build_lineage(
        self, tax_id: str, nodes: dict[str, tuple], names: dict[str, dict]
    ) -> tuple[list[str], list[str]]:
        """Build lineage from root to current tax_id.

        Args:
            tax_id: Starting taxonomy ID
            nodes: Dict from _parse_nodes
            names: Dict from _parse_names

        Returns:
            Tuple of (lineage_ids, lineage_names)
        """
        lineage_ids = []
        lineage_names = []
        current_id = tax_id
        visited = set()

        while current_id in nodes:
            if current_id in visited:
                break
            visited.add(current_id)

            lineage_ids.insert(0, current_id)

            sci_name = names.get(current_id, {}).get("scientific_name", f"TAX:{current_id}")
            lineage_names.insert(0, sci_name)

            parent_id, rank, division = nodes[current_id]

            if current_id == parent_id:
                break

            current_id = parent_id

        return lineage_ids, lineage_names

    def parse(
        self, filter_ranks: list[str] | None = None, filter_divisions: list[str] | None = None, limit: int = 0
    ) -> list[TaxonomyRecord]:
        """Parse taxonomy data and return list of records.

        Args:
            filter_ranks: Optional list of ranks to filter by (e.g., ["species", "genus"])
            filter_divisions: Optional list of division IDs to filter by
            limit: Maximum number of records to return (0 = no limit)

        Returns:
            List of TaxonomyRecord objects
        """
        nodes = self._parse_nodes()
        names = self._parse_names()
        self._parse_merged()

        print("\nBuilding taxonomy records...")
        if filter_ranks:
            print(f"Filtering by ranks: {filter_ranks}")
        if filter_divisions:
            print(f"Filtering by divisions: {filter_divisions}")

        records = []
        skipped_no_name = 0
        skipped_filter = 0

        for tax_id, (parent_tax_id, rank, division_id) in nodes.items():
            if filter_ranks and rank not in filter_ranks:
                skipped_filter += 1
                continue

            if filter_divisions and division_id not in filter_divisions:
                skipped_filter += 1
                continue

            name_info = names.get(tax_id, {})
            scientific_name = name_info.get("scientific_name")
            if not scientific_name:
                skipped_no_name += 1
                continue

            common_names = name_info.get("common_names", [])
            synonyms = name_info.get("synonyms", [])

            lineage_ids, lineage_names = self._build_lineage(tax_id, nodes, names)

            record = TaxonomyRecord(
                tax_id=tax_id,
                scientific_name=scientific_name,
                common_names=common_names,
                synonyms=synonyms,
                parent_tax_id=parent_tax_id,
                rank=rank,
                division_id=division_id,
                lineage=lineage_ids,
                lineage_names=lineage_names,
            )
            records.append(record)

            if limit and len(records) >= limit:
                break

        print(f"Built {len(records):,} taxonomy records")
        if skipped_no_name > 0:
            print(f"  Skipped {skipped_no_name:,} nodes without scientific names")
        if skipped_filter > 0:
            print(f"  Skipped {skipped_filter:,} nodes due to filters")

        return records
