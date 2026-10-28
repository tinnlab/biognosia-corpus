#!/usr/bin/env python3
"""
UBERON Anatomy OWL Parser

Parses UBERON OWL/XML file to extract anatomical term definitions and hierarchies.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass
class UberonRecord:
    """Represents an anatomical term from UBERON."""

    uberon_id: str
    label: str
    definition: str
    synonyms: list[str]
    parent_ids: list[str]
    subsets: list[str]
    dbxrefs: list[str]
    is_obsolete: bool


class UberonParser:
    """Parse UBERON OWL file."""

    NAMESPACES = {
        "owl": "http://www.w3.org/2002/07/owl#",
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "obo": "http://purl.obolibrary.org/obo/",
        "oboInOwl": "http://www.geneontology.org/formats/oboInOwl#",
    }

    def __init__(self, owl_file: str):
        """Initialize parser with OWL file path."""
        self.owl_file = Path(owl_file)
        if not self.owl_file.exists():
            raise FileNotFoundError(f"OWL file not found: {owl_file}")

        for prefix, uri in self.NAMESPACES.items():
            ET.register_namespace(prefix, uri)

    def _convert_uberon_id(self, uri: str) -> str | None:
        """Convert UBERON URI to standard format.

        Args:
            uri: http://purl.obolibrary.org/obo/UBERON_0000916

        Returns:
            UBERON:0000916 or None if not a UBERON identifier
        """
        if "UBERON_" in uri:
            uberon_part = uri.split("UBERON_")[-1]
            return f"UBERON:{uberon_part}"
        return None

    def parse(
        self, filter_subsets: list[str] | None = None, exclude_obsolete: bool = True, limit: int = 0
    ) -> list[UberonRecord]:
        """Parse OWL file and return list of anatomy records.

        Args:
            filter_subsets: Optional list of subset names to filter by
            exclude_obsolete: Exclude deprecated terms
            limit: Maximum number of terms to return (0 = no limit)

        Returns:
            List of UberonRecord objects
        """
        print(f"Parsing UBERON from {self.owl_file}...")

        tree = ET.parse(self.owl_file)
        root = tree.getroot()

        records = []

        for owl_class in root.findall(".//owl:Class", self.NAMESPACES):
            about_attr = owl_class.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about")

            if not about_attr:
                continue

            uberon_id = self._convert_uberon_id(about_attr)
            if not uberon_id:
                continue

            label_elem = owl_class.find("rdfs:label", self.NAMESPACES)
            if label_elem is None or not label_elem.text:
                continue
            label = label_elem.text.strip()

            deprecated_elem = owl_class.find("owl:deprecated", self.NAMESPACES)
            is_obsolete = deprecated_elem is not None and deprecated_elem.text == "true"
            if exclude_obsolete and is_obsolete:
                continue

            definition = ""
            for elem in owl_class.findall("obo:IAO_0000115", self.NAMESPACES):
                if elem.text:
                    definition = elem.text.strip()
                    break

            synonyms = []
            for syn_tag in [
                "oboInOwl:hasExactSynonym",
                "oboInOwl:hasRelatedSynonym",
                "oboInOwl:hasNarrowSynonym",
                "oboInOwl:hasBroadSynonym",
            ]:
                for elem in owl_class.findall(syn_tag, self.NAMESPACES):
                    if elem.text:
                        synonyms.append(elem.text.strip())

            parent_ids = []
            for subclass_elem in owl_class.findall("rdfs:subClassOf", self.NAMESPACES):
                resource_attr = subclass_elem.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource")
                if resource_attr:
                    parent_id = self._convert_uberon_id(resource_attr)
                    if parent_id:
                        parent_ids.append(parent_id)

            subsets = []
            for subset_elem in owl_class.findall("oboInOwl:inSubset", self.NAMESPACES):
                resource_attr = subset_elem.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource")
                if resource_attr:
                    if "#" in resource_attr:
                        subset_name = resource_attr.split("#")[-1]
                        subsets.append(subset_name)

            if filter_subsets and not any(s in subsets for s in filter_subsets):
                continue

            dbxrefs = []
            for dbxref_elem in owl_class.findall("oboInOwl:hasDbXref", self.NAMESPACES):
                if dbxref_elem.text:
                    dbxrefs.append(dbxref_elem.text.strip())

            record = UberonRecord(
                uberon_id=uberon_id,
                label=label,
                definition=definition,
                synonyms=synonyms,
                parent_ids=parent_ids,
                subsets=subsets,
                dbxrefs=dbxrefs,
                is_obsolete=is_obsolete,
            )

            records.append(record)

            if limit > 0 and len(records) >= limit:
                break

        print(f"Parsed {len(records)} anatomy terms")
        return records

    def extract_hierarchies(self, records: list[UberonRecord]) -> list[tuple[str, str]]:
        """Extract parent-child relationships (is_a relations).

        Args:
            records: List of parsed anatomy records

        Returns:
            List of (child_uberon_id, parent_uberon_id) tuples
        """
        hierarchies = []

        valid_uberon_ids = {record.uberon_id for record in records}

        for record in records:
            for parent_id in record.parent_ids:
                if parent_id in valid_uberon_ids:
                    hierarchies.append((record.uberon_id, parent_id))

        print(f"Extracted {len(hierarchies)} hierarchical relationships")
        return hierarchies
