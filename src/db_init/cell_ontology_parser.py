#!/usr/bin/env python3
"""
Cell Ontology OWL Parser

Parses Cell Ontology OWL/XML file to extract cell type definitions and hierarchies.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CellOntologyRecord:
    """Represents a cell type from Cell Ontology."""

    cl_id: str
    label: str
    definition: str
    synonyms: list[str]
    parent_ids: list[str]
    subsets: list[str]
    dbxrefs: list[str]
    is_obsolete: bool


class CellOntologyParser:
    """Parse Cell Ontology OWL file."""

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

    def _convert_cl_id(self, uri: str) -> str | None:
        """Convert CL URI to standard format.

        Args:
            uri: http://purl.obolibrary.org/obo/CL_0000005

        Returns:
            CL:0000005 or None if not a CL identifier
        """
        if "CL_" in uri:
            cl_part = uri.split("CL_")[-1]
            return f"CL:{cl_part}"
        return None

    def parse(self) -> list[CellOntologyRecord]:
        """Parse OWL file and return list of cell type records."""
        print(f"Parsing Cell Ontology from {self.owl_file}...")

        tree = ET.parse(self.owl_file)
        root = tree.getroot()

        records = []

        for owl_class in root.findall(".//owl:Class", self.NAMESPACES):
            about_attr = owl_class.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about")

            if not about_attr:
                continue

            cl_id = self._convert_cl_id(about_attr)
            if not cl_id:
                continue

            label_elem = owl_class.find("rdfs:label", self.NAMESPACES)
            if label_elem is None or not label_elem.text:
                continue
            label = label_elem.text.strip()

            deprecated_elem = owl_class.find("owl:deprecated", self.NAMESPACES)
            is_obsolete = deprecated_elem is not None and deprecated_elem.text == "true"

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
                    parent_id = self._convert_cl_id(resource_attr)
                    if parent_id:
                        parent_ids.append(parent_id)

            subsets = []
            for subset_elem in owl_class.findall("oboInOwl:inSubset", self.NAMESPACES):
                resource_attr = subset_elem.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource")
                if resource_attr:
                    if "#" in resource_attr:
                        subset_name = resource_attr.split("#")[-1]
                        subsets.append(subset_name)

            dbxrefs = []
            for dbxref_elem in owl_class.findall("oboInOwl:hasDbXref", self.NAMESPACES):
                if dbxref_elem.text:
                    dbxrefs.append(dbxref_elem.text.strip())

            record = CellOntologyRecord(
                cl_id=cl_id,
                label=label,
                definition=definition,
                synonyms=synonyms,
                parent_ids=parent_ids,
                subsets=subsets,
                dbxrefs=dbxrefs,
                is_obsolete=is_obsolete,
            )

            records.append(record)

        print(f"Parsed {len(records)} cell types")
        return records

    def extract_hierarchies(self, records: list[CellOntologyRecord]) -> list[tuple[str, str]]:
        """Extract parent-child relationships (is_a relations).

        Args:
            records: List of parsed cell type records

        Returns:
            List of (child_cl_id, parent_cl_id) tuples in CL:XXXXXXX format
        """
        hierarchies = []

        valid_cl_ids = {record.cl_id for record in records}

        for record in records:
            for parent_id in record.parent_ids:
                if parent_id in valid_cl_ids:
                    hierarchies.append((record.cl_id, parent_id))

        print(f"Extracted {len(hierarchies)} hierarchical relationships")
        return hierarchies
