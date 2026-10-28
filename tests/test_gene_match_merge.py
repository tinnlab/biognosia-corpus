from src.entity_extraction.ngram_entity_matcher import merge_gene_matches


def hit(entity_type, name, score):
    return {
        "entity_type": entity_type,
        "entity_name": name,
        "entity_label": name,
        "entity_content": "",
        "similarity_score_label": score,
    }


def test_best_gene_hit_wins_and_is_typed_as_genes():
    merged = merge_gene_matches([hit("Genes", "GENE:A", 0.80), hit("Gene_Synonyms", "GENE:B", 0.91)])
    assert len(merged) == 1
    assert merged[0]["entity_name"] == "GENE:B"
    assert merged[0]["entity_type"] == "Genes"


def test_tie_prefers_genes_collection():
    merged = merge_gene_matches([hit("Gene_Synonyms", "GENE:B", 0.9), hit("Genes", "GENE:A", 0.9)])
    assert [m["entity_name"] for m in merged] == ["GENE:A"]


def test_other_collections_are_untouched():
    merged = merge_gene_matches([hit("Disease", "DISEASE:D1", 0.88), hit("Genes", "GENE:A", 0.86)])
    assert {m["entity_name"] for m in merged} == {"DISEASE:D1", "GENE:A"}


def test_no_gene_hit_returns_input():
    matches = [hit("Disease", "DISEASE:D1", 0.88)]
    assert merge_gene_matches(matches) == matches
