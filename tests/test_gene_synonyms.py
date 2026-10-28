import hashlib

from build_gene_synonyms import build_records, row_id

GENE_ENTITIES = [
    {"entity_name": "GENE:TP53", "synonyms": "p53|LFS1|TRP53"},
    {"entity_name": "GENE:BRCA1", "synonyms": ""},
    {"entity_name": "GENE:MOUSE1", "synonyms": "m1"},
]

GENE_INFO = [
    "9606\t7157\tTP53\t-\tBCC7|LFS1|P53|TRP53\t-\t17\t17p13.1\ttumor protein p53\tprotein-coding\tTP53\ttumor protein p53\tO\t-\t20240101\tprotein-coding\n",
    "9606\t672\tBRCA1\t-\tBRCC1|FANCS\t-\t17\t17q21.31\tBRCA1 DNA repair associated\tprotein-coding\tBRCA1\tBRCA1 DNA repair associated\tO\t-\t20240101\tprotein-coding\n",
    "9606\t1\tNCRNA1\t-\tX1\t-\t1\t1p1\tsome ncRNA\tncRNA\t-\t-\tO\t-\t20240101\tncRNA\n",
    "10090\t22059\tTrp53\t-\tp53\t-\t11\t11 B3\ttransformation related protein 53\tprotein-coding\tTrp53\ttransformation related protein 53\tO\t-\t20240101\tprotein-coding\n",
]


def test_row_id_is_md5_of_entity_name_tab_text():
    assert row_id("GENE:TP53", "p53") == hashlib.md5(b"GENE:TP53\tp53").hexdigest()


def test_records_cover_symbol_synonyms_and_full_name():
    records = build_records(GENE_ENTITIES, GENE_INFO)
    tp53 = {r["label"]: r for r in records if r["entity_name"] == "GENE:TP53"}
    assert set(tp53) == {"TP53", "p53", "LFS1", "TRP53", "BCC7", "P53", "tumor protein p53"}
    assert tp53["TP53"]["text_type"] == "symbol"
    assert tp53["BCC7"]["source"] == "ncbi_human"
    assert tp53["tumor protein p53"]["text_type"] == "full_name"


def test_no_duplicate_pairs_and_unique_ids():
    records = build_records(GENE_ENTITIES, GENE_INFO)
    pairs = [(r["entity_name"], r["label"]) for r in records]
    assert len(pairs) == len(set(pairs))
    assert len({r["id"] for r in records}) == len(records)


def test_non_human_and_non_coding_genes_add_no_ncbi_rows():
    records = build_records(GENE_ENTITIES, GENE_INFO)
    mouse = [r for r in records if r["entity_name"] == "GENE:MOUSE1"]
    assert {r["label"] for r in mouse} == {"MOUSE1", "m1"}
    assert all(r["entity_name"] != "GENE:NCRNA1" for r in records)


def test_symbol_only_gene_still_gets_ncbi_synonyms():
    records = build_records(GENE_ENTITIES, GENE_INFO)
    brca1 = {r["label"] for r in records if r["entity_name"] == "GENE:BRCA1"}
    assert brca1 == {"BRCA1", "BRCC1", "FANCS", "BRCA1 DNA repair associated"}
