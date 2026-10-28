# BioGnosia corpus

Corpus processing and database insertion for the BioGnosia knowledge base.

The pipeline in this repository turns a list of full-text papers and a set of curated biomedical resources into the databases that [biognosia-rag](https://github.com/tinnlab/biognosia-rag) queries:

| Store | Content |
|---|---|
| Elasticsearch | Chunk text for BM25 keyword search |
| Redis (store) | Chunk text, document status, and the entity-to-chunk and chunk-to-entity links |
| Milvus | Chunk embeddings, reference entity collections (`entities_*`), entities linked to papers, relationships |
| Neo4j | Entity nodes and relationship edges with co-occurrence weights |

## Contents

- [Requirements](#requirements)
- [Quickstart](#quickstart)
- [Services](#services)
- [Pipeline steps](#pipeline-steps)
- [Data layout](#data-layout)
- [Configuration](#configuration)
- [Data sources](#data-sources)
- [Names written to the databases](#names-written-to-the-databases)
- [Use with biognosia-rag](#use-with-biognosia-rag)
- [Paper list](#paper-list)
- [Development](#development)
- [License](#license)

## Requirements

- Linux, Python 3.11, and Docker with the Compose plugin
- An NVIDIA GPU with CUDA for the embedding and chunking steps (CPU also works but is slow)
- `curl`, `unzip` and `tar` for `scripts/00_download_resources.sh`
- A Semantic Scholar API key for `scripts/03_download_papers.sh`
- Disk space for the databases and for the downloaded papers, chunks and model weights

Install the Python environment from the repository root:

```bash
conda env create -f environment.yml
conda activate biognosia-corpus
```

or, in an existing Python 3.11 environment:

```bash
pip install -r requirements.txt
```

## Quickstart

```bash
git clone https://github.com/tinnlab/biognosia-corpus.git
cd biognosia-corpus
cp .env.example .env          # set the passwords and the API key
docker compose up -d          # start the databases
scripts/run_all.sh            # run steps 00 to 10
```

`scripts/run_all.sh --from N --to M` runs a range of steps. Each step script can also be run on its own. The MSigDB file needs a manual download (see [Data sources](#data-sources)) before step 01.

## Services

`docker-compose.yml` starts the databases on `127.0.0.1`:

| Service | Image | Port |
|---|---|---|
| Milvus standalone (with etcd and MinIO) | `milvusdb/milvus:v2.6.2` | 19530 |
| Neo4j Community | `neo4j:2025.09.0-community` | 7687 (Bolt), 7474 (HTTP) |
| Redis (store) | `redis:8.2-alpine` | 6379 |
| Redis (n-gram cache, used while matching entities) | `redis:8.2-alpine` | 6380 |
| Elasticsearch | `docker.elastic.co/elasticsearch/elasticsearch:8.11.0` | 9200 |

Passwords come from `.env` (`NEO4J_PASSWORD`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`). Memory settings are controlled by `NEO4J_HEAP_SIZE`, `NEO4J_PAGECACHE_SIZE`, `ES_HEAP_SIZE`, `REDIS_MAXMEMORY` and `REDIS_CACHE_MAXMEMORY`. Database files are stored under `DATABASE_DIR` (default `./database`).

Milvus, Redis and Elasticsearch run without authentication and listen on `127.0.0.1` only.

## Pipeline steps

All paths are relative to `DATA_DIR` (default `./data`).

| Step | Script | Input | Output |
|---|---|---|---|
| 00 | `00_download_resources.sh` | URLs listed under [Data sources](#data-sources) | `resources/*`, `resources/VERSIONS.tsv`, NLTK data, model weights in `MODEL_CACHE_DIR` |
| 01 | `01_init_reference_db.sh` | `resources/*` | Milvus `entities_*` and `relationships_*` collections (genes, GO, CTD diseases and chemicals, KEGG, Reactome, MSigDB, Cell Ontology, Uberon, Taxonomy) |
| 02 | `02_build_gene_synonyms.sh` | Milvus `entities_Genes`, `resources/genes/gene_info.gz` | Milvus `entities_Gene_Synonyms` |
| 03 | `03_download_papers.sh` | `data/papers.tsv` | `papers/<paperId>.txt`, `papers-pdf/<paperId>.pdf`, `papers-download.tsv` |
| 04 | `04_preprocess.sh` | `papers/` | `papers-preprocessed/`, `preprocessing-stats/`, `preprocessing-error/` |
| 05 | `05_chunk.sh` | `papers-preprocessed/` | `chunk/<paperId>/<n>.txt`, `chunk-stats/chunk_stats.csv`, `chunk-stats/paper_stats.csv`, `chunk-error/` |
| 06 | `06_match_entities.sh` | `chunk/`, Milvus `entities_*` | `entities-ngram/<paperId>/<n>.tsv`, `entities-ngram-stats/matching_stats.csv` |
| 07 | `07_entity_statistics.sh` | `entities-ngram/` | `entities-ngram-stats/entity_statistics*.tsv` |
| 08 | `08_insert_chunks.sh` | `chunk/`, `chunk-stats/chunk_stats.csv` | Milvus `lightrag_chunks`, Redis chunk keys, Elasticsearch index `lightrag_chunks` |
| 09 | `09_insert_entities.sh` | `entities-ngram/`, `entities-ngram-stats/entity_statistics_GLOBAL.tsv`, Milvus `entities_*` | Milvus `lightrag_entities`, Redis entity-chunk links |
| 10 | `10_insert_relationships.sh` | `entities-ngram/`, Milvus `lightrag_entities` and `relationships_*` | Milvus `lightrag_relationships`, Neo4j nodes and edges |

Details of each step:

- **00** downloads the resources and records source URL, download date, size and SHA-256 of each file in `resources/VERSIONS.tsv`. It also downloads the NLTK data and the models `BAAI/bge-m3`, `ncbi/MedCPT-Query-Encoder` and `minishlab/potion-base-32M`.
- **01** runs one loader per resource from `src/db_init/`. Each loader embeds entity text with BAAI/bge-m3 (content, 1024-d, field `vector`) and entity labels with MedCPT-Query-Encoder (768-d, field `label_embedding`). Run `scripts/01_init_reference_db.sh genes kegg` to run a subset.
- **02** adds one row per gene symbol, synonym and NCBI full name to `entities_Gene_Synonyms`. The rows come from the `symbol` and `synonyms` of every `entities_Genes` entity and from the synonyms and full name of human protein-coding genes in `gene_info.gz`. Each row has `entity_name` = `GENE:<SYMBOL>`, `id` = md5 of `entity_name`, a tab and the text, `label` = the text, and `label_embedding` from MedCPT-Query-Encoder (masked mean pooling, L2-normalised, 768-d).
- **03** looks up the open-access PDF of each `paperId` through the Semantic Scholar Graph API (batch endpoint, `SEMANTIC_SCHOLAR_API_KEY`). If Semantic Scholar lists no usable PDF, it looks up the open-access PDF in Europe PMC by PMID or DOI. PDFs are converted to text with [pypdf](https://pypdf.readthedocs.io/). The script rate-limits its requests, skips papers that already have a text file, and records every attempt in `papers-download.tsv`. Use `--retry-failed` to retry papers that failed. Which papers are available as open access depends on the time of the download.
- **04** repairs the text encoding, converts LaTeX and Unicode to ASCII, removes inline citations and duplicate paragraphs, and normalises whitespace and hyphenation. Papers shorter than 100 characters, longer than 1,000,000 characters, or not in English are rejected and written to `preprocessing-error/`.
- **05** splits each paper into semantic chunks with the Chonkie `SemanticChunker` (`minishlab/potion-base-32M` embeddings, similarity window 4, at least 10 sentences per chunk, target `CHUNK_SIZE` tokens).
- **06** generates 1 to 5-grams of each chunk, filters them by stop words and part-of-speech tags, embeds them with MedCPT-Query-Encoder, and searches every `entities_*` collection on `label_embedding` (cosine). The best hit per collection is kept. Hits from `entities_Genes` and `entities_Gene_Synonyms` are reduced to the single best gene hit, and a synonym hit resolves to its `GENE:<SYMBOL>` entity. Matches with a cosine similarity of at least `NGRAM_MIN_SIMILARITY` (0.85) are written. Search results per n-gram are cached in the second Redis instance. Use `--clear-cache` after the entity collections change.
- **07** counts, for every matched entity, the papers and chunks that contain it.
- **08** embeds the chunks with BAAI/bge-m3 and inserts them. Chunk ids have the form `chunk-<paperId>-NNNNNN`.
- **09** inserts the matched entities that pass the threshold `ENTITY_SIMILARITY_THRESHOLD` (0.85) into the `lightrag_entities` collection, copying the entity records from the reference collections. It writes the forward link `entity id -> chunk ids` and the reverse link `chunk id -> entity names` to Redis.
- **10** pairs the entities matched in each paper, looks up the pairs that have a documented relationship in the `relationships_*` collections, and writes each found relationship to Milvus and Neo4j. The edge weight is the number of papers in which the pair occurs.

Steps 03 to 10 are resumable. Progress files `chunks-inserted.csv`, `entities-inserted.csv` and `relationships-inserted.csv` in `DATA_DIR` record the inserted papers. Pass `--drop-existing` to a step 08, 09 or 10 script to start that step from scratch.

## Data layout

```
DATA_DIR/
  resources/
    genes/              gene_info.gz, gene2go.gz
    ctd/                CTD_*.csv.gz
    reactome/           ReactomePathways.txt, ReactomePathways.gmt
    msigdb/             msigdb_v2025.1.Hs.db
    ontologies/         cl.owl, uberon.owl
    taxonomy/           nodes.dmp, names.dmp, merged.dmp
    nltk_data/
    VERSIONS.tsv
  models/               Hugging Face cache
  db-init/              loader metadata files, KEGG response cache
  papers/               <paperId>.txt
  papers-pdf/           <paperId>.pdf
  papers-download.tsv
  papers-preprocessed/  <paperId>.txt
  chunk/                <paperId>/<n>.txt
  chunk-stats/          chunk_stats.csv, paper_stats.csv
  entities-ngram/       <paperId>/<n>.tsv
  entities-ngram-stats/ matching_stats.csv, entity_statistics*.tsv
  chunks-inserted.csv, entities-inserted.csv, relationships-inserted.csv
```

## Configuration

`config/pipeline.conf` (steps 03 to 10) and `config/db_init.conf` (steps 00 to 02) are shell files that the scripts source and the Python modules parse. A value written as `${NAME:-default}` is read from the environment or from `.env` first.

Environment variables (`.env.example`):

| Variable | Use |
|---|---|
| `DATA_DIR`, `MODEL_CACHE_DIR`, `DATABASE_DIR` | Data, model cache and database volume directories |
| `NEO4J_PASSWORD`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY` | Service credentials |
| `SEMANTIC_SCHOLAR_API_KEY`, `CONTACT_EMAIL` | Paper download |
| `HF_TOKEN` | Optional Hugging Face token |
| `MILVUS_HOST`, `MILVUS_PORT`, `REDIS_HOST`, `REDIS_PORT`, `NGRAM_REDIS_HOST`, `NGRAM_REDIS_PORT`, `ELASTICSEARCH_HOST`, `ELASTICSEARCH_PORT`, `NEO4J_URI`, `NEO4J_USERNAME` | Service endpoints |
| `CUDA_VISIBLE_DEVICES`, `WORKERS` | GPUs and CPU workers |

Main keys of `config/pipeline.conf`:

| Key | Default | Step |
|---|---|---|
| `CHUNK_SIZE` | 300 | 05 |
| `NGRAM_K_VALUES` | 1,2,3,4,5 | 06 |
| `NGRAM_MIN_SIMILARITY` | 0.85 | 06, 07 |
| `NGRAM_PAPER_BATCH`, `NGRAM_EMBEDDING_BATCH`, `NGRAM_CHUNK_LOAD_WORKERS`, `NGRAM_EMBEDDING_WORKERS`, `NGRAM_QUERY_WORKERS`, `NGRAM_REDIS_BATCH_SIZE` | 512, 512, 32, 8, 1, 20000 | 06 |
| `CHUNK_EMBEDDING_WORKERS`, `EMBEDDING_SUB_BATCH_SIZE`, `CHUNKS_PER_BATCH`, `MILVUS_INSERT_BATCH_PAPERS`, `MAX_PENDING_BATCHES`, `INSERT_BATCH_SIZE` | 4, 64, 500, 500, 10, 1000 | 08 |
| `ENTITY_SIMILARITY_THRESHOLD` | 0.85 | 09, 10 |
| `ENTITY_STATS_FILE`, `MAX_PCT_PAPER`, `MAX_PCT_CHUNK` | see file | 09, 10 |
| `PAPERS_PER_BATCH`, `REDIS_BATCH_SIZE`, `REFERENCE_INSERT_BATCH_SIZE`, `INSERT_GENES` ... `INSERT_KEGG` | 5000, 500, 1000, false | 09 |
| `RELATIONSHIP_PAPER_BATCH_SIZE`, `RELATIONSHIP_UPSERT_BATCH`, `RELATIONSHIP_NUM_QUERY_WORKERS` | 640, 1000, 4 | 10 |
| `PAPER_DOWNLOAD_WORKERS`, `SEMANTIC_SCHOLAR_REQUEST_INTERVAL`, `EUROPE_PMC_REQUEST_INTERVAL` | 4, 1.1, 0.2 | 03 |
| `LIMIT`, `OFFSET` | 0 | 04 to 10 |

`config/db_init.conf` sets the embedding models and lengths, the organism, gene type and collection filters of each loader, the batch sizes, and the paths of the resource files.

## Data sources

| Resource | Files | URL |
|---|---|---|
| NCBI Gene | `gene_info.gz`, `gene2go.gz` | https://ftp.ncbi.nlm.nih.gov/gene/DATA/ |
| Comparative Toxicogenomics Database | `CTD_diseases`, `CTD_chemicals`, `CTD_curated_genes_diseases`, `CTD_diseases_pathways`, `CTD_chem_gene_ixns`, `CTD_curated_chemicals_diseases`, `CTD_chem_pathways_enriched`, `CTD_chem_go_enriched` (`.csv.gz`) | https://ctdbase.org/downloads/ |
| KEGG | Pathways, genes and links, fetched by the KEGG loader with caching under `db-init/kegg-cache` | https://rest.kegg.jp |
| Reactome | `ReactomePathways.txt`, `ReactomePathways.gmt` | https://reactome.org/download-data |
| MSigDB | SQLite release `msigdb_v2025.1.Hs.db` (registration required, manual download) | https://www.gsea-msigdb.org/gsea/downloads.jsp |
| Cell Ontology | `cl.owl` | http://purl.obolibrary.org/obo/cl.owl |
| Uberon | `uberon.owl` | http://purl.obolibrary.org/obo/uberon.owl |
| NCBI Taxonomy | `taxdump.tar.gz` (`nodes.dmp`, `names.dmp`, `merged.dmp`) | https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/ |
| Semantic Scholar Graph API | Open-access PDF links | https://api.semanticscholar.org/api-docs/graph |
| Europe PMC | Open-access PDFs | https://europepmc.org/RestfulWebService |

MSigDB requires a free registration. Register, download the human SQLite release, unzip it and place `msigdb_v2025.1.Hs.db` in `DATA_DIR/resources/msigdb/`. `scripts/00_download_resources.sh` prints these instructions when the file is missing. Each resource is subject to its own license and terms of use.

## Names written to the databases

The workspace prefix is `WORKSPACE` in the configuration files (`lightrag`).

| Store | Name |
|---|---|
| Elasticsearch | index `lightrag_chunks` |
| Milvus | `lightrag_chunks`, `lightrag_entities`, `lightrag_relationships`, `entities_<Type>`, `relationships_<TypeA>_<TypeB>` |
| Redis | `lightrag_text_chunks:<chunk id>`, `lightrag_doc_status:<paperId>`, `lightrag_entity_sources:<entity id>`, `lightrag_chunk_entities:<chunk id>` |
| Neo4j | node label `lightrag` with property `entity_id`, a type label per entity prefix, relationship type `DIRECTED` with `weight` |

## Use with biognosia-rag

[biognosia-rag](https://github.com/tinnlab/biognosia-rag) reads the stores built here. Point its `config/rag-web.conf` (or the matching `BIOGNOSIA_*` environment variables) at the same Milvus, Neo4j, Redis and Elasticsearch instances and keep `workspace = lightrag`. Its `entity_collections` setting lists the entity collections searched with `label_embedding`, including `Gene_Synonyms`.

biognosia-rag resolves citations through a MongoDB collection of paper metadata keyed by `paperId`. This repository does not build that collection.

## Paper list

`data/papers.tsv` lists the 134,864 papers of the corpus with the columns `paperId`, `doi`, `pmid`, `title`, `journal` and `publication_date`. `paperId` is the Semantic Scholar paper id and is the file name stem used in every step. For 111,490 papers the other columns are filled where the source has a value. For the remaining 23,374 papers only `paperId` is set.

## Development

```bash
ruff check src tests
ruff format --check src tests
pip install pytest
pytest
```

The tests run without a GPU or a database. The ruff configuration is in `pyproject.toml`.

## License

MIT, see `LICENSE`. Third-party models and data are listed in `NOTICE`.
