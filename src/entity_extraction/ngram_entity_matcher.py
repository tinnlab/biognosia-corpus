"""
N-gram generation, embedding and search against the Milvus entity collections.

1. Generate n-grams from text with character positions
2. Compute MedCPT-Query-Encoder embeddings for the n-grams
3. Search the entities_* collections on label_embedding
"""

import re
from dataclasses import dataclass

import numpy as np
import torch
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from pymilvus import Collection, connections, utility
from transformers import AutoModel, AutoTokenizer

try:
    from model2vec import StaticModel

    MODEL2VEC_AVAILABLE = True
except ImportError:
    MODEL2VEC_AVAILABLE = False
    StaticModel = None

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from db_init.embedding_utils import masked_mean_pooling, normalize_embeddings

GENE_COLLECTION = "Genes"
GENE_SYNONYM_COLLECTION = "Gene_Synonyms"


def merge_gene_matches(matches: list[dict]) -> list[dict]:
    """Reduce the hits from Genes and Gene_Synonyms to the single best gene hit."""
    gene_hits = [m for m in matches if m["entity_type"] in (GENE_COLLECTION, GENE_SYNONYM_COLLECTION)]
    if not gene_hits:
        return matches
    best = max(gene_hits, key=lambda m: (m["similarity_score_label"], m["entity_type"] == GENE_COLLECTION))
    others = [m for m in matches if m["entity_type"] not in (GENE_COLLECTION, GENE_SYNONYM_COLLECTION)]
    return others + [{**best, "entity_type": GENE_COLLECTION}]


@dataclass
class NGram:
    """N-gram with position information."""

    text: str
    start: int
    end: int
    k: int
    pos_tags: list = None


@dataclass
class EntityMatch:
    """Matched entity from Milvus search."""

    extracted_text: str
    start: int
    end: int
    entity_name: str
    entity_label: str
    entity_type: str
    entity_content: str
    similarity_score_label: float
    similarity_score_content: float


class NGramGenerator:
    """Generate n-grams from text with sentence boundaries and stop word filtering."""

    def __init__(
        self,
        k_values: list[int] = None,
        remove_stopwords: bool = True,
        filter_pos: bool = True,
        language: str = "english",
    ):
        """
        Initialize n-gram generator.

        Args:
            k_values: List of n-gram sizes (number of tokens)
            remove_stopwords: Whether to filter out n-grams containing only stop words
            filter_pos: Whether to filter n-grams based on POS tags
            language: Language for stop words (default: 'english')
        """
        if k_values is None:
            k_values = [1, 2, 3, 4, 5]
        self.k_values = sorted(k_values)
        self.remove_stopwords = remove_stopwords
        self.filter_pos = filter_pos

        if remove_stopwords:
            self.stop_words = set(stopwords.words(language))
        else:
            self.stop_words = set()

        self.invalid_start_tags = {
            "CC",
            "IN",
            "DT",
            "TO",
            "WDT",
            "WP",
            "WRB",
            "RB",
            "JJ",
            "JJR",
            "JJS",
            "VB",
            "VBD",
            "VBG",
            "VBN",
            "VBP",
            "VBZ",
            "MD",
            "CD",
            "PRP",
            "PRP$",
        }

        self.invalid_end_tags = {
            "CC",
            "IN",
            "DT",
            "TO",
            "VB",
            "VBD",
            "VBG",
            "VBN",
            "VBP",
            "VBZ",
            "MD",
            "RB",
            "CD",
            "PRP",
            "PRP$",
        }

        self.invalid_start_words = {
            "that",
            "it",
            "which",
            "where",
            "whereare",
            "there",
            "they",
            "these",
            "those",
            "this",
            "what",
            "when",
            "who",
            "whereas",
            "would",
            "will",
            "could",
            "can",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "ought",
        }

        self.invalid_end_words = {
            "that",
            "it",
            "which",
            "where",
            "whereare",
            "there",
            "they",
            "these",
            "those",
            "this",
            "what",
            "when",
            "who",
            "whereas",
            "would",
            "will",
            "could",
            "can",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "ought",
            "such",
            "as",
        }

        self.to_be_forms = {"is", "are", "was", "were", "am", "be", "been", "being"}

        self.pronouns = {
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            "my",
            "your",
            "his",
            "its",
            "our",
            "their",
            "mine",
            "yours",
            "hers",
            "ours",
            "theirs",
            "myself",
            "yourself",
            "himself",
            "herself",
            "itself",
            "ourselves",
            "yourselves",
            "themselves",
            "this",
            "that",
            "these",
            "those",
            "which",
            "who",
            "whom",
            "whose",
            "what",
            "where",
            "when",
            "why",
            "how",
        }

        self.comparison_temporal_words = {"then", "than", "more", "less", "most", "least"}

        self.comparative_tags = {"JJR", "JJS", "RBR", "RBS"}

        self.past_tense_tags = {"VBD", "VBN"}

    def generate(self, text: str) -> list[NGram]:
        """
        Generate n-grams from text with sentence-aware tokenization.

        This method:
        1. Protects single-letter abbreviations (e.g., "S. aureus")
        2. Splits text into sentences
        3. Tokenizes each sentence using NLTK
        4. Generates n-grams within sentence boundaries
        5. Filters out n-grams containing only stop words
        6. Maintains accurate character positions

        Args:
            text: Input text

        Returns:
            List of NGram objects with text and position info
        """
        ngrams = []

        abbrev_pattern = r"\b([A-Z])\.\s+([a-z])"
        abbrev_placeholder = r"\1__ABBREV__\2"
        text_protected = re.sub(abbrev_pattern, abbrev_placeholder, text)

        sentences = sent_tokenize(text_protected)

        current_pos = 0

        for sentence_protected in sentences:
            sentence = re.sub(r"([A-Z])__ABBREV__([a-z])", r"\1. \2", sentence_protected)

            sentence_start = text.find(sentence, current_pos)
            if sentence_start == -1:
                continue
            current_pos = sentence_start

            tokens_with_pos = []

            words = word_tokenize(sentence)

            search_start = sentence_start
            for word in words:
                if re.match(r"^[\W_]+$", word):
                    continue

                word_pos = text.find(word, search_start)
                if word_pos != -1:
                    tokens_with_pos.append((word, word_pos, word_pos + len(word)))
                    search_start = word_pos + len(word)

            for k in self.k_values:
                if k > len(tokens_with_pos):
                    continue

                for i in range(len(tokens_with_pos) - k + 1):
                    ngram_tokens = tokens_with_pos[i : i + k]

                    tokens = [t[0] for t in ngram_tokens]
                    start = ngram_tokens[0][1]
                    end = ngram_tokens[-1][2]
                    ngram_text = text[start:end].strip()

                    if not ngram_text:
                        continue

                    if "\n" in ngram_text or "\r" in ngram_text:
                        continue

                    if not ngram_text[0].isalnum() or not ngram_text[-1].isalnum():
                        continue

                    if self.remove_stopwords:
                        if all(t.lower() in self.stop_words for t in tokens):
                            continue

                    if len(ngram_text) < 3:
                        continue

                    if re.match(r"^[\d\W]+$", ngram_text):
                        continue

                    if all(len(t) == 1 for t in tokens):
                        continue

                    if not re.match(r"^[a-zA-Z0-9\-_/. ]+$", ngram_text):
                        continue

                    if ngram_text.endswith("."):
                        continue

                    pos_tags = None
                    if self.filter_pos:
                        pos_tags = pos_tag(tokens)

                        invalid_start_without_jj = self.invalid_start_tags - {"JJ"}
                        if pos_tags[0][1] in invalid_start_without_jj:
                            continue

                        if pos_tags[-1][1] in self.invalid_end_tags:
                            continue

                        if tokens[0].lower() in self.invalid_start_words:
                            continue

                        if tokens[-1].lower() in self.invalid_end_words:
                            continue

                        if any(token.lower() in self.to_be_forms for token in tokens):
                            continue

                        if any(token.lower() in self.pronouns for token in tokens):
                            continue

                        if any(token.lower() in self.comparison_temporal_words for token in tokens):
                            continue

                        if any(pos_tag[1] in self.comparative_tags for pos_tag in pos_tags):
                            continue

                        if any(pos_tag[1] in self.past_tense_tags for pos_tag in pos_tags):
                            continue

                    ngram_obj = NGram(text=ngram_text, start=start, end=end, k=k)
                    if pos_tags:
                        ngram_obj.pos_tags = pos_tags
                    ngrams.append(ngram_obj)

            current_pos = sentence_start + len(sentence)

        if self.filter_pos and ngrams:
            words_in_longer_ngrams = set()
            for ng in ngrams:
                if ng.k >= 2:
                    words_in_longer_ngrams.update(ng.text.split())

            filtered_ngrams = []
            for ng in ngrams:
                if ng.k >= 2:
                    filtered_ngrams.append(ng)
                    continue

                if ng.pos_tags and len(ng.pos_tags) > 0:
                    first_tag = ng.pos_tags[0][1]
                    if first_tag == "JJ" and ng.text in words_in_longer_ngrams:
                        continue

                filtered_ngrams.append(ng)

            ngrams = filtered_ngrams

        return ngrams


class NGramEmbedder:
    """Generate embeddings for n-grams using various models."""

    def __init__(self, model_name: str = "ncbi/MedCPT-Query-Encoder", device: str = "cuda:0", batch_size: int = 128):
        """
        Initialize embedder with specified model.

        Args:
            model_name: HuggingFace model name or model2vec model
            device: Device to use (cuda:0, cuda:1, cpu)
            batch_size: Batch size for embedding
        """
        self.device = device
        self.batch_size = batch_size
        self.model_name = model_name

        self.is_model2vec = "potion" in model_name.lower()

        if self.is_model2vec:
            if not MODEL2VEC_AVAILABLE:
                raise ImportError(
                    "model2vec library is required for Potion model. Install it with: pip install model2vec"
                )

            print(f"Loading model2vec model {model_name}...")
            self.model = StaticModel.from_pretrained(model_name)
            self.tokenizer = None
        else:
            print(f"Loading {model_name} on {device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(device)
            self.model.eval()

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Compute embeddings using the appropriate method for the model type.

        Args:
            texts: List of n-gram texts

        Returns:
            Numpy array of embeddings (N x D where D depends on model)
        """
        if not texts:
            return np.array([])

        if self.is_model2vec:
            embeddings = self.model.encode(texts)
            return embeddings
        else:
            embeddings_list = []

            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i : i + self.batch_size]

                inputs = self.tokenizer(
                    batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)

                embeddings = masked_mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
                embeddings = normalize_embeddings(embeddings)

                embeddings_list.append(embeddings.cpu().numpy())

            return np.vstack(embeddings_list)


OUTPUT_FIELDS = ["entity_name", "label", "content"]


class EntitySearcher:
    """Search Milvus entity collections for n-gram matches."""

    def __init__(self, milvus_host: str = "localhost", milvus_port: str = "19530", workspace: str = "lightrag"):
        """
        Initialize connection to Milvus.

        Args:
            milvus_host: Milvus host
            milvus_port: Milvus port
            workspace: Workspace prefix for collections
        """
        self.workspace = workspace

        connections.connect(alias="default", host=milvus_host, port=milvus_port)

        self.entity_collections = self.discover_entity_collections()
        print(f"Found {len(self.entity_collections)} entity collections")

    def discover_entity_collections(self) -> dict[str, Collection]:
        """
        Find all collections starting with 'entities_'.

        Returns:
            Dict mapping collection name to Collection object
        """
        all_collections = utility.list_collections()

        prefix = "entities_"
        entity_collection_names = [
            name for name in all_collections if name.startswith(prefix) and not name.startswith(f"{self.workspace}_")
        ]

        collections = {}
        for name in entity_collection_names:
            collection = Collection(name)
            collection.load()

            indexes = collection.indexes
            indexed_fields = {idx.field_name for idx in indexes}

            if "entity_name" not in indexed_fields:
                print(f"  Creating TRIE index on entity_name for {name}...")
                try:
                    collection.create_index(field_name="entity_name", index_params={"index_type": "TRIE"})
                    print(f"  Created index on {name}.entity_name")
                except Exception as e:
                    print(f"  Warning: Could not create index on {name}.entity_name: {e}")

            entity_type = name.replace(prefix, "")
            collections[entity_type] = collection

        return collections

    def get_collection_names(self) -> list[str]:
        """
        Get list of entity collection names.

        Returns:
            List of collection names (entity types)
        """
        return list(self.entity_collections.keys())

    def search_single_collection(
        self, collection_name: str, embeddings: np.ndarray, top_k: int = 1, min_similarity: float = 0.85
    ) -> list[list[dict]]:
        """
        Search a single entity collection.

        Args:
            collection_name: Name of the collection (entity type)
            embeddings: N-gram embeddings
            top_k: Number of top matches per n-gram
            min_similarity: Minimum similarity threshold

        Returns:
            List of search results, one per n-gram
        """
        if len(embeddings) == 0:
            return [[] for _ in range(len(embeddings))]

        if collection_name not in self.entity_collections:
            return [[] for _ in range(len(embeddings))]

        collection = self.entity_collections[collection_name]
        query_vectors = embeddings.tolist()

        results = [[] for _ in range(len(embeddings))]

        try:
            search_results = collection.search(
                data=query_vectors,
                anns_field="label_embedding",
                param={"metric_type": "COSINE", "params": {}},
                limit=top_k,
                output_fields=OUTPUT_FIELDS,
            )

            for ngram_idx, hits in enumerate(search_results):
                for hit in hits:
                    if hit.distance >= min_similarity:
                        results[ngram_idx].append(
                            {
                                "entity_type": collection_name,
                                "entity_name": hit.entity.get("entity_name"),
                                "entity_label": hit.entity.get("label", ""),
                                "entity_content": hit.entity.get("content", ""),
                                "similarity_score_label": float(hit.distance),
                            }
                        )

        except Exception as e:
            print(f"Warning: Error searching collection {collection_name}: {e}")

        return results

    def search_all_collections(
        self, embeddings: np.ndarray, top_k: int = 1, min_similarity: float = 0.85
    ) -> list[list[dict]]:
        """
        Search all entity collections for best matches.

        Args:
            embeddings: N-gram embeddings (N x 768)
            top_k: Number of top matches per n-gram
            min_similarity: Minimum similarity threshold

        Returns:
            List of search results, one per n-gram
            Each result is a list of dicts with entity info
        """
        if len(embeddings) == 0:
            return []

        query_vectors = embeddings.tolist()

        all_results = [[] for _ in range(len(embeddings))]

        for entity_type, collection in self.entity_collections.items():
            try:
                search_results = collection.search(
                    data=query_vectors,
                    anns_field="label_embedding",
                    param={"metric_type": "COSINE", "params": {}},
                    limit=top_k,
                    output_fields=OUTPUT_FIELDS,
                )

                for ngram_idx, hits in enumerate(search_results):
                    for hit in hits:
                        if hit.distance >= min_similarity:
                            all_results[ngram_idx].append(
                                {
                                    "entity_type": entity_type,
                                    "entity_name": hit.entity.get("entity_name"),
                                    "entity_label": hit.entity.get("label", ""),
                                    "entity_content": hit.entity.get("content", ""),
                                    "similarity_score_label": float(hit.distance),
                                }
                            )

            except Exception as e:
                print(f"Warning: Error searching collection {entity_type}: {e}")
                continue

        return [merge_gene_matches(r) for r in all_results]
