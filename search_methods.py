"""
search_methods.py

Three progressive search implementations:
  1. search_semantic     — embedding similarity only
  2. search_hybrid       — token match of product data + semantic, combined via alpha
  3. search_query_aware  — query-category relevance weights * hybrid scores per category

All return the same shape:
    [
        {
            "product_id":      str,
            "score":           float,
            "category_scores": {
                category_name: {
                    "hybrid":   float,   # hybrid-only methods; 0.0 for semantic
                    "semantic": float,
                    "token":    float,   # hybrid-only methods; 0.0 for semantic
                    "weight":   float,   # query-aware only; 1.0 for others
                }
            }
        },
        ...
    ]
"""

from collections import defaultdict
from scipy.spatial.distance import cosine
import category_profile_tokens
import re
import os
import time
import json
from nltk.corpus import stopwords
import numpy as np
from nltk.stem import SnowballStemmer

# ── SETUP ──────────────────────────────
# model         : SentenceTransformer
# collections   : dict[str, chromadb.Collection]
# compute_all_hybrid_scores(query, alpha) -> {pid: {cat: score_dict}}
# The per-category token sets (EPD_TOKENS, MATERIALS_TOKENS, etc.) for query-aware
# ─────────────────────────────────────────────────────────────────────────────


# ════════════════════════════════════════════════════════════════════
# 1. SEMANTIC ONLY
# ════════════════════════════════════════════════════════════════════

def search_semantic(
        query_embedding: np.ndarray,
        collections: dict,
        loaded_chunk_embeddings_by_pid: dict,
        top_k: int = 5
) -> list:
    """
    Pure embedding similarity between query and product data.
    No token matching involved.
    """
    # Phase 1 — nominate pids from top-k per collection
    nominated_pids    = set()
    collection_results = {}

    for doc_type, collection in collections.items():
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, collection.count()),
                include=["metadatas", "distances", "documents"]
            )
            collection_results[doc_type] = results
            for metadata in results["metadatas"][0]:
                nominated_pids.add(str(metadata["product_id"]))
        except Exception as e:
            print(f"[semantic] Error searching '{doc_type}': {e}")

    # Phase 2 — score every nominated pid across all collections
    raw_scores      = defaultdict(lambda: defaultdict(list))
    category_scores = defaultdict(dict)

    for pid in nominated_pids:
        for doc_type, collection in collections.items():
            try:
                # Look up preloaded embeddings for this pid in this category
                pid_embeddings = loaded_chunk_embeddings_by_pid.get(doc_type, {}).get(pid, [])

                if not pid_embeddings:
                    continue

                for doc_emb in pid_embeddings:
                    similarity = 1 - cosine(query_embedding, doc_emb)
                    if similarity < 0.1:
                        continue
                    raw_scores[pid][doc_type].append(similarity)

            except Exception as e:
                print(f"[semantic] Error scoring pid {pid} in '{doc_type}': {e}")

        # Build category_scores for this pid
        for doc_type, sims in raw_scores[pid].items():
            sem_score = sum(sims) / len(sims)
            category_scores[pid][doc_type] = {
                "hybrid":   0.0,   # not applicable
                "semantic": round(sem_score, 4),
                "token":    0.0,   # not applicable
                "weight":   1.0,   # uniform — no weighting in semantic
            }

    # Phase 3 — aggregate: uniform mean across categories that fired
    product_scores = {}
    for pid, cats in category_scores.items():
        if not cats:
            continue
        product_scores[pid] = sum(s["semantic"] for s in cats.values()) / len(cats)

    ranked = sorted(product_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    unique_scores_seen = []
    for pid, score in ranked:
        rounded = round(score, 4)
        if rounded not in unique_scores_seen:
            if len(unique_scores_seen) >= top_k:
                break
            unique_scores_seen.append(rounded)
        results.append({
            "product_id":      pid,
            "score":           rounded,
            "category_scores": category_scores[pid],
        })

    return results


# ════════════════════════════════════════════════════════════════════
# 2. HYBRID
# ════════════════════════════════════════════════════════════════════

def compute_semantic_scores_from_chroma(
        query_embedding: np.ndarray,
        collection
) -> dict[str, float]:
    """
    Compute cosine similarity between the query embedding and every
    document's embedding directly.
    """
    results = collection.get(include=["embeddings", "metadatas", "documents"])

    scores = {}
    for emb, metadata, document in zip(
        results["embeddings"], results["metadatas"], results["documents"]
    ):
        product_id = str(metadata["product_id"])
        similarity = 1 - cosine(query_embedding, np.array(emb))
        if similarity < 0.1:
            continue
        if product_id not in scores or similarity > scores[product_id]["semantic"]:
            scores[product_id] = {"semantic": similarity, "text": document}

    return scores

stemmer = SnowballStemmer("english")

def compute_token_match_score(query: str, product_text: str) -> tuple[list, float]:
    def tokenize(text):
        # hyphen -> space, not deleted, so "eco-label" tokenizes like "eco label"
        return re.sub(r'[^\w\s]', ' ', text.lower()).split()

    stop_words = set(stopwords.words('english'))

    def stem_filter(tokens):
        out = set()
        for w in tokens:
            if w in stop_words:
                continue
            # drop short pure-noise leftovers, but keep short alphanumeric
            # domain tokens (standard codes like "en", "iso")
            if len(w) <= 2 and not any(c.isdigit() for c in w) and w not in {"en", "iso"}:
                continue
            out.add(stemmer.stem(w))
        return out

    query_tokens = stem_filter(tokenize(query))
    text_tokens  = stem_filter(tokenize(product_text))

    matched = query_tokens & text_tokens

    if not query_tokens or not matched:
        return [], 0.0

    # normalized by how much of the query's meaningful tokens were covered
    score = round(len(matched) / len(query_tokens), 4)
    return sorted(matched), score

def compute_token_scores_from_collection(
        query: str, 
        collection
    ) -> dict[str, tuple[list, float]]:
    """
    Scan every document in the collection and compute token match score.
    Returns {product_id: (matched_tokens, score)}
    """
    all_docs = collection.get(include=["documents", "metadatas"])

    token_scores = {}
    for document, metadata in zip(all_docs["documents"], all_docs["metadatas"]):
        product_id = str(metadata["product_id"])
        matched, score = compute_token_match_score(query, document)

        # keep best score if product has multiple chunks
        if product_id not in token_scores or score > token_scores[product_id][1]:
            token_scores[product_id] = (matched, score)

    return token_scores

def compute_category_scores_for_product(
    query: str,
    query_embedding: np.ndarray,
    collection,
    alpha: float = 0.5
) -> dict[str, dict]:

    semantic_scores = compute_semantic_scores_from_chroma(query_embedding, collection)
    token_scores = compute_token_scores_from_collection(query, collection)
    
    # union of pids from both signals
    all_pids = set(semantic_scores.keys())
    all_pids |= set(token_scores.keys())

    combined = {}
    for product_id in all_pids:
        sem   = semantic_scores.get(product_id, {})
        s_sem = sem.get("semantic", 0.0)
        s_text = sem.get("text", "")

        matched, s_tok = token_scores.get(product_id, ([], 0.0))
        hybrid = alpha * s_sem + (1 - alpha) * s_tok

        # no 0.1 threshold here, only applied to semantic

        combined[product_id] = {
            "hybrid":        hybrid,
            "semantic":      s_sem,
            "semantic_text": s_text,
            "token":         s_tok,
            "token_text":    ", ".join(sorted(matched)),
        }            

    return combined

def compute_all_hybrid_scores(
    query: str,
    query_embedding: np.ndarray,
    collections: dict,
    alpha: float = 0.5,
) -> dict[str, dict[str, float]]:
    """
    Compute hybrid scores across all categories.
    Returns {product_id: {category_name: score}}
    """
    all_scores = {}
    for category_name, collection in collections.items():
        category_scores = compute_category_scores_for_product(
            query, query_embedding, collection, alpha=alpha
        )
        for product_id, score_dict in category_scores.items():
            if product_id not in all_scores:
                all_scores[product_id] = {}
            if score_dict["hybrid"] > 0:
                all_scores[product_id][category_name] = score_dict

    return all_scores

def search_hybrid(
    query: str, 
    query_embedding: np.ndarray,
    collections: dict,
    top_k: int = 5, 
    alpha: float = 0.5
) -> list:
    """
    Combines:
      - token match: product data vs category vocabulary tokens
      - semantic:    query embedding vs product data embedding
    Final per-category score = alpha * semantic + (1 - alpha) * token
    Aggregated as uniform mean across relevant categories (hybrid >= 0.1).
    """
    # compute_all_hybrid_scores returns
    # {pid: {cat: {"hybrid", "semantic", "token", "semantic_text", "token_text"}}}
    all_scores = compute_all_hybrid_scores(query, query_embedding, collections, alpha=alpha)

    if not all_scores:
        return []

    product_scores = {}
    for pid, cat_scores in all_scores.items():
        weighted_sum = sum(s["hybrid"] for s in cat_scores.values())
        total_weight = len(cat_scores)
        product_scores[pid] = (weighted_sum / total_weight) if total_weight > 0 else 0.0

    ranked = sorted(product_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    unique_scores_seen = []
    for pid, score in ranked:
        rounded = round(score, 4)
        if rounded not in unique_scores_seen:
            if len(unique_scores_seen) >= top_k:
                break
            unique_scores_seen.append(rounded)

        # Normalise shape to match the common return format
        cat_scores_out = {
            cat: {
                "hybrid":   round(s["hybrid"],   4),
                "semantic": round(s["semantic"],  4),
                "token":    round(s.get("token", 0.0), 4),
                "weight":   1.0,   # uniform — no query-aware weighting here
            }
            for cat, s in all_scores[pid].items()
        }
        results.append({
            "product_id":      pid,
            "score":           rounded,
            "category_scores": cat_scores_out,
        })

    return results


# ════════════════════════════════════════════════════════════════════
# 3. QUERY-AWARE
# ════════════════════════════════════════════════════════════════════

# Map each collection name to its curated token list

COLLECTION_01=os.getenv("COLLECTION_01")
COLLECTION_02=os.getenv("COLLECTION_02")
COLLECTION_03=os.getenv("COLLECTION_03")
COLLECTION_04=os.getenv("COLLECTION_04")
COLLECTION_05=os.getenv("COLLECTION_05")
COLLECTION_06=os.getenv("COLLECTION_06")
COLLECTION_07=os.getenv("COLLECTION_07")
COLLECTION_08=os.getenv("COLLECTION_08")
COLLECTION_09=os.getenv("COLLECTION_09")
COLLECTION_10=os.getenv("COLLECTION_10")
COLLECTION_11=os.getenv("COLLECTION_11")

CATEGORY_TOKEN_SETS = json.loads(os.getenv("CATEGORY_TOKENS"))


def compute_category_weight(
    query: str,
    query_embedding: np.ndarray,
    semantic_embeddings: np.ndarray | None,
    token_set: list[str],
    alpha_w: float = 0.5,
    top_k: int = 3
) -> float:
    """
    Compute relevance weight of a category for a given query.

    Combines:
      - token_w:    fraction of query tokens that match any token in the category's
                    vocabulary (query tokens vs category tokens)
      - semantic_w: cosine similarity between query embedding and the centroid
                    embedding of the category token set

    weight = alpha_w * semantic_w + (1 - alpha_w) * token_w
    """
    query_lower = query.lower()

    if not token_set or semantic_embeddings is None or len(semantic_embeddings) == 0:
        return 0.0

    # ── Token weight ──────────────────────────────────────────────
    token_w = 0.0
    matched = []
    for token in token_set:
        token_lower = token.lower()
        if token_lower in query_lower:
            matched.append(token)
    if matched:
        token_w = min(1.0, 0.7 + (len(matched) - 1) * 0.1)

    # ── Semantic weight ───────────────────────────────────────────
    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
    
    sims = [cosine_sim(query_embedding, emb) for emb in semantic_embeddings]
    sims_thresholded = [s if s >= 0.1 else 0.0 for s in sims]
    semantic_w = float(np.mean(sorted(sims_thresholded, reverse=True)[:top_k]))

    return alpha_w * semantic_w + (1 - alpha_w) * token_w


def compute_all_category_weights(
    query: str,
    query_embedding: np.ndarray,
    loaded_token_embeddings_dict: dict,
    loaded_chunk_embeddings_dict: dict
) -> dict[str, float]:
    weights = {}

    for category_name in CATEGORY_TOKEN_SETS.keys():
        tokens = CATEGORY_TOKEN_SETS.get(category_name, [])
        token_embeddings = loaded_token_embeddings_dict.get(category_name, None)
        pp_chunks = loaded_chunk_embeddings_dict.get("product_performance", {})
        chunk_embeddings = [emb for embs in pp_chunks.values() for emb in embs]

        weights[category_name] = compute_category_weight(
            query=query,
            query_embedding=query_embedding,
            semantic_embeddings=chunk_embeddings if category_name == "product_performance" else token_embeddings,
            token_set=tokens,
        )

    return weights


def compute_final_score(
    hybrid_scores_per_category: dict[str, float],
    category_weights: dict[str, float],
    weight_threshold: float = 0.05,
) -> float:
    """
    Weighted mean of per-category hybrid scores, using query-aware category weights.
    Categories whose weight falls below weight_threshold are excluded.
    """
    numerator   = 0.0
    denominator = 0.0

    for category_name, w_c in category_weights.items():
        if w_c < weight_threshold:
            continue
        score = hybrid_scores_per_category.get(category_name, 0.0)  # missing → 0
        numerator   += w_c * score
        denominator += w_c

    return numerator / denominator if denominator > 0 else 0.0


def search_query_aware(
    query: str, 
    query_embedding: np.ndarray,
    collections: dict, 
    loaded_token_embeddings_dict: dict,
    loaded_chunk_embeddings_dict: dict,
    top_k: int = 5,
    alpha: float = 0.5,
    weight_threshold: float = 0.05,
) -> list:
    """
    Query-aware category weighting:
      1. Compute hybrid scores per product per category (same as search_hybrid)
      2. Compute a relevance weight w_c for each category based on the query
         (token match of query vs category tokens + semantic similarity)
      3. Final score = weighted mean of per-category hybrid scores,
         excluding categories below weight_threshold
    """
    # Step 1 — hybrid scores per product per category
    all_scores = compute_all_hybrid_scores(query, query_embedding, collections, alpha=alpha)

    if not all_scores:
        return []

    # Step 2 — compute category weights from the query
    t_weight_start = time.perf_counter()
    category_weights = compute_all_category_weights(
        query=query, 
        query_embedding=query_embedding, 
        loaded_token_embeddings_dict=loaded_token_embeddings_dict, 
        loaded_chunk_embeddings_dict=loaded_chunk_embeddings_dict
    )
    weight_computation_s = time.perf_counter() - t_weight_start

    # Step 3 — compute final score per product
    product_scores = {}
    for pid, cat_scores in all_scores.items():
        hybrid_per_cat = {cat: s["hybrid"] for cat, s in cat_scores.items()}
        product_scores[pid] = compute_final_score(
            hybrid_per_cat, category_weights, weight_threshold=weight_threshold
        )

    ranked = sorted(product_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    unique_scores_seen = []
    for pid, score in ranked:
        rounded = round(score, 4)
        if rounded not in unique_scores_seen:
            if len(unique_scores_seen) >= top_k:
                break
            unique_scores_seen.append(rounded)

        cat_scores_out = {
            cat: {
                "hybrid":   round(s["hybrid"],   4),
                "semantic": round(s["semantic"],  4),
                "token":    round(s.get("token", 0.0), 4),
                "weight":   round(category_weights.get(cat, 0.0), 4),
            }
            for cat, s in all_scores[pid].items()
        }
        results.append({
            "product_id":      pid,
            "score":           rounded,
            "category_scores": cat_scores_out,
        })

    return results, weight_computation_s

