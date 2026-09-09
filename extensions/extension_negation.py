
# ────────────────────────────────────────────────────────────────────
# ------- EXTENSION 2: Negation --------------------------------------
# ────────────────────────────────────────────────────────────────────


"""
negation_extension.py

Proposed mitigation for the negation failure mode identified in
search_query_aware: query-aware's token+semantic scoring has no
awareness of negation cues ("not", "without", etc.), so a negated
term (e.g. "UK" in "not locally made in UK") can score as a strong
positive match and inflate both a category's token score and its
query-aware weight.

This module does NOT modify search_methods.py. Instead, it:
  1. Detects negation cues and their scope in the query (with a
     curated exception list for lexicalized "non-X" compounds that
     are themselves positive catalogue terms, e.g. "non-VOC").
  2. Builds a "cleaned" query string with negated tokens removed.
  3. Reuses the ORIGINAL, unmodified token-matching, weighting, and
     aggregation functions from search_methods.py, but feeds them the
     cleaned query wherever token-level matching occurs.

Scope / known limitations (see accompanying report section):
  - This only addresses the TOKEN and CATEGORY-WEIGHT contributions.
    The semantic (embedding) component still uses the ORIGINAL query
    embedding by default, so it can still carry some of the same bias,
    since sentence embeddings do not reliably encode negation/polarity.
    An optional `model` parameter is provided to re-embed the cleaned
    query for semantic scoring too, as a partial mitigation — pass
    your SentenceTransformer instance if you want to test this.
  - Only single-token negation cues are handled ("not", "no",
    "without", "excluding", "except"); scope runs to the next clause
    boundary (comma, "and", "or", ";", end of string).
  - Coordination ("without VOC or formaldehyde"), multi-word negation
    targets spanning conjunctions, and synonym-based exclusion
    ("VOC-free", "avoids X") are NOT detected by this cue-word
    approach — see report discussion for why a general solution would
    require a scoped dependency parse or a broader synonym-aware cue
    list.
"""

import re
from search_methods import (
    compute_semantic_scores_from_chroma,
    compute_token_scores_from_collection,
    compute_category_weight,
    compute_final_score,
    CATEGORY_TOKEN_SETS,
)

# ── Negation detection ────────────────────────────────────────────

NEGATION_CUES = {"not", "no", "without", "excluding", "except"}
CLAUSE_BOUNDARIES = {",", "and", "or", ";"}

# Curated exception list: known lexicalized "non-X" compounds that are
# themselves positive catalogue terms, not compositional negation.
# Extend this based on your actual catalogue vocabulary.
NON_EXCEPTIONS = {
    "non-voc", "non-combustible", "non-toxic", "non-hazardous",
    "non-flammable", "non-allergenic",
}


def tokenize_preserving_hyphens(text: str) -> list[str]:
    """
    Splits on whitespace, with commas/semicolons separated out as their
    own tokens, but keeps hyphenated compounds (e.g. 'non-VOC') intact.
    """
    text = re.sub(r"([,;])", r" \1 ", text)
    return [t for t in text.strip().split() if t]


def detect_negated_terms(query: str) -> tuple[list[str], set[str]]:
    """
    Returns (tokens, negated_tokens).

    Scope rule: everything after a negation cue, up to the next clause
    boundary (comma, 'and', 'or', ';', or end of string), is treated as
    negated — except tokens matching a known non-X lexicalized
    exception, which are treated as ordinary positive tokens.
    """
    raw_tokens = tokenize_preserving_hyphens(query.lower())
    negated = set()
    in_negation_scope = False

    for tok in raw_tokens:
        stripped = tok.strip(",;")

        if stripped in CLAUSE_BOUNDARIES:
            in_negation_scope = False
            continue

        if stripped in NEGATION_CUES:
            in_negation_scope = True
            continue  # the cue word itself is not a negated term

        if in_negation_scope:
            if stripped in NON_EXCEPTIONS:
                # lexicalized exception — positive term, not negated
                continue
            negated.add(stripped)

    return raw_tokens, negated


def build_cleaned_query(query: str) -> tuple[str, set[str]]:
    """
    Returns (cleaned_query, negated_tokens).

    cleaned_query has all negated tokens and clause-boundary symbols
    removed, leaving only the query's positive content — this is what
    gets passed to the ORIGINAL token-matching and weighting functions,
    so a negated term can no longer contribute a token match.
    """
    tokens, negated = detect_negated_terms(query)
    cleaned_tokens = [
        t for t in tokens
        if t.strip(",;") not in CLAUSE_BOUNDARIES 
        and t.strip(",;") not in negated
        and t.strip(",;") not in NEGATION_CUES
    ]
    cleaned_query = " ".join(cleaned_tokens)
    return cleaned_query, negated


# ── Negation-aware category scoring ───────────────────────────────

def compute_category_scores_for_product_negation_aware(
    query: str,
    query_embedding,
    collection,
    alpha: float = 0.5,
    model=None,
) -> dict[str, dict]:
    """
    Same shape/behaviour as search_methods.compute_category_scores_for_product,
    except token matching uses the negation-cleaned query, so negated
    tokens cannot contribute to the token score.

    If `model` is provided (a SentenceTransformer), the cleaned query is
    also re-embedded and used for the semantic score instead of the
    original query_embedding — a partial mitigation for the semantic
    component, which otherwise still carries some negation bias (see
    module docstring).
    """
    cleaned_query, negated_tokens = build_cleaned_query(query)

    semantic_query_embedding = query_embedding
    if model is not None and cleaned_query.strip():
        semantic_query_embedding = model.encode(cleaned_query)

    semantic_scores = compute_semantic_scores_from_chroma(semantic_query_embedding, collection)
    token_scores = compute_token_scores_from_collection(cleaned_query, collection)

    all_pids = set(semantic_scores.keys()) | set(token_scores.keys())

    combined = {}
    for product_id in all_pids:
        sem = semantic_scores.get(product_id, {})
        s_sem = sem.get("semantic", 0.0)
        s_text = sem.get("text", "")

        matched, s_tok = token_scores.get(product_id, ([], 0.0))
        hybrid = alpha * s_sem + (1 - alpha) * s_tok

        combined[product_id] = {
            "hybrid":        hybrid,
            "semantic":      s_sem,
            "semantic_text": s_text,
            "token":         s_tok,
            "token_text":    ", ".join(sorted(matched)),
        }

    return combined


def compute_all_hybrid_scores_negation_aware(
    query: str,
    query_embedding,
    collections: dict,
    alpha: float = 0.5,
    model=None,
) -> dict[str, dict[str, dict]]:
    """
    Negation-aware equivalent of search_methods.compute_all_hybrid_scores.
    """
    all_scores = {}
    for category_name, collection in collections.items():
        category_scores = compute_category_scores_for_product_negation_aware(
            query, query_embedding, collection, alpha=alpha, model=model
        )
        for product_id, score_dict in category_scores.items():
            if product_id not in all_scores:
                all_scores[product_id] = {}
            if score_dict["hybrid"] > 0:
                all_scores[product_id][category_name] = score_dict

    return all_scores


def compute_all_category_weights_negation_aware(
    query: str,
    query_embedding,
    loaded_token_embeddings_dict: dict,
    loaded_chunk_embeddings_dict: dict,
    model=None,
) -> dict[str, float]:
    """
    Negation-aware equivalent of search_methods.compute_all_category_weights.
    Uses the cleaned query for the token_w component of each category's
    weight (so a negated term like "UK" cannot inflate a category's
    weight), and reuses the ORIGINAL search_methods.compute_category_weight
    unmodified — only the query text/embedding fed into it changes.
    """
    cleaned_query, negated_tokens = build_cleaned_query(query)

    semantic_query_embedding = query_embedding
    if model is not None and cleaned_query.strip():
        semantic_query_embedding = model.encode(cleaned_query)

    weights = {}
    for category_name in CATEGORY_TOKEN_SETS.keys():
        tokens = CATEGORY_TOKEN_SETS.get(category_name, [])
        token_embeddings = loaded_token_embeddings_dict.get(category_name, None)
        pp_chunks = loaded_chunk_embeddings_dict.get("product_performance", {})
        chunk_embeddings = [emb for embs in pp_chunks.values() for emb in embs]

        weights[category_name] = compute_category_weight(
            query=cleaned_query,
            query_embedding=semantic_query_embedding,
            semantic_embeddings=chunk_embeddings if category_name == "product_performance" else token_embeddings,
            token_set=tokens,
        )

    return weights


# ── Top-level negation-aware search ───────────────────────────────

def search_query_aware_negation_aware(
    query: str,
    query_embedding,
    collections: dict,
    loaded_token_embeddings_dict: dict,
    loaded_chunk_embeddings_dict: dict,
    top_k: int = 5,
    alpha: float = 0.5,
    weight_threshold: float = 0.05,
    model=None,
) -> tuple[list, float, dict]:
    """
    Negation-aware equivalent of search_methods.search_query_aware.

    Returns (results, weight_computation_s, negation_info), where
    negation_info = {"cleaned_query": str, "negated_tokens": set}
    so you can display exactly what was stripped, for the report.
    """
    import time
    import pandas as pd

    cleaned_query, negated_tokens = build_cleaned_query(query)

    all_scores = compute_all_hybrid_scores_negation_aware(
        query, query_embedding, collections, alpha=alpha, model=model
    )

    if not all_scores:
        return [], 0.0, {"cleaned_query": cleaned_query, "negated_tokens": negated_tokens}

    t_weight_start = time.perf_counter()
    category_weights = compute_all_category_weights_negation_aware(
        query=query,
        query_embedding=query_embedding,
        loaded_token_embeddings_dict=loaded_token_embeddings_dict,
        loaded_chunk_embeddings_dict=loaded_chunk_embeddings_dict,
        model=model,
    )
    weight_computation_s = time.perf_counter() - t_weight_start

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

    summary_df = pd.DataFrame(
        [{"rank": i + 1, "product_id": r["product_id"], "score": r["score"], "cat_scores": r["category_scores"]} for i, r in enumerate(results)]
    )

    negation_info = {"cleaned_query": cleaned_query, "negated_tokens": negated_tokens}
    return summary_df, results, weight_computation_s, negation_info