"""
extensions.py

Presentation helpers for query-aware search results, plus a proposed
extension that re-sorts an already-selected top-k by product quality.
The original search_query_aware and compute_final_score are untouched —
this only operates on the results after they're returned.
"""

import pandas as pd


# ────────────────────────────────────────────────────────────────────
# ------- EXTENSION 1: Quality scoring / Implicit Expectations -------
# ────────────────────────────────────────────────────────────────────

TIER_SCORE_MAP = {
    "poor":        0.0,
    "fair":        0.2,
    "good":        0.4,
    "very good":   0.6,
    "excellent":   0.8,
    "exceptional": 1.0,
}


def compute_quality_score(
    product_id: str,
    attributes_df: pd.DataFrame,
    prior: float = 0.5,
    pseudo_count: float = 2.0,
) -> tuple[float, int]:
    """
    Returns (quality_score, n_attributes_populated) for a given product,
    using Bayesian shrinkage toward `prior` so sparse ratings don't look
    as confident as well-corroborated ones.
        shrunk = (n * raw_mean + pseudo_count * prior) / (n + pseudo_count)
    """
    rows = attributes_df[attributes_df["product_id"].astype(str) == product_id]
    scores = [TIER_SCORE_MAP[r] for r in rows["rating"] if TIER_SCORE_MAP.get(r) is not None]

    n = len(scores)
    raw_mean = sum(scores) / n if n > 0 else prior
    shrunk_score = (n * raw_mean + pseudo_count * prior) / (n + pseudo_count)

    return shrunk_score, n


# ── v1: re-sort an existing top-k by quality ─────

def search_query_aware_sorted_by_quality(
    query: str,
    query_embedding,
    collections: dict,
    loaded_token_embeddings_dict: dict,
    loaded_chunk_embeddings_dict: dict,
    attributes_df: pd.DataFrame,
    search_query_aware_fn,     # your ORIGINAL, unmodified search_query_aware
    top_k: int = 5,
    alpha: float = 0.5,
    weight_threshold: float = 0.05,
) -> tuple[list, float]:
    """
    Runs the original, unmodified search_query_aware to select the top-k
    purely by relevance (quality plays no role in selection), then re-sorts
    that same top-k set by quality_score, descending.

    This only changes presentation order among already-selected results —
    it cannot surface a product that relevance alone excluded from top-k.
    """
    results, weight_computation_s = search_query_aware_fn(
        query=query,
        query_embedding=query_embedding,
        collections=collections,
        loaded_token_embeddings_dict=loaded_token_embeddings_dict,
        loaded_chunk_embeddings_dict=loaded_chunk_embeddings_dict,
        top_k=top_k,
        alpha=alpha,
        weight_threshold=weight_threshold,
    )

    for r in results:
        quality_score, n_attrs = compute_quality_score(r["product_id"], attributes_df)
        r["quality_score"] = round(quality_score, 4)
        r["n_attributes_rated"] = n_attrs

    results_sorted = sorted(results, key=lambda r: r["quality_score"], reverse=True)

    return results_sorted, weight_computation_s

# ── v2: bounded quality boost (new, separate from v1) ──

def search_query_aware_bounded_quality_boost(
    query: str,
    query_embedding,
    collections: dict,
    loaded_token_embeddings_dict: dict,
    loaded_chunk_embeddings_dict: dict,
    attributes_df: pd.DataFrame,
    search_query_aware_fn, 
    top_k: int = 5,
    alpha: float = 0.5,           # relevance blend weight, passed through unchanged
    weight_threshold: float = 0.05,
    quality_boost_alpha: float = 0.15,   # bounds how much quality can reorder results
) -> tuple[list, float]:
    """
    Runs the original, unmodified search_query_aware to select the top-k
    purely by relevance (quality plays no role in selection), then re-ranks
    that same top-k set using a bounded multiplicative quality boost:

        final_score = relevance_score * (1 + quality_boost_alpha * quality_score)

    quality_score is already in [0, 1] (see compute_quality_score's shrinkage
    toward a 0.5 prior), so the boost factor is bounded to
    [1, 1 + quality_boost_alpha]. This guarantees quality can only re-order
    products whose relevance scores are within roughly quality_boost_alpha
    of each other (proportionally) — it cannot override a large relevance gap.

    This is distinct from search_query_aware_sorted_by_quality, which sorts
    purely by quality_score and is kept unmodified for side-by-side
    comparison in the write-up.

    Still only changes ranking among already-selected results — it cannot
    surface a product that relevance alone excluded from top-k.
    """
    results, weight_computation_s = search_query_aware_fn(
        query=query,
        query_embedding=query_embedding,
        collections=collections,
        loaded_token_embeddings_dict=loaded_token_embeddings_dict,
        loaded_chunk_embeddings_dict=loaded_chunk_embeddings_dict,
        top_k=top_k,
        alpha=alpha,
        weight_threshold=weight_threshold,
    )

    for r in results:
        quality_score, n_attrs = compute_quality_score(r["product_id"], attributes_df)
        r["quality_score"] = round(quality_score, 4)
        r["n_attributes_rated"] = n_attrs

        boost_factor = 1 + quality_boost_alpha * quality_score
        r["boost_factor"] = round(boost_factor, 4)
        r["final_score"] = round(r["score"] * boost_factor, 6)

    results_sorted = sorted(results, key=lambda r: r["final_score"], reverse=True)

    return results_sorted, weight_computation_s


# ── Display helpers ───────────────────────────────────────────────

def compare_ranking_strategies(
    query: str,
    query_embedding,
    collections: dict,
    loaded_token_embeddings_dict: dict,
    loaded_chunk_embeddings_dict: dict,
    attributes_df: pd.DataFrame,
    search_query_aware_fn,
    top_k: int = 5,
    alpha: float = 0.5,
    weight_threshold: float = 0.05,
    quality_boost_alpha: float = 0.15,
) -> pd.DataFrame:
    """
    Runs search_query_aware_fn ONCE, then re-ranks that single candidate
    set three ways: relevance-only (as returned), pure-quality-sort, and
    bounded-quality-boost. This guarantees all three strategies compare
    the same underlying candidates, even if search_query_aware_fn returns
    more than top_k items due to ties at the score cutoff.
    """
    base_results, weight_computation_s = search_query_aware_fn(
        query=query,
        query_embedding=query_embedding,
        collections=collections,
        loaded_token_embeddings_dict=loaded_token_embeddings_dict,
        loaded_chunk_embeddings_dict=loaded_chunk_embeddings_dict,
        top_k=top_k,
        alpha=alpha,
        weight_threshold=weight_threshold,
    )

    if len(base_results) > top_k:
        print(f"[NOTE] search_query_aware_fn returned {len(base_results)} "
              f"results for top_k={top_k} due to ties at the score cutoff. ")

    # attach quality_score to every candidate once, reuse everywhere
    for r in base_results:
        quality_score, n_attrs = compute_quality_score(r["product_id"], attributes_df)
        r["quality_score"] = round(quality_score, 4)
        r["n_attributes_rated"] = n_attrs

    # Strategy 1 — relevance-only, as returned (already sorted by score)
    relevance_only = sorted(base_results, key=lambda r: r["score"], reverse=True)

    # Strategy 2 — pure quality sort
    pure_quality = sorted(base_results, key=lambda r: r["quality_score"], reverse=True)

    # Strategy 3 — bounded quality boost
    bounded_boost = []
    for r in base_results:
        boost_factor = 1 + quality_boost_alpha * r["quality_score"]
        r["boost_factor"] = round(boost_factor, 4)
        r["final_score"] = round(r["score"] * boost_factor, 6)
        bounded_boost.append(r)
    bounded_boost = sorted(bounded_boost, key=lambda r: r["final_score"], reverse=True)

    rows = []
    for rank in range(len(base_results)):
        rows.append({
            "rank": rank + 1,
            "relevance_only_id": relevance_only[rank]["product_id"] if rank < len(relevance_only) else None,
            "relevance_only_score": relevance_only[rank]["score"] if rank < len(relevance_only) else None,
            "pure_quality_id": pure_quality[rank]["product_id"] if rank < len(pure_quality) else None,
            "pure_quality_score": pure_quality[rank]["quality_score"] if rank < len(pure_quality) else None,
            "bounded_boost_id": bounded_boost[rank]["product_id"] if rank < len(bounded_boost) else None,
            "bounded_boost_final_score": bounded_boost[rank]["final_score"] if rank < len(bounded_boost) else None,
        })

    return pd.DataFrame(rows)

def print_result_block(result: dict, rank: int = None):
    header = f"Rank {rank} — " if rank is not None else ""
    print("=" * 80)
    print(f"{header}Product ID: {result['product_id']}  |  Relevance score: {result['score']}")
    if "quality_score" in result:
        print(f"Quality score: {result['quality_score']}  "
              f"(based on {result['n_attributes_rated']} rated attribute(s))")
    print("=" * 80)

    cat_df = pd.DataFrame.from_dict(result["category_scores"], orient="index")
    cat_df = cat_df.sort_values("weight", ascending=False)
    print(cat_df)
    print()


def print_results(results: list):
    for i, r in enumerate(results, start=1):
        print_result_block(r, rank=i)


def results_to_df(results: list) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "rank": i + 1,
            "product_id": r["product_id"],
            "relevance_score": r["score"],
            "quality_score": r.get("quality_score"),
            "n_attributes_rated": r.get("n_attributes_rated"),
        }
        for i, r in enumerate(results)
    ])

