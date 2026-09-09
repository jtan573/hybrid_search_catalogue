"""
test_runner_v5.py

Three check types, each targeting a specific component of the three methods:

  token_check:
      Checks if the any_of tokens appear in the PRODUCT DATA field.
      Tests the token matching component present in hybrid and query_aware.
      Not meaningful for semantic-only (no token matching).
      Evaluated per product in top-k.

  semantic_check:
      Checks if the category fired in the pipeline (present in category_scores)
      AND the product data contains at least one of the any_of tokens.
      Tests the embedding component across all three methods.
      Includes both exact-match categories (guaranteed to fire if token matched)
      and synonym categories (tests if embedding bridged the gap).
      Evaluated per product in top-k.

  weight_check:
      Checks that relevant categories have a weight above a threshold,
      and that specified irrelevant categories stay below threshold.
      Only meaningful for query_aware (the only method with category weights).
      Evaluated at query level (weights are query-level, not per-product).
"""

from collections import defaultdict

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from property_checks import PROPERTY_CHECKS
import time


# ─── Field helpers ────────────────────────────────────────────────────────────

def _field_text(product_row: dict, category: str) -> str:
    val = product_row.get(category, "")
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).lower()


def _token_in_data(product_row: dict, category: str, any_of: list) -> bool:
    text = _field_text(product_row, category)
    return any(term.lower() in text for term in any_of)


def load_chunk_embeddings_from_chroma(collections: dict) -> dict[str, dict[str, list]]:
    """
    Returns {category_name: {pid (str): [np.ndarray, ...]}}
    """
    chunk_embeddings = {}

    for category_name, collection in collections.items():
        results = collection.get(include=["embeddings", "metadatas"])
        embeddings = results["embeddings"]
        metadatas  = results["metadatas"]

        if embeddings is None:
            print(f"Warning: no embeddings found for {category_name}")
            continue

        pid_map = defaultdict(list)
        for emb, meta in zip(embeddings, metadatas):
            pid = str(meta["product_id"])
            pid_map[pid].append(np.array(emb))

        chunk_embeddings[category_name] = pid_map
        print(f"Loaded {len(embeddings)} chunk embeddings for {category_name} "
              f"({len(pid_map)} unique products)")

    return chunk_embeddings


def load_token_embeddings(collections: dict) -> dict[str, np.ndarray]:
    token_embeddings = {}
    for category_name, _ in collections.items():
        path = f"category_profiles/{category_name}_embeddings.npy"
        if os.path.exists(path):
            token_embeddings[category_name] = np.load(path)
            print(f"Loaded token embeddings for {category_name}")
        else:
            print(f"Warning: no token embeddings found for {category_name}")
    return token_embeddings

def _summarize_category_scores(category_scores: dict) -> dict:
    """
    Collapse a product's per-category score dict into summary stats
    for the flat CSV (one row per product per query per method).
    """
    if not category_scores:
        return {
            "n_categories_fired": 0,
            "categories_fired": "",
            "avg_hybrid":   None,
            "avg_semantic": None,
            "avg_token":    None,
        }

    hybrids   = [s.get("hybrid", 0.0)   for s in category_scores.values()]
    semantics = [s.get("semantic", 0.0) for s in category_scores.values()]
    tokens    = [s.get("token", 0.0)    for s in category_scores.values()]

    return {
        "n_categories_fired": len(category_scores),
        "categories_fired":   ", ".join(sorted(category_scores.keys())),
        "avg_hybrid":         round(sum(hybrids) / len(hybrids), 4),
        "avg_semantic":       round(sum(semantics) / len(semantics), 4),
        "avg_token":          round(sum(tokens) / len(tokens), 4),
    }


# ─── Token check (product data) ───────────────────────────────────────────────

def evaluate_token_checks(check_def: dict, product_row: dict) -> dict:
    """
    Groups token_checks by 'concept'. A concept is satisfied if ANY of its
    categories' any_of tokens is found in the product's data for that
    category (OR within a concept). Coverage = satisfied_concepts / total_concepts,
    rather than the old flat per-category count — this stops queries with the
    same fact split across multiple categories from being penalized relative
    to queries that only test each fact once.
    """
    checks = check_def.get("token_checks", [])
    if not checks:
        return {"passed": None, "coverage": None, "n_passed": 0, "n_total": 0,
                "concept_results": {}, "details": ["No token checks defined"]}

    concept_results = {}
    details = []

    concepts = defaultdict(list)
    for c in checks:
        concepts[c.get("concept", c["category"])].append(c)

    for concept, entries in concepts.items():
        satisfied = False
        for c in entries:
            cat    = c["category"]
            any_of = c["any_of"]
            hit    = _token_in_data(product_row, cat, any_of)
            details.append(
                f"TOKEN [{concept}] {cat} any_of {any_of[:2]}{'...' if len(any_of) > 2 else ''}: "
                f"{'HIT' if hit else 'miss'}"
            )
            satisfied = satisfied or hit
        concept_results[concept] = satisfied

    n_total   = len(concept_results)
    n_passed  = sum(concept_results.values())
    coverage  = round(n_passed / n_total, 4) if n_total > 0 else None

    return {
        "passed":          coverage == 1.0 if coverage is not None else None,
        "coverage":        coverage,
        "n_passed":        n_passed,
        "n_total":         n_total,
        "concept_results": concept_results,
        "details":         details,
    }


# ─── Semantic check ───────────────────────────────────────────────────────────

def evaluate_semantic_checks(
    check_def: dict,
    product_row: dict,
    product_category_scores: dict,
) -> dict:
    """
    Groups semantic_checks by 'concept'. A concept is satisfied if ANY of its
    categories fired in the pipeline (present in category_scores) — OR within
    a concept, same logic as evaluate_token_checks. Coverage = satisfied
    concepts / total concepts.
    """
    sem_checks = check_def.get("semantic_checks", [])
    if not sem_checks:
        return {"coverage": None, "n_passed": 0, "n_expected": 0,
                "concept_results": {}, "details": ["No semantic checks defined"]}

    concept_results = {}
    details = []

    concepts = defaultdict(list)
    for c in sem_checks:
        concepts[c.get("concept", c["category"])].append(c)

    for concept, entries in concepts.items():
        satisfied = False
        for c in entries:
            cat  = c["category"]
            fired = cat in product_category_scores
            details.append(
                f"SEMANTIC [{concept}] {cat} | fired={fired} | {c.get('rationale','')[:70]}"
            )
            satisfied = satisfied or fired
        concept_results[concept] = satisfied

    n_expected = len(concept_results)
    n_passed   = sum(concept_results.values())
    coverage   = round(n_passed / n_expected, 4) if n_expected > 0 else None

    return {
        "coverage":        coverage,
        "n_passed":        n_passed,
        "n_expected":      n_expected,
        "concept_results": concept_results,
        "details":         details,
    }

# ─── Weight check (query-aware only) ─────────────────────────────────────────

def evaluate_weight_checks(
    check_def: dict,
    category_weights: dict,
    weight_threshold: float = 0.05,
) -> dict:
    """
    Only meaningful for query_aware. Reports exactly which categories
    matched/missed rather than just a pass rate:
      - relevant_matched / relevant_missed:   relevant_categories that
        did / didn't clear weight_threshold
      - irrelevant_ok / irrelevant_leaked:    irrelevant_categories that
        correctly stayed below / incorrectly leaked above weight_threshold
    """
    relevant   = check_def.get("relevant_categories",   [])
    irrelevant = check_def.get("irrelevant_categories", [])

    if not relevant and not irrelevant:
        return {
            "passed": None,
            "relevant_matched": [], "relevant_missed": [],
            "irrelevant_ok": [],    "irrelevant_leaked": [],
            "n_relevant_matched": 0, "n_relevant_total": 0,
            "n_irrelevant_ok": 0,    "n_irrelevant_total": 0,
            "details": ["No weight checks defined"],
        }

    details = []

    relevant_matched, relevant_missed = [], []
    for cat in relevant:
        w = category_weights.get(cat, 0.0)
        (relevant_matched if w >= weight_threshold else relevant_missed).append(cat)
        details.append(
            f"WEIGHT {cat} = {round(w,4)} >= {weight_threshold}: "
            f"{'PASS' if w >= weight_threshold else 'FAIL'}"
        )

    irrelevant_ok, irrelevant_leaked = [], []
    for cat in irrelevant:
        w = category_weights.get(cat, 0.0)
        (irrelevant_ok if w < weight_threshold else irrelevant_leaked).append(cat)
        details.append(
            f"WEIGHT {cat} = {round(w,4)} < {weight_threshold} (irrelevant): "
            f"{'PASS' if w < weight_threshold else 'FAIL'}"
        )

    n_passed = len(relevant_matched) + len(irrelevant_ok)
    n_total  = len(relevant) + len(irrelevant)
    passed   = (n_passed / n_total) >= 0.5 if n_total > 0 else None

    return {
        "passed":              passed,
        "relevant_matched":    relevant_matched,
        "relevant_missed":     relevant_missed,
        "irrelevant_ok":       irrelevant_ok,
        "irrelevant_leaked":   irrelevant_leaked,
        "n_relevant_matched":  len(relevant_matched),
        "n_relevant_total":    len(relevant),
        "n_irrelevant_ok":     len(irrelevant_ok),
        "n_irrelevant_total":  len(irrelevant),
        "details":             details,
    }


# ─── Single query runner ──────────────────────────────────────────────────────

def _dense_rank_top_k(raw_results: list, top_k: int) -> list:
    """
    Assign dense rank by score: products with the same (rounded) score
    share a rank instead of each consuming a separate slot. Keeps only
    products belonging to the first `top_k` distinct score groups.
    Assumes raw_results is already sorted by score descending, which
    all three search_* functions guarantee.
    Returns a list of (rank, product_dict) tuples.
    """
    ranked      = []
    seen_scores = []

    for r in raw_results:
        score = round(r.get("score", r.get("final_score", 0.0)), 4)
        if score not in seen_scores:
            if len(seen_scores) >= top_k:
                break
            seen_scores.append(score)
        rank = seen_scores.index(score) + 1
        ranked.append((rank, r))

    return ranked

def run_single_query_test(
    query: str,
    query_embedding: np.ndarray,
    collections: dict, 
    check_def: dict,
    product_lookup: dict,
    loaded_token_embeddings_dict: dict,
    loaded_chunk_embeddings_dict: dict,
    search_semantic_fn,
    search_hybrid_fn,
    search_query_aware_fn,
    top_k: int = 5,
    weight_threshold: float = 0.05,
) -> dict:

    methods = {
        "semantic":    search_semantic_fn,
        "hybrid":      search_hybrid_fn,
        "query_aware": search_query_aware_fn,
    }

    result = {
        "query":            query,
        "type":             check_def["type"],
        "expected_outcome": check_def.get("expected_outcome", "relevant_match"),
        "methods":          {},
    }

    for method_name, fn in methods.items():
        t0 = time.perf_counter()

        if method_name == "semantic":
            raw_results = fn(query_embedding, collections, loaded_chunk_embeddings_dict)
            weight_computation_s = None
        elif method_name == "hybrid":
            raw_results = fn(query, query_embedding, collections)
            weight_computation_s = None
        else:
            raw_results, weight_computation_s = fn(query, query_embedding, collections, loaded_token_embeddings_dict, loaded_chunk_embeddings_dict)

        method_time_s = time.perf_counter() - t0 
        ranked_top = _dense_rank_top_k(raw_results, top_k) if raw_results else []

        # ── Extract query-level category weights (query_aware only) ──
        category_weights = {}
        if method_name == "query_aware" and ranked_top:
            first_result = ranked_top[0][1]
            for cat, s in first_result.get("category_scores", {}).items():
                category_weights[cat] = s.get("weight", 0.0)

        # ── Weight check (query-level) ────────────────────────────────
        weight_eval = evaluate_weight_checks(
            check_def, category_weights, weight_threshold=weight_threshold
        ) if method_name == "query_aware" else {"passed": None, "details": ["N/A for this method"]}

        # ── Per-product checks ────────────────────────────────────────
        product_evals = []
        for rank, r in ranked_top:
            pid             = str(r.get("product_id", ""))
            product_row     = product_lookup.get(pid, {})
            category_scores = r.get("category_scores", {})

            token_eval    = evaluate_token_checks(check_def, product_row)
            semantic_eval = evaluate_semantic_checks(check_def, product_row, category_scores)
            cat_summary   = _summarize_category_scores(category_scores)

            product_evals.append({
                "rank":                rank,
                "product_id":          pid,
                "score":               r.get("score", r.get("final_score", None)),
                "token_check":         token_eval,
                "semantic_check":      semantic_eval,
                "category_summary":    cat_summary,
                "category_scores_raw": category_scores,
            })

        def precision(evals, key):
            eligible = [p for p in evals if p[key]["passed"] is not None]
            if not eligible:
                return None
            return round(sum(1 for p in eligible if p[key]["passed"]) / len(eligible), 4)

        def mean_coverage(evals, key):
            vals = [p[key]["coverage"] for p in evals if p[key]["coverage"] is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        result["methods"][method_name] = {
            "n_results":               len(ranked_top),
            "time_s":                  round(method_time_s, 4),
            "weight_computation_s":    round(weight_computation_s, 4) if weight_computation_s is not None else None,
            "token_precision_at_k":    precision(product_evals, "token_check"),
            "token_coverage_mean":     mean_coverage(product_evals, "token_check"),
            "semantic_coverage_mean":  mean_coverage(product_evals, "semantic_check"),
            "weight_check":            weight_eval,
            "products":                product_evals,
        }

    return result


# ─── Full suite ───────────────────────────────────────────────────────────────

def run_all_tests(
    product_lookup: dict,
    model,
    collections: dict,
    loaded_token_embeddings_dict: dict,
    loaded_chunk_embeddings_dict: dict,
    search_semantic_fn,
    search_hybrid_fn,
    search_query_aware_fn,
    top_k: int = 5,
    weight_threshold: float = 0.05,
    output_dir: str = "test_run_outputs",
) -> tuple:

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results  = []
    summary_rows = []
    detail_rows  = []          # NEW

    for i, check_def in enumerate(PROPERTY_CHECKS, start=1):
        query = check_def["query"]

        encode_start = time.perf_counter()
        query_embedding = model.encode(query)
        encode_time_s = time.perf_counter() - encode_start
        print(f"[{i}/{len(PROPERTY_CHECKS)}] {query}")

        result = run_single_query_test(
            query=query,
            query_embedding=query_embedding,
            check_def=check_def,
            collections=collections,
            product_lookup=product_lookup,
            loaded_token_embeddings_dict=loaded_token_embeddings_dict,
            loaded_chunk_embeddings_dict=loaded_chunk_embeddings_dict,
            search_semantic_fn=search_semantic_fn,
            search_hybrid_fn=search_hybrid_fn,
            search_query_aware_fn=search_query_aware_fn,
            top_k=top_k,
            weight_threshold=weight_threshold,
        )
        result["encode_time_s"] = encode_time_s
        all_results.append(result)

        for method_name, method_data in result["methods"].items():
            row = {
                "query":            query,
                "type":             result["type"],
                "expected_outcome": result["expected_outcome"],
                "method":           method_name,
                "encode_time_s":    result["encode_time_s"],
            }
            if "error" in method_data:
                row["error"] = method_data["error"]
            else:
                row["n_results"]               = method_data["n_results"]
                row["time_s"]                  = method_data.get("time_s")            # NEW — total method wall-clock
                row["weight_computation_s"]    = method_data.get("weight_computation_s")
                row["token_precision_at_k"]    = method_data["token_precision_at_k"]
                row["token_coverage_mean"]     = method_data["token_coverage_mean"]
                row["semantic_coverage_mean"]  = method_data["semantic_coverage_mean"]   # CHANGED
                wc = method_data["weight_check"]
                row["weight_check_passed"]      = wc.get("passed")
                row["weight_relevant_matched"]  = ", ".join(wc.get("relevant_matched",  []))
                row["weight_relevant_missed"]   = ", ".join(wc.get("relevant_missed",   []))
                row["weight_irrelevant_ok"]     = ", ".join(wc.get("irrelevant_ok",     []))
                row["weight_irrelevant_leaked"] = ", ".join(wc.get("irrelevant_leaked", []))
            summary_rows.append(row)

            # ── one row per distinct rank, ties collapsed into shared_with_pids ──
            products_by_rank = defaultdict(list)
            for p in method_data.get("products", []):
                products_by_rank[p["rank"]].append(p)

            for rank in sorted(products_by_rank.keys()):
                group = products_by_rank[rank]
                p  = group[0]
                cs = p["category_summary"]
                other_pids = [g["product_id"] for g in group[1:]]

                detail_rows.append({
                    "query":                     query,
                    "type":                      result["type"],
                    "expected_outcome":          result["expected_outcome"],
                    "method":                    method_name,
                    "rank":                      rank,
                    "product_id":                p["product_id"],
                    "shared_with_pids":          ", ".join(other_pids),
                    "n_tied":                    len(group),
                    "score":                     p["score"],
                    "n_categories_fired":        cs["n_categories_fired"],
                    "categories_fired":          cs["categories_fired"],
                    "avg_hybrid":                cs["avg_hybrid"],
                    "avg_semantic":              cs["avg_semantic"],
                    "avg_token":                 cs["avg_token"],
                    "token_check_coverage":      p["token_check"]["coverage"],
                    "token_concepts_missed":     ", ".join(k for k, v in p["token_check"]["concept_results"].items() if not v),      # NEW
                    "semantic_check_coverage":   p["semantic_check"]["coverage"],
                    "semantic_concepts_missed":  ", ".join(k for k, v in p["semantic_check"]["concept_results"].items() if not v),   # NEW
                    "semantic_check_n_passed":   p["semantic_check"]["n_passed"],
                    "semantic_check_n_expected": p["semantic_check"]["n_expected"],
                    "category_scores_json":      json.dumps(p["category_scores_raw"]),
                })

    summary_df   = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, f"test_summary_{timestamp}.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary CSV → {summary_path}")

    detail_df   = pd.DataFrame(detail_rows)                                    # NEW
    detail_path = os.path.join(output_dir, f"test_topk_detail_{timestamp}.csv")  # NEW
    detail_df.to_csv(detail_path, index=False)                                 # NEW
    print(f"Top-K detail CSV → {detail_path}")                                 # NEW

    json_path = os.path.join(output_dir, f"test_full_results_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Full JSON   → {json_path}")

    return summary_df, detail_df, all_results          # CHANGED — now 3 return values


# ─── Progression report ───────────────────────────────────────────────────────

def print_progression_report(summary_df: pd.DataFrame):
    relevant  = summary_df[summary_df["expected_outcome"] == "relevant_match"]
    col_order = ["semantic", "hybrid", "query_aware"]

    print("\n" + "=" * 70)
    print("PROGRESSION REPORT")
    print("=" * 70)

    print("\n── TOKEN PRECISION@K")
    print("   (product data contains expected token — hybrid/query_aware only)")
    print("   Expect: N/A for semantic | improvement hybrid→query_aware")
    pivot = relevant.pivot_table(
        index="type", columns="method", values="token_precision_at_k", aggfunc="mean"
    )
    cols = [c for c in col_order if c in pivot.columns]
    print(pivot[cols].round(4).to_string())

    print("\n── TOKEN COVERAGE MEAN")
    print("   (fraction of expected token categories matched in product data)")
    pivot2 = relevant.pivot_table(
        index="type", columns="method", values="token_coverage_mean", aggfunc="mean"
    )
    print(pivot2[cols].round(4).to_string())

    print("\n── SEMANTIC COVERAGE MEAN")
    print("   (mean fraction of expected categories that fired — no pass/fail cutoff)")
    print("   Expect: baseline from semantic | improvement hybrid→query_aware")
    pivot3 = relevant.pivot_table(
        index="type", columns="method", values="semantic_coverage_mean", aggfunc="mean"
    )
    print(pivot3[cols].round(4).to_string())

    print("\n── WEIGHT CHECK PASS RATE (query_aware only)")
    print("   (relevant categories above threshold, irrelevant below)")
    qaware = relevant[relevant["method"] == "query_aware"]
    if not qaware.empty:
        print(qaware.groupby("type")["weight_check_passed"].mean().round(4).to_string())
        print(f"  Overall: {qaware['weight_check_passed'].mean():.4f}")

    poor = summary_df[summary_df["expected_outcome"] == "poor_match"]
    if not poor.empty:
        print("\n── POOR_MATCH (low scores = correct)")
        print(poor[["query", "method", "token_precision_at_k", "semantic_coverage_mean"]]
              .to_string(index=False))


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from search_methods import search_semantic, search_hybrid, search_query_aware
    import chromadb
    from sentence_transformers import SentenceTransformer

    model  = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path="./chroma_db")
    collections = {
        name: client.get_or_create_collection(name=name)
        for name in [
            "epd", "materials", "certifications", "standards",
            "tags", "scores", "attributes",
            "product_compliance", "product_core",
            "product_properties", "product_performance"
        ]
    }

    LOADED_CHUNK_EMBEDDINGS = load_chunk_embeddings_from_chroma(collections)
    LOADED_TOKEN_EMBEDDINGS = load_token_embeddings(collections)

    combined_df = pd.read_csv("processed_chunks/combined_chunks.csv")
    combined_df["product_id"] = combined_df["product_id"].astype(str)
    product_lookup = combined_df.set_index("product_id").to_dict(orient="index")

    summary_df, detail_df, all_results = run_all_tests(   # was: summary_df, all_results = ...
        product_lookup=product_lookup,
        model=model,
        collections=collections,
        loaded_token_embeddings_dict=LOADED_TOKEN_EMBEDDINGS,
        loaded_chunk_embeddings_dict=LOADED_CHUNK_EMBEDDINGS,
        search_semantic_fn=search_semantic,
        search_hybrid_fn=search_hybrid,
        search_query_aware_fn=search_query_aware,
        top_k=5,
        weight_threshold=0.05,
    )
    print_progression_report(summary_df)
    print("Import and call run_all_tests() with your search functions and product_lookup.")
