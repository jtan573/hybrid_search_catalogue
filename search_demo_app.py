"""
search_demo_app.py

Lightweight Streamlit UI for demoing search_semantic / search_hybrid /
search_query_aware side by side, using existing pipeline
(search_methods.py + the loaders in test_runner_v5.py).

Run with:
    streamlit run search_demo_app.py

Expects to sit in the same directory as:
    search_methods.py
    test_runner.py
    category_profile_tokens.py
    processed_chunks/combined_chunks.csv
    chroma_db/                          
"""

import time
import pandas as pd
import streamlit as st

from search_methods import search_semantic, search_hybrid, search_query_aware
from test_runner import load_chunk_embeddings_from_chroma, load_token_embeddings

st.set_page_config(page_title="Search Method Comparison", layout="wide")

METHOD_LABELS = {
    "semantic":    "1 · Semantic only",
    "hybrid":      "2 · Hybrid (token + semantic)",
    "query_aware": "3 · Query-aware (weighted categories)",
}

NAME_FIELD_CANDIDATES = ["product_name", "name", "title", "product_title"]

PRESET_QUERIES = [
    "nylon polyamide flooring",
    "glass wool acoustic panel 40mm residential Sweden",
    "I need a carpet that is easy to clean and durable",
]


# ─────────────────────────────────────────────────────────────────────────────
# Cached resource loading — runs once per session, not on every search
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model + chroma collections...")
def load_pipeline(chroma_path: str, collection_names: tuple, csv_path: str):
    import chromadb
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=chroma_path)
    collections = {
        name: client.get_or_create_collection(name=name)
        for name in collection_names
    }

    loaded_chunk_embeddings = load_chunk_embeddings_from_chroma(collections)
    loaded_token_embeddings = load_token_embeddings(collections)

    combined_df = pd.read_csv(csv_path)
    combined_df["product_id"] = combined_df["product_id"].astype(str)
    product_lookup = combined_df.set_index("product_id").to_dict(orient="index")

    return model, collections, loaded_chunk_embeddings, loaded_token_embeddings, product_lookup


def product_display_name(pid: str, product_lookup: dict) -> str:
    row = product_lookup.get(pid, {})
    for field in NAME_FIELD_CANDIDATES:
        if field in row and pd.notna(row[field]) and str(row[field]).strip():
            return str(row[field])
    return pid


def dense_rank(raw_results: list, top_k: int) -> list:
    ranked, seen = [], []
    for r in raw_results:
        score = round(r.get("score", 0.0), 4)
        if score not in seen:
            if len(seen) >= top_k:
                break
            seen.append(score)
        ranked.append((seen.index(score) + 1, r))
    return ranked


def category_scores_df(category_scores: dict) -> pd.DataFrame:
    if not category_scores:
        return pd.DataFrame(columns=["category", "hybrid", "semantic", "token", "weight"])
    rows = []
    for cat, s in sorted(category_scores.items(), key=lambda kv: kv[1].get("hybrid", 0), reverse=True):
        rows.append({
            "category": cat,
            "hybrid":   s.get("hybrid", 0.0),
            "semantic": s.get("semantic", 0.0),
            "token":    s.get("token", 0.0),
            "weight":   s.get("weight", 1.0),
        })
    return pd.DataFrame(rows)


def build_overview_table(results_by_method: dict, top_k: int, product_lookup: dict) -> pd.DataFrame:
    """
    One row per rank, one column per method, showing 'name (score)' —
    a quick at-a-glance comparison before the detailed breakdown.
    """
    ranked_by_method = {
        method: dense_rank(results, top_k) for method, results in results_by_method.items()
    }

    rows = []
    for rank in range(1, top_k + 1):
        row = {"Rank": rank}
        for method, ranked in ranked_by_method.items():
            match = next((r for rk, r in ranked if rk == rank), None)
            if match is None:
                row[METHOD_LABELS[method]] = ""
            else:
                pid = str(match["product_id"])
                name = product_display_name(pid, product_lookup)
                row[METHOD_LABELS[method]] = f"{name}  ({match['score']})"
        rows.append(row)

    return pd.DataFrame(rows).set_index("Rank")


def render_method_column(col, method_key, raw_results, top_k, product_lookup, elapsed_s):
    with col:
        st.subheader(METHOD_LABELS[method_key])
        st.caption(f"{elapsed_s*1000:.0f} ms")

        if not raw_results:
            st.info("No results.")
            return

        ranked = dense_rank(raw_results, top_k)
        for rank, r in ranked:
            pid = str(r["product_id"])
            name = product_display_name(pid, product_lookup)
            st.markdown(f"**#{rank} · {name}**  \n`{pid}` — score `{r['score']}`")
            with st.expander("category breakdown"):
                df = category_scores_df(r.get("category_scores", {}))
                if df.empty:
                    st.write("No categories fired.")
                else:
                    st.dataframe(df, hide_index=True, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — config
# ─────────────────────────────────────────────────────────────────────────────

st.title("Search Method Comparison")
st.caption("Run one query through semantic, hybrid, and query-aware search side by side.")

with st.sidebar:
    st.header("Configuration")
    chroma_path = st.text_input("Chroma DB path", value="./chroma_db")
    csv_path = st.text_input("Product CSV path", value="processed_chunks/combined_chunks.csv")
    collection_names_input = st.text_area(
        "Collection names (one per line)",
        value="\n".join([
            "epd", "materials", "certifications", "standards", "tags",
            "scores", "attributes", "product_compliance", "product_core",
            "product_properties", "product_performance",
        ]),
        height=180,
    )
    top_k = st.slider("Top K", min_value=1, max_value=15, value=5)
    alpha = st.slider("Alpha (semantic vs token, hybrid/query-aware)", 0.0, 1.0, 0.5, 0.05)
    weight_threshold = st.slider("Weight threshold (query-aware)", 0.0, 0.5, 0.05, 0.01)

    load_clicked = st.button("Load / reload pipeline", type="primary")

collection_names = tuple(
    line.strip() for line in collection_names_input.splitlines() if line.strip()
)

if load_clicked:
    load_pipeline.clear()

if "pipeline_loaded" not in st.session_state:
    st.session_state["pipeline_loaded"] = False

try:
    model, collections, loaded_chunk_embeddings, loaded_token_embeddings, product_lookup = load_pipeline(
        chroma_path, collection_names, csv_path
    )
    st.session_state["pipeline_loaded"] = True
except Exception as e:
    st.session_state["pipeline_loaded"] = False
    st.error(f"Could not load pipeline: {e}")
    st.stop()

st.success(f"Pipeline ready — {len(collections)} collections, {len(product_lookup)} products.")

# ─────────────────────────────────────────────────────────────────────────────
# Query input
# ─────────────────────────────────────────────────────────────────────────────

if "query_input" not in st.session_state:
    st.session_state["query_input"] = ""

preset_col, custom_col = st.columns([1, 2])
with preset_col:
    preset_choice = st.selectbox(
        "Preset queries", options=["— custom —"] + PRESET_QUERIES, index=0
    )
    if preset_choice != "— custom —":
        st.session_state["query_input"] = preset_choice

with custom_col:
    query = st.text_input(
        "Query", key="query_input", placeholder="e.g. FSC certified oak flooring low VOC"
    )

run_clicked = st.button("Search", type="primary")

if run_clicked and query.strip():
    query_embedding = model.encode(query)

    t0 = time.perf_counter()
    semantic_results = search_semantic(query_embedding, collections, loaded_chunk_embeddings, top_k=top_k)
    t_semantic = time.perf_counter() - t0

    t0 = time.perf_counter()
    hybrid_results = search_hybrid(query, query_embedding, collections, top_k=top_k, alpha=alpha)
    t_hybrid = time.perf_counter() - t0

    t0 = time.perf_counter()
    query_aware_results, weight_computation_s = search_query_aware(
        query, query_embedding, collections,
        loaded_token_embeddings, loaded_chunk_embeddings,
        top_k=top_k, alpha=alpha, weight_threshold=weight_threshold,
    )
    t_query_aware = time.perf_counter() - t0

    st.divider()
    st.subheader("Overview — top product IDs by method")
    overview_df = build_overview_table(
        {"semantic": semantic_results, "hybrid": hybrid_results, "query_aware": query_aware_results},
        top_k, product_lookup,
    )
    st.dataframe(overview_df, use_container_width=True)

    st.divider()
    col1, col2, col3 = st.columns(3)
    render_method_column(col1, "semantic", semantic_results, top_k, product_lookup, t_semantic)
    render_method_column(col2, "hybrid", hybrid_results, top_k, product_lookup, t_hybrid)
    render_method_column(col3, "query_aware", query_aware_results, top_k, product_lookup, t_query_aware)

    # Query-aware category weights — the thing unique to method 3
    st.divider()
    st.subheader("Query-aware category weights")
    st.caption(f"Weight computation took {weight_computation_s*1000:.0f} ms")
    if query_aware_results:
        first_result_cats = query_aware_results[0].get("category_scores", {})
        weights = {cat: s.get("weight", 0.0) for cat, s in first_result_cats.items()}
        if weights:
            weights_df = pd.DataFrame(
                sorted(weights.items(), key=lambda kv: kv[1], reverse=True),
                columns=["category", "weight"],
            ).set_index("category")
            st.bar_chart(weights_df)
        else:
            st.info("No category weights available for this query.")
    else:
        st.info("No query-aware results to derive weights from.")

elif run_clicked:
    st.warning("Enter a query first.")