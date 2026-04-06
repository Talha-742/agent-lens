"""
app.py — AgentLens Streamlit UI
Run with:  streamlit run app.py
"""
from __future__ import annotations

import time
import pandas as pd
import streamlit as st

from config import APP_TITLE, APP_SUBTITLE, APP_VERSION, COST_COLORS, score_color, OLLAMA_MODEL
from agent_core import (
    get_recommendations,
    recommendations_to_dataframe,
    check_openai_client_status,
)
from ollama_utils import (
    list_local_models,
    get_ollama_status,
    test_model,
    score_local_model_for_workflow,
)

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgentLens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Global font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Header gradient */
.main-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #0f766e 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    color: white;
}
.main-header h1 { font-size: 2.4rem; font-weight: 700; margin: 0; }
.main-header p  { font-size: 1rem; opacity: 0.85; margin: 0.4rem 0 0; }

/* Model card */
.model-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.model-card:hover { border-color: #0f766e; }
.model-card h3 { color: #e2e8f0; margin: 0 0 0.3rem; font-size: 1.15rem; font-weight: 600; }
.model-card .provider { color: #94a3b8; font-size: 0.82rem; margin-bottom: 0.7rem; }
.model-card .desc { color: #cbd5e1; font-size: 0.9rem; line-height: 1.5; }

/* Badge */
.badge {
    display: inline-block;
    padding: 0.18rem 0.65rem;
    border-radius: 999px;
    font-size: 0.73rem;
    font-weight: 600;
    color: #fff;
    margin-right: 0.35rem;
}

/* Score ring */
.score-ring {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 52px; height: 52px;
    border-radius: 50%;
    border: 3px solid;
    font-size: 1rem;
    font-weight: 700;
    color: white;
}

/* Section title */
.section-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 1.5rem 0 0.8rem;
    border-left: 4px solid #0f766e;
    padding-left: 0.75rem;
}

/* Status pill */
.status-ok  { color: #22c55e; font-weight: 600; }
.status-err { color: #ef4444; font-weight: 600; }

/* Comparison table */
.comparison-table th {
    background: #0f172a !important;
    color: #94a3b8 !important;
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 0.05em;
}
</style>
""", unsafe_allow_html=True)


# ─── Session state ────────────────────────────────────────────────────────────
if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "local_models" not in st.session_state:
    st.session_state.local_models = []


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"## 🔍 {APP_TITLE}")
    st.caption(f"v{APP_VERSION}")
    st.divider()

    # System status
    st.markdown("### ⚙️ System Status")

    ollama_status = get_ollama_status()
    client_status = check_openai_client_status()

    if ollama_status["running"]:
        st.markdown(f'<span class="status-ok">{ollama_status["message"]}</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="status-err">{ollama_status["message"]}</span>', unsafe_allow_html=True)

    if client_status["connected"]:
        st.markdown(f'<span class="status-ok">{client_status["message"]}</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="status-err">{client_status["message"]}</span>', unsafe_allow_html=True)

    st.caption(f"Active model: **{OLLAMA_MODEL}**")
    st.divider()

    # Search history
    st.markdown("### 🕘 Search History")
    if st.session_state.search_history:
        for i, q in enumerate(reversed(st.session_state.search_history[-8:]), 1):
            st.markdown(f"`{i}.` {q[:55]}{'…' if len(q) > 55 else ''}")
    else:
        st.caption("No searches yet.")
    st.divider()

    # How it works
    with st.expander("ℹ️ How It Works"):
        st.markdown("""
1. **You describe** your agentic AI workflow in plain language.
2. **AgentLens searches** the web (DuckDuckGo) for up-to-date LLM info.
3. **Qwen3 (via Ollama)** analyses the results and produces structured recommendations.
4. **Local Ollama models** are listed and scored against your workflow.
5. **A comparison table** lets you evaluate all options side-by-side.
        """)


# ─── Main header ──────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
    <h1>{APP_TITLE}</h1>
    <p>{APP_SUBTITLE} — powered by Ollama + DuckDuckGo Search</p>
</div>
""", unsafe_allow_html=True)


# ─── Query input ──────────────────────────────────────────────────────────────
EXAMPLE_QUERIES = [
    "I'm building a marketing automation agent that handles campaign creation, audience targeting, A/B test analysis, and report generation.",
    "I need an LLM for a customer support agentic workflow with tool calling, RAG, and multi-turn memory.",
    "Which LLMs support tool calling for a coding agent that writes, tests, and deploys code?",
    "Best models for a financial analysis agent that reads PDFs, runs calculations, and generates reports.",
]

col_q, col_ex = st.columns([3, 1])
with col_q:
    user_query = st.text_area(
        "📝 Describe your agentic AI workflow",
        height=130,
        placeholder=EXAMPLE_QUERIES[0],
        help="Be specific: mention tasks, tools, data types, scale, and any special requirements.",
    )
with col_ex:
    st.markdown("**💡 Example queries**")
    for eq in EXAMPLE_QUERIES[:3]:
        if st.button(eq[:52] + "…", key=eq, use_container_width=True):
            st.session_state["_inject_query"] = eq
            st.rerun()

# Handle injected example query
if "_inject_query" in st.session_state:
    user_query = st.session_state.pop("_inject_query")

search_btn = st.button("🔍 Search LLMs", type="primary", use_container_width=False)

st.divider()


# ─── Helper: render a single model card ───────────────────────────────────────
def render_model_card(rec: dict) -> None:
    cost_color  = COST_COLORS.get(rec.get("cost_tier", "Unknown"), "#94a3b8")
    s_color     = score_color(float(rec.get("suitability_score", 5)))
    tool_color  = "#22c55e" if rec.get("tool_calling_support") == "Yes" else (
                  "#f59e0b" if rec.get("tool_calling_support") == "Partial" else "#ef4444")

    with st.container():
        c1, c2 = st.columns([10, 1])
        with c1:
            st.markdown(f"""
<div class="model-card">
  <h3>#{rec.get('rank','')} {rec.get('model_name','')}</h3>
  <div class="provider">🏢 {rec.get('provider','')} &nbsp;|&nbsp; 📐 {rec.get('parameters','')} params &nbsp;|&nbsp; 🖥️ {rec.get('model_type','')}</div>
  <div class="desc">{rec.get('description','')}</div>
  <br/>
  <span class="badge" style="background:{cost_color}">💰 {rec.get('cost_tier','')}</span>
  <span class="badge" style="background:{tool_color}">🔧 Tool Calling: {rec.get('tool_calling_support','')}</span>
  <span class="badge" style="background:#3b82f6">🪟 {rec.get('context_window','')}</span>
  <span class="badge" style="background:{s_color}">⭐ Score: {rec.get('suitability_score','')}/10</span>
</div>
""", unsafe_allow_html=True)

        with st.expander(f"📖 Details — {rec.get('model_name','')}"):
            d1, d2 = st.columns(2)
            with d1:
                st.markdown("**Key Features**")
                for feat in rec.get("key_features", []):
                    st.markdown(f"- {feat}")
                st.markdown(f"**Best For:** {rec.get('best_for','')}")
            with d2:
                st.markdown("**Tool Calling Details**")
                st.info(rec.get("tool_calling_details", "N/A"))
                st.markdown("**Why It Fits Your Workflow**")
                st.success(rec.get("suitability_reason", ""))


# ─── Main search flow ─────────────────────────────────────────────────────────
if search_btn and user_query.strip():

    # Add to history
    if user_query not in st.session_state.search_history:
        st.session_state.search_history.append(user_query)

    with st.spinner("🌐 Searching the web for latest LLM data…"):
        result = get_recommendations(user_query)
        st.session_state.last_result = result

    # Error guard
    if result.get("error"):
        st.error(f"⚠️ Error during recommendation: {result['error']}")

elif search_btn and not user_query.strip():
    st.warning("Please describe your workflow before searching.")

# ─── Render last result ───────────────────────────────────────────────────────
result = st.session_state.last_result
if result and not result.get("error"):

    recs = result.get("recommendations", [])

    # ── Workflow analysis ──
    st.markdown('<div class="section-title">🧠 Workflow Analysis</div>', unsafe_allow_html=True)
    st.info(result.get("workflow_analysis", ""))

    if result.get("key_requirements"):
        cols = st.columns(min(len(result["key_requirements"]), 4))
        for i, req in enumerate(result["key_requirements"][:4]):
            cols[i % 4].markdown(
                f'<span class="badge" style="background:#1d4ed8;font-size:0.82rem;padding:0.35rem 0.9rem;">'
                f'✓ {req}</span>', unsafe_allow_html=True)

    st.markdown("")

    # ── Cloud / All recommendations ──
    st.markdown('<div class="section-title">☁️ LLM Recommendations</div>', unsafe_allow_html=True)

    cloud_recs = [r for r in recs if r.get("model_type") in ("Cloud", "Both")]
    local_recs = [r for r in recs if r.get("model_type") == "Local"]
    all_recs   = recs

    tab_all, tab_cloud, tab_oss = st.tabs([
        f"🌐 All ({len(all_recs)})",
        f"☁️ Cloud ({len(cloud_recs)})",
        f"🏠 Open-Source ({len(local_recs)})",
    ])

    with tab_all:
        for rec in all_recs:
            render_model_card(rec)

    with tab_cloud:
        if cloud_recs:
            for rec in cloud_recs:
                render_model_card(rec)
        else:
            st.caption("No cloud-specific models in this result set.")

    with tab_oss:
        if local_recs:
            for rec in local_recs:
                render_model_card(rec)
        else:
            st.caption("No open-source-specific models in this result set.")

    # ── Comparison table ──
    st.markdown('<div class="section-title">📊 Comparison Table</div>', unsafe_allow_html=True)
    df = recommendations_to_dataframe(recs)
    if not df.empty:
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Suitability Score": st.column_config.ProgressColumn(
                    "Suitability Score", min_value=0, max_value=10, format="%.1f"
                ),
                "Rank": st.column_config.NumberColumn("Rank", width="small"),
            },
        )

    # ── Web Sources ──
    if result.get("search_results"):
        with st.expander("🔗 Web Sources Used"):
            for r in result["search_results"][:6]:
                st.markdown(f"- [{r.get('title',r.get('href',''))}]({r.get('href','#')})")

    st.divider()


# ─── Local Ollama models section ──────────────────────────────────────────────
st.markdown('<div class="section-title">🏠 Local Ollama Models</div>', unsafe_allow_html=True)

if st.button("🔄 Refresh local models"):
    st.session_state.local_models = list_local_models()

if not st.session_state.local_models:
    st.session_state.local_models = list_local_models()

local_models = st.session_state.local_models

if not local_models:
    st.warning("No local Ollama models found — make sure Ollama is running and you have pulled at least one model.")
else:
    # Extract workflow keywords from last query for scoring
    last_q = st.session_state.search_history[-1] if st.session_state.search_history else ""
    kw = last_q.lower().split() if last_q else []

    st.caption(f"{len(local_models)} model(s) installed locally")

    cols = st.columns(3)
    for i, m in enumerate(local_models):
        score, reason = score_local_model_for_workflow(m, kw)
        s_col = score_color(score)

        with cols[i % 3]:
            st.markdown(f"""
<div class="model-card">
  <h3>🏠 {m['name']}</h3>
  <div class="provider">Family: {m['family']} &nbsp;|&nbsp; {m['parameters']}</div>
  <div class="desc">
    Quantization: <b>{m['quantization']}</b><br>
    Size: <b>{m['size_gb']} GB</b> &nbsp;|&nbsp; Format: {m['format']}<br>
    Modified: {m['modified']}
  </div>
  <br/>
  <span class="badge" style="background:{s_col}">Workflow Score: {score}/10</span>
  <span class="badge" style="background:#1d4ed8">Free</span>
</div>
""", unsafe_allow_html=True)
            with st.expander(f"🧪 Test {m['name']}"):
                test_prompt = st.text_input(
                    "Test prompt",
                    value="Briefly introduce yourself and your capabilities.",
                    key=f"tp_{m['name']}",
                )
                if st.button("▶ Run", key=f"run_{m['name']}"):
                    with st.spinner(f"Running {m['name']}…"):
                        out = test_model(m["name"], test_prompt)
                    st.markdown(f"**Response:**\n\n{out}")

    # Local model comparison table
    st.markdown("#### Local Model Details")
    local_df = pd.DataFrame([{
        "Model":        m["name"],
        "Family":       m["family"],
        "Parameters":   m["parameters"],
        "Quantization": m["quantization"],
        "Size (GB)":    m["size_gb"],
        "Workflow Fit": f"{score_local_model_for_workflow(m, kw)[0]}/10",
    } for m in local_models])
    st.dataframe(local_df, use_container_width=True, hide_index=True)
