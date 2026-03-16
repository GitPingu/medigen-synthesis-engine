"""
Strategic Synthesis Engine — MediGen Corp Demo
A RAG-powered query interface over MediGen's internal document corpus.
Tier 1: Streaming, Chat History, Feedback, Phased Loading, Export
"""
import os
import time
import json
from datetime import datetime
import streamlit as st
import chromadb
import anthropic
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

def get_api_key():
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        try:
            key = st.secrets["ANTHROPIC_API_KEY"]
        except Exception:
            pass
    return key

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Strategic Synthesis Engine | MediGen Corp",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session State Init ───────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "feedback_log" not in st.session_state:
    st.session_state.feedback_log = []

# ── Dark Theme CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        background-color: #0A0A0B !important;
        color: #E8E8E8;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background-color: #111113 !important;
        border-right: 1px solid #1E1E22;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown small {
        color: #A0A0A8 !important;
    }
    section[data-testid="stSidebar"] label {
        color: #A0A0A8 !important;
    }

    /* ── Hero Header ── */
    .hero-container {
        position: relative;
        padding: 2rem 2.5rem 1.5rem;
        margin: -1rem -1rem 1.5rem -1rem;
        background: linear-gradient(135deg, #0A0A0B 0%, #141418 50%, #0A0A0B 100%);
        border-bottom: 1px solid #1E1E22;
        overflow: hidden;
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle at 30% 50%, rgba(255, 229, 1, 0.03) 0%, transparent 50%),
                    radial-gradient(circle at 70% 80%, rgba(255, 229, 1, 0.02) 0%, transparent 40%);
        pointer-events: none;
    }
    .hero-eyebrow {
        font-size: 0.7rem; font-weight: 600; letter-spacing: 3px;
        text-transform: uppercase; color: #FFE501; margin-bottom: 0.4rem;
    }
    .hero-title {
        font-size: 2.2rem; font-weight: 800; color: #FFFFFF;
        margin: 0; line-height: 1.15; letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 0.95rem; color: #6B6B75; margin-top: 0.5rem; font-weight: 400;
    }
    .hero-divider {
        width: 48px; height: 3px; background: #FFE501;
        margin-top: 1rem; border-radius: 2px;
    }

    /* ── Metric Cards ── */
    .metric-row {
        display: flex; gap: 0.8rem; margin-bottom: 1rem;
    }
    .stat-card {
        background: #141418; border: 1px solid #1E1E22;
        border-radius: 10px; padding: 0.8rem 1rem;
        text-align: center; flex: 1; transition: border-color 0.3s ease;
    }
    .stat-card:hover { border-color: #FFE501; }
    .stat-card .stat-value {
        font-size: 1.6rem; font-weight: 800; color: #FFE501; margin: 0; line-height: 1.2;
    }
    .stat-card .stat-label {
        font-size: 0.6rem; color: #6B6B75; text-transform: uppercase;
        letter-spacing: 1.5px; margin-top: 0.2rem;
    }

    /* ── Section Headers ── */
    .section-label {
        font-size: 0.65rem; font-weight: 600; letter-spacing: 2.5px;
        text-transform: uppercase; color: #FFE501;
        margin-bottom: 0.8rem; margin-top: 1.5rem;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: #141418 !important; color: #C8C8D0 !important;
        border: 1px solid #2A2A30 !important; border-radius: 8px !important;
        padding: 0.6rem 1rem !important; font-size: 0.82rem !important;
        font-family: 'Inter', sans-serif !important; font-weight: 400 !important;
        text-align: left !important; transition: all 0.25s ease !important;
        line-height: 1.4 !important;
    }
    .stButton > button:hover {
        border-color: #FFE501 !important; color: #FFFFFF !important;
        background: #1A1A1F !important;
        box-shadow: 0 0 20px rgba(255, 229, 1, 0.05) !important;
    }
    .stButton > button:active {
        background: #FFE501 !important; color: #0A0A0B !important;
        border-color: #FFE501 !important;
    }

    /* ── Text Input ── */
    .stTextInput > div > div > input {
        background: #141418 !important; color: #FFFFFF !important;
        border: 1px solid #2A2A30 !important; border-radius: 10px !important;
        padding: 0.8rem 1rem !important; font-size: 0.95rem !important;
        font-family: 'Inter', sans-serif !important;
        transition: border-color 0.25s ease !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #FFE501 !important;
        box-shadow: 0 0 0 1px #FFE501, 0 0 20px rgba(255, 229, 1, 0.08) !important;
    }
    .stTextInput > div > div > input::placeholder { color: #4A4A55 !important; }
    .stTextInput label { color: #A0A0A8 !important; font-size: 0.85rem !important; }

    /* ── Chat Input ── */
    .stChatInput > div {
        background: #141418 !important; border: 1px solid #2A2A30 !important;
        border-radius: 10px !important;
    }
    .stChatInput textarea {
        color: #FFFFFF !important; font-family: 'Inter', sans-serif !important;
    }
    .stChatInput button {
        color: #FFE501 !important;
    }

    /* ── Chat Messages ── */
    .stChatMessage {
        background: transparent !important;
        border: none !important;
    }
    [data-testid="stChatMessageAvatarUser"] {
        background: #FFE501 !important;
    }
    [data-testid="stChatMessageAvatarAssistant"] {
        background: #2A2A30 !important;
    }

    /* ── Source Cards ── */
    .source-card {
        background: #111113; border: 1px solid #1E1E22;
        border-left: 3px solid #FFE501; padding: 1rem 1.2rem;
        margin: 0.6rem 0; border-radius: 0 10px 10px 0;
        transition: all 0.25s ease;
        animation: fadeInUp 0.3s ease-out both;
    }
    .source-card:nth-child(2) { animation-delay: 0.05s; }
    .source-card:nth-child(3) { animation-delay: 0.1s; }
    .source-card:nth-child(4) { animation-delay: 0.15s; }
    .source-card:nth-child(5) { animation-delay: 0.2s; }
    .source-card:nth-child(6) { animation-delay: 0.25s; }
    .source-card:nth-child(7) { animation-delay: 0.3s; }
    .source-card:nth-child(8) { animation-delay: 0.35s; }
    .source-card:hover {
        border-color: #2A2A30; border-left-color: #FFE501; background: #141418;
    }
    .source-card .source-title {
        font-weight: 600; color: #FFFFFF; font-size: 0.85rem; margin-bottom: 0.4rem;
    }
    .source-card .source-meta {
        color: #6B6B75; font-size: 0.72rem; letter-spacing: 0.3px;
    }
    .source-card .source-excerpt {
        color: #8A8A95; margin-top: 0.5rem; font-size: 0.8rem; line-height: 1.5;
    }

    /* ── Relevance Badges ── */
    .confidence-badge {
        display: inline-block; padding: 0.15rem 0.55rem; border-radius: 20px;
        font-size: 0.65rem; font-weight: 600; letter-spacing: 0.5px;
        margin-left: 0.5rem; vertical-align: middle;
    }
    .conf-high {
        background: rgba(255, 229, 1, 0.15); color: #FFE501;
        border: 1px solid rgba(255, 229, 1, 0.3);
    }
    .conf-medium {
        background: rgba(255, 165, 0, 0.12); color: #FFA500;
        border: 1px solid rgba(255, 165, 0, 0.25);
    }
    .conf-low {
        background: rgba(255, 80, 80, 0.1); color: #FF6B6B;
        border: 1px solid rgba(255, 80, 80, 0.2);
    }

    /* ── Sidebar Metrics ── */
    .sidebar-metric {
        background: #141418; border: 1px solid #1E1E22;
        border-radius: 8px; padding: 0.8rem; text-align: center;
        margin-bottom: 0.5rem;
    }
    .sidebar-metric .sm-value {
        font-size: 1.4rem; font-weight: 800; color: #FFE501; margin: 0;
    }
    .sidebar-metric .sm-label {
        font-size: 0.6rem; color: #6B6B75; text-transform: uppercase;
        letter-spacing: 1.5px; margin-top: 0.15rem;
    }

    /* ── Performance Bar ── */
    .perf-bar {
        display: flex; gap: 1.5rem; padding: 0.5rem 0; margin-top: 0.5rem;
        font-size: 0.7rem; color: #4A4A55; letter-spacing: 0.5px;
    }
    .perf-bar span { color: #6B6B75; }

    /* ── Animations ── */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* ── Misc ── */
    .stSpinner > div { border-top-color: #FFE501 !important; }
    .stAlert {
        background-color: #141418 !important; border: 1px solid #1E1E22 !important;
        color: #6B6B75 !important; border-radius: 10px !important;
    }
    .stSlider label { color: #A0A0A8 !important; }
    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"] { color: #6B6B75 !important; }
    .stMultiSelect label { color: #A0A0A8 !important; }
    .stMultiSelect [data-baseweb="select"] {
        background-color: #141418 !important; border-color: #2A2A30 !important;
    }
    hr { border-color: #1E1E22 !important; }
    .sidebar-brand {
        padding: 0.8rem 0; margin-bottom: 1rem; border-bottom: 1px solid #1E1E22;
    }
    .sidebar-brand .brand-name {
        font-size: 0.65rem; font-weight: 600; letter-spacing: 3px;
        text-transform: uppercase; color: #FFE501;
    }
    .sidebar-brand .brand-sub {
        font-size: 0.7rem; color: #4A4A55; margin-top: 0.2rem;
    }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0A0A0B; }
    ::-webkit-scrollbar-thumb { background: #2A2A30; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #3A3A42; }

    /* ── Download Button ── */
    .stDownloadButton > button {
        background: #141418 !important; color: #FFE501 !important;
        border: 1px solid #2A2A30 !important; border-radius: 8px !important;
        font-size: 0.75rem !important; padding: 0.4rem 0.8rem !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stDownloadButton > button:hover {
        border-color: #FFE501 !important; background: #1A1A1F !important;
    }

    /* ── Feedback Buttons ── */
    .stFeedback button {
        background: #141418 !important; border: 1px solid #2A2A30 !important;
        color: #6B6B75 !important;
    }
    .stFeedback button:hover {
        border-color: #FFE501 !important; color: #FFE501 !important;
    }
    .stFeedback button[aria-pressed="true"] {
        background: rgba(255, 229, 1, 0.15) !important;
        border-color: #FFE501 !important; color: #FFE501 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-eyebrow">MediGen Corp</div>
    <div class="hero-title">Strategic Synthesis Engine</div>
    <div class="hero-subtitle">Query 1,050 internal documents across 8 systems. Get cited answers in seconds.</div>
    <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)

# ── Initialize Clients ───────────────────────────────────────────────────────
@st.cache_resource
def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection("medigen_docs")

@st.cache_resource
def get_anthropic_client():
    return anthropic.Anthropic(api_key=get_api_key())

try:
    collection = get_chroma_collection()
    llm_client = get_anthropic_client()
    corpus_ready = True
except Exception as e:
    corpus_ready = False
    st.error(f"Could not load document corpus. Run `python ingest.py` first.\n\nError: {e}")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="brand-name">MediGen Corp</div>
        <div class="brand-sub">Strategic Synthesis Engine</div>
    </div>
    """, unsafe_allow_html=True)

    n_results = st.slider("Sources to retrieve", 3, 15, 8)
    dept_filter = st.multiselect(
        "Filter by department",
        ["Research", "Clinical Development", "Regulatory Affairs", "Legal",
         "Manufacturing & CMC", "Quality", "Medical Affairs", "Commercial",
         "IT", "Corporate"],
        default=[],
        help="Leave empty to search all departments",
    )

    st.markdown("---")

    if st.button("New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    if corpus_ready:
        total_chunks = collection.count()
        st.markdown(f"""
        <div class="sidebar-metric"><div class="sm-value">{total_chunks:,}</div><div class="sm-label">Indexed Chunks</div></div>
        <div class="sidebar-metric"><div class="sm-value">1,050</div><div class="sm-label">Source Documents</div></div>
        <div class="sidebar-metric"><div class="sm-value">8</div><div class="sm-label">Connected Systems</div></div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        '<small style="color: #4A4A55;">Powered by Claude + ChromaDB</small>',
        unsafe_allow_html=True,
    )

# ── Stats Row ────────────────────────────────────────────────────────────────
if corpus_ready:
    total_chunks = collection.count()
    st.markdown(f"""
    <div class="metric-row">
        <div class="stat-card">
            <div class="stat-value">1,050</div>
            <div class="stat-label">Documents Ingested</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{total_chunks:,}</div>
            <div class="stat-label">Searchable Chunks</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">8</div>
            <div class="stat-label">Source Systems</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">&lt;5s</div>
            <div class="stat-label">Avg Response Time</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Example Queries (department-aware) ────────────────────────────────────────
dept_example_queries = {
    "Research": [
        "What is the best-tolerated dose from our prior antibody-drug conjugate animal studies?",
        "What were the efficacy results for MG-401 in HER3-expressing xenograft models?",
        "What pharmacokinetic data do we have on the MG-Link payload in primates?",
        "Which combination studies have been run for our pipeline compounds?",
        "What compound synthesis work has been done on MG-Link linker variants?",
        "What is the selectivity profile of our latest medicinal chemistry leads?",
    ],
    "Clinical Development": [
        "What is the current enrollment status of the MG-401 STELLAR trial?",
        "What were the efficacy results from the CARALYN Phase 2 study?",
        "What dose was selected for the Phase 2 recommended dose of MG-401?",
        "What biomarker findings have been reported from our clinical trials?",
        "What serious adverse events have been reported across our programs?",
        "What subgroup analyses are planned for the STELLAR trial?",
    ],
    "Regulatory Affairs": [
        "What did the FDA recommend in the MG-309 pre-IND meeting?",
        "What is the regulatory strategy for MG-401 in the United States?",
        "What Health Authority queries are currently outstanding?",
        "What CMC regulatory assessments have been completed for CARALYN?",
        "What post-marketing commitments exist for VELORIN?",
        "What labeling updates have been proposed based on new safety data?",
    ],
    "Legal": [
        "What patents does MediGen hold related to the MG-Link platform?",
        "What licensing agreements are currently active with external partners?",
        "What is the freedom-to-operate status for our antibody-drug conjugate programs?",
        "Which non-disclosure agreements are expiring in the next 90 days?",
        "What IP provisions exist in our co-development agreements?",
        "What patent prosecution actions are currently pending?",
    ],
    "Manufacturing & CMC": [
        "What stability data do we have for CARALYN drug product?",
        "What batch manufacturing records exist for MG-401 drug substance?",
        "What deviations have been reported at the Devens facility?",
        "What is the current drug-to-antibody ratio specification for our conjugates?",
        "What analytical methods have been validated for release testing?",
        "What process development work has been done on scale-up from 200L to 2000L?",
    ],
    "Quality": [
        "Which suppliers have been qualified for our manufacturing operations?",
        "What CAPA reports are currently open?",
        "What were the findings from our most recent quality management review?",
        "What out-of-specification investigations have been conducted?",
        "What is our current deviation rate and trend?",
        "What training compliance rates have been reported?",
    ],
    "Medical Affairs": [
        "What were the key insights from our most recent advisory board?",
        "What competitive intelligence has been gathered at recent oncology congresses?",
        "What real-world evidence do we have on VELORIN treatment patterns?",
        "What are the most common medical information inquiries for CARALYN?",
        "How many key opinion leader interactions did the field team complete last quarter?",
        "What scientific publications are planned for our pipeline programs?",
    ],
    "Commercial": [
        "What is VELORIN's current market share in second-line AML?",
        "What payer coverage does CARALYN have across commercial plans?",
        "What competitive threats have been identified for our marketed products?",
        "What is the launch readiness status for MG-401?",
        "What patient assistance program enrollment numbers are we seeing?",
        "What pricing analyses have been conducted for our pipeline products?",
    ],
    "IT": [
        "What system validations have been completed for Benchling?",
        "What IT security assessments have been conducted recently?",
        "What data migration plans are in progress?",
        "What software requirements have been defined for cross-system integration?",
        "What disaster recovery testing has been performed?",
        "What vendor assessments are currently underway?",
    ],
    "Corporate": [
        "What is the current R&D budget allocation across programs?",
        "What headcount plans have been approved for this year?",
        "What facility expansion plans exist for the Devens site?",
        "What business continuity updates have been made recently?",
        "What vendor contracts are up for renewal?",
        "What capital expenditure has been approved for manufacturing?",
    ],
}

default_queries = [
    "What is the best-tolerated dose from our prior antibody-drug conjugate animal studies?",
    "What were the efficacy results from the CARALYN Phase 2 study?",
    "What did the FDA recommend in the MG-309 pre-IND meeting?",
    "What patents does MediGen hold related to the MG-Link platform?",
    "What is the current enrollment status of the MG-401 STELLAR trial?",
    "What stability data do we have for CARALYN drug product?",
]

if dept_filter:
    example_queries = []
    for dept in dept_filter:
        example_queries.extend(dept_example_queries.get(dept, [])[:3])
    example_queries = example_queries[:6]
else:
    example_queries = default_queries

# Only show example queries if no conversation yet
if not st.session_state.messages:
    st.markdown('<div class="section-label">Try a question</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, q in enumerate(example_queries[:6]):
        col = cols[i % 3]
        if col.button(q, key=f"example_{i}", use_container_width=True):
            st.session_state["selected_query"] = q
            st.rerun()

# ── Helper: Render Source Cards ──────────────────────────────────────────────
def render_sources(sources):
    st.markdown('<div class="section-label">Sources Retrieved</div>', unsafe_allow_html=True)
    for i, src in enumerate(sources):
        relevance = src["relevance"]
        if relevance >= 70:
            conf_class, conf_label = "conf-high", "HIGH"
        elif relevance >= 40:
            conf_class, conf_label = "conf-medium", "MED"
        else:
            conf_class, conf_label = "conf-low", "LOW"

        st.markdown(f"""
        <div class="source-card">
            <div class="source-title">
                Source {i+1}: {src['filename']}
                <span class="confidence-badge {conf_class}">{conf_label} {relevance}%</span>
            </div>
            <div class="source-meta">
                {src['department']} &bull; {src['system']} &bull; {src['program']}
            </div>
            <div class="source-excerpt">{src['excerpt']}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Helper: Build Export Content ─────────────────────────────────────────────
def build_export(query, answer, sources, retrieval_ms, generation_ms):
    lines = [
        f"# Strategic Synthesis Engine — Query Export",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Query:** {query}",
        f"**Retrieval:** {retrieval_ms}ms | **Generation:** {generation_ms}ms",
        "",
        "---",
        "",
        "## Answer",
        "",
        answer,
        "",
        "---",
        "",
        "## Sources",
        "",
    ]
    for i, src in enumerate(sources):
        lines.append(f"**Source {i+1}:** {src['filename']} ({src['department']} / {src['system']}) — Relevance: {src['relevance']}%")
    lines.append("")
    lines.append("---")
    lines.append("*Exported from Strategic Synthesis Engine — MediGen Corp*")
    return "\n".join(lines)

# ── Display Chat History ─────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            # Performance metrics
            if "retrieval_ms" in msg:
                st.markdown(f"""<div class="perf-bar">
                    <span>Retrieved {len(msg['sources'])} sources in {msg['retrieval_ms']}ms</span>
                    <span>Answer generated in {msg['generation_ms']}ms</span>
                    <span>Total: {msg['retrieval_ms'] + msg['generation_ms']}ms</span>
                </div>""", unsafe_allow_html=True)

            # Export button
            export_content = build_export(
                msg.get("query", ""),
                msg["content"],
                msg["sources"],
                msg.get("retrieval_ms", 0),
                msg.get("generation_ms", 0),
            )
            st.download_button(
                "Export Answer",
                export_content,
                file_name=f"sse_export_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
                key=f"export_{msg.get('timestamp', id(msg))}",
            )

            # Feedback
            feedback = st.feedback(
                "thumbs",
                key=f"feedback_{msg.get('timestamp', id(msg))}",
            )
            if feedback is not None:
                st.session_state.feedback_log.append({
                    "query": msg.get("query", ""),
                    "rating": "positive" if feedback == 1 else "negative",
                    "timestamp": datetime.now().isoformat(),
                })

            # Source cards
            render_sources(msg["sources"])

# ── Chat Input ───────────────────────────────────────────────────────────────
selected = st.session_state.pop("selected_query", None)
if selected:
    prompt = selected
elif corpus_ready:
    prompt = st.chat_input("Ask a question across MediGen's knowledge base...")
else:
    prompt = None

if prompt and corpus_ready:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ── Assistant Response ────────────────────────────────────────────────
    with st.chat_message("assistant"):

        # Phase 1: Retrieval
        with st.status("Searching document corpus...", expanded=True) as status:
            st.write(f"Querying {total_chunks:,} document chunks...")
            t0 = time.perf_counter()

            where_filter = None
            if dept_filter:
                if len(dept_filter) == 1:
                    where_filter = {"department": dept_filter[0]}
                else:
                    where_filter = {"department": {"$in": dept_filter}}

            results = collection.query(
                query_texts=[prompt],
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )

            retrieved_docs = results["documents"][0]
            retrieved_metas = results["metadatas"][0]
            retrieved_distances = results["distances"][0]

            retrieval_ms = int((time.perf_counter() - t0) * 1000)
            st.write(f"Found {len(retrieved_docs)} relevant sources in {retrieval_ms}ms")
            status.update(label=f"Retrieved {len(retrieved_docs)} sources in {retrieval_ms}ms", state="complete")

        # Build source data
        sources_data = []
        for doc, meta, dist in zip(retrieved_docs, retrieved_metas, retrieved_distances):
            relevance = max(0, min(100, int((1 - dist / 2) * 100)))
            excerpt = doc[:300].replace("\n", " ").strip()
            if len(doc) > 300:
                excerpt += "..."
            sources_data.append({
                "filename": meta.get("filename", "unknown"),
                "department": meta.get("department", "unknown"),
                "system": meta.get("source", "unknown"),
                "program": meta.get("program", "N/A"),
                "relevance": relevance,
                "excerpt": excerpt,
            })

        # Build context for LLM
        context_parts = []
        for i, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_metas)):
            source_label = f"[Source {i+1}: {meta.get('filename', 'unknown')} | Dept: {meta.get('department', 'unknown')} | System: {meta.get('source', 'unknown')}]"
            context_parts.append(f"{source_label}\n{doc}")
        context = "\n\n---\n\n".join(context_parts)

        system_prompt = """You are the Strategic Synthesis Engine, an AI knowledge retrieval system for MediGen Corp, a biotech company. Your role is to answer questions using ONLY the retrieved document excerpts provided below. Follow these rules strictly:

1. ONLY use information from the provided source documents. Never use outside knowledge.
2. CITE every claim by referencing the source document (e.g., "Source 1: [filename]").
3. If the retrieved documents do not contain enough information to answer the question, say so explicitly.
4. Write in clear, professional language accessible to both scientific and business audiences.
5. Structure your answer with a direct response first, followed by supporting details.
6. At the end, list all sources used with their department and system of origin."""

        # Build messages with conversation context (last 3 turns for context)
        llm_messages = []
        recent_history = st.session_state.messages[:-1][-6:]  # last 3 exchanges
        for msg in recent_history:
            if msg["role"] in ("user", "assistant"):
                llm_messages.append({"role": msg["role"], "content": msg["content"]})

        # Current turn with retrieved context
        user_prompt = f"""RETRIEVED DOCUMENTS:

{context}

---

QUESTION: {prompt}

Provide a comprehensive, cited answer based solely on the retrieved documents above."""

        llm_messages.append({"role": "user", "content": user_prompt})

        # Phase 2: Streaming generation
        t1 = time.perf_counter()
        answer_placeholder = st.empty()
        full_answer = ""

        with llm_client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            system=system_prompt,
            messages=llm_messages,
        ) as stream:
            for text in stream.text_stream:
                full_answer += text
                answer_placeholder.markdown(full_answer + " ")

        answer_placeholder.markdown(full_answer)
        generation_ms = int((time.perf_counter() - t1) * 1000)

        # Performance metrics
        st.markdown(f"""<div class="perf-bar">
            <span>Retrieved {len(sources_data)} sources in {retrieval_ms}ms</span>
            <span>Answer generated in {generation_ms}ms</span>
            <span>Total: {retrieval_ms + generation_ms}ms</span>
        </div>""", unsafe_allow_html=True)

        # Export button
        timestamp = datetime.now().isoformat()
        export_content = build_export(prompt, full_answer, sources_data, retrieval_ms, generation_ms)
        st.download_button(
            "Export Answer",
            export_content,
            file_name=f"sse_export_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            key=f"export_current_{timestamp}",
        )

        # Feedback
        feedback = st.feedback("thumbs", key=f"feedback_current_{timestamp}")
        if feedback is not None:
            st.session_state.feedback_log.append({
                "query": prompt,
                "rating": "positive" if feedback == 1 else "negative",
                "timestamp": timestamp,
            })

        # Source cards
        render_sources(sources_data)

        # Store assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_answer,
            "sources": sources_data,
            "query": prompt,
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "timestamp": timestamp,
        })

elif not st.session_state.messages and not prompt:
    st.markdown("""
    <div style="text-align: center; padding: 3rem 0; color: #4A4A55;">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">&#8593;</div>
        <div style="font-size: 0.85rem;">Enter a question or select an example above to begin</div>
    </div>
    """, unsafe_allow_html=True)
