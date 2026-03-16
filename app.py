"""
Strategic Synthesis Engine — MediGen Corp Demo
A RAG-powered query interface over MediGen's internal document corpus.
Tiers 1-3: Streaming, Chat, Feedback, Loading, Export, Drill-down,
Follow-ups, Confidence, Animations, Enhanced Source Cards
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
CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "demo_corpus")

def get_api_key():
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        try:
            key = st.secrets["ANTHROPIC_API_KEY"]
        except Exception:
            pass
    return key

# ── Department Labels ────────────────────────────────────────────────────────
DEPT_ICONS = {
    "Research": "RES",
    "Clinical Development": "CLI",
    "Regulatory Affairs": "REG",
    "Legal": "LEG",
    "Manufacturing & CMC": "MFG",
    "Quality": "QA",
    "Medical Affairs": "MED",
    "Commercial": "COM",
    "IT": "IT",
    "Corporate": "CORP",
}

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
if "query_count" not in st.session_state:
    st.session_state.query_count = 0
if "total_retrieval_ms" not in st.session_state:
    st.session_state.total_retrieval_ms = 0
if "total_generation_ms" not in st.session_state:
    st.session_state.total_generation_ms = 0

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

    section[data-testid="stSidebar"] {
        background-color: #111113 !important;
        border-right: 1px solid #1E1E22;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown small { color: #A0A0A8 !important; }
    section[data-testid="stSidebar"] label { color: #A0A0A8 !important; }

    .hero-container {
        position: relative; padding: 2rem 2.5rem 1.5rem;
        margin: -1rem -1rem 1.5rem -1rem;
        background: linear-gradient(135deg, #0A0A0B 0%, #141418 50%, #0A0A0B 100%);
        border-bottom: 1px solid #1E1E22; overflow: hidden;
    }
    .hero-container::before {
        content: ''; position: absolute; top: -50%; left: -50%;
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
        animation: pulseGlow 3s ease-in-out infinite;
    }
    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 5px rgba(255, 229, 1, 0.2); }
        50% { box-shadow: 0 0 15px rgba(255, 229, 1, 0.5); }
    }

    .metric-row { display: flex; gap: 0.8rem; margin-bottom: 1rem; }
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

    .section-label {
        font-size: 0.65rem; font-weight: 600; letter-spacing: 2.5px;
        text-transform: uppercase; color: #FFE501;
        margin-bottom: 0.8rem; margin-top: 1.5rem;
    }

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

    .stTextInput > div > div > input {
        background: #141418 !important; color: #FFFFFF !important;
        border: 1px solid #2A2A30 !important; border-radius: 10px !important;
        padding: 0.8rem 1rem !important; font-size: 0.95rem !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #FFE501 !important;
        box-shadow: 0 0 0 1px #FFE501, 0 0 20px rgba(255, 229, 1, 0.08) !important;
    }
    .stTextInput > div > div > input::placeholder { color: #4A4A55 !important; }
    .stTextInput label { color: #A0A0A8 !important; font-size: 0.85rem !important; }

    .stChatInput > div {
        background: #141418 !important; border: 1px solid #2A2A30 !important;
        border-radius: 10px !important;
    }
    .stChatInput textarea { color: #FFFFFF !important; font-family: 'Inter', sans-serif !important; }
    .stChatInput button { color: #FFE501 !important; }

    .stChatMessage { background: transparent !important; border: none !important; }
    [data-testid="stChatMessageAvatarUser"] { background: #FFE501 !important; }
    [data-testid="stChatMessageAvatarAssistant"] { background: #2A2A30 !important; }

    .source-card {
        background: #111113; border: 1px solid #1E1E22;
        border-left: 3px solid #FFE501; padding: 1rem 1.2rem;
        margin: 0.6rem 0; border-radius: 0 10px 10px 0;
        transition: all 0.25s ease; animation: fadeInUp 0.3s ease-out both;
    }
    .source-card:nth-child(2) { animation-delay: 0.05s; }
    .source-card:nth-child(3) { animation-delay: 0.1s; }
    .source-card:nth-child(4) { animation-delay: 0.15s; }
    .source-card:nth-child(5) { animation-delay: 0.2s; }
    .source-card:nth-child(6) { animation-delay: 0.25s; }
    .source-card:nth-child(7) { animation-delay: 0.3s; }
    .source-card:nth-child(8) { animation-delay: 0.35s; }
    .source-card:hover { border-color: #2A2A30; border-left-color: #FFE501; background: #141418; }
    .source-card .source-header {
        display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.4rem;
    }
    .source-card .source-icon {
        font-size: 0.55rem; font-weight: 700; letter-spacing: 1px;
        color: #FFE501; background: rgba(255, 229, 1, 0.1);
        border: 1px solid rgba(255, 229, 1, 0.2);
        padding: 0.15rem 0.4rem; border-radius: 4px;
        min-width: 28px; text-align: center;
    }
    .source-card .source-title {
        font-weight: 600; color: #FFFFFF; font-size: 0.85rem; flex: 1;
    }
    .source-card .source-meta {
        color: #6B6B75; font-size: 0.72rem; letter-spacing: 0.3px;
    }
    .source-card .source-excerpt {
        color: #8A8A95; margin-top: 0.5rem; font-size: 0.8rem; line-height: 1.5;
    }
    .relevance-bar-container {
        margin-top: 0.5rem; display: flex; align-items: center; gap: 0.5rem;
    }
    .relevance-bar-bg {
        flex: 1; height: 4px; background: #1E1E22; border-radius: 2px; overflow: hidden;
    }
    .relevance-bar-fill {
        height: 100%; border-radius: 2px; transition: width 0.5s ease;
    }
    .relevance-bar-label {
        font-size: 0.65rem; font-weight: 600; min-width: 32px; text-align: right;
    }

    .confidence-badge {
        display: inline-block; padding: 0.15rem 0.55rem; border-radius: 20px;
        font-size: 0.65rem; font-weight: 600; letter-spacing: 0.5px;
        margin-left: 0.5rem; vertical-align: middle;
    }
    .conf-high { background: rgba(255, 229, 1, 0.15); color: #FFE501; border: 1px solid rgba(255, 229, 1, 0.3); }
    .conf-medium { background: rgba(255, 165, 0, 0.12); color: #FFA500; border: 1px solid rgba(255, 165, 0, 0.25); }
    .conf-low { background: rgba(255, 80, 80, 0.1); color: #FF6B6B; border: 1px solid rgba(255, 80, 80, 0.2); }

    .sidebar-metric {
        background: #141418; border: 1px solid #1E1E22; border-radius: 8px;
        padding: 0.8rem; text-align: center; margin-bottom: 0.5rem;
    }
    .sidebar-metric .sm-value { font-size: 1.4rem; font-weight: 800; color: #FFE501; margin: 0; }
    .sidebar-metric .sm-label {
        font-size: 0.6rem; color: #6B6B75; text-transform: uppercase;
        letter-spacing: 1.5px; margin-top: 0.15rem;
    }

    .perf-bar {
        display: flex; gap: 1.5rem; padding: 0.5rem 0; margin-top: 0.5rem;
        font-size: 0.7rem; color: #4A4A55; letter-spacing: 0.5px;
    }
    .perf-bar span { color: #6B6B75; }

    /* Confidence Meter */
    .confidence-meter {
        background: #141418; border: 1px solid #1E1E22; border-radius: 8px;
        padding: 0.8rem 1rem; margin: 0.8rem 0; display: flex;
        align-items: center; gap: 1rem; animation: fadeInUp 0.4s ease-out both;
    }
    .confidence-meter .cm-score {
        font-size: 1.5rem; font-weight: 800; min-width: 50px; text-align: center;
    }
    .confidence-meter .cm-details {
        flex: 1; font-size: 0.75rem; color: #6B6B75; line-height: 1.5;
    }
    .confidence-meter .cm-label {
        font-size: 0.7rem; font-weight: 600; letter-spacing: 1px;
        text-transform: uppercase; margin-bottom: 0.2rem;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

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
    .sidebar-brand { padding: 0.8rem 0; margin-bottom: 1rem; border-bottom: 1px solid #1E1E22; }
    .sidebar-brand .brand-name {
        font-size: 0.65rem; font-weight: 600; letter-spacing: 3px;
        text-transform: uppercase; color: #FFE501;
    }
    .sidebar-brand .brand-sub { font-size: 0.7rem; color: #4A4A55; margin-top: 0.2rem; }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0A0A0B; }
    ::-webkit-scrollbar-thumb { background: #2A2A30; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #3A3A42; }

    .stDownloadButton > button {
        background: #141418 !important; color: #FFE501 !important;
        border: 1px solid #2A2A30 !important; border-radius: 8px !important;
        font-size: 0.75rem !important; padding: 0.4rem 0.8rem !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stDownloadButton > button:hover { border-color: #FFE501 !important; background: #1A1A1F !important; }

    .stFeedback button {
        background: #141418 !important; border: 1px solid #2A2A30 !important; color: #6B6B75 !important;
    }
    .stFeedback button:hover { border-color: #FFE501 !important; color: #FFE501 !important; }
    .stFeedback button[aria-pressed="true"] {
        background: rgba(255, 229, 1, 0.15) !important;
        border-color: #FFE501 !important; color: #FFE501 !important;
    }

    /* Expander styling for drill-down */
    .stExpander {
        background: #111113 !important; border: 1px solid #1E1E22 !important;
        border-radius: 8px !important;
    }
    .stExpander summary { color: #A0A0A8 !important; font-size: 0.8rem !important; }
    .stExpander [data-testid="stExpanderDetails"] { color: #8A8A95 !important; }
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
        default=[], help="Leave empty to search all departments",
    )

    st.markdown("---")
    if st.button("New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.query_count = 0
        st.session_state.total_retrieval_ms = 0
        st.session_state.total_generation_ms = 0
        st.rerun()

    st.markdown("---")
    if corpus_ready:
        total_chunks = collection.count()
        st.markdown(f"""
        <div class="sidebar-metric"><div class="sm-value">{total_chunks:,}</div><div class="sm-label">Indexed Chunks</div></div>
        <div class="sidebar-metric"><div class="sm-value">1,050</div><div class="sm-label">Source Documents</div></div>
        <div class="sidebar-metric"><div class="sm-value">8</div><div class="sm-label">Connected Systems</div></div>
        """, unsafe_allow_html=True)

    # Session analytics
    if st.session_state.query_count > 0:
        st.markdown("---")
        avg_rt = st.session_state.total_retrieval_ms // st.session_state.query_count
        avg_gt = st.session_state.total_generation_ms // st.session_state.query_count
        pos = sum(1 for f in st.session_state.feedback_log if f["rating"] == "positive")
        neg = sum(1 for f in st.session_state.feedback_log if f["rating"] == "negative")
        st.markdown(f"""
        <div class="sidebar-metric"><div class="sm-value">{st.session_state.query_count}</div><div class="sm-label">Queries This Session</div></div>
        <div class="sidebar-metric"><div class="sm-value">{avg_rt + avg_gt}ms</div><div class="sm-label">Avg Response Time</div></div>
        """, unsafe_allow_html=True)
        if pos + neg > 0:
            st.markdown(f"""
            <div class="sidebar-metric"><div class="sm-value">{pos}/{pos+neg}</div><div class="sm-label">Positive Feedback</div></div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<small style="color: #4A4A55;">Powered by Claude + ChromaDB</small>', unsafe_allow_html=True)

# ── Stats Row ────────────────────────────────────────────────────────────────
if corpus_ready:
    total_chunks = collection.count()
    st.markdown(f"""
    <div class="metric-row">
        <div class="stat-card"><div class="stat-value">1,050</div><div class="stat-label">Documents Ingested</div></div>
        <div class="stat-card"><div class="stat-value">{total_chunks:,}</div><div class="stat-label">Searchable Chunks</div></div>
        <div class="stat-card"><div class="stat-value">8</div><div class="stat-label">Source Systems</div></div>
        <div class="stat-card"><div class="stat-value">&lt;5s</div><div class="stat-label">Avg Response Time</div></div>
    </div>
    """, unsafe_allow_html=True)

# ── Example Queries ──────────────────────────────────────────────────────────
dept_example_queries = {
    "Research": [
        "What is the safest dose we've tested for MG-401 in primates, and what side effects did we see?",
        "How did MG-401 perform against the competitor ADC in our xenograft studies?",
        "How long does the MG-Link payload stay in circulation compared to standard linkers?",
        "Walk me through MG-217's combination data with fulvestrant — did we see synergy?",
        "Is neutropenia a class effect across all our MG-Link ADC programs?",
        "How many NSCLC patients would be eligible for MG-401 based on our HER3 expression survey?",
    ],
    "Clinical Development": [
        "What is the STELLAR trial and where does enrollment stand?",
        "How did CARALYN perform in clinical trials? What was the response rate?",
        "Has MG-309 shown any early signs of efficacy in the Phase 1 dose escalation?",
        "What biomarker cutoff are we using for MG-401 patient selection and why?",
        "What safety signals has VELORIN shown in 5 years of post-marketing data?",
        "What subgroup analyses are pre-specified in the STELLAR statistical analysis plan?",
    ],
    "Regulatory Affairs": [
        "Summarize what the FDA told us in the MG-309 pre-IND meeting and how it changed our plans",
        "What regulatory pathway are we pursuing for MG-401 — standard or accelerated?",
        "Are there any outstanding Health Authority queries we need to respond to?",
        "What post-marketing commitments do we still owe for VELORIN?",
        "Have we filed a pediatric study plan for any of our programs?",
        "What labeling changes are being considered based on new safety data?",
    ],
    "Legal": [
        "What makes our MG-Link linker technology better than competitor approaches?",
        "Do we have any active licensing deals, and with which partners?",
        "Are there any competitor patents that could block our ADC programs?",
        "Which of our non-disclosure agreements are expiring soon?",
        "What IP protection do we have on veloritinib's crystal forms?",
        "What are the royalty terms in our co-development agreements?",
    ],
    "Manufacturing & CMC": [
        "How long can CARALYN drug product be stored and under what conditions?",
        "Have there been any manufacturing deviations at the Devens facility recently?",
        "What drug-to-antibody ratio are we targeting for our ADC conjugates?",
        "How did the 200L to 2000L scale-up go — did product quality hold?",
        "What viral clearance data do we have for our biologics manufacturing process?",
        "Which analytical methods have been validated for release testing?",
    ],
    "Quality": [
        "Which external suppliers have we audited and qualified recently?",
        "Are there any open corrective action reports that need attention?",
        "What did our most recent quality management review find?",
        "Have we had any out-of-specification investigations this year?",
        "What is our current right-first-time rate in manufacturing?",
        "Is everyone current on their GMP training requirements?",
    ],
    "Medical Affairs": [
        "What did our advisory board experts say about positioning VELORIN?",
        "What are competitors presenting at recent oncology congresses that we should know about?",
        "How are doctors actually using VELORIN in the real world compared to the label?",
        "What questions are healthcare providers asking most about CARALYN?",
        "How many field medical interactions did we have last quarter and on what topics?",
        "What publications are we planning for our pipeline data?",
    ],
    "Commercial": [
        "What market share does VELORIN hold in second-line AML right now?",
        "How is payer access looking for CARALYN — any coverage gaps?",
        "Which competitors are launching products that could impact our sales?",
        "Are we on track for the MG-401 commercial launch?",
        "How many patients are enrolled in our patient assistance programs?",
        "What is our current gross-to-net for VELORIN and CARALYN?",
    ],
    "IT": [
        "Has Benchling passed its system validation for GxP use?",
        "What vulnerabilities were found in our last IT security assessment?",
        "Are there any data migration projects currently in progress?",
        "What integration work has been scoped between Benchling and SharePoint?",
        "Did our last disaster recovery test meet the recovery time objectives?",
        "What new vendor systems are we evaluating?",
    ],
    "Corporate": [
        "How is our R&D budget split across pipeline programs this year?",
        "What new roles are we hiring for and in which departments?",
        "Are there any facility expansion plans for the Devens manufacturing site?",
        "What updates have been made to our business continuity plan?",
        "What is our current cash runway and revenue guidance?",
        "What capital expenditure has been approved for this year?",
    ],
}

default_queries = [
    "What is the safest dose we've tested for MG-401 in primates, and what side effects did we see?",
    "How did CARALYN perform in clinical trials? What was the response rate?",
    "Summarize what the FDA told us in the MG-309 pre-IND meeting and how it changed our plans",
    "What makes our MG-Link linker technology better than competitor approaches?",
    "What is the STELLAR trial and where does enrollment stand?",
    "Walk me through MG-217's combination data with fulvestrant — did we see synergy?",
]

if dept_filter:
    example_queries = []
    for dept in dept_filter:
        example_queries.extend(dept_example_queries.get(dept, [])[:3])
    example_queries = example_queries[:6]
else:
    example_queries = default_queries

if not st.session_state.messages:
    st.markdown('<div class="section-label">Try a question</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, q in enumerate(example_queries[:6]):
        col = cols[i % 3]
        if col.button(q, key=f"example_{i}", use_container_width=True):
            st.session_state["selected_query"] = q
            st.rerun()

# ── Helpers ──────────────────────────────────────────────────────────────────

def compute_confidence(sources_data, answer_text):
    """Compute composite confidence score from source relevance, agreement, and coverage."""
    if not sources_data:
        return 0, "No sources retrieved", {}

    avg_relevance = sum(s["relevance"] for s in sources_data) / len(sources_data)
    top3_relevance = sum(s["relevance"] for s in sources_data[:3]) / min(3, len(sources_data))

    sources_cited = sum(1 for i in range(len(sources_data)) if f"Source {i+1}" in answer_text)
    citation_rate = (sources_cited / len(sources_data)) * 100 if sources_data else 0

    hedging_phrases = ["insufficient", "not enough information", "cannot determine",
                       "no relevant", "unclear from", "not available in"]
    has_hedging = any(p in answer_text.lower() for p in hedging_phrases)

    score = (top3_relevance * 0.5) + (citation_rate * 0.3) + (20 if not has_hedging else 0)
    score = max(0, min(100, int(score)))

    components = {
        "top_source_relevance": int(top3_relevance),
        "sources_cited": sources_cited,
        "total_sources": len(sources_data),
        "hedging_detected": has_hedging,
    }
    if score >= 70:
        label = "High confidence — multiple corroborating sources"
    elif score >= 40:
        label = "Medium confidence — limited source coverage"
    else:
        label = "Low confidence — verify independently"

    return score, label, components


def render_confidence_meter(score, label, components):
    if score >= 70:
        color = "#FFE501"
    elif score >= 40:
        color = "#FFA500"
    else:
        color = "#FF6B6B"

    st.markdown(f"""
    <div class="confidence-meter">
        <div class="cm-score" style="color: {color};">{score}%</div>
        <div class="cm-details">
            <div class="cm-label" style="color: {color};">{label}</div>
            Top source relevance: {components.get('top_source_relevance', 0)}% &bull;
            Sources cited: {components.get('sources_cited', 0)}/{components.get('total_sources', 0)} &bull;
            {"Hedging detected" if components.get('hedging_detected') else "Direct answer"}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sources(sources, allow_drilldown=True):
    st.markdown('<div class="section-label">Sources Retrieved</div>', unsafe_allow_html=True)
    for i, src in enumerate(sources):
        relevance = src["relevance"]
        if relevance >= 70:
            conf_class, conf_label, bar_color = "conf-high", "HIGH", "#FFE501"
        elif relevance >= 40:
            conf_class, conf_label, bar_color = "conf-medium", "MED", "#FFA500"
        else:
            conf_class, conf_label, bar_color = "conf-low", "LOW", "#FF6B6B"

        icon = DEPT_ICONS.get(src["department"], "\U0001F4C4")

        st.markdown(f"""
        <div class="source-card">
            <div class="source-header">
                <span class="source-icon">{icon}</span>
                <span class="source-title">Source {i+1}: {src['filename']}</span>
                <span class="confidence-badge {conf_class}">{conf_label}</span>
            </div>
            <div class="source-meta">
                {src['department']} &bull; {src['system']} &bull; {src['program']}
            </div>
            <div class="relevance-bar-container">
                <div class="relevance-bar-bg">
                    <div class="relevance-bar-fill" style="width: {relevance}%; background: {bar_color};"></div>
                </div>
                <div class="relevance-bar-label" style="color: {bar_color};">{relevance}%</div>
            </div>
            <div class="source-excerpt">{src['excerpt']}</div>
        </div>
        """, unsafe_allow_html=True)

        # Document drill-down
        if allow_drilldown and "full_text" in src:
            with st.expander(f"View full document chunk — {src['filename']}"):
                st.code(src["full_text"], language=None)


def build_export(query, answer, sources, retrieval_ms, generation_ms, confidence_score=0):
    lines = [
        f"# Strategic Synthesis Engine — Query Export",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Query:** {query}",
        f"**Confidence:** {confidence_score}%",
        f"**Retrieval:** {retrieval_ms}ms | **Generation:** {generation_ms}ms",
        "", "---", "",
        "## Answer", "", answer, "", "---", "",
        "## Sources", "",
    ]
    for i, src in enumerate(sources):
        lines.append(f"**Source {i+1}:** {src['filename']} ({src['department']} / {src['system']}) — Relevance: {src['relevance']}%")
    lines.extend(["", "---", "*Exported from Strategic Synthesis Engine — MediGen Corp*"])
    return "\n".join(lines)


# ── Display Chat History ─────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            if "confidence_score" in msg:
                render_confidence_meter(msg["confidence_score"], msg["confidence_label"], msg.get("confidence_components", {}))

            if "retrieval_ms" in msg:
                st.markdown(f"""<div class="perf-bar">
                    <span>Retrieved {len(msg['sources'])} sources in {msg['retrieval_ms']}ms</span>
                    <span>Answer generated in {msg['generation_ms']}ms</span>
                    <span>Total: {msg['retrieval_ms'] + msg['generation_ms']}ms</span>
                </div>""", unsafe_allow_html=True)

            col1, col2 = st.columns([1, 5])
            with col1:
                export_content = build_export(
                    msg.get("query", ""), msg["content"], msg["sources"],
                    msg.get("retrieval_ms", 0), msg.get("generation_ms", 0),
                    msg.get("confidence_score", 0),
                )
                st.download_button(
                    "Export", export_content,
                    file_name=f"sse_export_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                    mime="text/markdown", key=f"export_{msg.get('timestamp', id(msg))}",
                )
            with col2:
                st.feedback("thumbs", key=f"feedback_{msg.get('timestamp', id(msg))}")

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
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

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
                query_texts=[prompt], n_results=n_results,
                where=where_filter, include=["documents", "metadatas", "distances"],
            )
            retrieved_docs = results["documents"][0]
            retrieved_metas = results["metadatas"][0]
            retrieved_distances = results["distances"][0]

            retrieval_ms = int((time.perf_counter() - t0) * 1000)
            st.write(f"Found {len(retrieved_docs)} relevant sources in {retrieval_ms}ms")
            status.update(label=f"Retrieved {len(retrieved_docs)} sources in {retrieval_ms}ms", state="complete")

        # Build source data with full text for drill-down
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
                "full_text": doc,
            })

        # Build LLM context
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
6. At the end, list all sources used with their department and system of origin.
7. After your complete answer, add a section "**Suggested follow-up questions:**" with exactly 3 follow-up questions the user might want to ask next, formatted as a numbered list."""

        llm_messages = []
        recent_history = st.session_state.messages[:-1][-6:]
        for msg in recent_history:
            if msg["role"] in ("user", "assistant"):
                llm_messages.append({"role": msg["role"], "content": msg["content"]})

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
            max_tokens=2000,
            system=system_prompt,
            messages=llm_messages,
        ) as stream:
            for text in stream.text_stream:
                full_answer += text
                answer_placeholder.markdown(full_answer + " ")

        answer_placeholder.markdown(full_answer)
        generation_ms = int((time.perf_counter() - t1) * 1000)

        # Confidence scoring
        conf_score, conf_label, conf_components = compute_confidence(sources_data, full_answer)
        render_confidence_meter(conf_score, conf_label, conf_components)

        # Performance metrics
        st.markdown(f"""<div class="perf-bar">
            <span>Retrieved {len(sources_data)} sources in {retrieval_ms}ms</span>
            <span>Answer generated in {generation_ms}ms</span>
            <span>Total: {retrieval_ms + generation_ms}ms</span>
        </div>""", unsafe_allow_html=True)

        # Export + Feedback row
        timestamp = datetime.now().isoformat()
        export_content = build_export(prompt, full_answer, sources_data, retrieval_ms, generation_ms, conf_score)
        col1, col2 = st.columns([1, 5])
        with col1:
            st.download_button(
                "Export", export_content,
                file_name=f"sse_export_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown", key=f"export_current_{timestamp}",
            )
        with col2:
            feedback = st.feedback("thumbs", key=f"feedback_current_{timestamp}")
            if feedback is not None:
                st.session_state.feedback_log.append({
                    "query": prompt, "rating": "positive" if feedback == 1 else "negative",
                    "timestamp": timestamp,
                })

        # Source cards with drill-down
        render_sources(sources_data, allow_drilldown=True)

        # Update session analytics
        st.session_state.query_count += 1
        st.session_state.total_retrieval_ms += retrieval_ms
        st.session_state.total_generation_ms += generation_ms

        # Store message (strip full_text from sources to save memory)
        sources_for_storage = [{k: v for k, v in s.items() if k != "full_text"} for s in sources_data]
        st.session_state.messages.append({
            "role": "assistant", "content": full_answer,
            "sources": sources_for_storage, "query": prompt,
            "retrieval_ms": retrieval_ms, "generation_ms": generation_ms,
            "confidence_score": conf_score, "confidence_label": conf_label,
            "confidence_components": conf_components, "timestamp": timestamp,
        })

elif not st.session_state.messages and not prompt:
    st.markdown("""
    <div style="text-align: center; padding: 3rem 0; color: #4A4A55;">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">&#8593;</div>
        <div style="font-size: 0.85rem;">Enter a question or select an example above to begin</div>
    </div>
    """, unsafe_allow_html=True)
