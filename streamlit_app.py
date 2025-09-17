# streamlit_app.py — clean Streamlit UI aligned with the updated bot
import os
import time
import json
import datetime
from typing import List, Tuple, Union, Dict, Any

import streamlit as st

from full_rag_bot import (
    RAGCustomerSupportBot,
    _classify_intent,
    _expand_with_synonyms,
    INTENT_TO_CATEGORIES,
    USE_T5_PARAPHRASE,
    QINDEX_ACCEPT,
    QINDEX_BORDER,
    CHUNK_ACCEPT,
)

st.set_page_config(page_title="Customer Support Bot", layout="wide")

# ---------------------------
# Session state
# ---------------------------
def init_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "bot" not in st.session_state:
        st.session_state.bot = None
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session_{int(time.time())}"
    if "kb_loaded_from" not in st.session_state:
        st.session_state.kb_loaded_from = "none"
    if "faq_categories" not in st.session_state:
        st.session_state.faq_categories: Dict[str, List[str]] = {}
    if "selected_category" not in st.session_state:
        st.session_state.selected_category = None
    if "last_selected_question" not in st.session_state:
        st.session_state.last_selected_question = None
    if "last_selected_category" not in st.session_state:
        st.session_state.last_selected_category = None
    if "latest_trace" not in st.session_state:
        st.session_state.latest_trace = {}

init_session_state()

# ---------------------------
# Paths
# ---------------------------
BASE_DIR   = os.path.abspath(os.path.dirname(__file__))
CHROMA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "chroma_db"))
os.makedirs(CHROMA_DIR, exist_ok=True)

def resolve_kb_path() -> str:
    env_json = os.environ.get("FAQ_JSON_PATH")
    if env_json:
        return os.path.abspath(env_json)
    candidates = [
        os.path.join(BASE_DIR, "faq_dataset.json"),
        os.path.join(BASE_DIR, "Ecommerce_FAQ_Chatbot_dataset.json"),
        os.path.join(BASE_DIR, "..", "Ecommerce_FAQ_Chatbot_dataset.json"),
        os.path.join(BASE_DIR, "..", "data", "Ecommerce_FAQ_Chatbot_dataset.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)
    return ""

JSON_KB_PATH = resolve_kb_path()

# ---------------------------
# KB parsing (JSON only)
# ---------------------------
def normalize(s: str) -> str:
    return " ".join((s or "").lower().strip().split())

def parse_categories_from_json(path: str) -> Dict[str, List[str]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data: Union[Dict[str, Any], List[Dict[str, Any]]] = json.load(f)
    except Exception:
        return {}

    cats: Dict[str, List[str]] = {}

    def add_q(cat: str, item: Dict[str, Any]):
        q = item.get("question")
        a = item.get("answer")
        if isinstance(q, str) and isinstance(a, str) and q.strip() and a.strip():
            cats.setdefault(cat, []).append(q.strip())

    if isinstance(data, dict) and "questions" in data and isinstance(data["questions"], list):
        for it in data["questions"]:
            if isinstance(it, dict):
                add_q("All", it)
        return cats

    if isinstance(data, dict):
        for cat, lst in data.items():
            if isinstance(lst, list):
                for it in lst:
                    if isinstance(it, dict):
                        add_q(cat, it)
        return cats

    if isinstance(data, list):
        has_blocks = False
        for block in data:
            if isinstance(block, dict) and "faqs" in block and isinstance(block["faqs"], list):
                has_blocks = True
                cat = block.get("category") or block.get("name") or "All"
                for it in block["faqs"]:
                    if isinstance(it, dict):
                        add_q(cat, it)
        if has_blocks:
            return cats
        for it in data:
            if isinstance(it, dict) and "question" in it and "answer" in it:
                add_q("All", it)
    return cats

CACHE_VERSION = "ui_e5_semantic_thresholds_v1"

@st.cache_resource(show_spinner=False)
def load_bot_and_kb(cache_version: str = CACHE_VERSION):
    bot = RAGCustomerSupportBot(persistent_path=CHROMA_DIR, fresh_start=True)
    kb_ok = False
    kb_used = "none"
    categories: Dict[str, List[str]] = {}

    if JSON_KB_PATH and os.path.exists(JSON_KB_PATH):
        kb_ok = bot.load_kb_json(JSON_KB_PATH)
        if kb_ok:
            kb_used = os.path.relpath(JSON_KB_PATH, BASE_DIR)
            categories = parse_categories_from_json(JSON_KB_PATH)

    return bot, categories, kb_ok, kb_used

# ---------------------------
# Backend trace (light mirror of routing)
# ---------------------------
def trace_backend(question: str) -> Dict[str, Any]:
    trace: Dict[str, Any] = {
        "question": question,
        "start_ts": time.time(),
        "phases": [],
        "intent": None,
        "lexical_hit": None,
        "lexical_text": None,
        "routing": {},
        "final": {},
    }

    bot = st.session_state.bot
    t0 = time.time()

    # Intent
    t_intent0 = time.time()
    intent = _classify_intent(question)
    t_intent1 = time.time()
    trace["intent"] = intent
    trace["phases"].append(("intent", t_intent1 - t_intent0))

    # Optional paraphrase
    t_para0 = time.time()
    try:
        q_par = bot._maybe_paraphrase(question)
    except Exception:
        q_par = question
    t_para1 = time.time()
    if USE_T5_PARAPHRASE:
        trace["phases"].append(("paraphrase", t_para1 - t_para0))

    # Lexical fast path
    t_lex0 = time.time()
    try:
        lex = bot._fast_answer(q_par) or bot._fast_answer(question)
    except Exception:
        lex = None
    t_lex1 = time.time()
    trace["lexical_hit"] = (lex is not None)
    trace["lexical_text"] = lex
    trace["phases"].append(("lexical", t_lex1 - t_lex0))

    # Build expanded query
    q_for_embed = _expand_with_synonyms(q_par, intent)

    # q_index + cluster vote
    t_qidx0 = time.time()
    best_qidx_ans, best_qidx_score = None, 0.0
    try:
        best_qidx_ans, best_qidx_score = bot._qindex_cluster_vote(q_for_embed, intent)
    except Exception:
        pass
    t_qidx1 = time.time()
    trace["phases"].append(("q_index_vote", t_qidx1 - t_qidx0))

    # chunk fallback
    t_chunk0 = time.time()
    best_chunk_ans, best_chunk_score = None, 0.0
    try:
        best_chunk_ans, best_chunk_score = bot._chunk_search_best(q_for_embed, intent)
    except Exception:
        pass
    t_chunk1 = time.time()
    trace["phases"].append(("chunk_search", t_chunk1 - t_chunk0))

    # Routing preview
    will_answer_qidx  = best_qidx_ans is not None and best_qidx_score >= QINDEX_ACCEPT
    will_answer_chunk = best_chunk_ans is not None and best_chunk_score >= CHUNK_ACCEPT
    trace["routing"] = {
        "intent_categories_preferred": INTENT_TO_CATEGORIES.get(intent, []),
        "qindex_best_score": round(best_qidx_score, 3),
        "qindex_accept": QINDEX_ACCEPT,
        "qindex_border": QINDEX_BORDER,
        "chunk_best_score": round(best_chunk_score, 3),
        "chunk_accept": CHUNK_ACCEPT,
        "will_answer_qidx": will_answer_qidx,
        "will_answer_chunk": will_answer_chunk,
        "t5_enabled": USE_T5_PARAPHRASE,
    }

    # Final call (actual bot)
    t_call0 = time.time()
    response = bot.generate_response(question)
    t_call1 = time.time()

    # Fallback detection
    try:
        fallback_text = bot._fallback()
        actually_fallback = (normalize(response) == normalize(fallback_text))
    except Exception:
        actually_fallback = False

    trace["final"] = {
        "response": response,
        "actually_fallback": actually_fallback,
        "total_latency_sec": t_call1 - t0,
        "generate_latency_sec": t_call1 - t_call0,
    }

    trace["end_ts"] = t_call1
    return trace

def render_backend_sidebar(latest_trace: Dict[str, Any]):
    with st.sidebar:
        st.header("⚙️ Backend view")
        if not latest_trace:
            st.caption("Ask something to see the backend timeline here.")
            return
        st.markdown(f"**Question:** {latest_trace.get('question','')}")
        st.markdown(f"**Intent:** `{latest_trace.get('intent')}`")

        st.subheader("Timeline")
        for name, dur in latest_trace.get("phases", []):
            st.write(f"- **{name}**: {dur*1000:.1f} ms")

        r = latest_trace.get("routing", {})
        with st.expander("Routing details"):
            st.write(f"- Preferred categories: `{r.get('intent_categories_preferred')}`")
            st.write(f"- q_index best score: **{r.get('qindex_best_score', 0)}** "
                     f"(accept **{r.get('qindex_accept')}**, border **{r.get('qindex_border')}**)")
            st.write(f"- chunk best score: **{r.get('chunk_best_score', 0)}** "
                     f"(accept **{r.get('chunk_accept')}**)")
            st.write(f"- Will answer via q_index: **{r.get('will_answer_qidx')}**")
            st.write(f"- Will answer via chunk: **{r.get('will_answer_chunk')}**")
            st.write(f"- T5 paraphrase enabled: **{r.get('t5_enabled')}**")

        f = latest_trace.get("final", {})
        st.subheader("Finalization")
        st.write(f"- Actually fallback: **{f.get('actually_fallback')}**")
        st.write(f"- generate_response latency: **{f.get('generate_latency_sec', 0):.3f}s**")
        st.write(f"- Total latency: **{f.get('total_latency_sec', 0):.3f}s**")

# ---------------------------
# UI helpers
# ---------------------------
def render_chat_history():
    for chat in st.session_state.chat_history:
        q = chat.get("question", "")
        if q:
            with st.chat_message("user"):
                st.write(q)
        r = chat.get("response")
        if r is not None:
            with st.chat_message("assistant"):
                st.write(r)

def answer_and_append(question_text: str):
    trace = trace_backend(question_text)
    response = trace["final"]["response"]
    st.session_state.chat_history.append({
        "question": question_text,
        "response": response,
        "timestamp": datetime.datetime.now(),
        "trace": trace,
    })
    st.session_state["latest_trace"] = trace

def render_category_picker():
    cats = st.session_state.faq_categories or {}
    if not cats:
        return
    st.markdown("### Explore FAQs by category")
    cat_names = sorted(cats.keys())
    cols_per_row = 3
    rows = (len(cat_names) + cols_per_row - 1) // cols_per_row
    idx = 0
    for _ in range(rows):
        cols = st.columns(cols_per_row)
        for c in cols:
            if idx >= len(cat_names):
                break
            name = cat_names[idx]
            with c:
                if st.button(f"{name}", key=f"cat_{name}"):
                    st.session_state.selected_category = name
                    st.rerun()
            idx += 1

def render_category_questions_dropdown():
    cats = st.session_state.faq_categories or {}
    sel = st.session_state.selected_category
    if not sel or sel not in cats:
        return
    st.markdown(f"#### {sel} FAQs")
    options = cats[sel]
    if not options:
        st.info("No questions in this category yet.")
        return

    placeholder = "— Select a question —"
    chosen = st.selectbox(
        "Pick a question to ask:",
        options=[placeholder] + options,
        index=0,
        key=f"sel_q_{sel}"
    )

    if chosen and chosen != placeholder:
        prev_q = st.session_state.get("last_selected_question")
        prev_c = st.session_state.get("last_selected_category")
        if chosen != prev_q or sel != prev_c:
            st.session_state.last_selected_question = chosen
            st.session_state.last_selected_category = sel
            answer_and_append(chosen)
            st.rerun()

# ---------------------------
# UI
# ---------------------------
st.title("Customer Support Bot")

render_backend_sidebar(st.session_state.get("latest_trace", {}))

col_reset, _ = st.columns([1, 9])
with col_reset:
    if st.button("Clear chat"):
        st.session_state.chat_history = []
        st.session_state.selected_category = None
        st.session_state.last_selected_question = None
        st.session_state.last_selected_category = None
        st.session_state["latest_trace"] = {}
        st.rerun()

if st.session_state.bot is None:
    bot, categories, kb_ok, kb_used = load_bot_and_kb()
    st.session_state.bot = bot
    st.session_state.faq_categories = categories
    st.session_state.kb_loaded_from = kb_used

render_chat_history()
render_category_picker()
render_category_questions_dropdown()

if st.session_state.chat_history:
    last = st.session_state.chat_history[-1]
    if last.get("response") and ("rate this conversation" in last["response"].lower()):
        feedback_val = None
        try:
            fb = st.feedback("thumbs", key=f"fb_{len(st.session_state.chat_history)}")
            if fb is not None:
                feedback_val = "up" if fb else "down"
        except Exception:
            c1, c2 = st.columns(2)
            with c1:
                if st.button("👍 Helpful", key=f"up_{len(st.session_state.chat_history)}"):
                    feedback_val = "up"
            with c2:
                if st.button("👎 Needs improvement", key=f"down_{len(st.session_state.chat_history)}"):
                    feedback_val = "down"
        if feedback_val:
            last["feedback"] = feedback_val
            st.toast("Thanks for your feedback!")

user_question = st.chat_input("Ask me anything about orders, refunds, shipping, payments...")
if user_question:
    if normalize(user_question) == "hi":
        st.session_state.chat_history.append({
            "question": user_question,
            "response": "Hi! Choose a category above to browse FAQs, or ask me anything directly.",
            "timestamp": datetime.datetime.now(),
        })
        st.session_state["latest_trace"] = {
            "question": user_question,
            "intent": "greeting",
            "phases": [],
            "routing": {},
            "final": {"response": "Hi!", "actually_fallback": False, "total_latency_sec": 0.0, "generate_latency_sec": 0.0},
        }
        st.rerun()
    else:
        answer_and_append(user_question)
        st.rerun()
