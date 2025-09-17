# rag_support_bot_e5.py — simple RAG bot with semantic chunks, typo-fix, T5 paraphrase, and strict fallback
import os, re, json
from typing import List, Dict, Optional, Tuple, Union
from collections import OrderedDict, defaultdict
from difflib import get_close_matches

import chromadb
from sentence_transformers import SentenceTransformer, util as st_util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# =========================
# Config
# =========================
COLLECTION_NAME = "support_docs"
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "intfloat/multilingual-e5-base")

# Semantic chunking
SEM_SIM_THRESHOLD = 0.75
MAX_CHUNK_CHARS   = 1200

# Paraphrase (enabled)
USE_T5_PARAPHRASE = True
T5_MODEL_NAME     = os.getenv("T5_MODEL_NAME", "google/flan-t5-small")
PARAPHRASE_MAX_NEW_TOKENS = 48

# Typo correction
ENABLE_TYPO_FIX   = True
TYPO_MIN_LEN      = 4      # only correct words this length or longer
TYPO_SIM          = 0.84   # closeness required to replace
TYPO_MAX_CHANGES  = 3      # cap replacements per query

# Confidence gates
QINDEX_ACCEPT = 0.65     # accept nearest-question (cluster vote) only if >= this
QINDEX_BORDER = 0.55     # informational (used in UI)
CHUNK_ACCEPT  = 0.60     # accept semantic-chunk only if >= this
MARGIN_ACCEPT = 0.08     # top1 - top2 similarity must be >= this

# Stricter bar for very short/generic queries
MIN_SHARED_CONTENT_TOKENS = 1
MIN_QUERY_CONTENT_TOKENS  = 2
STRICT_SHORT_SIM          = 0.72

# Lexical fuzzy
FUZZY_MATCH_THRESHOLD = 0.6

# Fallback copy
SUPPORT_EMAIL   = "support@gmail.com"
CATEGORIES_HINT = "Orders • Shipping • Returns • Refunds • Payments • Products"


# =========================
# Small utilities
# =========================
_STOP = {
    "the","a","an","and","or","to","of","in","on","for","with","at","by","from",
    "is","are","was","were","be","can","do","does","how","what","when","where",
    "which","who","whom","why","your","our","my","you","we","i"
}

# very generic nouns to ignore for overlap checks
_GENERIC = {
    "product","products","item","items","order","orders","help","support",
    "info","information","thing","things","stuff","page","website","site"
}

def _norm(s: str) -> str:
    if not s:
        return ""
    s = s.casefold()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _tokens(s: str) -> Tuple[str, frozenset]:
    n = _norm(s)
    toks = frozenset(w for w in n.split() if w and w not in _STOP)
    return n, toks

def _content_tokens(toks: frozenset) -> frozenset:
    return frozenset(t for t in toks if t not in _GENERIC)

def _shared_content_ok(q: str, cand: str) -> bool:
    _, q_t = _tokens(q); _, c_t = _tokens(cand)
    q_c = _content_tokens(q_t)
    c_c = _content_tokens(c_t)
    return len(q_c & c_c) >= MIN_SHARED_CONTENT_TOKENS

def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return 0.0 if inter == 0 else inter / len(a | b)

def sentence_split(text: str) -> List[str]:
    t = re.sub(r"\s+", " ", (text or "").strip())
    if not t:
        return []
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", t) if p.strip()]

def _is_junk(s: str) -> bool:
    if not s:
        return True
    t = _norm(s)
    if t in {"answer","question","n/a","na","none","null"}:
        return True
    return len(t) < 8


# =========================
# Intent + synonyms
# =========================
INTENT_KEYWORDS = {
    "shipping": {"ship","shipping","delivery","deliver","arrive","arrival","receive","eta","days","time"},
    "tracking": {"track","tracking","status","where","package","order"},
    "returns": {"return","returns","exchange","rma"},
    "refunds": {"refund","refunds","money","back","credit","credited","reimburse"},
    "payments": {"payment","pay","card","paypal","apple","google"},
}

INTENT_SYNONYMS = {
    "shipping": ["delivery", "arrive", "ETA"],
    "tracking": ["track", "status", "carrier"],
    "returns":  ["exchange", "send back", "RMA"],
    "refunds":  ["money back", "credited", "processing"],
    "payments": ["card", "wallet", "PayPal"],
    "generic":  [],
}

INTENT_TO_CATEGORIES = {
    "shipping": ["shipping_and_delivery"],
    "tracking": ["shipping_and_delivery"],
    "returns":  ["returns_and_exchanges"],
    "refunds":  ["returns_and_exchanges"],
    "payments": ["payments"],
    "generic":  [],
}

def _classify_intent(text: str) -> str:
    t = _norm(text)
    scores = {}
    for intent, kws in INTENT_KEYWORDS.items():
        sc = sum(1 for k in kws if k in t)
        if sc > 0:
            scores[intent] = sc
    return max(scores, key=scores.get) if scores else "generic"

def _expand_with_synonyms(query: str, intent: str) -> str:
    syns = INTENT_SYNONYMS.get(intent, [])
    tail = " ".join(syns[:3])
    return f"{query} {tail}".strip()


# =========================
# Optional paraphraser
# =========================
class T5Paraphraser:
    def __init__(self, model_name: str):
        import importlib
        torch = importlib.import_module("torch")
        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def paraphrase(self, q: str) -> str:
        try:
            prompt = (
                "Paraphrase the following user question into a short, standard form "
                "without changing its meaning. Keep one sentence.\n\n"
                f"Question: {q}\nParaphrase:"
            )
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            with self._torch.no_grad():
                out = self.model.generate(**inputs, do_sample=False, num_beams=1,
                                          max_new_tokens=PARAPHRASE_MAX_NEW_TOKENS, temperature=1.0)
            text = self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
            if "?" in q and "?" not in text:
                text = text.rstrip(".") + "?"
            return text if len(text) > 3 else q
        except Exception:
            return q


# =========================
# Semantic chunking
# =========================
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

def semantic_chunks(text: str, model: SentenceTransformer,
                    sim_th: float = 0.7, max_chars: int = 500) -> List[str]:
    sents = sentence_split(text)
    if not sents: return []
    E = model.encode(sents, show_progress_bar=False, convert_to_numpy=True)
    E /= np.linalg.norm(E, axis=1, keepdims=True)  # normalize
    sims = (E[:-1] * E[1:]).sum(1)  # consecutive cosine sims

    chunks, cur, cur_len = [], [sents[0]], len(sents[0])
    for s, sim in zip(sents[1:], sims):
        if sim >= sim_th and cur_len + 1 + len(s) <= max_chars:
            cur.append(s); cur_len += 1 + len(s)
        else:
            chunks.append(" ".join(cur)); cur, cur_len = [s], len(s)
    chunks.append(" ".join(cur))
    return chunks


# =========================
# E5 prefixes
# =========================
def _as_query(texts: List[str]) -> List[str]:
    return [f"query: {t}" for t in texts]

def _as_passage(texts: List[str]) -> List[str]:
    return [f"passage: {t}" for t in texts]


# =========================
# Bot
# =========================
class RAGCustomerSupportBot:
    def __init__(self, persistent_path: str, fresh_start: bool = True):
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        self.embed = SentenceTransformer(EMBED_MODEL_NAME)

        os.makedirs(persistent_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persistent_path)

        if fresh_start:
            try:
                self.client.delete_collection(name=COLLECTION_NAME)
            except Exception:
                pass

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

        self._fast_exact: Dict[str, str] = {}
        self._fast_fuzzy: List[Tuple[frozenset, str]] = []
        self.parent_full: Dict[str, str] = {}
        self.parent_question: Dict[str, str] = {}

        # Paraphrase clusters
        self._q_to_cluster: Dict[str, str] = {}
        self._cluster_to_canonical: Dict[str, str] = {}
        self._cluster_to_answer: Dict[str, str] = {}

        # Track last selection (for margin/overlap checks)
        self._last_qindex_best_cid: Optional[str] = None
        self._last_qindex_second_best: float = 0.0
        self._last_chunk_parent_id: Optional[str] = None

        # KB vocabulary for typo-fix
        self._kb_vocab: frozenset[str] = frozenset()

        # Optional paraphraser
        self.t5 = T5Paraphraser(T5_MODEL_NAME) if USE_T5_PARAPHRASE else None
        self._p_cache: "OrderedDict[str,str]" = OrderedDict()

    # ----- KB loading -----
    def load_kb_json(self, path: str) -> bool:
        if not os.path.exists(path):
            print(f"KB not found: {path}")
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                data: Union[Dict, List] = json.load(f)
        except Exception as e:
            print(f"JSON error: {e}")
            return False

        pairs: List[Tuple[str, str, str]] = []  # (category, question, answer)
        if isinstance(data, dict):
            for cat, items in data.items():
                if isinstance(items, list):
                    for it in items:
                        if isinstance(it, dict):
                            q, a = it.get("question"), it.get("answer")
                            if q and a and not _is_junk(q) and not _is_junk(a):
                                pairs.append((cat, q.strip(), a.strip()))
        elif isinstance(data, list):
            for it in data:
                if isinstance(it, dict) and "question" in it and "answer" in it:
                    pairs.append(("All", it["question"].strip(), it["answer"].strip()))

        if not pairs:
            print("No valid Q/A found.")
            return False

        self._ingest_pairs(pairs)
        return True

    # ----- Ingestion + clustering -----
    def _ingest_pairs(self, pairs: List[Tuple[str, str, str]]) -> None:
        questions = [q for (_, q, _) in pairs]
        q_embs = self.embed.encode(_as_passage(questions), show_progress_bar=False)

        cluster_id_of_idx: Dict[int, str] = {}
        cluster_reps: Dict[str, int] = {}
        cluster_members: Dict[str, List[int]] = defaultdict(list)

        next_cluster = 0
        for i in range(len(questions)):
            assigned = False
            for cid, rep_idx in cluster_reps.items():
                sim = float(st_util.cos_sim(q_embs[i], q_embs[rep_idx]).item())
                if sim >= 0.88:
                    cluster_id_of_idx[i] = cid
                    cluster_members[cid].append(i)
                    assigned = True
                    break
            if not assigned:
                cid = f"clu_{next_cluster}"; next_cluster += 1
                cluster_reps[cid] = i
                cluster_id_of_idx[i] = cid
                cluster_members[cid].append(i)

        for cid, idxs in cluster_members.items():
            best_idx = max(idxs, key=lambda k: len(questions[k]))
            self._cluster_to_canonical[cid] = questions[best_idx]

        ids, docs, metas = [], [], []
        kb_vocab_tokens = set()

        for i, (category, q, a) in enumerate(pairs):
            if _is_junk(q) or _is_junk(a):
                continue
            base_id = f"kb_{i}"
            cid = cluster_id_of_idx[i]

            self._q_to_cluster[_tokens(q)[0]] = cid
            self._cluster_to_answer.setdefault(cid, a)
            self.parent_full[base_id] = a
            self.parent_question[base_id] = q

            # collect content tokens for typo-fix vocab
            _, q_toks = _tokens(q)
            for t in _content_tokens(q_toks):
                kb_vocab_tokens.add(t)

            # qa_full
            ids.append(base_id)
            docs.append(f"Q: {q}\nA: {a}")
            metas.append({
                "type": "qa_full",
                "question": q,
                "answer": a,
                "parent_id": base_id,
                "category": category,
                "cluster_id": cid,
                "source": "kb_json",
            })

            # q_index
            qid = f"{base_id}_qidx"
            ids.append(qid)
            docs.append(q)
            metas.append({
                "type": "q_index",
                "question": q,
                "answer": a,
                "parent_id": base_id,
                "category": category,
                "cluster_id": cid,
                "source": "kb_json",
            })

            # semantic chunks
            chunks = semantic_chunks(a, self.embed)
            if not chunks:
                chunks = [a]
            for j, ch in enumerate(chunks):
                cid_chunk = f"{base_id}_chunk_{j}"
                ids.append(cid_chunk)
                docs.append(ch)
                metas.append({
                    "type": "semantic_chunk",
                    "question": q,
                    "answer": a,
                    "parent_id": base_id,
                    "chunk_index": j,
                    "category": category,
                    "cluster_id": cid,
                    "source": "kb_json",
                })

            # fast lexical
            self._add_fast_entry(q, a)

        # add embeddings to Chroma
        embs = self.embed.encode(_as_passage(docs), show_progress_bar=False)
        self.collection.add(ids=ids, documents=docs, embeddings=embs.tolist(), metadatas=metas)

        # extend vocab with intent synonyms
        for syns in INTENT_SYNONYMS.values():
            for s in syns:
                kb_vocab_tokens.add(_norm(s))
        self._kb_vocab = frozenset(kb_vocab_tokens)

    def _add_fast_entry(self, q: str, a: str) -> None:
        if _is_junk(q) or _is_junk(a):
            return
        n, toks = _tokens(q)
        if not n:
            return
        self._fast_exact.setdefault(n, a)
        self._fast_fuzzy.append((toks, a))
        if len(self._fast_fuzzy) > 5000:
            self._fast_fuzzy = self._fast_fuzzy[-5000:]

    # ----- Query -----
    def generate_response(self, user_q: str) -> str:
        if not user_q or len(user_q.strip()) < 2:
            return self._fallback()

        intent = _classify_intent(user_q)

        # typo-fix → paraphrase
        q0 = user_q.strip()
        q_fixed = self._typo_fix(q0)
        q_par = self._maybe_paraphrase(q_fixed)

        # lexical fast path
        fast = self._fast_answer(q_par) or self._fast_answer(q_fixed) or self._fast_answer(q0)
        if fast and not _is_junk(fast):
            return fast

        # synonyms expansion
        q_for_embed = _expand_with_synonyms(q_par, intent)

        # stricter bars for very short/generic queries
        _, q_toks = _tokens(q_par)
        q_content_len = len(_content_tokens(q_toks))
        short_query = q_content_len < MIN_QUERY_CONTENT_TOKENS

        # 1) nearest-question + cluster vote (strict gates + margin + overlap)
        best_ans, best_score = self._qindex_cluster_vote(q_for_embed, intent)
        accept_qidx = False
        if best_ans:
            best_cid = self._last_qindex_best_cid
            second = self._last_qindex_second_best
            if best_cid:
                canonical_q = self._cluster_to_canonical.get(best_cid, "")
                overlap_ok = _shared_content_ok(q_par, canonical_q)
                margin_ok = (best_score - float(second)) >= MARGIN_ACCEPT
                sim_ok = best_score >= (STRICT_SHORT_SIM if short_query else QINDEX_ACCEPT)
                accept_qidx = overlap_ok and margin_ok and sim_ok
        if accept_qidx:
            return best_ans

        # 2) semantic-chunk fallback (strict gates + overlap)
        chunk_ans, chunk_score = self._chunk_search_best(q_for_embed, intent)
        accept_chunk = False
        if chunk_ans:
            pid = self._last_chunk_parent_id
            parent_q = self.parent_question.get(pid or "", "")
            overlap_ok = _shared_content_ok(q_par, parent_q)
            sim_ok = chunk_score >= (STRICT_SHORT_SIM if short_query else CHUNK_ACCEPT)
            accept_chunk = overlap_ok and sim_ok
        if accept_chunk:
            return chunk_ans

        return self._fallback()

    # helpers
    def _maybe_paraphrase(self, q: str) -> str:
        if not USE_T5_PARAPHRASE or self.t5 is None or len(q) < 3:
            return q
        key = q.strip().lower()
        if key in self._p_cache:
            v = self._p_cache.pop(key); self._p_cache[key] = v
            return v
        v = self.t5.paraphrase(q)
        self._p_cache[key] = v
        if len(self._p_cache) > 500:
            self._p_cache.popitem(last=False)
        return v

    def _typo_fix(self, q: str) -> str:
        """Replace misspelled words with the closest KB token (conservative)."""
        if not ENABLE_TYPO_FIX or not q or not self._kb_vocab:
            return q
        words = _norm(q).split()
        changed = 0
        out = []
        for w in words:
            if len(w) < TYPO_MIN_LEN or w in self._kb_vocab:
                out.append(w); continue
            cand = get_close_matches(w, self._kb_vocab, n=1, cutoff=TYPO_SIM)
            if cand:
                out.append(cand[0]); changed += 1
                if changed >= TYPO_MAX_CHANGES:
                    # append remaining original words and stop
                    idx = len(out)
                    out.extend(words[idx:])
                    break
            else:
                out.append(w)
        return " ".join(out)

    def _fast_answer(self, user_q: str) -> Optional[str]:
        n, toks = _tokens(user_q)
        if not n:
            return None
        if n in self._fast_exact:
            cand = self._fast_exact[n]
            return None if _is_junk(cand) else cand
        best_score, best_ans = 0.0, None
        for qtoks, ans in self._fast_fuzzy:
            sc = _jaccard(toks, qtoks)
            if sc > best_score and not _is_junk(ans):
                best_score, best_ans = sc, ans
        return best_ans if best_score >= FUZZY_MATCH_THRESHOLD else None

    def _qindex_cluster_vote(self, query: str, intent: str, top_k: int = 12) -> Tuple[Optional[str], float]:
        # reset tracking
        self._last_qindex_best_cid = None
        self._last_qindex_second_best = 0.0

        q_emb = self.embed.encode(_as_query([query]), show_progress_bar=False).tolist()
        where_q = {"type": {"$eq": "q_index"}}
        pref = INTENT_TO_CATEGORIES.get(intent, [])

        for pass_i in range(2):
            kwargs = {
                "query_embeddings": q_emb,
                "n_results": top_k,
                "include": ["metadatas", "distances", "documents"],
                "where": where_q
            }
            if pass_i == 0 and pref:
                kwargs["where"] = {"$and": [where_q, {"category": {"$in": pref}}]}
            res = self.collection.query(**kwargs)
            metas = res.get("metadatas", [[]])[0]
            dists = res.get("distances", [[]])[0]

            cluster_best: Dict[str, float] = {}
            for m, d in zip(metas, dists):
                cid = (m or {}).get("cluster_id")
                if not cid:
                    continue
                sim = 1.0 - float(d or 0.0)
                if cid not in cluster_best or sim > cluster_best[cid]:
                    cluster_best[cid] = sim

            if cluster_best:
                items = sorted(cluster_best.items(), key=lambda x: x[1], reverse=True)
                best_cid, best_sim = items[0]
                second_best = items[1][1] if len(items) > 1 else 0.0
                self._last_qindex_best_cid = best_cid
                self._last_qindex_second_best = second_best
                best_ans = self._cluster_to_answer.get(best_cid)
                return (best_ans, best_sim)

        return (None, 0.0)

    def _chunk_search_best(self, query: str, intent: str, top_k: int = 8) -> Tuple[Optional[str], float]:
        # reset tracking
        self._last_chunk_parent_id = None

        q_emb = self.embed.encode(_as_query([query]), show_progress_bar=False).tolist()
        where_c = {"type": {"$eq": "semantic_chunk"}}
        pref = INTENT_TO_CATEGORIES.get(intent, [])

        best_answer, best_score = None, 0.0

        for pass_i in range(2):
            kwargs = {
                "query_embeddings": q_emb,
                "n_results": top_k,
                "include": ["metadatas", "distances", "documents"],
                "where": where_c
            }
            if pass_i == 0 and pref:
                kwargs["where"] = {"$and": [where_c, {"category": {"$in": pref}}]}
            res = self.collection.query(**kwargs)
            metas = res.get("metadatas", [[]])[0]
            dists = res.get("distances", [[]])[0]

            best_by_parent: Dict[str, Tuple[float, str]] = {}
            for m, d in zip(metas, dists):
                pid = (m or {}).get("parent_id")
                if not pid:
                    continue
                sim = 1.0 - float(d or 0.0)
                cur = best_by_parent.get(pid)
                if not cur or sim > cur[0]:
                    best_by_parent[pid] = (sim, self.parent_full.get(pid, ""))

            if best_by_parent:
                best_pid, (sim, full_ans) = max(best_by_parent.items(), key=lambda x: x[1][0])
                if full_ans and not _is_junk(full_ans) and sim > best_score:
                    best_answer, best_score = full_ans.strip(), sim
                    self._last_chunk_parent_id = best_pid

            if best_answer:
                break

        return best_answer, best_score

    def _fallback(self) -> str:
        return ("Unfortunately it's out of my scope. Here is the support email: "
                f"{SUPPORT_EMAIL}. You can contact them anytime and they'll definitely "
                "be able to help better! So, can I help you with something else?")


# =========================
# Example
# =========================
if __name__ == "__main__":
    kb_path   = os.environ.get("FAQ_JSON_PATH", "faq_dataset.json")
    store_dir = os.environ.get("CHROMA_STORE", "./chroma_store")

    bot = RAGCustomerSupportBot(persistent_path=store_dir, fresh_start=True)
    ok = bot.load_kb_json(kb_path)
    print(f"KB loaded: {ok}")

    tests = [
        "How long will it take for my order to arrive?",
        "When will my pakage arive?",              # typo → fix → answer
        "how long dos shipin tek",                 # heavy typos → fix → answer
        "Do you offer express shipping?",
        "refund timeline please",
        "what do you sell",                        # out of scope → fallback
        "what is the products that you sel",       # out of scope → fallback
    ]
    for q in tests:
        print(f"\nQ: {q}\nA: {bot.generate_response(q)}\n" + "-"*50)

