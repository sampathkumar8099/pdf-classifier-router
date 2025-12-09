# app.py
"""
AI PDF Classifier & Router (single-file)
- PDF-only input (PyPDF2)
- OpenRouter (optional) + robust local evidence scorer
- Routes to 1 dept if confident, else up to 2 depts
- Provides solid, evidence-backed explanation with page-level matches
- Uses Plotly for visualization
- Sidebar dev API key (env or paste)
"""

import os
import io
import re
import json
import math
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PyPDF2 import PdfReader
from typing import List, Dict, Tuple, Optional

# ---------- Config ----------
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "amazon/nova-2-lite-v1:free"
API_KEY_ENV = "OPENROUTER_API_KEY"

# thresholds (tuneable) — changed defaults per your request
SINGLE_CONFIDENCE_THRESHOLD = 90    # >= this -> route to that single dept
SECONDARY_MIN_THRESHOLD = 10        # second-highest must be >= this to consider 2-dept routing
MAX_SECONDARY = 2                   # at most 2 departments
DEFAULT_MODEL_WEIGHT = 1.0          # model weight default as shown (1.0)

st.set_page_config(page_title="PDF Classifier & Router", layout="wide", page_icon="📄")
st.set_option('client.showErrorDetails', True)

# ---------- Helpers: PDF extraction ----------
def extract_pages_from_pdf_bytes(pdf_bytes: bytes) -> List[str]:
    """Return list of page-texts (strings)."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        return pages
    except Exception:
        return []

# ---------- Local evidence-based scorer ----------
KEYWORDS = {
    "Invoice": [r"\binvoice\b", r"invoice number", r"\bdue date\b", r"amount due", r"total due", r"bill to", r"remit to", r"tax\b", r"subtotal"],
    "Purchase Order": [r"purchase order", r"\bpo\b", r"po number", r"qty\b", r"unit price", r"ship to", r"vendor", r"line item"],
    "Contract": [r"\bagreement\b", r"terms and conditions", r"\bwhereas\b", r"\beffective date\b", r"\bparty\b", r"\bsignature\b", r"\bterm\b"]
}

def find_evidence_by_page(pages: List[str]) -> Dict[str, Dict]:
    """
    For each doc type, count matches across pages and capture example snippets with page numbers.
    Returns dict:
      { "Invoice": {"count": int, "examples": [(page_no, snippet), ...]}, ... }
    """
    results = {}
    for dtype, pats in KEYWORDS.items():
        total = 0
        examples = []
        for p_idx, ptext in enumerate(pages):
            page_lower = ptext.lower()
            for pat in pats:
                for m in re.finditer(pat, page_lower):
                    total += 1
                    start = max(0, m.start() - 40)
                    end = min(len(page_lower), m.end() + 40)
                    snippet = page_lower[start:end].strip().replace("\n", " ")
                    examples.append({"page": p_idx+1, "phrase": m.group(0), "snippet": snippet})
        results[dtype] = {"count": total, "examples": examples}
    return results

def local_scores_from_evidence(evidence: Dict[str, Dict]) -> Dict[str, int]:
    """
    Normalize counts into 0..100 integer scores.
    If no matches present, produce low baseline scores.
    """
    counts = {k: evidence[k]["count"] for k in evidence}
    max_c = max(counts.values()) if counts else 0
    if max_c == 0:
        # baseline low uncertainty scores
        return {k: 8 for k in counts}
    # produce a non-linear mapping: score = 20 + (count/max_count)*80
    scores = {}
    for k, c in counts.items():
        if c == 0:
            sc = 6
        else:
            frac = c / max_c
            sc = int(round(20 + frac * 80))
        sc = max(0, min(100, sc))
        scores[k] = sc
    return scores

# ---------- LLM prompt (multi-label with scores) ----------
def craft_messages_for_multilabel(document_text: str) -> list:
    system = (
        "You are an evidence-oriented document analyzer. Analyze the document and "
        "return a single JSON object only with keys: predictions (list), top_types (list), notes (string).\n\n"
        "predictions: list of objects each {document_type: 'Invoice'|'Purchase Order'|'Contract', score: integer 0-100, reason: short 1-2 sentence reason referencing evidence}\n"
        "top_types: list of document types with highest non-zero scores sorted desc (max 2).\n"
        "notes: optional short note about ambiguity.\n\n"
        "Return strictly valid JSON and nothing else."
    )
    user = f"DOCUMENT_TEXT_START\n{document_text}\nDOCUMENT_TEXT_END"
    return [{"role":"system","content":system},{"role":"user","content":user}]

def call_openrouter(messages: list, api_key: str, model: str = MODEL, timeout: int = 60) -> Tuple[bool, dict]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": 0.0, "max_tokens": 900}
    try:
        r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
    except requests.RequestException as e:
        return False, {"error": str(e), "raw": None}
    try:
        j = r.json()
    except Exception:
        return False, {"error": f"Non-JSON response ({r.status_code})", "raw": r.text}
    if not r.ok:
        return False, {"error": j.get("error") or j.get("message") or str(j), "raw": j}
    # extract content
    try:
        choices = j.get("choices", [])
        if not choices:
            return False, {"error": "No choices returned", "raw": j}
        first = choices[0]
        content = None
        if isinstance(first.get("message"), dict):
            content = first["message"].get("content")
        if not content:
            content = first.get("text") or first.get("delta", {}).get("content") or json.dumps(first)
        return True, {"content": content, "raw": j}
    except Exception as e:
        return False, {"error": str(e), "raw": j}

def parse_model_json(output: str) -> Tuple[bool, Optional[dict]]:
    if not output: return False, None
    s = output.strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return True, obj
    except Exception:
        pass
    # fallback: find balanced {...}
    start = s.find("{")
    if start == -1: return False, None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{": depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                cand = s[start:i+1]
                try:
                    obj = json.loads(cand)
                    return True, obj
                except Exception:
                    return False, None
    return False, None

# ---------- Decision logic ----------
def combine_scores(model_scores: Dict[str, int], local_scores: Dict[str, int], model_weight: float = 0.6) -> Dict[str, int]:
    """
    Weighted average: final = model_weight*model + (1-model_weight)*local
    If model_scores is empty (no model), rely on local entirely.
    """
    final = {}
    if not model_scores:
        return local_scores.copy()
    for k in local_scores.keys():
        m = model_scores.get(k, 0)
        l = local_scores.get(k, 0)
        val = model_weight * m + (1 - model_weight) * l
        final[k] = int(round(max(0, min(100, val))))
    return final

def choose_routing(final_scores: Dict[str, int]) -> Tuple[List[str], str]:
    """
    Return list of 1 or 2 routing departments to send to (max 2), and reason_short label.
    Logic:
      - If top score >= SINGLE_CONFIDENCE_THRESHOLD -> single route to that type's dept
      - Else if second score >= SECONDARY_MIN_THRESHOLD -> route to top 2
      - Else route to top 1 only (conservative)
    """
    items = sorted(final_scores.items(), key=lambda kv: kv[1], reverse=True)
    if not items:
        return ([], "no-evidence")
    top_type, top_score = items[0]
    second_type, second_score = items[1]
    if top_score >= SINGLE_CONFIDENCE_THRESHOLD:
        return ([top_type], "single-high")
    # allow two if second is reasonably high
    if second_score >= SECONDARY_MIN_THRESHOLD and len(items) >= 2:
        return ([top_type, second_type], "top-two")
    # otherwise route to top only (conservative)
    return ([top_type], "single-conservative")

# ---------- UI and flow ----------
st.title("PDF Classifier & Router")
st.write("Upload a PDF. The app analyzes the whole document, shows per-type confidence, and routes to at most 2 departments. Explanation includes page-level evidence and final decision logic.")

col1, col2 = st.columns([1, 2])
with col1:
    uploaded = st.file_uploader("Upload PDF (single)", type=["pdf"])
    st.markdown("### API Key (dev)")
    ui_key = st.text_input("Paste OpenRouter API key (optional for dev)", type="password")
    env_key = os.getenv(API_KEY_ENV)
    api_key = env_key or (ui_key if ui_key else None)
    if env_key:
        st.success("Using API key from environment")
    elif ui_key:
        st.info("Using API key pasted in UI (dev)")
    else:
        st.info("No API key — using local scorer only")

    model_weight = st.slider("Model weight vs Local evidence (when model present)", min_value=0.0, max_value=1.0, value=DEFAULT_MODEL_WEIGHT, step=0.1)
    st.write("Routing thresholds")
    sc = st.number_input("Single confidence threshold (%)", value=SINGLE_CONFIDENCE_THRESHOLD, min_value=50, max_value=100, step=1)
    st.write("If top >= this, route to single dept.")
    st.markdown("Secondary minimum for second dept:")
    sec = st.number_input("Secondary minimum (%)", value=SECONDARY_MIN_THRESHOLD, min_value=10, max_value=80, step=1)

with col2:
    st.markdown("### Results")

if st.button("Analyze PDF"):
    if not uploaded:
        st.error("Please upload a PDF.")
    else:
        # extract pages
        pdf_bytes = uploaded.read()
        pages = extract_pages_from_pdf_bytes(pdf_bytes)
        extracted_text = "\n\n".join(pages)
        if not extracted_text.strip():
            st.error("No extractable text from PDF — scanned images require OCR. Ask me to add OCR support.")
        else:
            # local evidence
            evidence = find_evidence_by_page(pages)
            local_scores = local_scores_from_evidence(evidence)

            # try model
            model_scores = {}
            used_model = False
            raw_api = None
            if api_key:
                messages = craft_messages_for_multilabel(extracted_text)
                ok, payload = call_openrouter(messages, api_key)
                if ok:
                    used_model = True
                    raw_api = payload.get("raw")
                    content = payload.get("content", "")
                    okp, parsed = parse_model_json(content)
                    if okp and parsed:
                        preds = parsed.get("predictions", [])
                        # normalize to dict
                        for p in preds:
                            try:
                                score = int(round(float(p.get("score", 0))))
                            except Exception:
                                score = 0
                            model_scores[p.get("document_type")] = max(0, min(100, score))
                    else:
                        # warn but continue with local
                        st.warning("Model returned unparsable JSON; using local evidence.")
                        used_model = False
                else:
                    st.warning(f"Model call failed: {payload.get('error')}; using local evidence.")
                    used_model = False

            final_scores = combine_scores(model_scores if used_model else {}, local_scores, model_weight=model_weight)
            # choose routing (use thresholds from UI)
            SINGLE_CONFIDENCE_THRESHOLD = sc
            SECONDARY_MIN_THRESHOLD = sec
            routed_types, route_mode = choose_routing(final_scores)

            # map doc types to departments
            mapping = {"Invoice":"Finance", "Purchase Order":"Procurement", "Contract":"Legal"}
            routed_depts = [mapping.get(t, "Unknown") for t in routed_types]

            # Display top area: cards
            # Use plotly bar
            labels = list(final_scores.keys())
            values = [final_scores[k] for k in labels]
            fig = px.bar(x=labels, y=values, labels={'x':'Document Type','y':'Confidence (%)'}, range_y=[0,100], text=values)
            fig.update_traces(marker_color=['#0ea5a4' if l=="Invoice" else '#7c3aed' if l=="Purchase Order" else '#ef4444' for l in labels])
            fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=320)

            # left summary and right detailed evidence
            sum_col1, sum_col2 = st.columns([1,1])
            with sum_col1:
                st.markdown("#### Final Confidence (combined model + local)")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("### Routing Decision")
                if routed_depts:
                    if len(routed_depts) == 1:
                        st.success(f"Route to: **{routed_depts[0]}** (from {routed_types[0]}) — mode: {route_mode}")
                    else:
                        st.success(f"Route to: **{routed_depts[0]}** and **{routed_depts[1]}** — mode: {route_mode}")
                else:
                    st.warning("No routing decision could be made automatically.")
                # Provide a concise, solid reasoning summary
                st.markdown("#### Why this routing?")
                reason_lines = []
                # top scores summary
                sorted_final = sorted(final_scores.items(), key=lambda kv: kv[1], reverse=True)
                for t, s in sorted_final:
                    reason_lines.append(f"- {t}: {s}% confidence")
                st.markdown("\n".join(reason_lines))
                st.markdown("**Decision rule applied:**")
                if route_mode == "single-high":
                    st.markdown(f"- Top score >= {SINGLE_CONFIDENCE_THRESHOLD}% → routed to single department.")
                elif route_mode == "top-two":
                    st.markdown(f"- Top two scores are sufficiently high (second >= {SECONDARY_MIN_THRESHOLD}%) → routed to two departments.")
                else:
                    st.markdown("- Conservatively routing to highest scoring single department.")

            with sum_col2:
                st.markdown("### Evidence & Examples (page-level)")
                # For each type, show count and top example snippets
                for dtype, info in evidence.items():
                    cnt = info["count"]
                    st.markdown(f"**{dtype}** — matches: {cnt} — local score: {local_scores.get(dtype,0)}")
                    # show up to 4 examples with page numbers
                    examples = info["examples"][:6]
                    if examples:
                        for ex in examples:
                            page = ex["page"]
                            phrase = ex["phrase"]
                            snippet = ex["snippet"]
                            st.markdown(f"- (p{page}) ...{snippet}...  **[{phrase}]**")
                    else:
                        st.markdown("- No direct keyword matches found.")

                # Note: model raw JSON output removed (per request) — the app will not show model_output anymore.

            # Detailed explanation block
            st.markdown("### Full explanation (concise)")
            expl = []
            if routed_types:
                if len(routed_types) == 1:
                    expl.append(f"The document is primarily identified as **{routed_types[0]}** (confidence {final_scores[routed_types[0]]}%).")
                    # include sample evidence
                    examp = evidence[routed_types[0]]["examples"][:3]
                    if examp:
                        expl.append("Key evidence (page: phrase): " + ", ".join([f"p{e['page']}: '{e['phrase']}'" for e in examp]))
                    # include possible secondary note
                    sec_score = sorted(final_scores.items(), key=lambda kv: kv[1], reverse=True)[1][1]
                    if sec_score >= 15 and sec_score < SECONDARY_MIN_THRESHOLD:
                        expl.append(f"There is some weaker evidence for another type ({sec_score}%), but not strong enough to route to that department.")
                else:
                    expl.append(f"Content is mixed: **{routed_types[0]}** ({final_scores[routed_types[0]]}%) and **{routed_types[1]}** ({final_scores[routed_types[1]]}%). Routing to both departments so each can review.")
                    # include sample evidence for both
                    for t in routed_types[:2]:
                        examp = evidence[t]["examples"][:2]
                        if examp:
                            expl.append(f"Evidence for {t}: " + ", ".join([f"p{e['page']}: '{e['phrase']}'" for e in examp]))
            else:
                expl.append("No decisive evidence found. Escalate to manual review.")

            st.info("\n\n".join(expl))

            # Save session history for quick lookups
            if "history" not in st.session_state:
                st.session_state["history"] = []
            st.session_state["history"].insert(0, {
                "name": uploaded.name if hasattr(uploaded, "name") else "uploaded",
                "routed": routed_depts,
                "scores": final_scores
            })
            # compact history
            with st.expander("Session history"):
                for idx, h in enumerate(st.session_state["history"][:8]):
                    top = ", ".join(h["routed"])
                    st.write(f"{idx+1}. {h['name']} → {top} — scores: {h['scores']}")

# Footer note
st.markdown("---")
st.caption("Notes: This app combines LLM outputs (if available) with deterministic local evidence for robustness. For scanned PDFs, OCR is required — I can add OCR integration on request.")
