# chatbot.py


import logging
import os
import re
import shutil
from pathlib import Path
from difflib import SequenceMatcher
from typing import List, Tuple, Optional, Dict

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------
# Config
# -----------------------
BASE = Path(__file__).resolve().parent
STATIC_UPLOADS = BASE / "static" / "uploads"
STATIC_UPLOADS.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.18           # minimum score to be confident
MIN_QUERY_TOKENS = 1       # require at least this many tokens in cleaned query

# regex for tokenization
TOKEN_RE = re.compile(r"[a-z0-9]+")

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatbot")

# -----------------------
# Optional: copy one uploaded image from this environment into static (keeps previous behavior)
# -----------------------
UPLOADED_LOCAL_PATH = Path(r"/mnt/data/a5cc2881-9bd8-4e96-9c34-d3f34ebcaf39.png")
UPLOADED_IMAGE_URL = None
if UPLOADED_LOCAL_PATH.exists():
    try:
        dst_name = UPLOADED_LOCAL_PATH.name.replace(" ", "_")
        dst = STATIC_UPLOADS / dst_name
        shutil.copy2(str(UPLOADED_LOCAL_PATH), str(dst))
        UPLOADED_IMAGE_URL = f"/static/uploads/{dst_name}"
        logger.info("Copied uploaded image to: %s", dst)
    except Exception as e:
        logger.warning("Could not copy uploaded image: %s", e)

# -----------------------
# FAQ (expanded)
# -----------------------
FAQ = [
    {"question": "How can I reset my password?",
     "answer": "Click 'Forgot Password' on the login page and follow the email instructions."},

    {"question": "How do I track my order?",
     "answer": "Open My Orders, select the order and click 'Track Order' for live updates.",
     **({"image": UPLOADED_IMAGE_URL} if UPLOADED_IMAGE_URL else {})},

    {"question": "What is your return policy?",
     "answer": "We accept returns within 30 days of delivery for unopened and unused items; some categories are excluded."},

    {"question": "How long does delivery take?",
     "answer": "Standard delivery: 3–7 business days. Express delivery: 1–2 business days depending on the destination."},

    {"question": "Do you ship internationally?",
     "answer": "Yes — we ship to many countries. Shipping fees and delivery times vary by destination."},

    {"question": "How can I contact customer support?",
     "answer": "You can email support@example.com or use the contact form on our website for faster routing."},

    {"question": "What payment methods do you accept?",
     "answer": "We accept credit/debit cards, PayPal, UPI, netbanking and major wallets."},

    {"question": "Why was my payment declined?",
     "answer": "Payment declines can be due to insufficient funds, bank rules, or incorrect card details — try another method or contact your bank."},

    {"question": "Can I cancel my order?",
     "answer": "Orders can be cancelled within 1 hour of placement from the My Orders page; after dispatch cancellation may not be possible."},

    {"question": "How do I change my shipping address?",
     "answer": "You can change the address within 1 hour of placing the order from My Orders. After that, please contact support."},

    {"question": "Do you offer Cash on Delivery (COD)?",
     "answer": "COD is available in selected pin codes — it will show as a payment option during checkout."},

    {"question": "Do you offer warranty or guarantees?",
     "answer": "Certain products include manufacturer warranties — warranty details are listed on the product page."},

    {"question": "How do I return a faulty item?",
     "answer": "Open a return request in My Orders, select 'Faulty' and follow the instructions. We'll arrange collection or provide a return label."},

    {"question": "How do I apply a promo code?",
     "answer": "Enter the promo code in the 'Apply Coupon' field on the cart page before checkout."},

    {"question": "Where can I find the invoice for my order?",
     "answer": "Invoices are available in My Orders → Order Details. You can download them as PDF."},

    {"question": "Do you have subscription plans?",
     "answer": "Yes — we offer subscription options for select product categories. Check the Subscriptions page for details."},

    {"question": "What should I do if I received the wrong item?",
     "answer": "Sorry for the trouble — open a return/replace request in My Orders and choose 'Wrong item received'."},

    {"question": "Are there shipping charges?",
     "answer": "Shipping charges depend on the order amount, weight and destination. Free shipping may apply above a threshold."},

    {"question": "How can I change my account email?",
     "answer": "Go to Account Settings → Edit Profile → Change email. Verify the new email to apply changes."},

    {"question": "How do I subscribe to order notifications?",
     "answer": "Enable notifications in Account Settings or allow notifications in the browser/device prompt."},

    # add more as needed...
]

# -----------------------
# Helper utilities
# -----------------------
def clean_text(s: str) -> str:
    """Lowercase and keep only alpha-numeric tokens."""
    s = (s or "").lower()
    return " ".join(TOKEN_RE.findall(s))

def token_set(s: str):
    return set(TOKEN_RE.findall((s or "").lower()))

def fuzzy_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

# -----------------------
# Prepare retrieval structures (TF-IDF)
# -----------------------
CORPUS = [clean_text(item.get("question", "") + " " + item.get("answer", "")) for item in FAQ]
NON_EMPTY = [c for c in CORPUS if c.strip()]

VECT: Optional[TfidfVectorizer] = None
X = None

if NON_EMPTY:
    try:
        VECT = TfidfVectorizer(ngram_range=(1, 2), stop_words="english").fit(NON_EMPTY)
        X = VECT.transform(NON_EMPTY)
        logger.info("TF-IDF vectorizer trained on FAQ corpus (%d items)", len(NON_EMPTY))
    except Exception as e:
        logger.exception("Failed to build TF-IDF: %s", e)
        VECT = None
        X = None
else:
    logger.warning("FAQ corpus empty after cleaning; TF-IDF disabled.")

FAQ_TOKEN_SETS = [token_set(item.get("question", "") + " " + item.get("answer", "")) for item in FAQ]

# -----------------------
# Flask app
# -----------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/")
def index():
    # if you have templates/index.html, Flask will serve it; otherwise return a small message
    index_path = BASE / "templates" / "index.html"
    if index_path.exists():
        return render_template("index.html")
    return "<h3>Chatbot backend is running. POST to /api/message</h3>"


@app.route("/faqs")
def faqs():
    # Return the FAQ list (question only) for frontend population if needed
    return jsonify([{"id": i, "question": item["question"]} for i, item in enumerate(FAQ)])


def rank_by_tfidf(query: str) -> Tuple[Optional[int], float]:
    """Return best index and score by TF-IDF cosine similarity."""
    if VECT is None or X is None:
        return None, 0.0
    try:
        v = VECT.transform([query])
        sims = cosine_similarity(v, X).flatten()
        if sims.size:
            best_idx = int(sims.argmax())
            return best_idx, float(sims[best_idx])
    except Exception as e:
        logger.exception("TF-IDF matching failed: %s", e)
    return None, 0.0


def rank_by_token_overlap(query_tokens: set) -> Tuple[Optional[int], float]:
    best_idx = None
    best_score = 0.0
    for i, faq_tokens in enumerate(FAQ_TOKEN_SETS):
        if not faq_tokens:
            continue
        overlap = len(query_tokens & faq_tokens) / max(len(query_tokens), 1)
        if overlap > best_score:
            best_score = overlap
            best_idx = i
    return best_idx, float(best_score)


def rank_by_fuzzy(query: str) -> Tuple[Optional[int], float]:
    best_idx = None
    best_score = 0.0
    for i, c in enumerate(CORPUS):
        score = fuzzy_ratio(query, c)
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx, float(best_score)


@app.route("/api/message", methods=["POST"])
def api_message():
    try:
        payload = request.get_json(force=True, silent=True) or {}
        msg = (payload.get("message") or "").strip()
        if not msg:
            return jsonify({"reply": "Please type a message.", "score": 0.0, "method": "empty"}), 200

        # quick rule-based greetings for UX
        low = msg.lower()
        if any(g in low for g in ["hi", "hello", "hey"]):
            return jsonify({"reply": "Hello! How can I help you today?", "score": 1.0, "method": "greeting"}), 200
        if "thank" in low:
            return jsonify({"reply": "You're welcome!", "score": 1.0, "method": "thanks"}), 200
        if any(x in low for x in ["bye", "goodbye"]):
            return jsonify({"reply": "Goodbye! Have a great day.", "score": 1.0, "method": "bye"}), 200
        if any(x in low for x in ["human", "agent", "support", "representative"]):
            return jsonify({"reply": "I can connect you to a human agent. Please provide contact details.", "score": 1.0, "method": "handoff"}), 200

        # preprocess
        q_clean = clean_text(msg)
        q_tokens = set(TOKEN_RE.findall(q_clean))
        if len(q_tokens) < MIN_QUERY_TOKENS:
            return jsonify({"reply": "I didn't quite get that. Could you rephrase?", "score": 0.0, "method": "empty_clean"}), 200

        # 1) TF-IDF
        best_idx, best_score = rank_by_tfidf(q_clean)
        best_method = "tfidf" if best_idx is not None else None

        # 2) Token-overlap may beat TF-IDF for short queries
        tok_idx, tok_score = rank_by_token_overlap(q_tokens)
        if tok_score > best_score:
            best_idx, best_score = tok_idx, tok_score
            best_method = "token_overlap"

        # 3) Fuzzy fallback
        fuzzy_idx, fuzzy_score = rank_by_fuzzy(q_clean)
        if fuzzy_score > best_score:
            best_idx, best_score = fuzzy_idx, fuzzy_score
            best_method = "fuzzy"

        # Defensive: no match found
        if best_idx is None:
            return jsonify({"reply": "I don't have enough information to answer that. Would you like human support?", "score": 0.0, "method": "no_match"}), 200

        # Map to faq item (CORPUS and FAQ align by index)
        faq_item = FAQ[best_idx]
        reply = faq_item.get("answer", "")
        resp: Dict = {"reply": reply, "score": float(best_score), "method": best_method, "intent_id": int(best_idx)}

        # include image if present
        if faq_item.get("image"):
            resp["image"] = faq_item["image"]

        # If score below threshold, give fallback suggestion
        if best_score < THRESHOLD:
            resp["reply"] = "I don't have an exact answer. Would you like me to connect you to a human agent?"
            resp["method"] = best_method
            return jsonify(resp), 200

        return jsonify(resp), 200

    except Exception as e:
        logger.exception("Unhandled error in /api/message: %s", e)
        return jsonify({"reply": "Server error. Please try again later.", "score": 0.0, "method": "error"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting chatbot on http://127.0.0.1:%s", port)
    app.run(host="127.0.0.1", port=port, debug=True, use_reloader=False)

