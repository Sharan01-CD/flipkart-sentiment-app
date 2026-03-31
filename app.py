import streamlit as st
import numpy as np
import pickle
import nltk
import re
import os
import csv
import gdown
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

st.set_page_config(
    page_title="Flipkart Sentiment Analyzer",
    page_icon="🛒",
    layout="centered"
)

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words('english'))
MAX_LEN     = 100
RETRAIN_EVERY = 50  # retrain after every 50 new reviews
DATA_FILE   = "new_reviews.csv"

# ── Download model files from Google Drive if not present ───────
ONNX_FILE      = "lstm_model.onnx"
TOKENIZER_FILE = "simple_tokenizer.pkl"

ONNX_FILE_ID      = "YOUR_ONNX_FILE_ID"       # ← replace this
TOKENIZER_FILE_ID = "YOUR_TOKENIZER_FILE_ID"   # ← replace this

if not os.path.exists(ONNX_FILE):
    with st.spinner("Downloading model... (first run only)"):
        gdown.download(f"https://drive.google.com/uc?id={ONNX_FILE_ID}", ONNX_FILE, quiet=False)

if not os.path.exists(TOKENIZER_FILE):
    with st.spinner("Downloading tokenizer... (first run only)"):
        gdown.download(f"https://drive.google.com/uc?id={TOKENIZER_FILE_ID}", TOKENIZER_FILE, quiet=False)
# ────────────────────────────────────────────────────────────────

# ── Text cleaning ────────────────────────────────────────────────
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# ── Tokenizer ────────────────────────────────────────────────────
def texts_to_padded(texts, word_index, maxlen, oov_index=1):
    padded = np.zeros((len(texts), maxlen), dtype=np.float32)
    for i, text in enumerate(texts):
        seq = [word_index.get(w, oov_index) for w in text.lower().split()]
        seq = seq[:maxlen]
        padded[i, :len(seq)] = seq
    return padded

# ── Load model ───────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    import onnxruntime as ort
    session = ort.InferenceSession(ONNX_FILE)
    with open(TOKENIZER_FILE, "rb") as f:
        tokenizer = pickle.load(f)
    return session, tokenizer

def predict_sentiment(text, session, tokenizer):
    cleaned    = clean_text(text)
    word_index = tokenizer['word_index']
    padded     = texts_to_padded([cleaned], word_index, MAX_LEN)
    proba      = session.run(None, {'input_layer': padded})[0][0]
    pred       = int(np.argmax(proba))
    return pred, proba

# ── Save new review to CSV ───────────────────────────────────────
def save_review(text, predicted_label):
    file_exists = os.path.exists(DATA_FILE)
    with open(DATA_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['review', 'label'])
        writer.writerow([text, predicted_label])

def get_review_count():
    if not os.path.exists(DATA_FILE):
        return 0
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        return max(0, sum(1 for _ in f) - 1)

# ── Retrain adaptive model with new data ─────────────────────────
def retrain_model(tokenizer):
    import pandas as pd
    from sklearn.linear_model import SGDClassifier

    try:
        df = pd.read_csv(DATA_FILE)
        if len(df) < 10:
            return False, "Not enough data yet"

        word_index = tokenizer['word_index']
        X = texts_to_padded(df['review'].tolist(), word_index, MAX_LEN)
        X_flat = X.reshape(len(X), -1)
        y = df['label'].values

        clf = SGDClassifier(loss='modified_huber', max_iter=100, random_state=42)
        clf.fit(X_flat, y)

        with open("adaptive_model.pkl", "wb") as f:
            pickle.dump(clf, f)

        return True, f"✅ Retrained on {len(df)} reviews!"
    except Exception as e:
        return False, str(e)

def predict_adaptive(text, tokenizer):
    if not os.path.exists("adaptive_model.pkl"):
        return None, None
    with open("adaptive_model.pkl", "rb") as f:
        clf = pickle.load(f)
    word_index = tokenizer['word_index']
    cleaned = clean_text(text)
    padded = texts_to_padded([cleaned], word_index, MAX_LEN)
    X_flat = padded.reshape(1, -1)
    pred = int(clf.predict(X_flat)[0])
    proba = clf.predict_proba(X_flat)[0] if hasattr(clf, 'predict_proba') else None
    return pred, proba

# ── UI ───────────────────────────────────────────────────────────
st.title("🛒 Flipkart Review Sentiment Analyzer")
st.markdown("Powered by **Deep Learning (BiLSTM)** · Auto-learns from new reviews!")
st.markdown("---")

with st.spinner("Loading model..."):
    session, tokenizer = load_artifacts()

st.success("Model ready!", icon="✅")

# Stats
review_count = get_review_count()
next_retrain = RETRAIN_EVERY - (review_count % RETRAIN_EVERY) if review_count > 0 else RETRAIN_EVERY
col1, col2 = st.columns(2)
col1.metric("📦 Reviews Collected", review_count)
col2.metric("🔄 Next Retrain In", f"{next_retrain} reviews")

st.subheader("📝 Enter Your Review")
review_input = st.text_area(
    label="Review",
    placeholder="e.g. Worst product ever, complete waste of money!",
    height=150,
    label_visibility="collapsed"
)

predict_btn = st.button("Analyze Sentiment", type="primary", use_container_width=True)

if predict_btn:
    if not review_input.strip():
        st.warning("Please enter a review before clicking Analyze.")
    else:
        # Use adaptive model if available, else use base ONNX model
        adaptive_pred, adaptive_proba = predict_adaptive(review_input, tokenizer)
        if adaptive_pred is not None:
            pred, proba = adaptive_pred, adaptive_proba
            st.info("🧠 Prediction by Adaptive Model", icon="🔄")
        else:
            pred, proba = predict_sentiment(review_input, session, tokenizer)
            st.info("🤖 Prediction by Base BiLSTM Model", icon="📦")

        # Save review + prediction
        save_review(review_input, pred)
        review_count = get_review_count()

        # Auto retrain every 50 reviews
        if review_count % RETRAIN_EVERY == 0 and review_count > 0:
            with st.spinner("🔄 Auto-retraining model with new data..."):
                success, msg = retrain_model(tokenizer)
            if success:
                st.success(msg)
            else:
                st.warning(f"Retrain skipped: {msg}")

        label_map  = {2: "Positive", 1: "Neutral", 0: "Negative"}
        emoji_map  = {2: "😊", 1: "😐", 0: "😠"}
        color_map  = {2: "#d4edda", 1: "#fff3cd", 0: "#f8d7da"}
        border_map = {2: "green", 1: "orange", 0: "red"}
        text_map   = {2: "green", 1: "#856404", 0: "red"}

        st.markdown("---")
        st.subheader("🔍 Result")
        st.markdown(
            f"""
            <div style="
                background-color: {color_map[pred]};
                border-left: 6px solid {border_map[pred]};
                padding: 20px 24px;
                border-radius: 8px;
                margin-bottom: 16px;
            ">
                <h2 style="margin:0; color:{text_map[pred]};">{emoji_map[pred]} {label_map[pred]}</h2>
                <p style="margin:4px 0 0 0; color:#555;">Sentiment detected in your review</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.subheader("📊 Confidence Scores")
        if proba is not None and len(proba) == 3:
            for label, score in zip(["Negative 😠", "Neutral 😐", "Positive 😊"], proba):
                st.markdown(f"**{label}** — {score*100:.1f}%")
                st.progress(float(score))
        else:
            st.markdown(f"**Predicted:** {label_map[pred]}")

st.markdown("---")
st.caption("Built with Streamlit · BiLSTM Deep Learning · Auto-Adaptive Learning · Trained on 205k Real Flipkart Reviews")