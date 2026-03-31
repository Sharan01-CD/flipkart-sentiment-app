import streamlit as st
import numpy as np
import pickle
import nltk
import re
import onnxruntime as ort
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

st.set_page_config(
    page_title="Flipkart Sentiment Analyzer",
    page_icon="🛒",
    layout="centered"
)

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
MAX_LEN = 100

# File names (local repo)
ONNX_FILE = "lstm_model.onnx"
TOKENIZER_FILE = "simple_tokenizer.pkl"

# ── Load model (cached) ─────────────────────────────
@st.cache_resource
def load_artifacts():
    session = ort.InferenceSession(ONNX_FILE)
    with open(TOKENIZER_FILE, "rb") as f:
        tokenizer = pickle.load(f)
    return session, tokenizer

# ── Text cleaning ───────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# ── Tokenizer ───────────────────────────────────────
def texts_to_padded(texts, word_index, maxlen, oov_index=1):
    padded = np.zeros((len(texts), maxlen), dtype=np.float32)
    for i, text in enumerate(texts):
        seq = [word_index.get(w, oov_index) for w in text.split()]
        seq = seq[:maxlen]
        padded[i, :len(seq)] = seq
    return padded

# ── Prediction ──────────────────────────────────────
def predict_sentiment(text, session, tokenizer):
    cleaned = clean_text(text)
    word_index = tokenizer['word_index']
    padded = texts_to_padded([cleaned], word_index, MAX_LEN)

    input_name = session.get_inputs()[0].name

    # ✅ Ensure correct datatype (VERY IMPORTANT)
    padded = padded.astype(np.float32)

    proba = session.run(None, {input_name: padded})[0][0]

    pred = int(np.argmax(proba))
    return pred, proba

# ── UI ──────────────────────────────────────────────
st.title("🛒 Flipkart Review Sentiment Analyzer")
st.markdown("Powered by **Deep Learning (BiLSTM)**")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Model Management")

    if st.button("🔄 Reload Model", use_container_width=True):
        st.cache_resource.clear()
        st.success("✅ Model reloaded!")
        st.rerun()

# Load model
with st.spinner("Loading model..."):
    session, tokenizer = load_artifacts()

st.success("Model ready!", icon="✅")

# Input
review_input = st.text_area("Enter Review")

if st.button("Analyze Sentiment"):
    if not review_input.strip():
        st.warning("Please enter a review")
    else:
        pred, proba = predict_sentiment(review_input, session, tokenizer)

        labels = ["Negative 😠", "Neutral 😐", "Positive 😊"]

        st.subheader(f"Prediction: {labels[pred]}")

        for label, score in zip(labels, proba):
            st.write(f"{label}: {score*100:.2f}%")
            st.progress(float(score))