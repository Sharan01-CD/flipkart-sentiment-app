import streamlit as st
import numpy as np
import pickle
import nltk
import re
import os
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

ONNX_FILE      = "lstm_model.onnx"
TOKENIZER_FILE = "simple_tokenizer.pkl"

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
    input_name = session.get_inputs()[0].name  # ✅ dynamic input name
    proba      = session.run(None, {input_name: padded})[0][0]
    pred       = int(np.argmax(proba))
    return pred, proba

# ── UI ───────────────────────────────────────────────────────────
st.title("🛒 Flipkart Review Sentiment Analyzer")
st.markdown("Powered by **Deep Learning (BiLSTM)** · Trained on 205k real Flipkart reviews")
st.markdown("---")

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Model Management")
    st.markdown("After retraining in Colab, push new model files to GitHub and click reload:")

    if st.button("🔄 Reload Model", use_container_width=True):
        st.cache_resource.clear()
        st.success("✅ Model reloaded!")
        st.rerun()

    st.markdown("---")
    st.markdown("**Workflow:**")
    st.markdown("1. Update dataset in Colab")
    st.markdown("2. Retrain & export new ONNX")
    st.markdown("3. Replace files & push to GitHub")
    st.markdown("4. Click **Reload Model**")

# ── Load model ───────────────────────────────────────────────────
with st.spinner("Loading model..."):
    session, tokenizer = load_artifacts()

st.success("Model ready!", icon="✅")

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
        pred, proba = predict_sentiment(review_input, session, tokenizer)

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
        for label, score in zip(["Negative 😠", "Neutral 😐", "Positive 😊"], proba):
            st.markdown(f"**{label}** — {score*100:.1f}%")
            st.progress(float(score))

st.markdown("---")
st.caption("Built with Streamlit · BiLSTM Deep Learning · 95.17% Accuracy · Trained on 205k Real Flipkart Reviews")
