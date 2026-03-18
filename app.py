import streamlit as st
import numpy as np
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ─────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flipkart Sentiment Analyzer",
    page_icon="🛒",
    layout="centered"
)

# ─────────────────────────────────────────────────────────────
# NLTK Downloads
# ─────────────────────────────────────────────────────────────
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# ─────────────────────────────────────────────────────────────
# Globals
# ─────────────────────────────────────────────────────────────
lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words('english'))
MAX_LEN     = 50
INPUT_NAME  = 'input_layer_3'

# ─────────────────────────────────────────────────────────────
# Load Model & Tokenizer
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    import onnxruntime as ort
    from keras.preprocessing.sequence import pad_sequences
    session   = ort.InferenceSession("lstm_model.onnx")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return session, tokenizer

# ─────────────────────────────────────────────────────────────
# Predict
# ─────────────────────────────────────────────────────────────
def predict_sentiment(text, session, tokenizer):
    from keras.preprocessing.sequence import pad_sequences
    seq    = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    input_data = padded.astype(np.float32)
    proba  = session.run(None, {INPUT_NAME: input_data})[0][0]
    pred   = np.argmax(proba)
    return pred, proba

# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────
st.title("🛒 Flipkart Review Sentiment Analyzer")
st.markdown("Powered by **Deep Learning (BiLSTM)** · Trained on 205k real Flipkart reviews")
st.markdown("---")

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
