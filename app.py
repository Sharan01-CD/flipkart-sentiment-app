import streamlit as st
import re
import pickle
import nltk
import os
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
nltk.download('omw-1.4', quiet=True)

# ─────────────────────────────────────────────────────────────
# Globals
# ─────────────────────────────────────────────────────────────
lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words('english'))

# ─────────────────────────────────────────────────────────────
# Text Cleaning
# ─────────────────────────────────────────────────────────────
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# ─────────────────────────────────────────────────────────────
# Load Pre-trained Model
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("sentiment_model.pkl", "rb") as f:
        return pickle.load(f)

# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────
st.title("🛒 Flipkart Review Sentiment Analyzer")
st.markdown("Enter a product review below and get instant sentiment prediction.")
st.markdown("---")

with st.spinner("Loading model..."):
    model = load_model()

st.success("Model ready!", icon="✅")

st.subheader("📝 Enter Your Review")
review_input = st.text_area(
    label="Review text",
    placeholder="e.g. Worst product ever, complete waste of money!",
    height=150,
    label_visibility="collapsed"
)

predict_btn = st.button("Analyze Sentiment", type="primary", use_container_width=True)

if predict_btn:
    if not review_input.strip():
        st.warning("Please enter a review before clicking Analyze.")
    else:
        cleaned = clean_text(review_input)
        pred    = model.predict([cleaned])[0]
        proba   = model.predict_proba([cleaned])[0]

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

        with st.expander("🔎 See cleaned text used for prediction"):
            st.code(cleaned if cleaned else "(empty after cleaning)")

st.markdown("---")
st.caption("Built with Streamlit · Logistic Regression + TF-IDF · Trained on 205k Real Flipkart Reviews")
