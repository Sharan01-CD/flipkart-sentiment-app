import streamlit as st
import pandas as pd
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ─────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flipkart Sentiment Analyzer",
    page_icon="🛒",
    layout="centered"
)

# ─────────────────────────────────────────────────────────────
# NLTK Downloads (runs once)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def download_nltk():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

download_nltk()

# ─────────────────────────────────────────────────────────────
# Text Cleaning
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_cleaner():
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    return lemmatizer, stop_words

def clean_text(text, lemmatizer, stop_words):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# ─────────────────────────────────────────────────────────────
# Load or Train Model
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    import os
    lemmatizer, stop_words = get_cleaner()

    # Try loading saved model first
    if os.path.exists("sentiment_model.pkl"):
        with open("sentiment_model.pkl", "rb") as f:
            return pickle.load(f), lemmatizer, stop_words

    # Otherwise train from scratch
    df = pd.read_csv("product_reviews_.csv")
    df['cleaned_review']  = df['review'].apply(lambda x: clean_text(x, lemmatizer, stop_words))
    df['cleaned_summary'] = df['summary'].apply(lambda x: clean_text(x, lemmatizer, stop_words))
    df['combined_text']   = df['cleaned_review'] + " " + df['cleaned_summary']
    df = df[df['combined_text'].str.strip() != ""].reset_index(drop=True)

    label_map = {'positive': 2, 'neutral': 1, 'negative': 0}
    df['label'] = df['sentiment'].str.lower().map(label_map)

    X, y = df['combined_text'], df['label']

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True
        )),
        ('clf', LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver='lbfgs',
            multi_class='multinomial',
            random_state=42
        ))
    ])
    pipeline.fit(X, y)

    # Save for next time
    with open("sentiment_model.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    return pipeline, lemmatizer, stop_words

# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────
st.title("🛒 Flipkart Review Sentiment Analyzer")
st.markdown("Enter a product review below and get instant sentiment prediction.")
st.markdown("---")

# Load model with spinner
with st.spinner("Loading model... (first run may take ~30 sec to train)"):
    model, lemmatizer, stop_words = load_model()

st.success("Model ready!", icon="✅")

# ─── Input ───
st.subheader("📝 Enter Your Review")
review_input = st.text_area(
    label="Review text",
    placeholder="e.g. The product quality is amazing, totally worth the price!",
    height=150,
    label_visibility="collapsed"
)

predict_btn = st.button("Analyze Sentiment", type="primary", use_container_width=True)

# ─── Prediction ───
if predict_btn:
    if not review_input.strip():
        st.warning("Please enter a review before clicking Analyze.")
    else:
        cleaned = clean_text(review_input, lemmatizer, stop_words)
        pred    = model.predict([cleaned])[0]
        proba   = model.predict_proba([cleaned])[0]

        label_map    = {2: "Positive", 1: "Neutral", 0: "Negative"}
        emoji_map    = {2: "😊", 1: "😐", 0: "😠"}
        color_map    = {2: "green", 1: "orange", 0: "red"}

        sentiment = label_map[pred]
        emoji     = emoji_map[pred]
        color     = color_map[pred]

        st.markdown("---")
        st.subheader("🔍 Result")

        st.markdown(
            f"""
            <div style="
                background-color: {'#d4edda' if pred==2 else '#fff3cd' if pred==1 else '#f8d7da'};
                border-left: 6px solid {'green' if pred==2 else 'orange' if pred==1 else 'red'};
                padding: 20px 24px;
                border-radius: 8px;
                margin-bottom: 16px;
            ">
                <h2 style="margin:0; color:{color};">{emoji} {sentiment}</h2>
                <p style="margin:4px 0 0 0; color:#555;">Sentiment detected in your review</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Confidence scores
        st.subheader("📊 Confidence Scores")
        conf_df = pd.DataFrame({
            "Sentiment": ["Negative 😠", "Neutral 😐", "Positive 😊"],
            "Confidence": [f"{proba[0]*100:.1f}%", f"{proba[1]*100:.1f}%", f"{proba[2]*100:.1f}%"],
            "Score": [proba[0], proba[1], proba[2]]
        })

        for _, row in conf_df.iterrows():
            st.markdown(f"**{row['Sentiment']}** — {row['Confidence']}")
            st.progress(float(row['Score']))

        # Show cleaned text (expandable)
        with st.expander("🔎 See cleaned text used for prediction"):
            st.code(cleaned if cleaned else "(empty after cleaning)")

# ─── Footer ───
st.markdown("---")
st.caption("Built with Streamlit · Logistic Regression + TF-IDF · Trained on Flipkart Reviews")
