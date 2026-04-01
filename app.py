import streamlit as st
import numpy as np
import pickle
import nltk
import re
import os
import csv
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

ONNX_FILE        = "lstm_model.onnx"
TOKENIZER_FILE   = "simple_tokenizer.pkl"
FEEDBACK_FILE    = "feedback_data.csv"
ADAPTIVE_MODEL   = "adaptive_model.pkl"

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

# ── Load base model ──────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    import onnxruntime as ort
    session = ort.InferenceSession(ONNX_FILE)
    with open(TOKENIZER_FILE, "rb") as f:
        tokenizer = pickle.load(f)
    return session, tokenizer

# ── Base model prediction ────────────────────────────────────────
def predict_base(text, session, tokenizer):
    cleaned    = clean_text(text)
    word_index = tokenizer['word_index']
    padded     = texts_to_padded([cleaned], word_index, MAX_LEN)
    input_name = session.get_inputs()[0].name
    proba      = session.run(None, {input_name: padded})[0][0]
    pred       = int(np.argmax(proba))
    confidence = float(np.max(proba))
    return pred, proba, confidence

# ── Adaptive model prediction ────────────────────────────────────
def predict_adaptive(text, tokenizer):
    if not os.path.exists(ADAPTIVE_MODEL):
        return None, None, None
    with open(ADAPTIVE_MODEL, "rb") as f:
        clf = pickle.load(f)
    word_index = tokenizer['word_index']
    cleaned = clean_text(text)
    padded = texts_to_padded([cleaned], word_index, MAX_LEN)
    X_flat = padded.reshape(1, -1)
    pred = int(clf.predict(X_flat)[0])
    proba = clf.predict_proba(X_flat)[0]
    confidence = float(np.max(proba))
    return pred, proba, confidence

# ── Save feedback ────────────────────────────────────────────────
def save_feedback(text, correct_label):
    file_exists = os.path.exists(FEEDBACK_FILE)
    with open(FEEDBACK_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['review', 'label'])
        writer.writerow([text, correct_label])

def get_feedback_count():
    if not os.path.exists(FEEDBACK_FILE):
        return 0
    with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
        return max(0, sum(1 for _ in f) - 1)

# ── Retrain adaptive model ───────────────────────────────────────
def retrain_adaptive(tokenizer):
    import pandas as pd
    from sklearn.linear_model import SGDClassifier

    try:
        df = pd.read_csv(FEEDBACK_FILE)
        if len(df) < 5:
            return False
        word_index = tokenizer['word_index']
        X = texts_to_padded(df['review'].tolist(), word_index, MAX_LEN)
        X_flat = X.reshape(len(X), -1)
        y = df['label'].values

        if os.path.exists(ADAPTIVE_MODEL):
            with open(ADAPTIVE_MODEL, "rb") as f:
                clf = pickle.load(f)
            clf.partial_fit(X_flat, y, classes=[0, 1, 2])
        else:
            clf = SGDClassifier(loss='modified_huber', max_iter=100, random_state=42)
            clf.fit(X_flat, y)

        with open(ADAPTIVE_MODEL, "wb") as f:
            pickle.dump(clf, f)
        return True
    except:
        return False

# ── UI ───────────────────────────────────────────────────────────
st.title("🛒 Flipkart Review Sentiment Analyzer")
st.markdown("Powered by **BiLSTM** · Learns from your feedback in real time!")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Model Stats")
    feedback_count = get_feedback_count()
    st.metric("📝 Feedback Collected", feedback_count)
    adaptive_active = os.path.exists(ADAPTIVE_MODEL)
    st.metric("🧠 Adaptive Model", "Active ✅" if adaptive_active else "Not trained yet")
    st.markdown("---")
    st.markdown("**Workflow:**")
    st.markdown("1. Enter a review")
    st.markdown("2. Model predicts sentiment")
    st.markdown("3. Confirm or correct it")
    st.markdown("4. Model learns from feedback!")
    if st.button("🔄 Reload Base Model", use_container_width=True):
        st.cache_resource.clear()
        st.success("✅ Reloaded!")
        st.rerun()

# Load model
with st.spinner("Loading model..."):
    session, tokenizer = load_artifacts()
st.success("Model ready!", icon="✅")

# ── Review Input ─────────────────────────────────────────────────
st.subheader("📝 Enter Your Review")
review_input = st.text_area(
    label="Review",
    placeholder="e.g. Worst product ever, complete waste of money!",
    height=150,
    label_visibility="collapsed"
)

predict_btn = st.button("Analyze Sentiment", type="primary", use_container_width=True)

if predict_btn and review_input.strip():
    st.session_state['review']    = review_input
    st.session_state['predicted'] = None
    st.session_state['proba']     = None
    st.session_state['feedback_given'] = False

if 'review' in st.session_state and st.session_state['review']:
    review = st.session_state['review']

    # Predict
    adaptive_pred, adaptive_proba, adaptive_conf = predict_adaptive(review, tokenizer)
    if adaptive_pred is not None:
        pred, proba, confidence = adaptive_pred, adaptive_proba, adaptive_conf
        model_used = "🧠 Adaptive Model"
    else:
        pred, proba, confidence = predict_base(review, session, tokenizer)
        model_used = "🤖 Base BiLSTM Model"

    st.session_state['predicted'] = pred
    st.session_state['proba']     = proba

    label_map  = {2: "Positive", 1: "Neutral", 0: "Negative"}
    emoji_map  = {2: "😊", 1: "😐", 0: "😠"}
    color_map  = {2: "#d4edda", 1: "#fff3cd", 0: "#f8d7da"}
    border_map = {2: "green", 1: "orange", 0: "red"}
    text_map   = {2: "green", 1: "#856404", 0: "red"}

    st.markdown("---")
    st.caption(f"Prediction by: {model_used}")
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

    st.markdown(f"**Model confidence: {confidence*100:.1f}%**")
    st.progress(float(confidence))

    st.subheader("📊 Confidence Scores")
    for label, score in zip(["Negative 😠", "Neutral 😐", "Positive 😊"], proba):
        st.markdown(f"**{label}** — {score*100:.1f}%")
        st.progress(float(score))

    # ── Feedback Section ─────────────────────────────────────────
    if not st.session_state.get('feedback_given', False):
        st.markdown("---")
        st.subheader("💬 Was this prediction correct?")
        st.markdown("Your feedback trains the model in real time — even a single correction helps.")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("👍 Yes, correct", use_container_width=True):
                save_feedback(review, pred)
                retrain_adaptive(tokenizer)
                st.session_state['feedback_given'] = True
                st.success("✅ Prediction confirmed and model updated. Thank you!")
                st.rerun()

        with col2:
            if st.button("👎 No, it was wrong", use_container_width=True):
                st.session_state['show_correction'] = True

        if st.session_state.get('show_correction', False):
            st.markdown("**What was the correct sentiment?**")
            correct = st.radio(
                "Select correct label:",
                options=[0, 1, 2],
                format_func=lambda x: {0: "😠 Negative", 1: "😐 Neutral", 2: "😊 Positive"}[x],
                horizontal=True,
                label_visibility="collapsed"
            )
            if st.button("✅ Submit Correction", use_container_width=True):
                save_feedback(review, correct)
                retrain_adaptive(tokenizer)
                st.session_state['feedback_given'] = True
                st.session_state['show_correction'] = False
                st.success(f"✅ Correction saved! Model updated with your feedback. Thank you!")
                st.rerun()
    else:
        st.success("✅ Feedback already given for this review. Thank you!")

st.markdown("---")
st.caption("Built with Streamlit · BiLSTM Deep Learning · Real-Time Adaptive Learning · Trained on 205k Real Flipkart Reviews")
