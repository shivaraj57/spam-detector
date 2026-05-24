import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Spam Detector",
    page_icon="📩",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #0f172a;
}

.title {
    font-size: 52px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg,#38bdf8,#818cf8,#ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}

.subtitle {
    text-align:center;
    font-size:20px;
    color:#cbd5e1;
    margin-bottom:30px;
}

.result-spam {
    background-color:#7f1d1d;
    padding:20px;
    border-radius:15px;
    color:white;
    font-size:26px;
    text-align:center;
    font-weight:bold;
}

.result-ham {
    background-color:#14532d;
    padding:20px;
    border-radius:15px;
    color:white;
    font-size:26px;
    text-align:center;
    font-weight:bold;
}

.acc-box {
    background:#1e293b;
    padding:15px;
    border-radius:12px;
    text-align:center;
    font-size:22px;
    color:#ffffff;
    margin-bottom:20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("spam.csv", encoding="latin-1")

df = df[['v1', 'v2']]
df.columns = ['label', 'message']

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# ---------------- MODEL ----------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# ---------------- UI ----------------
st.markdown('<div class="title">📩 Email & SMS Spam Detector</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">Detect spam messages instantly using Machine Learning</div>',
    unsafe_allow_html=True
)

st.markdown(
    f'<div class="acc-box">🎯 Model Accuracy: {round(accuracy*100,2)}%</div>',
    unsafe_allow_html=True
)

message = st.text_area(
    "✍️ Enter your message below",
    height=180,
    placeholder="Type your email or SMS here..."
)

if st.button("🚀 Check Message"):

    if message.strip() != "":
        msg_vector = vectorizer.transform([message])
        prediction = model.predict(msg_vector)

        if prediction[0] == 1:
            st.markdown(
                '<div class="result-spam">🚫 SPAM MESSAGE DETECTED</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-ham">✅ NOT SPAM MESSAGE</div>',
                unsafe_allow_html=True
            )
    else:
        st.warning("Please enter a message.")

st.markdown("---")
