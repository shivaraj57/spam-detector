import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# PAGE
st.set_page_config(page_title="Spam Detector", page_icon="📩", layout="centered")

# DATA
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# MODEL
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

# STYLE
st.markdown("""
<style>
.stApp{
    background: linear-gradient(135deg,#030712,#0f172a);
}

.title{
    text-align:center;
    font-size:60px;
    font-weight:900;
    color:white;
    text-shadow:0 0 18px #00e5ff;
}

.acc{
    text-align:center;
    font-size:24px;
    color:white;
    padding:15px;
    border-radius:16px;
    background:rgba(255,255,255,0.05);
    margin-bottom:20px;
}

.stButton>button{
    width:100%;
    border-radius:14px;
    font-size:22px;
    background:linear-gradient(90deg,#06b6d4,#8b5cf6);
    color:white;
    border:none;
    padding:14px;
}

.popup{
    position:fixed;
    top:50%;
    left:50%;
    transform:translate(-50%,-50%);
    width:420px;
    padding:35px;
    border-radius:22px;
    background:#111827;
    color:white;
    text-align:center;
    z-index:9999;
    box-shadow:0 0 50px rgba(0,229,255,.35);
    animation: pop 0.35s ease;
}

@keyframes pop{
    from{opacity:0;transform:translate(-50%,-50%) scale(.7);}
    to{opacity:1;transform:translate(-50%,-50%) scale(1);}
}
</style>
""", unsafe_allow_html=True)

# UI
st.markdown(
    '<div class="title">📩 SPAM DETECTOR</div>',
    unsafe_allow_html=True
)

st.markdown(
    f'<div class="acc">🎯 Accuracy: {accuracy*100:.2f}%</div>',
    unsafe_allow_html=True
)

message = st.text_area(
    "✍️ Enter your message:",
    height=200,
    placeholder="Type your email or SMS here..."
)

if st.button("🚀 Check Message"):
    if message.strip():
        pred = model.predict(vectorizer.transform([message]))[0]

        if pred == 1:
            st.markdown("""
            <div class="popup">
            <h1>🚨 SPAM MESSAGE</h1>
            <p>This message looks suspicious.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="popup">
            <h1>✅ NOT SPAM</h1>
            <p>This message appears safe.</p>
            </div>
            """, unsafe_allow_html=True)
