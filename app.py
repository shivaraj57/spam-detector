import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# ---------------- PAGE ----------------
st.set_page_config(
    page_title="Spam Detector",
    page_icon="📩",
    layout="centered"
)

# ---------------- DATA ----------------
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

accuracy = accuracy_score(y_test, model.predict(X_test))

# ---------------- CSS ----------------
st.markdown("""
<style>

.stApp {
    background: #050816;
}

/* glowing background blobs */
.stApp::before{
content:'';
position:fixed;
width:420px;
height:420px;
border-radius:50%;
background:#00e5ff;
filter: blur(150px);
top:-120px;
left:-120px;
opacity:0.25;
z-index:-1;
}

.stApp::after{
content:'';
position:fixed;
width:420px;
height:420px;
border-radius:50%;
background:#8b5cf6;
filter: blur(150px);
bottom:-120px;
right:-120px;
opacity:0.25;
z-index:-1;
}

/* title */
.title{
font-size:64px;
font-weight:900;
text-align:center;
color:white;
text-shadow:
0 0 10px #00e5ff,
0 0 25px #00e5ff,
0 0 40px #8b5cf6;
animation:pulse 2s infinite alternate;
margin-bottom:10px;
}

/* animation */
@keyframes pulse{
from{transform:scale(1);}
to{transform:scale(1.03);}
}

.sub{
text-align:center;
font-size:20px;
color:#cbd5e1;
margin-bottom:25px;
}

/* glass card */
.acc-box{
background: rgba(255,255,255,0.05);
border:1px solid rgba(255,255,255,0.1);
backdrop-filter: blur(14px);
padding:18px;
border-radius:18px;
text-align:center;
font-size:28px;
color:white;
box-shadow: 0 0 30px rgba(0,229,255,.15);
margin-bottom:20px;
}

/* textarea */
textarea{
background: rgba(255,255,255,0.05) !important;
color:white !important;
border-radius:18px !important;
border:1px solid rgba(255,255,255,0.1) !important;
}

/* button */
.stButton>button{
width:100%;
padding:14px;
font-size:22px;
font-weight:bold;
border:none;
border-radius:16px;
color:white;
background: linear-gradient(90deg,#06b6d4,#3b82f6,#9333ea);
box-shadow:0 0 25px rgba(59,130,246,.45);
transition:0.3s;
}

.stButton>button:hover{
transform:scale(1.03);
box-shadow:0 0 45px rgba(147,51,234,.6);
}

/* result box */
.result{
padding:22px;
border-radius:18px;
font-size:30px;
font-weight:bold;
text-align:center;
margin-top:25px;
box-shadow:0 0 30px rgba(255,255,255,.15);
}

</style>
""", unsafe_allow_html=True)

# ---------------- UI ----------------
st.markdown(
    '<div class="title">📩 SPAM DETECTOR</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub">Detect Email & SMS Spam Instantly using Machine Learning</div>',
    unsafe_allow_html=True
)

st.markdown(
    f'<div class="acc-box">🎯 Model Accuracy: {accuracy*100:.2f}%</div>',
    unsafe_allow_html=True
)

message = st.text_area(
    "✍️ Enter your message:",
    height=200,
    placeholder="Type your email or SMS here..."
)

if st.button("🚀 Check Message"):
    if message.strip():
        prediction = model.predict(
            vectorizer.transform([message])
        )[0]

        if prediction == 1:
            st.markdown(
                '<div class="result" style="background:#3f0d1a;color:#fb7185;">🚨 SPAM MESSAGE</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result" style="background:#052e16;color:#4ade80;">✅ NOT SPAM</div>',
                unsafe_allow_html=True
            )
