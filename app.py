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

# SESSION
if "show_popup" not in st.session_state:
    st.session_state.show_popup = False

if "popup_message" not in st.session_state:
    st.session_state.popup_message = ""

# STYLE
st.markdown("""
<style>
.stApp{
background: linear-gradient(135deg,#030712,#0f172a);
}

.title{
text-align:center;
font-size:58px;
font-weight:900;
color:white;
text-shadow:0 0 20px #00e5ff;
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
font-size:20px;
padding:14px;
background:linear-gradient(90deg,#06b6d4,#8b5cf6);
color:white;
border:none;
}
</style>
""", unsafe_allow_html=True)

# MAIN SCREEN
st.markdown('<div class="title">📩 SPAM DETECTOR</div>', unsafe_allow_html=True)

st.markdown(
    f'<div class="acc">🎯 Model Accuracy: {accuracy*100:.2f}%</div>',
    unsafe_allow_html=True
)

message = st.text_area(
    "✍️ Enter your message:",
    height=180,
    placeholder="Type your email or SMS here..."
)

if st.button("🚀 Check Message"):
    if message.strip():
        pred = model.predict(vectorizer.transform([message]))[0]

        if pred == 1:
            st.session_state.popup_message = "🚨 SPAM MESSAGE DETECTED"
        else:
            st.session_state.popup_message = "✅ NOT SPAM MESSAGE"

        st.session_state.show_popup = True

# POPUP
if st.session_state.show_popup:
    with st.modal("Detection Result"):
        st.markdown(
            f"<h2 style='text-align:center'>{st.session_state.popup_message}</h2>",
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Check Another"):
                st.session_state.show_popup = False
                st.rerun()

        with col2:
            if st.button("🏠 Home / Exit"):
                st.session_state.show_popup = False
                st.rerun()
