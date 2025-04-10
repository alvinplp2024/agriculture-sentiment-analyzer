import streamlit as st
import joblib

# Load model and vectorizer
sentiment_model = joblib.load("sentiment_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# Custom CSS Styling
st.markdown("""
    <style>
        body {
            background-color: white;
            color: black;
        }
        .main {
            background-color: white;
        }
        textarea {
            background-color: #1e1e1e;
            color: white;
        }
        .stTextArea textarea {
            background-color: #f0f0f0;
            color: black;
        }
        .stButton > button {
            background-color: red;
            color: white;
            font-weight: bold;
            border-radius: 8px;
        }
        .stButton > button:hover {
            background-color: darkred;
        }
        .sentiment-box {
            background-color: #f2f2f2;
            padding: 1rem;
            border-radius: 10px;
            margin-top: 20px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Page Title and Headers
st.image("https://cdn-icons-png.flaticon.com/512/776/776556.png", width=50)
st.title("🌾 ALVIN - DATA SCIENCE & AI")
st.header("🌿 Agriculture Sentiment Analyzer")
st.caption("Analyze farmers' sentiments about crops and planting")

# User Input
user_text = st.text_area("Type your message here:", placeholder="e.g. The weather is good for planting wheat.")

if st.button("Analyze Sentiment"):
    if user_text.strip():
        X = vectorizer.transform([user_text])
        prediction = sentiment_model.predict(X)[0]

        st.markdown(f"""
            <div class="sentiment-box">
                🧠 The sentiment of the text is: <span style='color: green;'>{prediction.upper()}</span>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter some text to analyze.")
