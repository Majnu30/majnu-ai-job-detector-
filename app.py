import streamlit as st
import pickle
import time
import re
import pandas as pd

# ===============================
# LOAD MODELS
# ===============================
text_model = pickle.load(open("text_model.pkl", "rb"))
url_model = pickle.load(open("url_model.pkl", "rb"))
text_vectorizer = pickle.load(open("text_vectorizer.pkl", "rb"))
url_vectorizer = pickle.load(open("url_vectorizer.pkl", "rb"))

# ===============================
# SESSION
# ===============================
if "history" not in st.session_state:
    st.session_state.history = []

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="God Level AI Job Detector", layout="wide")

# ===============================
# HEADER
# ===============================
st.title("👑 GOD LEVEL AI JOB DETECTOR")
st.caption("Advanced AI System with Explainability & Smart Risk Score")

# ===============================
# INPUT SECTION
# ===============================
col1, col2 = st.columns(2)

with col1:
    url = st.text_input("🔗 Enter Job URL")

with col2:
    file = st.file_uploader("📄 Upload Job PDF", type=["txt", "pdf"])

desc = st.text_area("📝 Enter Job Description")

# ===============================
# KEYWORD LIST (AI EXPLANATION)
# ===============================
suspicious_words = [
    "urgent hiring", "work from home", "no experience",
    "quick money", "easy job", "free registration"
]

def highlight_keywords(text):
    found = []
    for word in suspicious_words:
        if word in text.lower():
            found.append(word)
    return found

# ===============================
# ANALYZE BUTTON
# ===============================
if st.button("🚀 Analyze Job"):

    with st.spinner("🤖 AI is thinking..."):
        time.sleep(2)

        results = []
        probs = []

        # URL MODEL
        if url:
            vec = url_vectorizer.transform([url])
            pred = url_model.predict(vec)[0]
            prob = url_model.predict_proba(vec)[0].max()
            results.append(pred)
            probs.append(prob)

        # TEXT MODEL
        if desc:
            vec = text_vectorizer.transform([desc])
            pred = text_model.predict(vec)[0]
            prob = text_model.predict_proba(vec)[0].max()
            results.append(pred)
            probs.append(prob)

        if results:
            final = max(results)
            confidence = round(max(probs)*100, 2)
        else:
            st.warning("Please provide input")
            st.stop()

        # ===============================
        # SMART RISK SCORE
        # ===============================
        risk_score = confidence if final == 1 else 100 - confidence

        # ===============================
        # RESULT DISPLAY
        # ===============================
        if final == 1:
            st.error(f"🚨 FAKE JOB DETECTED")
        else:
            st.success(f"✅ REAL JOB")

        st.metric("Confidence", f"{confidence}%")
        st.metric("Risk Score", f"{risk_score}%")

        # ===============================
        # KEYWORD HIGHLIGHT
        # ===============================
        keywords = highlight_keywords(desc)

        if keywords:
            st.warning(f"⚠️ Suspicious Keywords Found: {', '.join(keywords)}")
        else:
            st.info("No suspicious keywords detected")

        # ===============================
        # AI EXPLANATION
        # ===============================
        if final == 1:
            explanation = "This job looks suspicious due to risky patterns and keywords."
        else:
            explanation = "This job appears safe based on trained data patterns."

        st.info(f"🧠 AI Explanation: {explanation}")

        # ===============================
        # SAVE HISTORY
        # ===============================
        st.session_state.history.append({
            "URL": url,
            "Result": "Fake" if final else "Real",
            "Confidence": confidence
        })

# ===============================
# HISTORY TABLE
# ===============================
st.subheader("📊 History")

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df)
else:
    st.info("No history yet")

# ===============================
# AI CHAT ASSISTANT
# ===============================
st.subheader("🤖 Ask AI")

question = st.text_input("Ask something about job scams")

if question:
    if "fake job" in question.lower():
        st.write("Fake jobs often ask for money, personal info, or unrealistic offers.")
    else:
        st.write("AI Assistant: Please ask about job detection.")
