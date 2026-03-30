import streamlit as st
import pickle
import time
import requests
from streamlit_lottie import st_lottie

# ===============================
# LOAD MODELS
# ===============================
text_model = pickle.load(open("text_model.pkl", "rb"))
url_model = pickle.load(open("url_model.pkl", "rb"))
text_vectorizer = pickle.load(open("text_vectorizer.pkl", "rb"))
url_vectorizer = pickle.load(open("url_vectorizer.pkl", "rb"))

# ===============================
# LOAD LOTTIE ANIMATION
# ===============================
def load_lottie(url):
    r = requests.get(url)
    return r.json()

lottie_ai = load_lottie("https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json")

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title=" AI JOB DETECTOR", layout="wide")

# ===============================
# PREMIUM CSS (GLASS UI)
# ===============================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}
.glass {
    background: rgba(255,255,255,0.1);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
}
.result {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# HEADER
# ===============================
col1, col2 = st.columns([2,1])

with col1:
    st.title(" AI JOB DETECTOR")
    st.caption("Next-gen Fake Job Detection with AI Intelligence")

with col2:
    st_lottie(lottie_ai, height=150)

# ===============================
# INPUT UI
# ===============================
st.markdown('<div class="glass">', unsafe_allow_html=True)

url = st.text_input("🔗 Enter Job URL")
desc = st.text_area("📝 Enter Job Description")

st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# BUTTON
# ===============================
if st.button("🚀 Analyze Job"):

    with st.spinner("🤖 AI is analyzing..."):
        time.sleep(2)

        results = []
        probs = []

        if url:
            vec = url_vectorizer.transform([url])
            pred = url_model.predict(vec)[0]
            prob = url_model.predict_proba(vec)[0].max()
            results.append(pred)
            probs.append(prob)

        if desc:
            vec = text_vectorizer.transform([desc])
            pred = text_model.predict(vec)[0]
            prob = text_model.predict_proba(vec)[0].max()
            results.append(pred)
            probs.append(prob)

        if not results:
            st.warning("Please enter input")
            st.stop()

        final = max(results)
        confidence = round(max(probs)*100, 2)

    # ===============================
    # ANIMATED RESULT
    # ===============================
    if final == 1:
        st.markdown(f'<div class="result" style="background:#ff4b4b;">🚨 FAKE JOB<br>{confidence}% Confidence</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result" style="background:#00c853;">✅ REAL JOB<br>{confidence}% Confidence</div>', unsafe_allow_html=True)

    # ===============================
    # PROGRESS BAR
    # ===============================
    st.progress(int(confidence))

    # ===============================
    # AI EXPLANATION
    # ===============================
    if final == 1:
        st.info("⚠️ AI detected suspicious patterns like unrealistic offers or phishing signals.")
    else:
        st.info("✔️ AI found no major risk patterns.")

# ===============================
# FOOTER
# ===============================
st.markdown("""
<hr>
<center>🚀 Developed by Majnu & Team | AI Powered System</center>
""", unsafe_allow_html=True)
