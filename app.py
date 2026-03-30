import streamlit as st
import pickle
import time

# ===============================
# LOAD MODELS
# ===============================
text_model = pickle.load(open("text_model.pkl", "rb"))
url_model = pickle.load(open("url_model.pkl", "rb"))
text_vectorizer = pickle.load(open("text_vectorizer.pkl", "rb"))
url_vectorizer = pickle.load(open("url_vectorizer.pkl", "rb"))

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title=" AI Job Detector",
    page_icon="🤖",
    layout="centered"
)

# ===============================
# CUSTOM CSS (GRADIENT + MOBILE)
# ===============================
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* Title */
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #00e5ff;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #d1d1d1;
    margin-bottom: 20px;
}

/* Card UI */
.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
}

/* Result box */
.result {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 40px;
    color: #aaaaaa;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# ===============================
# LOGO + HEADER
# ===============================
st.markdown('<div class="title"> AI Job Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Smart AI to Detect Fake Job Posts Instantly</div>', unsafe_allow_html=True)

# ===============================
# INPUT SECTION
# ===============================
st.markdown('<div class="card">', unsafe_allow_html=True)

url_input = st.text_input("🔗 Enter Job URL")
desc_input = st.text_area("📝 Enter Job Description")

st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# BUTTON
# ===============================
if st.button("🚀 Analyze Job", use_container_width=True):

    if url_input == "" and desc_input == "":
        st.warning("⚠️ Please enter URL or Description")

    else:
        with st.spinner("🔍 Analyzing... Please wait"):
            time.sleep(2)  # Loading animation

            results = []
            confidences = []

            # URL Prediction
            if url_input:
                url_vec = url_vectorizer.transform([url_input])
                url_pred = url_model.predict(url_vec)[0]
                url_prob = url_model.predict_proba(url_vec)[0].max()
                results.append(url_pred)
                confidences.append(url_prob)

            # TEXT Prediction
            if desc_input:
                text_vec = text_vectorizer.transform([desc_input])
                text_pred = text_model.predict(text_vec)[0]
                text_prob = text_model.predict_proba(text_vec)[0].max()
                results.append(text_pred)
                confidences.append(text_prob)

            final_result = max(results)
            confidence = round(max(confidences) * 100, 2)

        # ===============================
        # OUTPUT
        # ===============================
        if final_result == 1:
            st.markdown(
                f'<div class="result" style="background:#ff4b4b;">🚨 FAKE JOB DETECTED<br>Confidence: {confidence}%</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result" style="background:#00c853;">✅ REAL JOB<br>Confidence: {confidence}%</div>',
                unsafe_allow_html=True
            )

# ===============================
# FOOTER
# ===============================
st.markdown(
    '<div class="footer">Developed by batch 11 Team</div>',
    unsafe_allow_html=True
)
