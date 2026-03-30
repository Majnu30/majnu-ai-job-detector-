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
st.set_page_config(page_title="Majnu AI Job Detector", page_icon="🤖", layout="wide")

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("🤖 Majnu AI")
menu = st.sidebar.radio("Navigation", ["Home", "History", "About"])

# ===============================
# CUSTOM CSS
# ===============================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #141E30, #243B55);
    color: white;
}
.result {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# SESSION STATE (HISTORY)
# ===============================
if "history" not in st.session_state:
    st.session_state.history = []

# ===============================
# HOME PAGE
# ===============================
if menu == "Home":

    st.title("🚀 AI Fake Job Detection System")
    st.caption("Detect Fake Jobs using AI (URL + Description)")

    tab1, tab2, tab3 = st.tabs(["🔗 URL Check", "📝 Description Check", "⚡ Combined Check"])

    # -----------------------------
    # URL TAB
    # -----------------------------
    with tab1:
        url = st.text_input("Enter Job URL")

        if st.button("Check URL"):
            with st.spinner("Analyzing URL..."):
                time.sleep(1.5)
                vec = url_vectorizer.transform([url])
                pred = url_model.predict(vec)[0]
                prob = url_model.predict_proba(vec)[0].max()

                if pred == 1:
                    st.error(f"🚨 Fake URL ({round(prob*100,2)}%)")
                else:
                    st.success(f"✅ Safe URL ({round(prob*100,2)}%)")

                st.progress(int(prob*100))

                st.session_state.history.append(("URL", url, pred))

    # -----------------------------
    # TEXT TAB
    # -----------------------------
    with tab2:
        desc = st.text_area("Enter Job Description")

        if st.button("Check Description"):
            with st.spinner("Analyzing Description..."):
                time.sleep(1.5)
                vec = text_vectorizer.transform([desc])
                pred = text_model.predict(vec)[0]
                prob = text_model.predict_proba(vec)[0].max()

                if pred == 1:
                    st.error(f"🚨 Fake Job ({round(prob*100,2)}%)")
                else:
                    st.success(f"✅ Real Job ({round(prob*100,2)}%)")

                st.progress(int(prob*100))

                st.session_state.history.append(("TEXT", desc[:50], pred))

    # -----------------------------
    # COMBINED TAB
    # -----------------------------
    with tab3:
        url = st.text_input("Enter URL (optional)")
        desc = st.text_area("Enter Description (optional)")

        if st.button("Analyze Both"):
            with st.spinner("Analyzing..."):
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

                final = max(results)
                confidence = round(max(probs)*100, 2)

                if final == 1:
                    st.error(f"🚨 FAKE JOB DETECTED ({confidence}%)")
                    st.info("⚠️ Reason: Suspicious keywords / malicious URL patterns")
                else:
                    st.success(f"✅ REAL JOB ({confidence}%)")
                    st.info("✔️ Looks safe based on trained patterns")

                st.progress(int(confidence))

                st.session_state.history.append(("COMBINED", "Check", final))

# ===============================
# HISTORY PAGE
# ===============================
elif menu == "History":

    st.title("📊 Prediction History")

    if st.session_state.history:
        for item in st.session_state.history:
            st.write(item)
    else:
        st.info("No history yet")

# ===============================
# ABOUT PAGE
# ===============================
elif menu == "About":

    st.title("📌 About Project")

    st.write("""
    This project uses Machine Learning to detect fake job postings.
    
    Models used:
    - Logistic Regression
    - Random Forest
    
    Features:
    - URL Analysis
    - Job Description Analysis
    - Real-time Prediction
    
    Developed by ❤️ Majnu and Team
    """)
