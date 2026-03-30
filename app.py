import streamlit as st
import pickle
import time
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Majnu AI Job Detector", layout="wide")

# ===============================
# LOAD MODELS
# ===============================
text_model = pickle.load(open("text_model.pkl", "rb"))
url_model = pickle.load(open("url_model.pkl", "rb"))
text_vectorizer = pickle.load(open("text_vectorizer.pkl", "rb"))
url_vectorizer = pickle.load(open("url_vectorizer.pkl", "rb"))

# ===============================
# SESSION STATE
# ===============================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "history" not in st.session_state:
    st.session_state.history = []

# ===============================
# LOGIN PAGE
# ===============================
if not st.session_state.logged_in:
    st.title("🔐 Login - Majnu AI")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "majnu" and password == "1234":
            st.session_state.logged_in = True
            st.success("Login Successful")
            st.rerun()
        else:
            st.error("Invalid Credentials")

# ===============================
# MAIN APP
# ===============================
else:

    # SIDEBAR
    st.sidebar.title("🤖 Majnu AI")
    menu = st.sidebar.radio("Menu", ["Dashboard", "Analyzer", "History", "Analytics", "Logout"])

    # ===============================
    # DASHBOARD
    # ===============================
    if menu == "Dashboard":
        st.title("📊 Dashboard")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Checks", len(st.session_state.history))
        col2.metric("Fake Detected", sum(1 for x in st.session_state.history if x[2] == 1))
        col3.metric("Real Jobs", sum(1 for x in st.session_state.history if x[2] == 0))

    # ===============================
    # ANALYZER
    # ===============================
    elif menu == "Analyzer":

        st.title("🚀 AI Job Analyzer")

        tab1, tab2, tab3 = st.tabs(["🔗 URL", "📝 Description", "⚡ Combined"])

        # URL
        with tab1:
            url = st.text_input("Enter URL")

            if st.button("Check URL"):
                with st.spinner("Analyzing..."):
                    time.sleep(1.5)

                    vec = url_vectorizer.transform([url])
                    pred = url_model.predict(vec)[0]
                    prob = url_model.predict_proba(vec)[0].max()

                    if pred == 1:
                        st.error(f"🚨 Fake URL ({round(prob*100,2)}%)")
                    else:
                        st.success(f"✅ Safe URL ({round(prob*100,2)}%)")

                    st.progress(int(prob*100))
                    st.info("🔍 Reason: Suspicious patterns detected")

                    st.session_state.history.append(("URL", url, pred))

        # TEXT
        with tab2:
            desc = st.text_area("Enter Description")

            if st.button("Check Description"):
                with st.spinner("Analyzing..."):
                    time.sleep(1.5)

                    vec = text_vectorizer.transform([desc])
                    pred = text_model.predict(vec)[0]
                    prob = text_model.predict_proba(vec)[0].max()

                    if pred == 1:
                        st.error(f"🚨 Fake Job ({round(prob*100,2)}%)")
                    else:
                        st.success(f"✅ Real Job ({round(prob*100,2)}%)")

                    st.progress(int(prob*100))
                    st.info("🔍 Reason: Suspicious keywords found")

                    st.session_state.history.append(("TEXT", desc[:50], pred))

        # COMBINED
        with tab3:
            url = st.text_input("URL (optional)")
            desc = st.text_area("Description (optional)")

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
                        st.error(f"🚨 FAKE JOB ({confidence}%)")
                    else:
                        st.success(f"✅ REAL JOB ({confidence}%)")

                    st.progress(int(confidence))
                    st.info("🔍 AI Decision based on patterns")

                    st.session_state.history.append(("COMBINED", "Check", final))

    # ===============================
    # HISTORY
    # ===============================
    elif menu == "History":
        st.title("📜 History")

        if st.session_state.history:
            df = pd.DataFrame(st.session_state.history, columns=["Type", "Input", "Result"])
            st.dataframe(df)
        else:
            st.info("No history available")

    # ===============================
    # ANALYTICS (CHART)
    # ===============================
    elif menu == "Analytics":
        st.title("📊 Analytics")

        if st.session_state.history:
            results = [x[2] for x in st.session_state.history]

            labels = ["Real", "Fake"]
            values = [results.count(0), results.count(1)]

            plt.figure()
            plt.bar(labels, values)
            st.pyplot(plt)
        else:
            st.info("No data to analyze")

    # ===============================
    # LOGOUT
    # ===============================
    elif menu == "Logout":
        st.session_state.logged_in = False
        st.rerun()
