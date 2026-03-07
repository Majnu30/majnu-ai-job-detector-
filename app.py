import streamlit as st
import pickle

st.title("AI Fake Job Detection System")

# Load models
text_model = pickle.load(open("text_model.pkl", "rb"))
text_vectorizer = pickle.load(open("text_vectorizer.pkl", "rb"))

url_model = pickle.load(open("url_model.pkl", "rb"))
url_vectorizer = pickle.load(open("url_vectorizer.pkl", "rb"))

st.write("Check whether a job posting or URL is Fake or Real.")

# ---------------- URL CHECK (TOP) ----------------

st.subheader("Check Job URL")

job_url = st.text_input("Enter Job URL")

if st.button("Check URL"):
    
    if job_url.strip() == "":
        st.warning("Please enter a URL.")
    else:
        url_data = url_vectorizer.transform([job_url])
        prediction = url_model.predict(url_data)

        if prediction[0] == 1:
            st.error("⚠ Suspicious / Fake Job URL")
        else:
            st.success("✅ Safe Job URL")

# ---------------- DESCRIPTION CHECK (BOTTOM) ----------------

st.subheader("Check Job Description")

job_desc = st.text_area("Enter Job Description")

if st.button("Check Description"):
    
    if job_desc.strip() == "":
        st.warning("Please enter a job description.")
    else:
        data = text_vectorizer.transform([job_desc])
        prediction = text_model.predict(data)

        if prediction[0] == 1:
            st.error("⚠ Fake Job Posting")
        else:
            st.success("✅ Real Job Posting")

# Footer
st.markdown("---")
st.markdown("<center>Developed by Majnu and Team</center>", unsafe_allow_html=True)
