import streamlit as st
import pickle

# Load models
text_model = pickle.load(open("text_model.pkl", "rb"))
text_vectorizer = pickle.load(open("text_vectorizer.pkl", "rb"))

url_model = pickle.load(open("url_model.pkl", "rb"))
url_vectorizer = pickle.load(open("url_vectorizer.pkl", "rb"))

# Title
st.title("AI Fake Job Detection System")

st.write("Enter the Job URL and Job Description to verify whether the job is Fake or Real.")

# URL Input (Top)
st.subheader("Job URL")
job_url = st.text_input("Enter Job URL")

# Description Input (Bottom)
st.subheader("Job Description")
job_desc = st.text_area("Enter Job Description")

# Single Button
if st.button("Check Job Authenticity"):

    if job_url.strip() == "" and job_desc.strip() == "":
        st.warning("Please enter at least a URL or Job Description.")

    else:

        # URL Prediction
        if job_url.strip() != "":
            url_data = url_vectorizer.transform([job_url])
            url_prediction = url_model.predict(url_data)

            if url_prediction[0] == 1:
                st.error("⚠ Suspicious / Fake Job URL")
            else:
                st.success("✅ Job URL looks Safe")

        # Description Prediction
        if job_desc.strip() != "":
            desc_data = text_vectorizer.transform([job_desc])
            desc_prediction = text_model.predict(desc_data)

            if desc_prediction[0] == 1:
                st.error("⚠ Fake Job Description")
            else:
                st.success("✅ Job Description looks Real")

# Footer
st.markdown("---")
st.markdown("<center>Developed  by Majnu and Team</center>", unsafe_allow_html=True)
