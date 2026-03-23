import streamlit as st
import pickle
from urllib.parse import urlparse

trusted_domains = [
    "tcs.com",
    "infosys.com",
    "wipro.com",
    "amazon.jobs",
    "google.com",
    "linkedin.com",
    "naukri.com",
    "internshala.com"
]

def is_trusted(url):
    domain = urlparse(url).netloc.lower()
    for trusted in trusted_domains:
        if trusted in domain:
            return True
    return False

# Load models
text_model = pickle.load(open("text_model.pkl", "rb"))
text_vectorizer = pickle.load(open("text_vectorizer.pkl", "rb"))

url_model = pickle.load(open("url_model.pkl", "rb"))
url_vectorizer = pickle.load(open("url_vectorizer.pkl", "rb"))

# Page config
st.set_page_config(page_title="AI Fake Job Detector", page_icon="🤖", layout="centered")

# Title
st.markdown("<h1 style='text-align:center;'>🤖 AI Fake Job Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Detect fake job postings using Machine Learning</p>", unsafe_allow_html=True)

st.markdown("---")

# URL Input
st.subheader("🔗 Job URL")
job_url = st.text_input("Paste the job website link")

# Description Input
st.subheader("📝 Job Description")
job_desc = st.text_area("Paste the job description here")

st.markdown("---")

# Single Button
if st.button("🚀 Check Job Authenticity"):

    if job_url.strip() == "" and job_desc.strip() == "":
        st.warning("Please enter a Job URL or Job Description.")
    else:

        st.subheader("🔍 Analysis Result")

        # URL prediction
        if job_url.strip() != "":
            if is_trusted(job_url):
                st.success("✅ Legitimate Job Website (Trusted Domain)")
    
            else:
                url_data = url_vectorizer.transform([job_url])
                url_pred = url_model.predict(url_data)

                if url_pred[0] == 1:
                    st.error("⚠ Suspicious / Fake Job URL")
                else:
                    st.success("✅ Job URL looks Safe")
            

        # Description prediction
        if job_desc.strip() != "":
            desc_data = text_vectorizer.transform([job_desc])
            desc_pred = text_model.predict(desc_data)

            if desc_pred[0] == 1:
                st.error("⚠ Fake Job Description")
            else:
                st.success("✅ Job Description looks Real")

st.markdown("---")

# Footer
st.markdown(
    "<p style='text-align:center;'>Developed by Batch 11</b></p>",
    unsafe_allow_html=True
)
