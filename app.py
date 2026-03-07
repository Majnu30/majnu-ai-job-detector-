import streamlit as st
import pickle

st.title("AI Fake Job Detection System")

st.write("Enter Job Description and URL to check whether the job posting is Fake or Real.")

# Load models
text_model = pickle.load(open("text_model.pkl", "rb"))
text_vectorizer = pickle.load(open("text_vectorizer.pkl", "rb"))

url_model = pickle.load(open("url_model.pkl", "rb"))
url_vectorizer = pickle.load(open("url_vectorizer.pkl", "rb"))

# Inputs
job_desc = st.text_area("Job Description")
job_url = st.text_input("Job URL")

# Single Button
if st.button("Check Job Authenticity"):

    if job_desc.strip() == "" or job_url.strip() == "":
        st.warning("Please enter both Job Description and Job URL.")
    
    else:
        # Description prediction
        desc_data = text_vectorizer.transform([job_desc])
        desc_prediction = text_model.predict(desc_data)

        # URL prediction
        url_data = url_vectorizer.transform([job_url])
        url_prediction = url_model.predict(url_data)

        st.subheader("Results")

        if desc_prediction[0] == 1:
            st.error("⚠ Job Description looks Fake")
        else:
            st.success("✅ Job Description looks Real")

        if url_prediction[0] == 1:
            st.error("⚠ Job URL looks Suspicious")
        else:
            st.success("✅ Job URL looks Safe")

# Footer
st.markdown("---")
st.markdown("<center>Developed by #Majnu and Team</center>", unsafe_allow_html=True)
