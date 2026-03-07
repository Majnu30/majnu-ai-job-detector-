import streamlit as st
import pickle
import os

st.title("AI Fake Job Detection System")

# Load models
text_model = pickle.load(open("text_model.pkl","rb"))
text_vectorizer = pickle.load(open("text_vectorizer.pkl","rb"))

url_model = pickle.load(open("url_model.pkl","rb"))
url_vectorizer = pickle.load(open("url_vectorizer.pkl","rb"))

st.write("Check whether a Job Posting or URL is Fake or Real.")

# ---- Job Description Section ----
st.subheader("Check Job Description")

job_desc = st.text_area("Enter Job Description")

if st.button("Check Job Description"):
    
    if job_desc.strip() == "":
        st.warning("Please enter a job description.")
    else:
        data = text_vectorizer.transform([job_desc])
        prediction = text_model.predict(data)

        if prediction[0] == 1:
            st.error("⚠ Fake Job Posting")
        else:
            st.success("✅ Real Job Posting")


# ---- URL Section ----
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
