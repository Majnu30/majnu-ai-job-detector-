import streamlit as st
import pickle

# Load saved model and vectorizer
text_model = pickle.load(open("text_model.pkl", "rb"))
vectorizer = pickle.load(open("text_vectorizer.pkl", "rb"))

# Page title
st.title("AI Fake Job Detection System")

st.write("Enter a job description to check whether it is Real or Fake.")

# Text input
job_desc = st.text_area("Job Description")

# Button
if st.button("Check Job"):

    if job_desc.strip() == "":
        st.warning("Please enter a job description.")
    
    else:
        data = vectorizer.transform([job_desc])
        prediction = text_model.predict(data)

        if prediction[0] == 1:
            st.error("⚠ This looks like a FAKE job posting.")
        else:
            st.success("✅ This job posting appears to be REAL.")
