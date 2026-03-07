import streamlit as st
import pickle
import os

st.title("AI Fake Job Detection System")

model_path = "text_model.pkl"
vectorizer_path = "text_vectorizer.pkl"

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    st.error("Model files not found. Please upload text_model.pkl and text_vectorizer.pkl.")
else:
    text_model = pickle.load(open(model_path, "rb"))
    vectorizer = pickle.load(open(vectorizer_path, "rb"))

    st.write("Enter a job description to detect if it is Fake or Real.")

    job_desc = st.text_area("Job Description")

    if st.button("Check Job"):
        if job_desc.strip() == "":
            st.warning("Please enter a job description.")
        else:
            data = vectorizer.transform([job_desc])
            prediction = text_model.predict(data)

            if prediction[0] == 1:
                st.error("⚠ Fake Job Posting")
            else:
                st.success("✅ Real Job Posting")
