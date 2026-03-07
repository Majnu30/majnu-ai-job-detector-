import streamlit as st
import pickle
import requests
from bs4 import BeautifulSoup
import os

# Load the trained model safely
model_path = "text_model.pkl"

if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
else:
    st.error("Model file not found. Please upload text_model.pkl to the repository.")
    st.stop()

# Title
st.title("AI Fake Job Detection System")

# URL Input (Top)
st.subheader("Check Job URL")
url = st.text_input("Enter Job URL")

# Description Input (Bottom)
st.subheader("Check Job Description")
description = st.text_area("Enter Job Description")

# Single Button
if st.button("Check Job"):

    text_to_check = ""

    # Extract text from URL
    if url:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            text_to_check += soup.get_text()
        except:
            st.warning("Could not read the URL")

    # Add description text
    if description:
        text_to_check += " " + description

    if text_to_check.strip() == "":
        st.warning("Please enter a URL or job description")
    else:
        prediction = model.predict([text_to_check])

        if prediction[0] == 1:
            st.error("⚠️ This Job Posting is FAKE")
        else:
            st.success("✅ This Job Posting is REAL")

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ by Majnu and Team")
