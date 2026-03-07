import streamlit as st
import pickle
import requests
from bs4 import BeautifulSoup

# Load model
model = pickle.load(open("text_model.pkl", "rb"))

st.title("AI Fake Job Detection System")

# URL Input (Top)
st.subheader("Check Job URL")
url = st.text_input("Enter Job URL")

# Description Input (Below)
st.subheader("Check Job Description")
description = st.text_area("Enter Job Description")

# Single Button for both
if st.button("Check Job"):
    
    text_to_check = ""

    # If URL is provided, extract text
    if url:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            text_to_check += soup.get_text()
        except:
            st.error("Unable to fetch URL")

    # Add description text
    if description:
        text_to_check += " " + description

    # If nothing entered
    if text_to_check.strip() == "":
        st.warning("Please enter a URL or Description")
    else:
        prediction = model.predict([text_to_check])

        if prediction[0] == 1:
            st.error("⚠️ This Job Posting is FAKE")
        else:
            st.success("✅ This Job Posting is REAL")

st.markdown("---")
st.markdown("Developed with ❤️ by Majnu and Team")
