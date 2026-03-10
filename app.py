import streamlit as st
import pickle

# 1. Page Configuration
st.set_page_config(
    page_title="AI Fake Job Detector",
    page_icon="🛡️",
    layout="centered"
)

# 2. Custom CSS for Dark Mode Design
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    /* Sidebar Background */
    section[data-testid="stSidebar"] {
        background-color: #161B22 !important;
    }

    /* Input Fields */
    .stTextArea textarea, .stTextInput input {
        background-color: #21262D !important;
        color: white !important;
        border: 1px solid #30363D !important;
        border-radius: 8px !important;
    }

    /* Titles and Text */
    h1, h2, h3, p, span, label {
        color: #F0F6FC !important;
        font-family: 'Inter', sans-serif;
    }

    /* Button Styling (Neon Blue) */
    .stButton>button {
        background-color: #238636;
        color: white;
        border-radius: 8px;
        border: none;
        height: 3.5em;
        width: 100%;
        font-weight: bold;
        font-size: 1.1em;
        transition: 0.3s;
        box-shadow: 0px 4px 15px rgba(35, 134, 54, 0.3);
    }
    
    .stButton>button:hover {
        background-color: #2ea043;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0px 6px 20px rgba(35, 134, 54, 0.5);
    }

    /* Result Boxes */
    div
