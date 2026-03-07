import streamlit as st
import pickle
import pandas as pd

# 1. Page Configuration
st.set_page_config(page_title="Job Guardian AI", page_icon="🛡️", layout="wide")

# 2. Advanced CSS for Gradient Headers and Gauges
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stProgress > div > div > div > div { background-image: linear-gradient(to right, #4caf50, #f44336); }
    .report-card {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border-top: 5px solid #6c63ff;
    }
    .footer { text-align: center; padding: 20px; color: #888; font-size: 0.9em; }
    </style>
    """, unsafe_allow_html=True)

# --- Load models ---
@st.cache_resource
def load_models():
    # Note: Ensure your models were trained with probability=True if using SVM
    t_model = pickle.load(open("text_model.pkl", "rb"))
    t_vec = pickle.load(open("text_vectorizer.pkl", "rb"))
    u_model = pickle.load(open("url_model.pkl", "rb"))
    u_vec = pickle.load(open("url_vectorizer.pkl", "rb"))
    return t_model, t_vec, u_model, u_vec

text_model, text_vectorizer, url_model, url_vectorizer = load_models()

# --- UI Header ---
st.title("🛡️ Job Guardian: AI Fraud Detection")
st.write("Analyze job postings with deep-learning linguistics and URL heuristics.")

# --- Layout ---
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 Job Details")
    job_desc = st.text_area("Job Description", height=250, placeholder="Paste the text here...")
    job_url = st.text_input("Job URL", placeholder="https://linkedin.com/jobs/...")

with col2:
    st.markdown("### 🛠️ Analysis Control")
    st.write("Click analyze to run the dual-engine scan.")
    run_button = st.button("🚀 Run Full Diagnostic")
    
    if run_button:
        if not job_desc or not job_url:
            st.error("Missing input fields!")
        else:
            # --- Processing ---
            desc_vec = text_vectorizer.transform([job_desc])
            url_vec = url_vectorizer.transform([job_url])
            
            # Get Probabilities (Index 1 is usually 'Fake')
            desc_prob = text_model.predict_proba(desc_vec)[0][1] * 100
            url_prob = url_model.predict_proba(url_vec)[0][1] * 100
            
            st.session_state['desc_score'] = desc_prob
            st.session_state['url_score'] = url_prob
            st.session_state['analyzed'] = True

# --- Results Section ---
if st.session_state.get('analyzed'):
    st.divider()
    st.subheader("📊 Risk Assessment Report")
    
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        score = st.session_state['desc_score']
        st.markdown(f"<div class='report-card'>", unsafe_allow_html=True)
        st.write("**Description Risk Score**")
        st.metric(label="Risk Level", value=f"{score:.1f}%", delta="- Safe" if score < 50 else "+ Risk", delta_color="inverse")
        st.progress(score / 100)
        
        if score > 70:
            st.warning("🚨 High probability of linguistic manipulation (scam patterns detected).")
        elif score > 40:
            st.info("🟡 Moderate risk. Proceed with caution and verify company email.")
        else:
            st.success("🟢 Low risk. The text appears naturally written and professional.")
        st.markdown("</div>", unsafe_allow_html=True)

    with res_col2:
        u_score = st.session_state['url_score']
        st.markdown(f"<div class='report-card'>", unsafe_allow_html=True)
        st.write("**URL Trust Score**")
        st.metric(label="Suspicion Level", value=f"{u_score:.1f}%", delta="- Safe" if u_score < 50 else "+ Risk", delta_color="inverse")
        st.progress(u_score / 100)
        
        if u_score > 70:
            st.error("🛑 Highly Suspicious URL structure or domain.")
        else:
            st.success("✅ The domain structure follows standard trusted patterns.")
        st.markdown("</div>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("<div class='footer'>Developed with ❤️ by Majnu and Team | © 2026 Safeguard AI</div>", unsafe_allow_html=True)
