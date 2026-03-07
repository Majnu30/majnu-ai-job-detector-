import streamlit as st
import pickle

# 1. Page Configuration
st.set_page_config(page_title="AI Fake Job Detection", page_icon="🕵️‍♂️", layout="wide")

# 2. Custom CSS for Colors and Design
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3.5em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
        font-size: 1.1em;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        color: white;
        transform: translateY(-2px);
    }
    .report-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border-top: 5px solid #007bff;
        margin-bottom: 20px;
    }
    footer {text-align: center; padding: 20px; color: #666;}
    </style>
    """, unsafe_allow_html=True)

# --- Load models ---
@st.cache_resource
def load_models():
    try:
        t_model = pickle.load(open("text_model.pkl", "rb"))
        t_vec = pickle.load(open("text_vectorizer.pkl", "rb"))
        u_model = pickle.load(open("url_model.pkl", "rb"))
        u_vec = pickle.load(open("url_vectorizer.pkl", "rb"))
        return t_model, t_vec, u_model, u_vec
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

text_model, text_vectorizer, url_model, url_vectorizer = load_models()

# --- Main App Interface ---
st.title("🛡️ AI Fake Job Detection System")
st.write("Enter the job details below to analyze the risk of fraud using AI.")

# --- Layout: Inputs ---
col_in1, col_in2 = st.columns([2, 1])

with col_in1:
    job_desc = st.text_area("📄 Job Description", height=250, placeholder="Paste the full job description here...")

with col_in2:
    job_url = st.text_input("🔗 Job URL", placeholder="e.g., https://company-careers.com/job/123")
    st.write("---")
    # THE UPDATED BUTTON
    run_analysis = st.button("🔍 Check Job Authenticity")

# --- Analysis Logic ---
if run_analysis:
    if not job_desc.strip() or not job_url.strip():
        st.warning("⚠️ Please provide both the Job Description and the URL to proceed.")
    else:
        with st.spinner('Analyzing patterns and calculating risk...'):
            # Text analysis
            desc_vec = text_vectorizer.transform([job_desc])
            # Getting probability of class 1 (Fake)
            desc_prob = text_model.predict_proba(desc_vec)[0][1] * 100
            
            # URL analysis
            url_vec = url_vectorizer.transform([job_url])
            # Getting probability of class 1 (Suspicious)
            url_prob = url_model.predict_proba(url_vec)[0][1] * 100

        # --- Display Results ---
        st.markdown("### 📊 Security Assessment Report")
        res_col1, res_col2 = st.columns(2)

        # Description Result Card
        with res_col1:
            st.markdown("<div class='report-card'>", unsafe_allow_html=True)
            st.write("**Content Fraud Probability**")
            st.header(f"{desc_prob:.1f}%")
            st.progress(desc_prob / 100)
            
            if desc_prob > 75:
                st.error("❌ **High Risk:** This description matches known scam templates.")
            elif desc_prob > 40:
                st.warning("⚠️ **Moderate Risk:** Some phrases seem unusual. Verify the recruiter.")
            else:
                st.success("✅ **Low Risk:** The content appears legitimate.")
            st.markdown("</div>", unsafe_allow_html=True)

        # URL Result Card
        with res_col2:
            st.markdown("<div class='report-card'>", unsafe_allow_html=True)
            st.write("**URL Suspicion Level**")
            st.header(f"{url_prob:.1f}%")
            st.progress(url_prob / 100)
            
            if url_prob > 75:
                st.error("❌ **Dangerous URL:** This link structure is highly suspicious.")
            elif url_prob > 40:
                st.warning("⚠️ **Caution:** The URL doesn't match standard corporate formats.")
            else:
                st.success("✅ **Safe URL:** The domain appears to be trustworthy.")
            st.markdown("</div>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.markdown("<center>Developed  by <b>#Majnu and Team</b></center>", unsafe_allow_html=True)
