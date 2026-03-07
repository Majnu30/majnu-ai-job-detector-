import streamlit as st
import pickle

# 1. Page Configuration (Adds Browser Tab Icon & Title)
st.set_page_config(
    page_title="AI Fake Job Detector",
    page_icon="🔍",
    layout="centered"
)

# 2. Custom CSS for Colors and Designs
st.markdown("""
    <style>
    /* Change background color */
    .stApp {
        background-color: #f4f7f9;
    }
    
    /* Title styling */
    h1 {
        color: #1E3A8A;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        text-align: center;
    }

    /* Button styling */
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 10px;
        border: none;
        height: 3em;
        width: 100%;
        font-weight: bold;
        transition: 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #1E40AF;
        border: none;
        color: white;
        transform: scale(1.02);
    }

    /* Input box styling */
    .stTextArea textarea, .stTextInput input {
        border-radius: 10px !important;
        border: 1px solid #CBD5E1 !important;
    }
    
    /* Footer styling */
    .footer {
        color: #64748B;
        text-align: center;
        padding: 20px;
        font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Logo and Header
# You can replace the URL with your own local logo file path
st.markdown("<center><img src='https://cdn-icons-png.flaticon.com/512/1063/1063376.png' width='80'></center>", unsafe_allow_html=True)
st.title("AI Fake Job Detection System")

st.markdown("<p style='text-align: center; color: #475569;'>Enter Job Description and URL to check whether the job posting is Fake or Real.</p>", unsafe_allow_html=True)

# --- Sidebar Logo/Design ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2706/2706950.png", width=100)
    st.title("Guardian AI")
    st.info("This system uses Machine Learning to protect job seekers from phishing and fraudulent listings.")

# --- Load models ---
# (Added st.cache_resource to make it faster)
@st.cache_resource
def load_data():
    t_model = pickle.load(open("text_model.pkl", "rb"))
    t_vec = pickle.load(open("text_vectorizer.pkl", "rb"))
    u_model = pickle.load(open("url_model.pkl", "rb"))
    u_vec = pickle.load(open("url_vectorizer.pkl", "rb"))
    return t_model, t_vec, u_model, u_vec

text_model, text_vectorizer, url_model, url_vectorizer = load_data()

# --- Inputs ---
job_desc = st.text_area("📄 Job Description", placeholder="Paste the job requirements here...")
job_url = st.text_input("🔗 Job URL", placeholder="https://example.com/careers/job-id")

# --- Single Button ---
if st.button("Check Job Authenticity"):

    if job_desc.strip() == "" or job_url.strip() == "":
        st.warning("⚠️ Please enter both Job Description and Job URL.")
    
    else:
        # Description prediction
        desc_data = text_vectorizer.transform([job_desc])
        desc_prediction = text_model.predict(desc_data)

        # URL prediction
        url_data = url_vectorizer.transform([job_url])
        url_prediction = url_model.predict(url_data)

        st.markdown("### 📊 Results")
        
        col1, col2 = st.columns(2)

        with col1:
            if desc_prediction[0] == 1:
                st.error("🚨 Job Description looks **Fake**")
            else:
                st.success("✅ Job Description looks **Real**")

        with col2:
            if url_prediction[0] == 1:
                st.error("🚨 Job URL looks **Suspicious**")
            else:
                st.success("✅ Job URL looks **Safe**")

# --- Footer ---
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("<div class='footer'>Developed  by <b>#Majnu and Team</b></div>", unsafe_allow_html=True)
