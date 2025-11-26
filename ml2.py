import streamlit as st
import pickle
from datetime import datetime
import time
import pandas as pd
import plotly.express as px
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Import the URL analysis function
from url_analysis import analyze_url

# Download NLTK data
nltk.download('stopwords')

# Page config - removed sidebar
st.set_page_config(
    page_title="TRUTHLENS",
    page_icon="📰",
    layout="wide"
)

st.markdown("""
<style>
    body { background: linear-gradient(135deg, #0d1b2a, #1b263b); color: #e0e7ff; animation: fadeIn 1s ease-in; } /* Deep navy to dark blue gradient */
    .main { background: transparent; padding: 10px 20px 20px 20px; } /* Reduced top padding */
    .hero { 
        text-align: center; 
        padding: 30px 30px; /* Reduced padding */
        margin-bottom: 20px; 
        color: #e0e7ff; 
        animation: fadeIn 2s ease-out; 
        position: relative; 
    }
    .hero h1 { 
        color: #00d4ff; 
        font-size: 3em; 
        margin-bottom: 10px; 
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.5); 
        /* Removed pulse animation */
    }
    .hero h3 { 
        color: #48cae4; 
        font-size: 1.5em; 
        margin-bottom: 20px; 
    }
    .hero p { 
        font-size: 1.2em; 
        color: #caf0f8; 
    }
    .card { background: linear-gradient(135deg, #1b263b, #415a77); border-radius: 10px; padding: 20px; margin: 10px 0; border: 2px solid #00d4ff; animation: slideInUp 1s ease-out; } /* Dark blue to slate, cyan border */
    .stButton>button { background: linear-gradient(45deg, #00d4ff, #0096c7); color: white; border-radius: 10px; border: none; padding: 10px 20px; font-weight: bold; transition: all 0.3s ease; animation: fadeIn 0.5s ease-in; } /* Cyan to darker cyan */
    .stButton>button:hover { background: linear-gradient(45deg, #0096c7, #00d4ff); transform: scale(1.05); animation: pulse 0.5s; }
    .stTextArea textarea, .stTextInput input { border-radius: 10px; border: 2px solid #00d4ff; padding: 10px; background: #415a77; color: #e0e7ff; transition: all 0.3s ease; animation: fadeIn 0.5s ease-in; } /* Cyan border, slate background */
    .stTextArea textarea:focus, .stTextInput input:focus { border-color: #48cae4; box-shadow: 0 0 10px rgba(0,212,255,0.5); } /* Lighter cyan focus */
    .stSuccess { background: linear-gradient(45deg, #2d6a4f, #52b788); color: #d8f3dc; border-radius: 5px; padding: 10px; animation: slideInLeft 0.5s ease-out; } /* Dark green to teal */
    .stWarning { background: linear-gradient(45deg, #f77f00, #fcbf49); color: #fff3bf; border-radius: 5px; padding: 10px; animation: slideInRight 0.5s ease-out; } /* Orange to yellow */
    .stError { background: linear-gradient(45deg, #d00000, #e63946); color: #ffccd5; border-radius: 5px; padding: 10px; animation: slideInLeft 0.5s ease-out; } /* Dark red to coral */
    .stInfo { background: linear-gradient(45deg, #0077b6, #00b4d8); color: #caf0f8; border-radius: 5px; padding: 10px; animation: slideInRight 0.5s ease-out; } /* Blue to cyan */
    .footer { text-align: center; margin-top: 30px; padding: 15px; background: linear-gradient(135deg, #1b263b, #415a77); border-radius: 10px; animation: fadeIn 1s ease-in; color: #e0e7ff; } /* Dark blue to slate */
    /* Tabs styling for full width with gaps, bigger text, no white strips */
    .stTabs { width: 100%; margin: 0; padding: 0; } /* Removed margins */
    .stTabs [data-baseweb="tab-list"] { display: flex; justify-content: space-between; width: 100%; margin: 0; padding: 0; } /* No margins */
    .stTabs [data-baseweb="tab"] { flex: 1; margin: 0 10px; font-size: 1.5em; font-weight: bold; color: #e0e7ff; } /* Bigger text, gap between tabs, light text */
    /* Distinguish tabs with unique borders */
    .stTabs [data-baseweb="tab"]:nth-child(1) { border-bottom: 3px solid #00d4ff; } /* Text Analysis - cyan */
    .stTabs [data-baseweb="tab"]:nth-child(2) { border-bottom: 3px solid #0096c7; } /* URL Analysis - darker cyan */
    .stTabs [data-baseweb="tab"]:nth-child(3) { border-bottom: 3px solid #48cae4; } /* History & Stats - light cyan */
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    @keyframes bounceIn { 0% { transform: scale(0.3); opacity: 0; } 50% { transform: scale(1.05); } 70% { transform: scale(0.9); } 100% { transform: scale(1); opacity: 1; } }
    @keyframes slideInUp { from { transform: translateY(100%); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
    @keyframes slideInLeft { from { transform: translateX(-100%); opacity: 0; } to { transform: translateX(0); opacity: 1; } }
    @keyframes slideInRight { from { transform: translateX(100%); opacity: 0; } to { transform: translateX(0); opacity: 1; } }
    @keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.1); } 100% { transform: scale(1); } }
</style>
""", unsafe_allow_html=True)

# Load model
try:
    model = pickle.load(open('fake_news_model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    # Removed sidebar warning
except FileNotFoundError:
    st.error("Model files not found.")
    st.stop()

# Hero - Updated without box, in new style
st.markdown("""
<div class="hero">
    <h1>TRUTHLENS</h1>
    <h3>📰 Fake News Detector</h3>
    <p>Check news authenticity with AI. Paste text or enter a URL.</p>
</div>
""", unsafe_allow_html=True)


# Prediction function
def predict_news(text, source_type, source_url=None):
    if not text or text.strip() == "":
        st.warning("Please enter text.")
        return
    with st.spinner("Analyzing..."):
        time.sleep(1)
        ps = PorterStemmer()
        cleaned_text = re.sub('[^a-zA-Z]', ' ', text).lower().split()
        cleaned_text = [ps.stem(word) for word in cleaned_text if word not in stopwords.words('english')]
        cleaned_text = ' '.join(cleaned_text)
        transformed_news = vectorizer.transform([cleaned_text]).toarray()
        result = model.predict(transformed_news)[0].upper()
        confidence = 0
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(transformed_news)[0]
            confidence = max(prob) * 100
            if confidence < 60:
                st.warning("Uncertain. Verify manually.")
                result = "UNCERTAIN"
        if result == "FAKE":
            st.error(f"This news is {result}!")
        elif result == "REAL":
            st.success(f"This news is {result}!")
        if hasattr(model, 'predict_proba') and result != "UNCERTAIN":
            st.info(f"Confidence: {confidence:.2f}%")
            st.progress(int(confidence))
        if 'history' not in st.session_state:
            st.session_state.history = []
        entry = {
            'timestamp': datetime.now(),
            'type': source_type,
            'source': source_url if source_url else 'Pasted Text',
            'result': result,
            'confidence': confidence,
            'text_preview': text[:100] + '...' if len(text) > 100 else text
        }
        st.session_state.history.append(entry)

# Tabs with full width and gaps
tab1, tab2, tab3 = st.tabs(["📝 Text Analysis", "🔗 URL Analysis", "📈 History & Stats"])

with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Analyze News Text")
    news = st.text_area("Paste news text:", height=150)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Predict", key="predict_text"):
            predict_news(news, "Text")
    with col2:
        if st.button("Clear", key="clear_text"):
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Analyze News URL")
    url = st.text_input("Enter URL:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Fetch & Predict", key="fetch_predict_url"):
            with st.spinner("Fetching..."):
                time.sleep(1)
                result_data = analyze_url(url)
                if "error" in result_data:
                    st.error(result_data['error'])
                else:
                    st.info(f"Title: {result_data['title']}")
                    if result_data['uncertain']:
                        st.warning("Uncertain.")
                    elif result_data['result'] == "FAKE":
                        st.error(f"News is {result_data['result']}!")
                    elif result_data['result'] == "REAL":
                        st.success(f"News is {result_data['result']}!")
                    if result_data['confidence'] > 0:
                        st.info(f"Confidence: {result_data['confidence']:.2f}%")
                        st.progress(int(result_data['confidence']))
                    if 'history' not in st.session_state:
                        st.session_state.history = []
                    entry = {
                        'timestamp': datetime.now(),
                        'type': 'URL',
                        'source': url,
                        'result': result_data['result'],
                        'confidence': result_data['confidence'],
                        'text_preview': result_data['text_preview']
                    }
                    st.session_state.history.append(entry)
    with col2:
        if st.button("Clear", key="clear_url"):
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("History & Stats")
    if 'history' in st.session_state and st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df)
        total = len(df)
        fake_count = len(df[df['result'] == 'FAKE'])
        real_count = len(df[df['result'] == 'REAL'])
        st.markdown(f"Total: {total}, Fake: {fake_count}, Real: {real_count}")
        fig = px.pie(values=[fake_count, real_count], names=['Fake', 'Real'], title='Distribution')
        st.plotly_chart(fig)
    else:
        st.info("No history yet.")
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>TRUTHLENS 2025 </p>
    <p>POWERED BY - JASVEER SINGH , KAMAL CHAUDHARY , KARANDEEP SINGH</p>
</div>
""", unsafe_allow_html=True)
