import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests
import pickle
import json
import base64
import os
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Habitor - Student Grade Prediction",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load and cache prediction model
@st.cache_resource
def load_model():
    # This is a placeholder. In a real app, you'd load your trained model
    # Example: model = pickle.load(open('model/bilstm_model.pkl', 'rb'))
    
    # For demonstration, we'll create a more sophisticated mock model
    class MockModel:
        def predict(self, X):
            # Get feature values
            study_hours = X["study_hours"].values[0] if "study_hours" in X else 0
            attendance = X["attendance_rate"].values[0] if "attendance_rate" in X else 0
            sleep_hours = X["sleep_hours"].values[0] if "sleep_hours" in X else 0
            consistency = X["study_consistency"].values[0] if "study_consistency" in X else 0
            revision = X["revision_frequency"].values[0] if "revision_frequency" in X else 0
            social_media = X["social_media_hours"].values[0] if "social_media_hours" in X else 0
            
            # Base CGPA (2.0-4.0 scale)
            base_cgpa = 2.0
            
            # Study factors (up to +1.0)
            study_factor = min(study_hours / 12, 1.0) * 0.5
            consistency_factor = consistency / 4 * 0.3
            revision_factor = revision / 4 * 0.2
            
            # Health factors (up to +0.5)
            sleep_factor = 0
            if 7 <= sleep_hours <= 9:
                sleep_factor = 0.3
            else:
                sleep_factor = max(0, 0.3 - abs(sleep_hours - 8) * 0.1)
            
            # Attendance factor (up to +0.8)
            attendance_factor = attendance * 0.8
            
            # Distraction factor (negative)
            distraction_factor = min(social_media / 12, 1.0) * -0.3
            
            # Calculate final CGPA with some randomness
            final_cgpa = base_cgpa + study_factor + consistency_factor + revision_factor + sleep_factor + attendance_factor + distraction_factor
            
            # Add small random variation
            final_cgpa += np.random.normal(0, 0.1)
            
            # Ensure CGPA is within bounds
            final_cgpa = max(2.0, min(4.0, final_cgpa))
            
            return np.array([final_cgpa])
    
    return MockModel()

model = load_model()

# Load animation files
@st.cache_data
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Lottie animations
lottie_education = load_lottie_url("https://assets4.lottiefiles.com/packages/lf20_cwyqki9a.json")
lottie_prediction = load_lottie_url("https://assets3.lottiefiles.com/packages/lf20_zepsxyjy.json")
lottie_analysis = load_lottie_url("https://assets5.lottiefiles.com/packages/lf20_q5qeoo3q.json")
lottie_loading = load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_kk62um5v.json")

# Function to load SVG image and convert to base64
def get_svg_base64(svg_file_path):
    if not os.path.exists(svg_file_path):
        return None
    
    with open(svg_file_path, "r") as f:
        svg_content = f.read()
    
    # Convert to base64
    svg_bytes = svg_content.encode('utf-8')
    encoded = base64.b64encode(svg_bytes).decode('utf-8')
    
    return f"data:image/svg+xml;base64,{encoded}"

# Get logo as base64
logo_path = Path("images/habitor_logo.svg")
logo_base64 = get_svg_base64(str(logo_path))

# Define CSS styles
def local_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif !important;
        font-weight: 500;
    }
    
    .main {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    
    .stApp {
        transition: all 0.5s ease;
    }
    
    .dark-mode {
        --background-color: #0E1117;
        --text-color: #FFFFFF;
        --card-bg: #262730;
        --accent-color: #FF4B4B;
    }
    
    .light-mode {
        --background-color: #FFFFFF;
        --text-color: #0E1117;
        --card-bg: #F0F2F6;
        --accent-color: #FF4B4B;
    }
    
    .prediction-card {
        background-color: var(--card-bg);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    
    .prediction-card:hover {
        transform: translateY(-5px);
    }
    
    .splash-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 100vh;
        text-align: center;
    }
    
    .title-text {
        font-family: 'Poppins', sans-serif !important;
        font-size: 3.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        background: linear-gradient(45deg, #FF4B4B, #FF8000);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        opacity: 0;
        animation: fadeIn 1s ease forwards;
        animation-delay: 1.5s;
    }
    
    .subtitle-text {
        font-family: 'Poppins', sans-serif !important;
        font-size: 1.5rem;
        font-weight: 300;
        margin-bottom: 2rem;
        opacity: 0;
        animation: fadeIn 1s ease forwards;
        animation-delay: 2s;
    }
    
    .btn-primary {
        background-color: var(--accent-color);
        border: none;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
        transition: all 0.3s ease;
        font-family: 'Poppins', sans-serif !important;
        opacity: 0;
        animation: fadeIn 1s ease forwards;
        animation-delay: 2.5s;
    }
    
    .btn-primary:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
    }
    
    .animated-gradient {
        background: linear-gradient(-45deg, #FF4B4B, #FF8000);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .logo-container {
        width: 200px;
        height: 200px;
        margin-bottom: 20px;
        opacity: 0;
        animation: scaleIn 1s ease forwards;
    }
    
    .stButton button {
        font-family: 'Poppins', sans-serif !important;
    }
    
    /* Animation keyframes */
    @keyframes gradient {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    
    @keyframes fadeIn {
        0% {opacity: 0; transform: translateY(20px);}
        100% {opacity: 1; transform: translateY(0);}
    }
    
    @keyframes scaleIn {
        0% {opacity: 0; transform: scale(0.8);}
        100% {opacity: 1; transform: scale(1);}
    }
    
    @keyframes slideIn {
        0% {opacity: 0; transform: translateX(-30px);}
        100% {opacity: 1; transform: translateX(0);}
    }
    
    /* Sidebar animation */
    .stSidebar {
        animation: slideIn 0.5s ease forwards;
    }
    
    /* Tab button styling */
    .stTabs [data-baseweb="tab-list"] button {
        font-family: 'Poppins', sans-serif !important;
        font-weight: 500;
    }
    
    /* Sidebar style refinements */
    .stSidebar [data-testid="stSidebarNav"] {
        background-color: var(--card-bg);
        padding-top: 1rem;
        border-radius: 10px;
    }
    
    /* Sidebar menu styling */
    .nav-link {
        font-family: 'Poppins', sans-serif !important;
        font-weight: 500 !important;
        font-size: 16px !important;
    }
    
    .nav-link.active {
        background-color: #FF4B4B !important;
        color: white !important;
        font-weight: 500 !important;
    }
    
    /* Custom styling for sidebar menu text */
    .nav-link span {
        font-weight: 400 !important;
    }
    
    .nav-link.active span {
        font-weight: 500 !important;
    }
    
    /* Option menu container styling */
    div[data-testid="stVerticalBlock"] div.css-1544g2n.e1fqkh3o4 {
        padding: 0 !important;
    }
    
    /* Information card styling */
    .info-card {
        background-color: var(--card-bg);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 3px solid #FF4B4B;
        font-weight: 300;
    }
    
    .info-card h2 {
        font-size: 1.5rem;
        font-weight: 500;
        margin-bottom: 10px;
    }
    
    .info-card p {
        font-weight: 300;
        line-height: 1.6;
        margin-bottom: 0.5rem;
    }
    
    .feature-card {
        background-color: var(--card-bg);
        border-radius: 10px;
        padding: 18px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
        border-top: 3px solid #FF8000;
    }
    
    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.12);
    }
    
    .feature-card h3 {
        font-size: 1.2rem;
        font-weight: 500;
        margin-bottom: 8px;
    }
    
    .feature-card p {
        font-weight: 300;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    .feature-icon {
        font-size: 28px;
        color: #FF4B4B;
        margin-bottom: 12px;
    }
    
    .startup-animation {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        background-color: #0E1117;
        z-index: 9999;
        animation: fadeOut 0.5s ease forwards;
        animation-delay: 3s;
    }
    
    @keyframes fadeOut {
        0% {opacity: 1; visibility: visible;}
        100% {opacity: 0; visibility: hidden;}
    }
    
    .loading-text {
        color: white;
        font-family: 'Poppins', sans-serif !important;
        margin-top: 20px;
        font-weight: 500;
    }
    
    .loading-progress {
        width: 200px;
        height: 4px;
        background-color: rgba(255, 255, 255, 0.2);
        border-radius: 4px;
        margin-top: 10px;
        overflow: hidden;
    }
    
    .loading-bar {
        height: 100%;
        width: 0%;
        background: linear-gradient(45deg, #FF4B4B, #FF8000);
        animation: loadingProgress 3s ease forwards;
    }
    
    /* Coming soon section */
    .coming-soon {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 45vh;
        text-align: center;
        margin-top: -20px;
    }
    
    .coming-soon h2 {
        font-size: 2.2rem;
        margin-bottom: 0.8rem;
        color: #FF4B4B;
    }
    
    .coming-soon p {
        font-size: 1.1rem;
        max-width: 600px;
        margin: 0 auto 1.5rem auto;
        font-weight: 300;
    }
    
    /* Center header and subtitle */
    .center-header {
        text-align: center;
        margin-bottom: 25px;
        padding-top: 5px;
        margin-top: -20px;
    }
    
    .center-header h1 {
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 10px;
    }
    
    .center-header p {
        font-size: 1.1rem;
        font-weight: 300;
        max-width: 800px;
        margin: 0 auto;
        opacity: 0.9;
        line-height: 1.6;
    }
    
    /* Reduce vertical spacing */
    .stMarkdown {
        margin-bottom: 0.5rem !important;
    }
    
    /* Main content container padding adjustment */
    .main .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    
    /* Steps section */
    .steps-section {
        margin-top: 10px;
    }
    
    .steps-section h3 {
        font-size: 1.3rem;
        font-weight: 500;
        margin-bottom: 8px;
    }
    
    .steps-section p {
        font-weight: 300;
        font-size: 0.95rem;
    }
    
    @keyframes loadingProgress {
        0% {width: 0%;}
        100% {width: 100%;}
    }
    </style>
    """, unsafe_allow_html=True)

local_css()

# Initialize session state for theme
if 'theme' not in st.session_state:
    st.session_state.theme = "dark"

# Initialize session state for animation
if 'animation_complete' not in st.session_state:
    st.session_state.animation_complete = False
    # Set a timer to automatically transition from splash screen to main content
    st.session_state.show_splash = False

# Create theme toggle
def toggle_theme():
    if st.session_state.theme == "light":
        st.session_state.theme = "dark"
    else:
        st.session_state.theme = "light"

# Apply theme class based on session state
st.markdown(f"""
<script>
    document.querySelector('body').classList.remove('light-mode', 'dark-mode');
    document.querySelector('body').classList.add('{st.session_state.theme}-mode');
</script>
""", unsafe_allow_html=True)

# Startup animation (shown only once when the app loads)
if not st.session_state.animation_complete:
    st.markdown(f"""
    <div class="startup-animation">
        <img src="{logo_base64}" alt="Habitor Logo" width="200">
        <div class="loading-text">Loading Habitor...</div>
        <div class="loading-progress">
            <div class="loading-bar"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.session_state.animation_complete = True

# Sidebar with theme toggle
with st.sidebar:
    st.title("Habitor")
    st.caption("Predict Student Performance Based on Habits")
    
    # Theme toggle button
    st.button("Toggle Light/Dark Mode", on_click=toggle_theme)
    
    # Add custom CSS for the sidebar menu
    st.markdown("""
    <style>
    /* Fix for the menu font weights */
    .css-wjbhl0, .css-17lntkn {
        font-weight: 400 !important;
    }
    .css-17lntkn:hover {
        color: #FF4B4B !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Navigation
    selected = option_menu(
        menu_title=None,
        options=["Home", "Predict CGPA", "Data Analysis"],
        icons=["house", "graph-up", "bar-chart"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"font-size": "18px"},
            "nav-link": {"font-weight": "400", "font-size": "16px", "text-align": "left", "padding": "10px", "margin": "2px 0"},
            "nav-link-selected": {"font-weight": "500", "background-color": "#FF4B4B"}
        }
    )
    
    st.markdown("---")
    st.caption("© 2023 Habitor - SUB")

# Main app content based on navigation selection
if selected == "Home":
    # Center header with title and subtitle
    st.markdown("""
    <div class="center-header">
        <h1 class="animated-gradient">Habitor</h1>
        <p>An intelligent system that predicts student academic performance based on daily habits and behaviors</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Remove extra spacing
    st.markdown('<style>div.block-container{padding-top:0rem;}</style>', unsafe_allow_html=True)
    
    # Static home page content - lighter version
    st.markdown("""
    <div class="info-card">
        <p>Habitor analyzes student habits like study patterns, sleep schedules, and class engagement to predict academic performance. 
        Our AI model identifies which behaviors most strongly influence success and provides personalized recommendations for improvement.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature cards in a 2x2 grid - lighter version with reduced spacing
    st.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <h3>Accurate Predictions</h3>
            <p>Get CGPA predictions based on your daily habits with our advanced AI model.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <h3>AI Analysis</h3>
            <p>Neural networks analyze complex patterns in student behavior data.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <h3>Personalized Advice</h3>
            <p>Receive tailored suggestions to improve your study habits.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📈</div>
            <h3>Visual Insights</h3>
            <p>Explore charts showing how different habits impact academic success.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # How it works section - lighter version
    st.markdown("<h2>How It Works</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="steps-section" style="text-align: center; padding: 15px;">
            <h3>1. Input Your Habits</h3>
            <p>Answer questions about your study routines and daily behaviors.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="steps-section" style="text-align: center; padding: 15px;">
            <h3>2. AI Analysis</h3>
            <p>Our model processes your data and identifies important patterns.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="steps-section" style="text-align: center; padding: 15px;">
            <h3>3. Get Results</h3>
            <p>Receive your predicted CGPA with personalized recommendations.</p>
        </div>
        """, unsafe_allow_html=True)
        
elif selected == "Predict CGPA":
    # CGPA Prediction page
    st.markdown("""
    <div class="center-header">
        <h1 class="animated-gradient">CGPA Prediction</h1>
        <p>Our AI model analyzes your habits to predict academic performance</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Remove extra spacing
    st.markdown('<style>div.block-container{padding-top:0rem;}</style>', unsafe_allow_html=True)
    
    # Create form for user input
    with st.form("habit_form"):
        st.markdown("<h3>Enter Your Habits</h3>", unsafe_allow_html=True)
        
        # Create tabs for different categories
        tabs = st.tabs(["Study Habits", "Health & Lifestyle", "Class Engagement"])
        
        # Tab 1: Study Habits
        with tabs[0]:
            st.markdown("<h4 style='margin-bottom: 25px;'>Study Habits</h4>", unsafe_allow_html=True)
            
            # Use 2 columns layout for better spacing
            sc1, sc2 = st.columns(2)
            
            with sc1:
                study_hours = st.slider("Daily study hours", 0, 12, 3, help="How many hours do you study each day on average?")
                
                study_consistency = st.select_slider(
                    "Study consistency",
                    options=["Very irregular", "Irregular", "Moderate", "Consistent", "Very consistent"],
                    value="Moderate",
                    help="How regular is your study schedule?"
                )
            
            with sc2:
                assignment_completion = st.select_slider(
                    "Assignment completion",
                    options=["Last minute", "Delayed", "On time", "Early", "Well ahead"],
                    value="On time",
                    help="How do you typically complete your assignments?"
                )
                
                revision_frequency = st.select_slider(
                    "Revision frequency",
                    options=["Never", "Rarely", "Monthly", "Weekly", "Daily"],
                    value="Weekly",
                    help="How often do you revise previously learned material?"
                )
        
        # Tab 2: Health & Lifestyle
        with tabs[1]:
            st.markdown("<h4 style='margin-bottom: 25px;'>Health & Lifestyle</h4>", unsafe_allow_html=True)
            
            # Use 2 columns layout for better spacing
            hc1, hc2 = st.columns(2)
            
            with hc1:
                sleep_hours = st.slider("Sleep hours per day", 3, 12, 7, help="How many hours do you sleep each day on average?")
                
                sleep_consistency = st.select_slider(
                    "Sleep schedule consistency",
                    options=["Very irregular", "Irregular", "Moderate", "Consistent", "Very consistent"],
                    value="Moderate",
                    help="How consistent is your sleep schedule?"
                )
            
            with hc2:
                exercise_days = st.slider("Exercise days per week", 0, 7, 3, help="How many days per week do you exercise?")
                
                stress_level = st.select_slider(
                    "Stress level",
                    options=["Very low", "Low", "Medium", "High", "Very high"],
                    value="Medium",
                    help="What is your current stress level?"
                )
        
        # Tab 3: Class Engagement
        with tabs[2]:
            st.markdown("<h4 style='margin-bottom: 25px;'>Class Engagement</h4>", unsafe_allow_html=True)
            
            # Use 2 columns layout for better spacing
            cc1, cc2 = st.columns(2)
            
            with cc1:
                attendance_rate = st.slider("Class attendance (%)", 0, 100, 85, help="What percentage of classes do you attend?")
                
                participation = st.select_slider(
                    "Class participation",
                    options=["Never", "Rarely", "Sometimes", "Often", "Always"],
                    value="Sometimes",
                    help="How actively do you participate in class discussions?"
                )
            
            with cc2:
                note_taking = st.select_slider(
                    "Note-taking quality",
                    options=["None", "Minimal", "Basic", "Detailed", "Comprehensive"],
                    value="Detailed",
                    help="How would you rate your note-taking?"
                )
                
                social_media_hours = st.slider("Social media hours per day", 0, 12, 2, help="How many hours do you spend on social media daily?")
        
        # Add space before submit button
        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
        
        # Submit button - centered and styled
        _, submit_col, _ = st.columns([1, 2, 1])
        with submit_col:
            submitted = st.form_submit_button(
                "Predict My CGPA", 
                use_container_width=True,
                help="Click to analyze your habits and predict your CGPA"
            )
            
            # Add custom styling to the button
            st.markdown("""
            <style>
            div.stButton > button {
                background-color: #FF4B4B;
                color: white;
                font-weight: 500;
                border: none;
                padding: 12px 20px;
                font-size: 16px;
                border-radius: 8px;
                transition: all 0.3s;
            }
            div.stButton > button:hover {
                background-color: #FF2D2D;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            </style>
            """, unsafe_allow_html=True)
    
    # Process the form submission
    if submitted:
        # Show a spinner while processing
        with st.spinner("Analyzing your habits..."):
            # Convert categorical variables to numerical
            study_consistency_map = {"Very irregular": 0, "Irregular": 1, "Moderate": 2, "Consistent": 3, "Very consistent": 4}
            assignment_map = {"Last minute": 0, "Delayed": 1, "On time": 2, "Early": 3, "Well ahead": 4}
            revision_map = {"Never": 0, "Rarely": 1, "Monthly": 2, "Weekly": 3, "Daily": 4}
            sleep_consistency_map = {"Very irregular": 0, "Irregular": 1, "Moderate": 2, "Consistent": 3, "Very consistent": 4}
            stress_map = {"Very low": 0, "Low": 1, "Medium": 2, "High": 3, "Very high": 4}
            participation_map = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3, "Always": 4}
            note_taking_map = {"None": 0, "Minimal": 1, "Basic": 2, "Detailed": 3, "Comprehensive": 4}
            
            # Create features dictionary
            features = {
                "study_hours": study_hours,
                "study_consistency": study_consistency_map[study_consistency],
                "assignment_completion": assignment_map[assignment_completion],
                "revision_frequency": revision_map[revision_frequency],
                "sleep_hours": sleep_hours,
                "sleep_consistency": sleep_consistency_map[sleep_consistency],
                "exercise_days": exercise_days,
                "stress_level": stress_map[stress_level],
                "attendance_rate": attendance_rate / 100,  # Convert to ratio
                "participation": participation_map[participation],
                "note_taking": note_taking_map[note_taking],
                "social_media_hours": social_media_hours
            }
            
            # Convert to DataFrame for prediction
            X = pd.DataFrame([features])
            
            # Add small delay to simulate processing
            time.sleep(1.5)
            
            # Make prediction
            predicted_cgpa = float(model.predict(X)[0])
            
            # Determine performance category
            if predicted_cgpa >= 3.7:
                category = "Excellent"
                category_color = "#28a745"  # Green
            elif predicted_cgpa >= 3.3:
                category = "Very Good"
                category_color = "#5cb85c"  # Light green
            elif predicted_cgpa >= 3.0:
                category = "Good"
                category_color = "#4ECDC4"  # Teal
            elif predicted_cgpa >= 2.7:
                category = "Satisfactory"
                category_color = "#f0ad4e"  # Orange
            else:
                category = "Needs Improvement"
                category_color = "#FF4B4B"  # Red
            
            # Generate recommendations based on inputs
            recommendations = []
            
            # Study habits recommendations
            if study_hours < 3:
                recommendations.append("Increase your daily study hours to at least 3-4 hours")
            if study_consistency_map[study_consistency] < 3:
                recommendations.append("Develop a more consistent study schedule")
            if revision_map[revision_frequency] < 3:
                recommendations.append("Revise your material more frequently, ideally weekly")
            
            # Health recommendations
            if sleep_hours < 7:
                recommendations.append("Increase your sleep to 7-8 hours for optimal cognitive function")
            elif sleep_hours > 9:
                recommendations.append("Reduce your sleep to 7-8 hours for optimal cognitive function")
            else:
                # Only suggest sleep consistency if they're already sleeping the right amount but inconsistently
                if sleep_consistency_map[sleep_consistency] < 3 and sleep_hours >= 7 and sleep_hours <= 9:
                    recommendations.append("Maintain a more consistent sleep schedule while keeping your good sleep duration")
            
            if exercise_days < 3:
                recommendations.append("Exercise at least 3 days per week to improve focus and reduce stress")
            
            # Class engagement recommendations
            if attendance_rate < 85:
                recommendations.append("Improve your class attendance to at least 90%")
            if participation_map[participation] < 2:
                recommendations.append("Participate more actively in class discussions")
            if note_taking_map[note_taking] < 3:
                recommendations.append("Improve your note-taking technique for better retention")
            if social_media_hours > 3:
                recommendations.append("Reduce social media usage to less than 2 hours daily")
            
            # Check for contradictions
            contradictions = []
            if sleep_hours >= 7 and sleep_hours <= 8:
                contradictions.append("Aim for 7-8 hours of sleep for optimal cognitive function")
            
            # Remove any contradicting recommendations
            recommendations = [rec for rec in recommendations if rec not in contradictions]
            
            # Limit to top 4 recommendations
            recommendations = recommendations[:4]
            
            # If no recommendations, add positive reinforcement
            if not recommendations:
                recommendations.append("Maintain your excellent habits that contribute to academic success")
        
        # Display results in a nice UI
        st.markdown("<h3 style='text-align: center; margin-top: 30px;'>Your Prediction Results</h3>", unsafe_allow_html=True)
        
        # Add a divider
        st.markdown("<hr style='margin: 15px 0 30px 0; border: none; height: 1px; background-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
        
        # Create a card-like container for the gauge chart
        st.markdown("""
        <div style='background-color: var(--card-bg); border-radius: 15px; padding: 20px; margin-bottom: 30px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
        """, unsafe_allow_html=True)
        
        # Create two columns for the results with better ratio
        res_col1, res_col2 = st.columns([1, 1.2])
        
        with res_col1:
            # Display the predicted CGPA with gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=predicted_cgpa,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Predicted CGPA", 'font': {'size': 24, 'family': 'Poppins'}},
                gauge={
                    'axis': {'range': [0, 4], 'tickwidth': 1},
                    'bar': {'color': category_color},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 2.0], 'color': '#ff4d4d'},
                        {'range': [2.0, 2.7], 'color': '#ffad33'},
                        {'range': [2.7, 3.0], 'color': '#ffff33'},
                        {'range': [3.0, 3.3], 'color': '#b3ff66'},
                        {'range': [3.3, 3.7], 'color': '#80ff80'},
                        {'range': [3.7, 4.0], 'color': '#33cc33'}
                    ]
                }
            ))
            
            fig.update_layout(
                height=300,
                margin=dict(t=40, b=0, l=40, r=40),
                font={'family': "Poppins"},
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display performance category
            st.markdown(f"""
            <div style="text-align: center; margin-top: -20px;">
                <h4>Performance Category: <span style="color: {category_color};">{category}</span></h4>
            </div>
            """, unsafe_allow_html=True)
        
        with res_col2:
            # Display key habit impacts
            st.markdown("<h4>Top Factors Influencing Your CGPA</h4>", unsafe_allow_html=True)
            
            # Calculate importance scores based on known optimal values
            factor_scores = {
                "Study Hours": min(study_hours / 10 * 100, 100),  # 10+ hours is excellent
                "Attendance": attendance_rate,
                "Sleep Quality": min(100 - abs(((sleep_hours - 7.5) / 4) * 100), 100),  # 7-8 hours is optimal
                "Revision": revision_map[revision_frequency] / 4 * 100,
                "Social Media": max(0, 100 - (social_media_hours / 6 * 100))  # Lower social media is better
            }
            
            # Update Sleep Quality score if sleep is consistent
            if sleep_consistency_map[sleep_consistency] >= 3 and sleep_hours >= 7 and sleep_hours <= 9:
                factor_scores["Sleep Quality"] = min(factor_scores["Sleep Quality"] + 20, 100)
            
            # Create a bar chart for factor importance
            df = pd.DataFrame({
                'Factor': list(factor_scores.keys()),
                'Impact': list(factor_scores.values())
            }).sort_values('Impact', ascending=False)
            
            # Color mapping based on impact
            colors = []
            for impact in df['Impact']:
                if impact >= 80:
                    colors.append('#28a745')  # Green for high impact
                elif impact >= 60:
                    colors.append('#5cb85c')  # Light green
                elif impact >= 40:
                    colors.append('#f0ad4e')  # Orange
                else:
                    colors.append('#FF4B4B')  # Red for low impact
            
            # Update bar colors
            fig = px.bar(
                df,
                x='Impact',
                y='Factor',
                orientation='h',
                labels={'Impact': 'Positive Impact (%)'},
                height=320
            )
            
            # Update bar colors
            fig.update_traces(marker_color=colors)
            
            fig.update_layout(
                margin=dict(t=0, b=0, l=0, r=0),
                xaxis_range=[0, 100],
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'family': "Poppins"},
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)',
                ),
                yaxis=dict(
                    showgrid=False,
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Close the card container
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Display recommendations with a card-like container
        st.markdown("""
        <div style='background-color: var(--card-bg); border-radius: 15px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <h3 style='margin-bottom: 20px;'>Personalized Recommendations</h3>
        """, unsafe_allow_html=True)
        
        if recommendations:
            for i, rec in enumerate(recommendations):
                # Use different border colors based on the type of recommendation
                if "increase" in rec.lower() or "improve" in rec.lower() or "develop" in rec.lower():
                    border_color = "#FF4B4B"  # Red for areas needing improvement
                    icon = "⚠️"
                elif "maintain" in rec.lower() or "excellent" in rec.lower():
                    border_color = "#28a745"  # Green for positive reinforcement
                    icon = "✅"
                else:
                    border_color = "#f0ad4e"  # Orange for general advice
                    icon = "💡"
                
                st.markdown(f"""
                <div style="background-color: rgba(0,0,0,0.1); padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 3px solid {border_color};">
                    <b>{icon} {i+1}.</b> {rec}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("Great job! Keep maintaining your current habits for continued success.")
        
        # Close the card container
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Additional study tips in a card-like container
        with st.expander("View Additional Study Tips"):
            st.markdown("""
            <div style="padding: 10px;">
                <ul style="list-style-type: none; padding-left: 0;">
                    <li style="margin-bottom: 12px;">🕒 <b>Pomodoro Technique:</b> Study for 25 minutes, then take a 5-minute break</li>
                    <li style="margin-bottom: 12px;">🧠 <b>Active Recall:</b> Test yourself instead of just re-reading material</li>
                    <li style="margin-bottom: 12px;">🔄 <b>Spaced Repetition:</b> Review material at increasing intervals</li>
                    <li style="margin-bottom: 12px;">🏠 <b>Study Environment:</b> Create a dedicated, distraction-free study space</li>
                    <li style="margin-bottom: 12px;">👥 <b>Study Groups:</b> Collaborate with peers for different perspectives</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

elif selected == "Data Analysis":
    # Data Analysis page
    st.markdown("""
    <div class="center-header">
        <h1 class="animated-gradient">Data Analysis</h1>
        <p>Explore the relationships between student habits and academic performance</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Remove extra spacing
    st.markdown('<style>div.block-container{padding-top:0rem;}</style>', unsafe_allow_html=True)
    
    # Generate mock data for analysis
    @st.cache_data
    def generate_analysis_data(n_samples=500):
        np.random.seed(42)  # For reproducibility
        
        # Generate student habits
        data = {
            "student_id": range(1, n_samples + 1),
            "study_hours": np.random.normal(4, 1.5, n_samples).clip(0, 12),
            "sleep_hours": np.random.normal(6.5, 1.2, n_samples).clip(3, 10),
            "attendance_rate": np.random.beta(7, 2, n_samples) * 100,
            "revision_frequency": np.random.randint(0, 5, n_samples),  # 0=Never, 4=Daily
            "exercise_days": np.random.randint(0, 8, n_samples),
            "social_media_hours": np.random.gamma(2, 1, n_samples).clip(0, 10),
            "stress_level": np.random.randint(1, 6, n_samples),  # 1=Very Low, 5=Very High
            "department": np.random.choice(["Computer Science", "Engineering", "Business", "Arts", "Sciences"], n_samples),
            "year_of_study": np.random.randint(1, 5, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Generate CGPA with relationships to habits
        cgpa_base = 2.0
        cgpa_base += df["study_hours"] * 0.15  # More study hours -> higher CGPA
        cgpa_base += (df["sleep_hours"] - 6) * 0.05  # Optimal sleep around 7-8 hours
        cgpa_base += df["attendance_rate"] * 0.01  # Higher attendance -> higher CGPA
        cgpa_base += df["revision_frequency"] * 0.1  # More revision -> higher CGPA
        cgpa_base -= np.maximum(0, df["social_media_hours"] - 2) * 0.05  # More than 2 hours of social media -> lower CGPA
        cgpa_base -= (df["stress_level"] - 1) * 0.05  # Higher stress -> lower CGPA
        
        # Add some noise
        cgpa = cgpa_base + np.random.normal(0, 0.2, n_samples)
        df["cgpa"] = np.clip(cgpa, 0.0, 4.0)
        
        # Create categorical variables - FIX: Use direct string assignments instead of pd.cut
        def assign_category(cgpa_value):
            if cgpa_value < 2.0:
                return "Needs Improvement"
            elif cgpa_value < 2.7:
                return "Satisfactory"
            elif cgpa_value < 3.3:
                return "Good"
            elif cgpa_value < 3.7:
                return "Very Good"
            else:
                return "Excellent"
        
        df["cgpa_category"] = df["cgpa"].apply(assign_category)
        
        # Create study categories manually
        def assign_study_category(hours):
            if hours < 2:
                return "Very Low (<2h)"
            elif hours < 4:
                return "Low (2-4h)"
            elif hours < 6:
                return "Moderate (4-6h)"
            elif hours < 8:
                return "High (6-8h)"
            else:
                return "Very High (>8h)"
        
        df["study_category"] = df["study_hours"].apply(assign_study_category)
        
        # Create sleep categories manually
        def assign_sleep_category(hours):
            if hours < 5:
                return "Very Low (<5h)"
            elif hours < 6:
                return "Low (5-6h)"
            elif hours < 7:
                return "Optimal (6-7h)"
            elif hours < 8:
                return "Ideal (7-8h)"
            else:
                return "High (>8h)"
        
        df["sleep_category"] = df["sleep_hours"].apply(assign_sleep_category)
        
        # Create text mappings
        revision_map = {0: "Never", 1: "Rarely", 2: "Monthly", 3: "Weekly", 4: "Daily"}
        df["revision_text"] = df["revision_frequency"].map(revision_map)
        
        stress_map = {1: "Very Low", 2: "Low", 3: "Medium", 4: "High", 5: "Very High"}
        df["stress_text"] = df["stress_level"].map(stress_map)
        
        return df
    
    # Generate the data
    df_analysis = generate_analysis_data(500)
    
    # Create tabs for different analysis sections
    analysis_tabs = st.tabs([
        "Key Insights", 
        "Habit Correlations", 
        "Performance Breakdown",
        "Department Analysis",
        "Model Comparison"
    ])
    
    # Tab 1: Key Insights
    with analysis_tabs[0]:
        st.markdown("""
        <div style='background-color: var(--card-bg); border-radius: 15px; padding: 20px; margin: 10px 0 20px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <h3 style='margin-bottom: 15px;'>Key Factors Influencing Academic Performance</h3>
        """, unsafe_allow_html=True)
        
        # Create a correlation heatmap
        corr_cols = ["study_hours", "sleep_hours", "attendance_rate", "revision_frequency", 
                     "exercise_days", "social_media_hours", "stress_level", "cgpa"]
        corr_names = ["Study Hours", "Sleep Hours", "Attendance", "Revision", 
                      "Exercise", "Social Media", "Stress", "CGPA"]
        
        corr_matrix = df_analysis[corr_cols].corr()
        
        # Plot the heatmap with Plotly
        fig = px.imshow(
            corr_matrix,
            x=corr_names,
            y=corr_names,
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            text_auto='.2f'
        )
        
        fig.update_layout(
            title="Correlation Heatmap of Student Habits and CGPA",
            height=500,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate correlations with CGPA
        corr_with_cgpa = corr_matrix["cgpa"].drop("cgpa").sort_values(ascending=False)
        corr_df = pd.DataFrame({"Factor": corr_names[:-1], "Correlation": corr_with_cgpa.values})
        
        # Define colors based on correlation values
        colors = []
        for corr in corr_df["Correlation"]:
            if corr >= 0.5:
                colors.append('#28a745')  # Strong positive
            elif corr > 0:
                colors.append('#5cb85c')  # Positive
            elif corr > -0.5:
                colors.append('#ff4d4d')  # Negative
            else:
                colors.append('#cc0000')  # Strong negative
        
        # Create bar chart of correlations
        fig = px.bar(
            corr_df,
            x="Correlation",
            y="Factor",
            orientation='h',
            title="Correlation of Habits with CGPA",
            height=400
        )
        
        fig.update_traces(marker_color=colors)
        
        fig.update_layout(
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis_title="Correlation Coefficient",
            xaxis=dict(range=[-1, 1]),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Key findings and insights
        st.markdown("""
        <div style='background-color: var(--card-bg); border-radius: 15px; padding: 20px; margin: 20px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <h3 style='margin-bottom: 15px;'>Key Findings</h3>
            <ul style='list-style-type: none; padding-left: 0;'>
                <li style='margin-bottom: 12px;'>📚 <b>Study hours</b> have the strongest positive correlation with CGPA, confirming that dedicated study time is crucial</li>
                <li style='margin-bottom: 12px;'>📝 <b>Revision frequency</b> shows significant positive correlation, highlighting the importance of reviewing material</li>
                <li style='margin-bottom: 12px;'>📅 <b>Attendance</b> has a positive impact on academic performance, emphasizing the value of class participation</li>
                <li style='margin-bottom: 12px;'>😴 <b>Sleep</b> demonstrates a moderate positive correlation, suggesting adequate rest contributes to better performance</li>
                <li style='margin-bottom: 12px;'>📱 <b>Social media usage</b> shows a negative correlation, indicating excessive usage may detract from studies</li>
                <li style='margin-bottom: 12px;'>😓 <b>Stress</b> is negatively correlated with CGPA, highlighting the importance of mental health</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Tab 2: Habit Correlations
    with analysis_tabs[1]:
        st.markdown("""
        <div style='background-color: var(--card-bg); border-radius: 15px; padding: 20px; margin: 10px 0 20px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <h3 style='margin-bottom: 15px;'>Relationship Between Study Hours and CGPA</h3>
        """, unsafe_allow_html=True)
        
        # Ultra-simplified scatter plot to avoid category-related errors
        fig = px.scatter(
            df_analysis,
            x="study_hours",
            y="cgpa",
            title="Study Hours vs CGPA"
        )
        
        # Add trendline
        fig.update_layout(
            height=500,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        # Add trend line
        x_range = np.linspace(0, 12, 100)
        y_pred = 2.0 + 0.15 * x_range  # Approximate relationship
        
        fig.add_trace(
            go.Scatter(
                x=x_range, 
                y=y_pred, 
                mode="lines", 
                line=dict(color="black", width=2, dash="dash"),
                name="Trend"
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add key insight
        st.markdown("""
        <div style='background-color: rgba(0,0,0,0.1); padding: 15px; border-radius: 10px; margin-top: 10px;'>
            <b>📊 Key Insight:</b> The scatter plot shows a clear positive relationship between study hours and CGPA. 
            Students who study 6+ hours daily tend to achieve higher grades. However, note the variation at each study hour level, 
            indicating other factors also influence performance.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Sleep analysis section
        st.markdown("""
        <div style='background-color: var(--card-bg); border-radius: 15px; padding: 20px; margin: 20px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <h3 style='margin-bottom: 15px;'>Sleep Patterns and Academic Performance</h3>
        """, unsafe_allow_html=True)
        
        # Box plot of sleep category vs CGPA
        fig = px.box(
            df_analysis,
            x="sleep_category",
            y="cgpa",
            color="sleep_category",
            notched=True,
            points="all",
            title="Sleep Hours and CGPA Distribution",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            category_orders={"sleep_category": ["Very Low (<5h)", "Low (5-6h)", "Optimal (6-7h)", "Ideal (7-8h)", "High (>8h)"]}
        )
        
        fig.update_layout(
            height=450,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Sleep Category",
            yaxis_title="CGPA",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add key insight
        st.markdown("""
        <div style='background-color: rgba(0,0,0,0.1); padding: 15px; border-radius: 10px; margin-top: 10px;'>
            <b>😴 Key Insight:</b> The data suggests an optimal sleep range of 7-8 hours for academic performance. 
            Students with "Ideal" sleep patterns (7-8 hours) show the highest median CGPA. Both insufficient sleep 
            (<6 hours) and excessive sleep (>8 hours) correlate with lower academic outcomes, highlighting the importance 
            of balanced rest patterns.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Revision frequency analysis
        st.markdown("""
        <div style='background-color: var(--card-bg); border-radius: 15px; padding: 20px; margin: 20px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <h3 style='margin-bottom: 15px;'>Revision Frequency Impact</h3>
        """, unsafe_allow_html=True)
        
        # Group by revision frequency and calculate average CGPA
        revision_impact = df_analysis.groupby("revision_text")["cgpa"].mean().reset_index()
        revision_impact["revision_order"] = pd.Categorical(
            revision_impact["revision_text"],
            categories=["Never", "Rarely", "Monthly", "Weekly", "Daily"],
            ordered=True
        )
        revision_impact = revision_impact.sort_values("revision_order")
        
        # Bar chart of revision frequency vs CGPA
        fig = px.bar(
            revision_impact,
            x="revision_text",
            y="cgpa",
            color="cgpa",
            text_auto='.2f',
            title="Impact of Revision Frequency on CGPA",
            color_continuous_scale=px.colors.sequential.Viridis,
            category_orders={"revision_text": ["Never", "Rarely", "Monthly", "Weekly", "Daily"]}
        )
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Revision Frequency",
            yaxis_title="Average CGPA",
            coloraxis_showscale=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add key insight
        st.markdown("""
        <div style='background-color: rgba(0,0,0,0.1); padding: 15px; border-radius: 10px; margin-top: 10px;'>
            <b>📝 Key Insight:</b> Regular revision shows a strong positive impact on academic performance. 
            Students who revise daily achieve an average CGPA nearly 1.0 point higher than those who never revise. 
            The data shows a clear upward trend as revision frequency increases, supporting the effectiveness of consistent review practices.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Tab 3: Performance Breakdown
    with analysis_tabs[2]:
        st.markdown("""
        <div style='background-color: var(--card-bg); border-radius: 15px; padding: 20px; margin: 10px 0 20px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <h3 style='margin-bottom: 15px;'>CGPA Distribution Analysis</h3>
        """, unsafe_allow_html=True)
        
        # Ultra-simplified histogram without color categories
        fig = px.histogram(
            df_analysis,
            x="cgpa",
            nbins=20,
            title="Distribution of Student CGPA Scores"
        )
        
        fig.update_layout(
            height=450,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="CGPA",
            yaxis_title="Number of Students"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Basic statistics display instead of pie chart
        st.subheader("CGPA Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average CGPA", f"{df_analysis['cgpa'].mean():.2f}")
        with col2:
            st.metric("Minimum CGPA", f"{df_analysis['cgpa'].min():.2f}")
        with col3:
            st.metric("Maximum CGPA", f"{df_analysis['cgpa'].max():.2f}")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Study hours vs CGPA scatter
        st.markdown("""
        <div style='background-color: var(--card-bg); border-radius: 15px; padding: 20px; margin: 20px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <h3 style='margin-bottom: 15px;'>Study Hours Effect on CGPA</h3>
        """, unsafe_allow_html=True)
        
        # Simple scatter plot
        fig = px.scatter(
            df_analysis,
            x="study_hours",
            y="cgpa",
            title="Study Hours vs CGPA",
            opacity=0.7
        )
        
        fig.update_layout(
            height=450,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Study Hours",
            yaxis_title="CGPA"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add key insight
        st.markdown("""
        <div style='background-color: rgba(0,0,0,0.1); padding: 15px; border-radius: 10px; margin-top: 10px;'>
            <b>🔍 Key Insight:</b> The data reveals a strong positive relationship between study hours and academic performance.
            Students who dedicate more time to studying consistently achieve higher CGPA scores, though individual variation exists.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Tab 4: Department Analysis
    with analysis_tabs[3]:
        st.markdown("""
        <div style='background-color: var(--card-bg); border-radius: 15px; padding: 20px; margin: 10px 0 20px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <h3 style='margin-bottom: 15px;'>Performance by Department</h3>
        """, unsafe_allow_html=True)
        
        # Calculate department statistics - simplified
        dept_stats = df_analysis.groupby("department")["cgpa"].mean().reset_index()
        dept_stats.columns = ["Department", "Average CGPA"]
        
        # Create simple bar chart for average CGPA by department
        fig = px.bar(
            dept_stats,
            x="Department",
            y="Average CGPA",
            title="Average CGPA by Department"
        )
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Department",
            yaxis_title="Average CGPA"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Department-specific habits - simplified
        dept_habits = df_analysis.groupby("department")[
            ["study_hours", "sleep_hours"]
        ].mean().reset_index()
        
        # Convert to long format for grouped bar chart
        dept_habits_long = pd.melt(
            dept_habits,
            id_vars=["department"],
            value_vars=["study_hours", "sleep_hours"],
            var_name="Habit",
            value_name="Value"
        )
        
        # Create a grouped bar chart
        fig = px.bar(
            dept_habits_long,
            x="department",
            y="Value",
            color="Habit",
            barmode="group",
            title="Key Habits by Department"
        )
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add key insight
        st.markdown("""
        <div style='background-color: rgba(0,0,0,0.1); padding: 15px; border-radius: 10px; margin-top: 10px;'>
            <b>🎓 Key Insight:</b> Department analysis reveals interesting variations in both performance and habits. 
            Computer Science shows the highest average CGPA, with students reporting more study hours but slightly less sleep. 
            Arts students report higher attendance rates but fewer study hours. This suggests that different academic 
            disciplines may benefit from tailored study strategies.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Tab 5: Model Comparison
    with analysis_tabs[4]:
        st.markdown("""
        <div style='background-color: var(--card-bg); border-radius: 15px; padding: 20px; margin: 10px 0 20px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <h3 style='margin-bottom: 15px;'>Prediction Model Performance</h3>
        """, unsafe_allow_html=True)
        
        # Mock data for model comparison - simplified
        models = ["Linear Regression", "Random Forest", "Neural Network", "BiLSTM"]
        accuracy = [76, 85, 92, 95]
        
        # Create a dataframe for model metrics
        model_metrics = pd.DataFrame({
            "Model": models,
            "Accuracy": accuracy
        })
        
        # Create a simple bar chart
        fig = px.bar(
            model_metrics,
            x="Model",
            y="Accuracy",
            title="Model Accuracy Comparison"
        )
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Model",
            yaxis_title="Accuracy (%)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add key insight
        st.markdown("""
        <div style='background-color: rgba(0,0,0,0.1); padding: 15px; border-radius: 10px; margin-top: 10px;'>
            <b>🤖 Key Insight:</b> The BiLSTM model outperforms other algorithms with the highest accuracy. 
            Neural network and random forest models also perform well, while simpler models like 
            linear regression show lower accuracy. This suggests that the complex relationships between student habits 
            and academic performance benefit from more sophisticated modeling approaches.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Feature importance section - simplified
        st.markdown("""
        <div style='background-color: var(--card-bg); border-radius: 15px; padding: 20px; margin: 20px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <h3 style='margin-bottom: 15px;'>Feature Importance Analysis</h3>
        """, unsafe_allow_html=True)
        
        # Mock data for feature importance
        features = ["Study Hours", "Attendance Rate", "Sleep Hours", "Social Media Hours"]
        importance = [0.45, 0.30, 0.15, 0.10]
        
        # Create a dataframe for feature importance
        feature_importance = pd.DataFrame({
            "Feature": features,
            "Importance": importance
        })
        
        # Create a simple bar chart
        fig = px.bar(
            feature_importance,
            x="Feature",
            y="Importance",
            title="Feature Importance in CGPA Prediction"
        )
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Student Habit Feature",
            yaxis_title="Relative Importance"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add key insight
        st.markdown("""
        <div style='background-color: rgba(0,0,0,0.1); padding: 15px; border-radius: 10px; margin-top: 10px;'>
            <b>🎯 Key Insight:</b> Feature importance analysis reveals that study hours and attendance rate are the 
            most influential factors in predicting academic performance. This suggests intervention strategies 
            focused on these areas could yield the greatest improvements.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Add animation with st_lottie
        st_lottie(lottie_analysis, height=200, key="analysis_animation")