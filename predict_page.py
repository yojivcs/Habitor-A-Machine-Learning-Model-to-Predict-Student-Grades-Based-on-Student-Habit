import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import requests
import time

# Load animation
@st.cache_data
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_prediction = load_lottie_url("https://assets3.lottiefiles.com/packages/lf20_ydo1amjm.json")
lottie_results = load_lottie_url("https://assets6.lottiefiles.com/packages/lf20_jOH2nv.json")

def show_predict_page(model):
    # Two columns layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Tell us about your habits")
        st.write("Fill in the form below to get a personalized prediction of your academic performance.")
        
        # Create form for user input
        with st.form("habit_form"):
            # Create tabs for different categories of habits
            tabs = st.tabs(["Study Habits", "Sleep & Health", "Class Engagement", "Time Management"])
            
            # Tab 1: Study Habits
            with tabs[0]:
                st.subheader("Study Habits")
                
                study_hours = st.slider("Average daily study hours (outside class)", 0, 12, 2)
                
                study_regularity = st.radio(
                    "How regular is your study schedule?",
                    ["Very irregular", "Somewhat irregular", "Somewhat regular", "Very regular"]
                )
                
                revision_frequency = st.select_slider(
                    "How often do you revise previously learned material?",
                    options=["Never", "Rarely", "Monthly", "Weekly", "Daily"]
                )
                
                assignment_time = st.slider(
                    "How many days before the deadline do you typically complete assignments?",
                    0, 14, 1
                )
                
                study_environment = st.multiselect(
                    "Where do you typically study? (Select all that apply)",
                    ["Library", "Dorm/Home", "Classroom", "Coffee shop", "Study group", "Outdoors"],
                    ["Dorm/Home"]
                )
            
            # Tab 2: Sleep & Health
            with tabs[1]:
                st.subheader("Sleep & Health")
                
                sleep_hours = st.slider("Average daily sleep hours", 3, 12, 7)
                
                sleep_schedule = st.radio(
                    "How consistent is your sleep schedule?",
                    ["Very inconsistent", "Somewhat inconsistent", "Somewhat consistent", "Very consistent"]
                )
                
                meal_regularity = st.radio(
                    "How regular are your meal times?",
                    ["Very irregular", "Somewhat irregular", "Somewhat regular", "Very regular"]
                )
                
                exercise_frequency = st.select_slider(
                    "How many days per week do you exercise?",
                    options=["0", "1-2", "3-4", "5-6", "7"]
                )
                
                caffeine_consumption = st.slider("How many caffeinated drinks do you consume daily?", 0, 10, 2)
            
            # Tab 3: Class Engagement
            with tabs[2]:
                st.subheader("Class Engagement")
                
                attendance_rate = st.slider("Class attendance rate (%)", 0, 100, 80)
                
                participation = st.radio(
                    "How actively do you participate in class discussions?",
                    ["Never", "Rarely", "Sometimes", "Often", "Always"]
                )
                
                note_taking = st.radio(
                    "How do you take notes during class?",
                    ["I don't take notes", "Minimal notes", "Key points only", "Comprehensive notes", "Detailed notes with personal insights"]
                )
                
                question_asking = st.radio(
                    "How often do you ask questions in class?",
                    ["Never", "Rarely", "Sometimes", "Often", "Always"]
                )
                
                attention_span = st.slider("Average attention span during lectures (minutes)", 5, 120, 45)
            
            # Tab 4: Time Management
            with tabs[3]:
                st.subheader("Time Management")
                
                planning_tools = st.multiselect(
                    "Which planning tools do you use? (Select all that apply)",
                    ["None", "Physical planner", "Digital calendar", "To-do lists", "Time blocking apps", "Pomodoro technique"],
                    ["None"]
                )
                
                procrastination = st.select_slider(
                    "How often do you procrastinate on academic tasks?",
                    options=["Never", "Rarely", "Sometimes", "Often", "Always"]
                )
                
                deadline_adherence = st.radio(
                    "How well do you adhere to self-imposed deadlines?",
                    ["Always miss them", "Usually miss them", "Sometimes meet them", "Usually meet them", "Always meet them"]
                )
                
                social_media_hours = st.slider("Hours spent on social media daily", 0, 12, 2)
                
                extracurricular_hours = st.slider("Hours spent on extracurricular activities weekly", 0, 30, 5)
            
            # Submit button
            submitted = st.form_submit_button("Predict My CGPA")
        
        # Process form submission
        if submitted:
            # Collect all inputs into a feature vector
            # In a real application, you would need to preprocess these inputs to match your model's expected format
            features = {
                "study_hours": study_hours,
                "study_regularity": ["Very irregular", "Somewhat irregular", "Somewhat regular", "Very regular"].index(study_regularity),
                "revision_frequency": ["Never", "Rarely", "Monthly", "Weekly", "Daily"].index(revision_frequency),
                "assignment_time": assignment_time,
                "sleep_hours": sleep_hours,
                "sleep_consistency": ["Very inconsistent", "Somewhat inconsistent", "Somewhat consistent", "Very consistent"].index(sleep_schedule),
                "meal_regularity": ["Very irregular", "Somewhat irregular", "Somewhat regular", "Very regular"].index(meal_regularity),
                "exercise_frequency": ["0", "1-2", "3-4", "5-6", "7"].index(exercise_frequency),
                "caffeine_consumption": caffeine_consumption,
                "attendance_rate": attendance_rate / 100,
                "participation": ["Never", "Rarely", "Sometimes", "Often", "Always"].index(participation),
                "note_taking": ["I don't take notes", "Minimal notes", "Key points only", "Comprehensive notes", "Detailed notes with personal insights"].index(note_taking),
                "question_asking": ["Never", "Rarely", "Sometimes", "Often", "Always"].index(question_asking),
                "attention_span": attention_span / 60,  # Convert to hours for consistency
                "planning_tools_count": len(planning_tools) if "None" not in planning_tools else 0,
                "procrastination": ["Never", "Rarely", "Sometimes", "Often", "Always"].index(procrastination),
                "deadline_adherence": ["Always miss them", "Usually miss them", "Sometimes meet them", "Usually meet them", "Always meet them"].index(deadline_adherence),
                "social_media_hours": social_media_hours,
                "extracurricular_hours": extracurricular_hours
            }
            
            # Convert to DataFrame (needed for most ML models)
            X = pd.DataFrame([features])
            
            # Show loading animation
            with st.spinner("Analyzing your habits..."):
                # Simulate processing time
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Make prediction
                predicted_cgpa = float(model.predict(X)[0])
                
                # Determine CGPA category and associated recommendations
                if predicted_cgpa >= 3.7:
                    category = "Excellent"
                    performance_level = "exceptional"
                    recommendations = [
                        "Continue your current habits, they're working well",
                        "Consider peer mentoring to help other students",
                        "Explore advanced coursework or research opportunities",
                        "Balance your excellence with adequate rest and recreation"
                    ]
                elif predicted_cgpa >= 3.3:
                    category = "Very Good"
                    performance_level = "very good"
                    recommendations = [
                        f"Increase study hours from {study_hours} to {min(study_hours + 1, 12)} hours daily",
                        "Improve sleep consistency for better cognitive performance",
                        "Enhance class participation to solidify understanding",
                        "Use more structured planning tools for assignments"
                    ]
                elif predicted_cgpa >= 3.0:
                    category = "Good"
                    performance_level = "good"
                    recommendations = [
                        f"Increase study regularity - aim for daily consistent sessions",
                        "Improve note-taking techniques for better retention",
                        "Reduce social media time from {social_media_hours} to {max(social_media_hours - 1, 0)} hours",
                        "Start assignments earlier - aim for at least {min(assignment_time + 3, 14)} days before deadline"
                    ]
                elif predicted_cgpa >= 2.7:
                    category = "Satisfactory"
                    performance_level = "satisfactory"
                    recommendations = [
                        f"Significantly increase study hours from {study_hours} to {min(study_hours + 2, 12)} hours daily",
                        "Improve attendance rate to at least 90%",
                        "Establish a consistent sleep schedule of 7-8 hours nightly",
                        "Utilize academic resources like tutoring and study groups"
                    ]
                else:
                    category = "Needs Improvement"
                    performance_level = "in need of improvement"
                    recommendations = [
                        "Create a structured daily study schedule with at least 4 hours",
                        "Attend all classes and actively participate",
                        "Seek academic counseling and tutoring services",
                        "Reduce distractions and procrastination with time management techniques",
                        "Establish healthier sleep and nutrition habits"
                    ]
            
            # Display results
            st.success("Analysis complete!")
            
            # Create expandable results section
            with st.expander("View Your Prediction Results", expanded=True):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"### Predicted CGPA: {predicted_cgpa:.2f}")
                    st.markdown(f"#### Performance Category: {category}")
                    st.write(f"Based on your habits, your academic performance is predicted to be {performance_level}.")
                    
                    # Create gauge chart for CGPA
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = predicted_cgpa,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Predicted CGPA", 'font': {'size': 24}},
                        gauge = {
                            'axis': {'range': [0, 4], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': "royalblue"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 2.0], 'color': 'firebrick'},
                                {'range': [2.0, 2.7], 'color': 'darkorange'},
                                {'range': [2.7, 3.3], 'color': 'gold'},
                                {'range': [3.3, 3.7], 'color': 'yellowgreen'},
                                {'range': [3.7, 4.0], 'color': 'forestgreen'}
                            ],
                        }
                    ))
                    
                    fig.update_layout(
                        height=300,
                        margin=dict(l=20, r=20, t=50, b=20),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st_lottie(lottie_results, height=200, key="results_anim")
                
                # Personalized recommendations
                st.markdown("### Personalized Recommendations")
                st.write("Based on your habits, here are some recommendations to improve your academic performance:")
                
                for i, recommendation in enumerate(recommendations):
                    st.markdown(f"**{i+1}.** {recommendation}")
                
                # Habit impact analysis
                st.markdown("### Habit Impact Analysis")
                
                # Calculate feature importance (in a real app, this would come from the model)
                # Here we're simulating feature importance
                important_features = {
                    "Study Hours": study_hours * 0.8,
                    "Attendance Rate": attendance_rate * 0.6 / 100,
                    "Sleep Consistency": ["Very inconsistent", "Somewhat inconsistent", "Somewhat consistent", "Very consistent"].index(sleep_schedule) * 0.5 / 3,
                    "Procrastination Level": (4 - ["Never", "Rarely", "Sometimes", "Often", "Always"].index(procrastination)) * 0.4 / 4,
                    "Time Management": ["Always miss them", "Usually miss them", "Sometimes meet them", "Usually meet them", "Always meet them"].index(deadline_adherence) * 0.3 / 4
                }
                
                # Create horizontal bar chart for feature importance
                df_importance = pd.DataFrame({
                    'Habit': list(important_features.keys()),
                    'Impact': list(important_features.values())
                })
                df_importance = df_importance.sort_values('Impact', ascending=False)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=df_importance['Habit'],
                    x=df_importance['Impact'],
                    orientation='h',
                    marker_color=['royalblue', 'cornflowerblue', 'lightsteelblue', 'lavender', 'thistle']
                ))
                
                fig.update_layout(
                    title="Habits with Highest Impact on Your Performance",
                    xaxis_title="Relative Impact",
                    height=300,
                    margin=dict(l=20, r=20, t=50, b=20),
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st_lottie(lottie_prediction, height=300, key="predict_anim")
        
        # Info box
        st.info("""
        ### How We Predict
        
        Our AI model analyzes the complex relationships between your daily habits and academic performance using Bidirectional LSTM neural networks.
        
        The model has been trained on data from hundreds of students, identifying patterns that most significantly influence academic success.
        
        The prediction is based on:
        - Study patterns
        - Sleep habits
        - Class engagement
        - Time management
        - Health factors
        
        **Note:** This prediction is an estimate based on statistical patterns and may vary based on individual circumstances.
        """)
        
        # Tips box
        st.markdown("""
        ### Quick Tips for Success
        
        🕒 **Consistency over quantity** - Regular short study sessions are more effective than cramming
        
        😴 **Prioritize sleep** - 7-8 hours of quality sleep improves memory and learning
        
        ✋ **Active engagement** - Participate in class and ask questions
        
        📱 **Manage distractions** - Set aside dedicated distraction-free study time
        
        📝 **Effective note-taking** - Develop a system that works for your learning style
        """) 