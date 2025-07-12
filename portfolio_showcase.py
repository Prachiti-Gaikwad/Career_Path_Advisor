import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import io
from PIL import Image
import hashlib
import sqlite3
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any, Optional
import requests
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Data Science Portfolio",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for portfolio styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
        letter-spacing: -2px;
    }
    
    .portfolio-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        margin: 1.5rem 0;
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .portfolio-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
    }
    
    .skill-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.25rem;
        display: inline-block;
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
    }
    
    .project-header {
        color: #2c3e50;
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        margin: 1rem 0;
    }
    
    .contact-card {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        margin: 2rem 0;
    }
    
    .demo-container {
        background: #ffffff;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .achievement-badge {
        background: linear-gradient(135deg, #ffd89b 0%, #19547b 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .github-button {
        background: #24292e;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        text-decoration: none;
        font-weight: 600;
        display: inline-block;
        margin: 0.5rem;
        transition: all 0.3s ease;
    }
    
    .github-button:hover {
        background: #444d56;
        transform: translateY(-2px);
    }
    
    .login-container {
        max-width: 400px;
        margin: 0 auto;
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Data classes
@dataclass
class Project:
    title: str
    description: str
    technologies: List[str]
    github_url: str
    demo_type: str
    image_url: str = None
    metrics: Dict[str, Any] = None

@dataclass
class Experience:
    company: str
    role: str
    duration: str
    description: str
    achievements: List[str]

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'portfolio_data' not in st.session_state:
    st.session_state.portfolio_data = load_portfolio_data()

def load_portfolio_data():
    """Load portfolio data"""
    return {
        'projects': [
            Project(
                title="AI-Powered Sales Forecasting",
                description="Advanced time series forecasting using LSTM networks and ensemble methods to predict sales with 94% accuracy.",
                technologies=["Python", "TensorFlow", "LSTM", "Prophet", "Plotly"],
                github_url="https://github.com/username/sales-forecasting",
                demo_type="time_series",
                metrics={"accuracy": 94.2, "mae": 156.3, "rmse": 203.1}
            ),
            Project(
                title="Customer Churn Prediction",
                description="Machine learning pipeline for predicting customer churn using feature engineering and ensemble methods.",
                technologies=["Python", "Scikit-learn", "XGBoost", "Pandas", "Streamlit"],
                github_url="https://github.com/username/churn-prediction",
                demo_type="classification",
                metrics={"accuracy": 89.7, "precision": 0.87, "recall": 0.92}
            ),
            Project(
                title="Real-time Sentiment Analysis",
                description="NLP pipeline for real-time sentiment analysis of social media data using transformers and streaming architecture.",
                technologies=["Python", "BERT", "Apache Kafka", "Docker", "FastAPI"],
                github_url="https://github.com/username/sentiment-analysis",
                demo_type="nlp",
                metrics={"f1_score": 0.91, "throughput": "10k tweets/min"}
            ),
            Project(
                title="Computer Vision Pipeline",
                description="End-to-end computer vision solution for object detection and classification using CNN architectures.",
                technologies=["Python", "PyTorch", "OpenCV", "YOLO", "Flask"],
                github_url="https://github.com/username/cv-pipeline",
                demo_type="computer_vision",
                metrics={"mAP": 0.85, "inference_time": "23ms"}
            )
        ],
        'experience': [
            Experience(
                company="TechCorp Inc.",
                role="Senior Data Scientist",
                duration="2022 - Present",
                description="Leading ML initiatives and building production-ready models for business growth.",
                achievements=[
                    "Improved model accuracy by 25% through advanced feature engineering",
                    "Deployed 8 ML models to production serving 1M+ users",
                    "Led a team of 4 data scientists on enterprise projects"
                ]
            ),
            Experience(
                company="DataStart Solutions",
                role="Data Scientist",
                duration="2020 - 2022",
                description="Developed predictive models and analytics solutions for various clients.",
                achievements=[
                    "Built customer segmentation model increasing revenue by 18%",
                    "Automated reporting pipeline saving 20 hours/week",
                    "Implemented A/B testing framework for product optimization"
                ]
            )
        ],
        'skills': {
            'Programming': ['Python', 'R', 'SQL', 'JavaScript', 'Scala'],
            'Machine Learning': ['Scikit-learn', 'XGBoost', 'TensorFlow', 'PyTorch', 'Keras'],
            'Data Engineering': ['Apache Spark', 'Kafka', 'Airflow', 'Docker', 'Kubernetes'],
            'Visualization': ['Plotly', 'Streamlit', 'Tableau', 'D3.js', 'Matplotlib'],
            'Cloud Platforms': ['AWS', 'GCP', 'Azure', 'Databricks', 'Snowflake']
        },
        'certifications': [
            "AWS Certified Machine Learning - Specialty",
            "Google Cloud Professional Data Engineer",
            "TensorFlow Developer Certificate",
            "Databricks Certified Data Scientist"
        ]
    }

def authenticate_user(username: str, password: str) -> bool:
    """Simple authentication (in production, use proper authentication)"""
    # Demo credentials
    demo_users = {
        "demo": "password123",
        "admin": "admin123",
        "guest": "guest123"
    }
    return demo_users.get(username) == password

def login_page():
    """Login page interface"""
    st.markdown('<h1 class="main-header">🔐 Portfolio Access</h1>', unsafe_allow_html=True)
    
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown('<div class="login-container">', unsafe_allow_html=True)
            
            st.markdown("### Welcome to My Data Science Portfolio")
            st.markdown("Please login to access the full portfolio experience.")
            
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                if st.button("🚀 Login", type="primary", use_container_width=True):
                    if authenticate_user(username, password):
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials!")
            
            with col_b:
                if st.button("👨‍💻 Demo Access", use_container_width=True):
                    st.session_state.authenticated = True
                    st.session_state.username = "demo_user"
                    st.rerun()
            
            st.markdown("---")
            st.markdown("**Demo Credentials:**")
            st.code("Username: demo\nPassword: password123")
            
            st.markdown('</div>', unsafe_allow_html=True)

def main_portfolio():
    """Main portfolio interface"""
    # Header
    st.markdown('<h1 class="main-header">🚀 Data Science Portfolio</h1>', unsafe_allow_html=True)
    st.markdown(f"### Welcome back, **{st.session_state.username}**! 👋")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Navigation")
        
        if st.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()
        
        st.markdown("---")
        
        # Quick stats
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("#### 📊 Quick Stats")
        st.metric("Projects Completed", "15+")
        st.metric("Years of Experience", "4+")
        st.metric("Models in Production", "8")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Contact info
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("#### 📞 Contact")
        st.markdown("📧 john.doe@email.com")
        st.markdown("💼 [LinkedIn](https://linkedin.com/in/johndoe)")
        st.markdown("🐙 [GitHub](https://github.com/johndoe)")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Skills overview
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("#### 🛠️ Top Skills")
        for skill in ["Python", "Machine Learning", "Deep Learning", "Data Engineering", "MLOps"]:
            st.markdown(f'<span class="skill-badge">{skill}</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏠 Overview", 
        "🚀 Projects", 
        "💼 Experience", 
        "🎯 Skills", 
        "🎮 Interactive Demos",
        "📊 Analytics"
    ])
    
    with tab1:
        overview_section()
    
    with tab2:
        projects_section()
    
    with tab3:
        experience_section()
    
    with tab4:
        skills_section()
    
    with tab5:
        interactive_demos_section()
    
    with tab6:
        analytics_section()

def overview_section():
    """Portfolio overview section"""
    st.header("👨‍💻 About Me")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Senior Data Scientist & ML Engineer
        
        Passionate data scientist with 4+ years of experience in building end-to-end machine learning solutions. 
        Specialized in developing production-ready models that drive business value and solve complex problems.
        
        **What I Do:**
        - 🤖 Design and deploy machine learning models at scale
        - 📊 Build data pipelines and analytics platforms
        - 🚀 Lead cross-functional teams on data-driven initiatives
        - 🔬 Research and implement cutting-edge ML techniques
        
        **My Approach:**
        I believe in the power of data to transform businesses. My methodology combines rigorous statistical analysis 
        with practical engineering solutions to deliver measurable impact.
        """)
    
    with col2:
        # Profile metrics
        st.markdown('<div class="metric-card"><h3>15+</h3><p>Projects Completed</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-card"><h3>8</h3><p>Models in Production</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-card"><h3>94%</h3><p>Average Model Accuracy</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-card"><h3>1M+</h3><p>Users Served</p></div>', unsafe_allow_html=True)
    
    # Recent achievements
    st.markdown("---")
    st.subheader("🏆 Recent Achievements")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="achievement-badge"><strong>Best ML Model</strong><br>Internal Hackathon 2023</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="achievement-badge"><strong>AWS Certified</strong><br>ML Specialty</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="achievement-badge"><strong>Team Lead</strong><br>ML Infrastructure</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="achievement-badge"><strong>Published Paper</strong><br>ICML 2023</div>', unsafe_allow_html=True)

def projects_section():
    """Projects showcase section"""
    st.header("🚀 Featured Projects")
    
    for i, project in enumerate(st.session_state.portfolio_data['projects']):
        st.markdown('<div class="portfolio-card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f'<div class="project-header">{project.title}</div>', unsafe_allow_html=True)
            st.markdown(project.description)
            
            # Technologies
            st.markdown("**Technologies Used:**")
            tech_badges = "".join([f'<span class="skill-badge">{tech}</span>' for tech in project.technologies])
            st.markdown(tech_badges, unsafe_allow_html=True)
            
            # GitHub link
            st.markdown(f'<a href="{project.github_url}" class="github-button" target="_blank">📁 View on GitHub</a>', unsafe_allow_html=True)
        
        with col2:
            if project.metrics:
                st.markdown("**Key Metrics:**")
                for metric, value in project.metrics.items():
                    if isinstance(value, float):
                        st.metric(metric.title(), f"{value:.1f}")
                    else:
                        st.metric(metric.title(), str(value))
            
            # Demo button
            if st.button(f"🎮 Try Demo", key=f"demo_{i}"):
                st.session_state.selected_demo = project.demo_type
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")

def experience_section():
    """Experience section"""
    st.header("💼 Professional Experience")
    
    for exp in st.session_state.portfolio_data['experience']:
        st.markdown('<div class="portfolio-card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"### {exp.role}")
            st.markdown(f"**{exp.company}**")
            st.markdown(exp.description)
            
            st.markdown("**Key Achievements:**")
            for achievement in exp.achievements:
                st.markdown(f"• {achievement}")
        
        with col2:
            st.markdown(f"**Duration**")
            st.markdown(exp.duration)
        
        st.markdown('</div>', unsafe_allow_html=True)

def skills_section():
    """Skills section"""
    st.header("🎯 Technical Skills")
    
    skills_data = st.session_state.portfolio_data['skills']
    
    for category, skills_list in skills_data.items():
        st.subheader(f"🔧 {category}")
        
        # Create skill level visualization
        skill_levels = np.random.uniform(70, 95, len(skills_list))  # Mock skill levels
        
        skill_df = pd.DataFrame({
            'Skill': skills_list,
            'Proficiency': skill_levels
        })
        
        fig = px.bar(
            skill_df, 
            x='Proficiency', 
            y='Skill',
            orientation='h',
            title=f"{category} Proficiency",
            color='Proficiency',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Certifications
    st.subheader("📜 Certifications")
    
    for cert in st.session_state.portfolio_data['certifications']:
        st.markdown(f'<span class="skill-badge">✅ {cert}</span>', unsafe_allow_html=True)

def interactive_demos_section():
    """Interactive demos section"""
    st.header("🎮 Interactive Demos")
    
    demo_type = st.selectbox(
        "Choose a demo to explore:",
        ["Time Series Forecasting", "Classification Model", "NLP Sentiment Analysis", "Computer Vision"]
    )
    
    if demo_type == "Time Series Forecasting":
        time_series_demo()
    elif demo_type == "Classification Model":
        classification_demo()
    elif demo_type == "NLP Sentiment Analysis":
        nlp_demo()
    elif demo_type == "Computer Vision":
        cv_demo()

def time_series_demo():
    """Time series forecasting demo"""
    st.markdown('<div class="demo-container">', unsafe_allow_html=True)
    st.subheader("📈 Sales Forecasting Demo")
    
    # Generate sample time series data
    dates = pd.date_range('2020-01-01', periods=365*3, freq='D')
    trend = np.linspace(1000, 1500, len(dates))
    seasonal = 200 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
    noise = np.random.normal(0, 50, len(dates))
    
    sales_data = pd.DataFrame({
        'date': dates,
        'sales': trend + seasonal + noise
    })
    
    # Interactive parameters
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_days = st.slider("Forecast Days", 30, 180, 90)
        seasonality_strength = st.slider("Seasonality Strength", 0.5, 2.0, 1.0)
    
    with col2:
        trend_strength = st.slider("Trend Strength", 0.5, 2.0, 1.0)
        noise_level = st.slider("Noise Level", 0.1, 1.0, 0.3)
    
    # Generate forecast
    future_dates = pd.date_range(sales_data['date'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='D')
    future_trend = np.linspace(sales_data['sales'].iloc[-1], sales_data['sales'].iloc[-1] * trend_strength, forecast_days)
    future_seasonal = seasonality_strength * 200 * np.sin(2 * np.pi * np.arange(forecast_days) / 365)
    future_sales = future_trend + future_seasonal
    
    # Plot
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=sales_data['date'],
        y=sales_data['sales'],
        mode='lines',
        name='Historical Sales',
        line=dict(color='blue')
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_sales,
        mode='lines',
        name='Forecast',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title="Sales Forecasting with LSTM",
        xaxis_title="Date",
        yaxis_title="Sales ($)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Forecast Accuracy", "94.2%")
    with col2:
        st.metric("MAE", "$156.30")
    with col3:
        st.metric("RMSE", "$203.10")
    
    st.markdown('</div>', unsafe_allow_html=True)

def classification_demo():
    """Classification model demo"""
    st.markdown('<div class="demo-container">', unsafe_allow_html=True)
    st.subheader("🎯 Customer Churn Prediction Demo")
    
    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    feature_names = [f'Feature_{i+1}' for i in range(10)]
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Interactive prediction
    st.subheader("🔮 Make a Prediction")
    
    col1, col2 = st.columns(2)
    
    input_features = []
    for i in range(5):
        with col1 if i < 3 else col2:
            feature_val = st.slider(f"{feature_names[i]}", float(X[:, i].min()), float(X[:, i].max()), float(X[:, i].mean()))
            input_features.append(feature_val)
    
    # Add remaining features with default values
    input_features.extend([0.0] * (10 - len(input_features)))
    
    if st.button("🎯 Predict Churn"):
        prediction = model.predict([input_features])[0]
        probability = model.predict_proba([input_features])[0]
        
        if prediction == 1:
            st.error(f"⚠️ High Churn Risk (Probability: {probability[1]:.2%})")
        else:
            st.success(f"✅ Low Churn Risk (Probability: {probability[0]:.2%})")
    
    # Model performance
    st.subheader("📊 Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, title="Confusion Matrix", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Feature importance
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title="Feature Importance")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def nlp_demo():
    """NLP sentiment analysis demo"""
    st.markdown('<div class="demo-container">', unsafe_allow_html=True)
    st.subheader("💬 Sentiment Analysis Demo")
    
    # Sample texts for demo
    sample_texts = [
        "I absolutely love this product! It's amazing!",
        "This is terrible, worst experience ever.",
        "It's okay, nothing special but does the job.",
        "Outstanding quality and excellent customer service!",
        "Could be better, but not bad overall."
    ]
    
    # Text input
    selected_text = st.selectbox("Choose a sample text or enter your own:", ["Custom"] + sample_texts)
    
    if selected_text == "Custom":
        user_text = st.text_area("Enter your text for sentiment analysis:", height=100)
    else:
        user_text = selected_text
        st.text_area("Text to analyze:", value=user_text, height=100, disabled=True)
    
    if st.button("🔍 Analyze Sentiment") and user_text:
        # Simulate sentiment analysis (in production, use actual model)
        sentiment_score = np.random.uniform(-1, 1)
        
        if sentiment_score > 0.1:
            sentiment = "Positive 😊"
            color = "green"
        elif sentiment_score < -0.1:
            sentiment = "Negative 😞"
            color = "red"
        else:
            sentiment = "Neutral 😐"
            color = "gray"
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Sentiment", sentiment)
        with col2:
            st.metric("Confidence", f"{abs(sentiment_score)*100:.1f}%")
        with col3:
            st.metric("Score", f"{sentiment_score:.3f}")
        
        # Sentiment gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = sentiment_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Sentiment Score"},
            gauge = {
                'axis': {'range': [-1, 1]},
                'bar': {'color': color},
                'steps': [
                    {'range': [-1, -0.1], 'color': "lightcoral"},
                    {'range': [-0.1, 0.1], 'color': "lightgray"},
                    {'range': [0.1, 1], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.9
                }
            }
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def cv_demo():
    """Computer vision demo"""
    st.markdown('<div class="demo-container">', unsafe_allow_html=True)
    st.subheader("👁️ Object Detection Demo")
    
    st.markdown("""
    **Computer Vision Pipeline Features:**
    - Real-time object detection using YOLO
    - Image classification with CNN
    - Custom model training capabilities
    - Edge deployment optimization
    """)
    
    # File uploader for images
    uploaded_file = st.file_uploader("Upload an image for analysis", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Simulate object detection results
        if st.button("🔍 Detect Objects"):
            # Mock detection results
            detections = [
                {"object": "person", "confidence": 0.95, "bbox": [100, 50, 200, 300]},
                {"object": "car", "confidence": 0.87, "bbox": [300, 150, 500, 400]},
                {"object": "bicycle", "confidence": 0.72, "bbox": [50, 200, 150, 350]}
            ]
            
            st.subheader("🎯 Detection Results")
            
            # Results table
            results_df = pd.DataFrame(detections)
            st.dataframe(results_df[['object', 'confidence']], use_container_width=True)
            
            # Confidence chart
            fig = px.bar(
                results_df, 
                x='object', 
                y='confidence',
                title="Detection Confidence by Object",
                color='confidence',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Model metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("mAP Score", "0.85")
    with col2:
        st.metric("Inference Time", "23ms")
    with col3:
        st.metric("Model Size", "45MB")
    
    st.markdown('</div>', unsafe_allow_html=True)

def analytics_section():
    """Portfolio analytics section"""
    st.header("📊 Portfolio Analytics")
    
    # Generate mock analytics data
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    views_data = pd.DataFrame({
        'date': dates,
        'page_views': np.random.poisson(50, len(dates)),
        'unique_visitors': np.random.poisson(30, len(dates)),
        'project_demos': np.random.poisson(10, len(dates))
    })
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Views", f"{views_data['page_views'].sum():,}")
    with col2:
        st.metric("Unique Visitors", f"{views_data['unique_visitors'].sum():,}")
    with col3:
        st.metric("Demo Interactions", f"{views_data['project_demos'].sum():,}")
    with col4:
        avg_engagement = (views_data['project_demos'].sum() / views_data['page_views'].sum()) * 100
        st.metric("Engagement Rate", f"{avg_engagement:.1f}%")
    
    # Time series charts
    st.subheader("📈 Traffic Trends")
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Page Views Over Time', 'Project Demo Interactions'),
        vertical_spacing=0.1
    )
    
    fig.add_trace(
        go.Scatter(x=views_data['date'], y=views_data['page_views'], name='Page Views'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=views_data['date'], y=views_data['project_demos'], name='Project Demos', line=dict(color='orange')),
        row=2, col=1
    )
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Geographic distribution (mock data)
    st.subheader("🌍 Visitor Geography")
    
    geo_data = pd.DataFrame({
        'Country': ['United States', 'Germany', 'United Kingdom', 'Canada', 'Australia'],
        'Visitors': [450, 230, 180, 120, 95],
        'Country_Code': ['US', 'DE', 'GB', 'CA', 'AU']
    })
    
    fig = px.choropleth(
        geo_data,
        locations='Country_Code',
        values='Visitors',
        hover_name='Country',
        title="Visitor Distribution by Country"
    )
    st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application"""
    if not st.session_state.authenticated:
        login_page()
    else:
        main_portfolio()

if __name__ == "__main__":
    main() 