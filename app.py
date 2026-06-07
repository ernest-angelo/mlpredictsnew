import sys
import os
import time

# 1. Scikit-learn Version Pickle-Loading Hook
# The trained model was serialized using scikit-learn 1.6.1.
# On newer versions (like 1.9.0), unpickling raises ModuleNotFoundError: No module named '_loss'.
# We intercept and map the top-level '_loss' to 'sklearn._loss._loss' dynamically.
try:
    import sklearn._loss._loss
    sys.modules['_loss'] = sklearn._loss._loss
except ImportError:
    pass

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="Mental Fatigue Predictor AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling (Navy, Cyan, White, Glassmorphism)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    /* Core Layout Styles */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        color: #f8fafc;
    }
    
    /* Custom Card Design (Frosted Glassmorphism) */
    .custom-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(6, 182, 212, 0.15);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
    }
    
    .custom-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px 0 rgba(6, 182, 212, 0.1);
        border-color: rgba(6, 182, 212, 0.3);
    }
    
    /* Hero Banner styling */
    .hero-banner {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8), rgba(30, 41, 59, 0.8)), linear-gradient(45deg, #0f172a, #0891b2);
        border: 1px solid rgba(6, 182, 212, 0.3);
        border-radius: 20px;
        padding: 40px 30px;
        text-align: left;
        margin-bottom: 30px;
        box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.4);
    }
    
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #f8fafc, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    
    .hero-subtitle {
        font-size: 1.25rem;
        color: #94a3b8;
        font-weight: 400;
        margin-top: 0;
    }
    
    /* Metrics panel */
    .metric-container {
        display: flex;
        justify-content: space-between;
        gap: 15px;
        margin-bottom: 25px;
    }
    
    .metric-card {
        flex: 1;
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #06b6d4;
        font-family: 'Outfit', sans-serif;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }
    
    /* Form sections */
    .form-section-title {
        color: #06b6d4;
        font-size: 1.2rem;
        font-weight: 600;
        border-bottom: 1px solid rgba(6, 182, 212, 0.2);
        padding-bottom: 8px;
        margin-top: 25px;
        margin-bottom: 15px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Risk badges */
    .risk-badge {
        display: inline-block;
        padding: 6px 14px;
        font-weight: 700;
        border-radius: 30px;
        text-transform: uppercase;
        font-size: 0.9rem;
        letter-spacing: 1px;
    }
    
    .risk-low {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .risk-medium {
        background-color: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    .risk-high {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    /* Custom Callouts */
    .info-callout {
        background: rgba(6, 182, 212, 0.05);
        border-left: 4px solid #06b6d4;
        padding: 16px;
        border-radius: 0 12px 12px 0;
        margin: 20px 0;
    }
    
    .warning-callout {
        background: rgba(245, 158, 11, 0.05);
        border-left: 4px solid #f59e0b;
        padding: 16px;
        border-radius: 0 12px 12px 0;
        margin: 20px 0;
    }
    
    /* Recommendation Card Styling */
    .recommendation-card {
        background: rgba(15, 23, 42, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 12px;
    }
    
    /* Navigation Sidebar styling */
    .sidebar-header {
        font-size: 1.4rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 20px;
        background: linear-gradient(90deg, #f8fafc, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        letter-spacing: 0.5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 2. Pipeline Loading and Re-scaling (Cached on Startup)
@st.cache_resource
def load_ml_pipeline():
    # Load raw dataset to replicate the exact fit of the StandardScaler
    df = pd.read_csv("sleep_mobile_stress_dataset_15000.csv")
    
    # Categorical columns
    categorical_cols = ['gender', 'occupation']
    
    # One-hot encode categorical features (exactly like the training notebook)
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Define X and y (excluding user_id and target mental_fatigue_score)
    X = df_encoded.drop(['mental_fatigue_score', 'user_id'], axis=1)
    y = df_encoded['mental_fatigue_score']
    
    # Recreate the exact train-test split (random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaler was fit ONLY on the numerical columns of the training set
    numerical_cols = ['age', 'daily_screen_time_hours', 'phone_usage_before_sleep_minutes',
                      'sleep_duration_hours', 'sleep_quality_score', 'stress_level',
                      'caffeine_intake_cups', 'physical_activity_minutes', 'notifications_received_per_day']
    
    scaler = StandardScaler()
    scaler.fit(X_train[numerical_cols])
    
    # Load the optimized model pickle
    model = joblib.load("final_mental_fatigue_model (1).pkl")
    
    # Model evaluation metrics from notebook for the landing page
    # Original RF: R2 0.8985, RMSE 0.8703, MAE 0.6796
    # Optimized RF: R2 0.9015, RMSE 0.8573, MAE 0.6683
    # GBR (Best): R2 0.9025, RMSE 0.8526, MAE 0.6646
    # XGBoost: R2 0.8892, RMSE 0.9091, MAE 0.7032
    metrics = {
        'best_model_name': 'Gradient Boosting Regressor',
        'r2_score': 0.9025,
        'rmse': 0.8526,
        'mae': 0.6646,
        'dataset_size': 15000,
        'test_size_pct': 20,
    }
    
    return model, scaler, X_train.columns, numerical_cols, df, metrics

try:
    model, scaler, feature_cols, numerical_cols, raw_df, metrics = load_ml_pipeline()
except Exception as e:
    st.error(f"Critical error loading model pipeline: {e}")
    st.info("Ensure the model pickle 'final_mental_fatigue_model (1).pkl' and dataset 'sleep_mobile_stress_dataset_15000.csv' are in the application root directory.")
    st.stop()

# 3. Sidebar Navigation Control
with st.sidebar:
    st.markdown('<div class="sidebar-header">🧠 CLINICAL PORTAL</div>', unsafe_allow_html=True)
    
    # Custom Navigation menu
    nav_selection = st.radio(
        "Navigate Application Sections:",
        ["🏠 Project Overview", "📋 Take Assessment", "📊 Analysis Dashboard"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 📊 Model Vital Stats")
    st.markdown(f"**Algorithm:** `{metrics['best_model_name']}`")
    st.markdown(f"**Validation R²:** `{metrics['r2_score']:.4f}`")
    st.markdown(f"**Mean Abs Error (MAE):** `{metrics['mae']:.4f}`")
    st.markdown(f"**Dataset N:** `{metrics['dataset_size']:,}`")
    
    st.markdown("---")
    st.markdown("### 🧪 Quick Controlled Scenarios")
    st.markdown("Load pre-configured test profiles to verify multi-factor behavior:")
    
    # Button triggers to pre-fill session states
    if st.button("CASE A: High Stress, Healthy Lifestyle"):
        st.session_state.age = 30
        st.session_state.gender = "Female"
        st.session_state.occupation = "Student"
        st.session_state.screen_time = 2.0
        st.session_state.phone_usage = 10
        st.session_state.sleep_duration = 8.0
        st.session_state.sleep_quality = 9.0
        st.session_state.stress_level = 9.0
        st.session_state.caffeine = 0
        st.session_state.activity = 90
        st.session_state.notifications = 30
        st.success("CASE A loaded! Go to the 'Take Assessment' tab.")
        
    if st.button("CASE B: Low Stress, Unhealthy Lifestyle"):
        st.session_state.age = 30
        st.session_state.gender = "Male"
        st.session_state.occupation = "Software Engineer"
        st.session_state.screen_time = 9.0
        st.session_state.phone_usage = 100
        st.session_state.sleep_duration = 4.0
        st.session_state.sleep_quality = 3.0
        st.session_state.stress_level = 2.0
        st.session_state.caffeine = 4
        st.session_state.activity = 10
        st.session_state.notifications = 250
        st.success("CASE B loaded! Go to the 'Take Assessment' tab.")
        
    if st.button("CASE D: Student Burnout Profile"):
        st.session_state.age = 25
        st.session_state.gender = "Male"
        st.session_state.occupation = "Student"
        st.session_state.screen_time = 10.0
        st.session_state.phone_usage = 110
        st.session_state.sleep_duration = 5.0
        st.session_state.sleep_quality = 2.0
        st.session_state.stress_level = 10.0
        st.session_state.caffeine = 3
        st.session_state.activity = 15
        st.session_state.notifications = 280
        st.success("CASE D loaded! Go to the 'Take Assessment' tab.")

# 4. TAB 1: LANDING PAGE (PROJECT OVERVIEW)
if nav_selection == "🏠 Project Overview":
    st.markdown(
        """
        <div class="hero-banner">
            <h1 class="hero-title">Mental Fatigue Predictor AI</h1>
            <p class="hero-subtitle">Academic research-backed cognitive fatigue analysis using optimized machine learning.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("🔬 Clinical Research Scope")
        st.markdown(
            """
            This clinical intelligence system is built to analyze the complex interactions between 
            **behavioral routines, sleep quality, digital activity, and daily stress levels** to predict cognitive fatigue scores.
            
            Cognitive fatigue is a state of mental exhaustion that reduces productivity, increases mistakes, 
            and lowers psychological well-being. It is highly prevalent among students and demanding modern professions. 
            By utilizing a highly optimized machine learning pipeline, this system assesses behavioral biomarkers to classify 
            risk levels and provide clinical-grade actionable insights.
            """
        )
        
        st.markdown(
            """
            <div class="info-callout">
                <h4>🛡️ HOW PREDICTION WORKS (Multi-Factor Principle)</h4>
                <p>
                    <b>Important Clinical Context:</b> This system is designed on <b>multi-factor learning principles</b>. 
                    Unlike simple rule-based apps, our model evaluates all parameters simultaneously. 
                    For example, an individual experiencing <b>high stress (stress level = 9)</b> but maintaining <b>healthy lifestyle habits</b> 
                    (8 hours of sleep, 9/10 sleep quality, regular physical activity, and restricted phone exposure before bed) 
                    may show controlled fatigue levels. Conversely, someone with <b>low stress (stress level = 2)</b> but <b>poor sleep, high notifications, 
                    and excessive digital exposure</b> can suffer from noticeable cognitive fatigue.
                </p>
                <p>
                    This prevents "shortcut learning" where stress level is treated as the sole proxy for fatigue, 
                    retaining proper, balanced machine learning behaviors.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.subheader("📊 Model Optimization & Performance")
        st.markdown(
            """
            During the research phase, multiple regression algorithms were trained and evaluated on 
            **15,000 empirical patient and student records**. The **Gradient Boosting Regressor** emerged as 
            the most accurate model, providing robust generalization and avoiding overfitting.
            """
        )
        
        # Display comparison table
        results_data = {
            "Model Name": ["Gradient Boosting Regressor (Best)", "Optimized Random Forest", "Original Random Forest", "XGBoost Regressor"],
            "R² Score": [0.9025, 0.9015, 0.8985, 0.8892],
            "Root Mean Squared Error (RMSE)": [0.8526, 0.8573, 0.8703, 0.9091],
            "Mean Absolute Error (MAE)": [0.6646, 0.6683, 0.6796, 0.7032]
        }
        st.table(pd.DataFrame(results_data))
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("📈 Statistical Feature Importances")
        st.markdown(
            """
            This chart demonstrates the statistical importance of each feature in the Gradient Boosting model. 
            While stress level represents the primary predictive anchor, sleep patterns and digital activities serve as crucial secondary weights.
            """
        )
        
        # Feature importance chart
        feat_importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)
        # Filter out very minor ones for visual clarity, or show top 9
        top_importances = feat_importances.tail(10)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=top_importances.index,
            x=top_importances.values,
            orientation='h',
            marker=dict(
                color=top_importances.values,
                colorscale='YlGnBu'
            ),
            text=[f"{val*100:.2f}%" for val in top_importances.values],
            textposition='auto'
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f8fafc', size=10),
            margin=dict(l=0, r=0, t=10, b=10),
            height=300,
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Relative Gini Importance"),
            yaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Clinical Study Demographics (using raw data)
        st.subheader("👥 Participant Profile Insights")
        gender_counts = raw_df['gender'].value_counts()
        fig_gender = px.pie(
            names=gender_counts.index, 
            values=gender_counts.values,
            color_discrete_sequence=px.colors.sequential.Cyan
        )
        fig_gender.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f8fafc'),
            margin=dict(l=0, r=0, t=30, b=0),
            height=200,
            showlegend=True
        )
        st.plotly_chart(fig_gender, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# 5. TAB 2: ASSESSMENT FORM
elif nav_selection == "📋 Take Assessment":
    st.markdown('<h2 style="margin-bottom: 5px;">📋 Lifestyle & Behavioral Assessment</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color:#94a3b8; margin-bottom: 25px;">Please fill out your parameters below. All values are restricted to numeric boundaries to eliminate subjective entry bias.</p>', unsafe_allow_html=True)
    
    # Setup initial session values if not set by quick scenarios
    if 'age' not in st.session_state: st.session_state.age = 28
    if 'gender' not in st.session_state: st.session_state.gender = "Female"
    if 'occupation' not in st.session_state: st.session_state.occupation = "Software Engineer"
    if 'screen_time' not in st.session_state: st.session_state.screen_time = 6.0
    if 'phone_usage' not in st.session_state: st.session_state.phone_usage = 30
    if 'sleep_duration' not in st.session_state: st.session_state.sleep_duration = 7.0
    if 'sleep_quality' not in st.session_state: st.session_state.sleep_quality = 6.0
    if 'stress_level' not in st.session_state: st.session_state.stress_level = 5.0
    if 'caffeine' not in st.session_state: st.session_state.caffeine = 2
    if 'activity' not in st.session_state: st.session_state.activity = 30
    if 'notifications' not in st.session_state: st.session_state.notifications = 120

    with st.form("assessment_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="form-section-title">A. Personal Information</div>', unsafe_allow_html=True)
            age = st.slider(
                "Age", 
                min_value=18, 
                max_value=80, 
                value=int(st.session_state.age), 
                help="Your chronological age in years."
            )
            
            gender_options = ["Female", "Male", "Other"]
            gender = st.selectbox(
                "Gender Identity", 
                options=gender_options,
                index=gender_options.index(st.session_state.gender),
                help="Clinical biological sex or identity."
            )
            
            occupation_options = ["Software Engineer", "Student", "Designer", "Teacher", "Manager", "Freelancer", "Doctor", "Researcher"]
            occupation = st.selectbox(
                "Occupation", 
                options=occupation_options,
                index=occupation_options.index(st.session_state.occupation),
                help="Primary daily activity domain."
            )
            
            st.markdown('<div class="form-section-title">B. Sleep Habits</div>', unsafe_allow_html=True)
            sleep_duration = st.slider(
                "Sleep Duration (Hours)", 
                min_value=3.0, 
                max_value=12.0, 
                value=float(st.session_state.sleep_duration), 
                step=0.5,
                help="Average total hours slept in a typical 24-hour cycle."
            )
            
            sleep_quality = st.slider(
                "Sleep Quality Score", 
                min_value=1.0, 
                max_value=10.0, 
                value=float(st.session_state.sleep_quality), 
                step=0.5,
                help="Subjective sleep rating: 1 = Terrible/Restless, 10 = Fully Refreshed."
            )
            
        with col2:
            st.markdown('<div class="form-section-title">C. Digital Behavior</div>', unsafe_allow_html=True)
            screen_time = st.slider(
                "Daily Screen Time (Hours)", 
                min_value=0.0, 
                max_value=16.0, 
                value=float(st.session_state.screen_time), 
                step=0.5,
                help="Average total active device screen interaction time per day."
            )
            
            phone_usage = st.slider(
                "Phone Usage Before Sleep (Minutes)", 
                min_value=0, 
                max_value=120, 
                value=int(st.session_state.phone_usage), 
                step=5,
                help="Active phone/tablet interactions in bed prior to sleep onset."
            )
            
            notifications = st.slider(
                "Notifications Received Per Day", 
                min_value=0, 
                max_value=500, 
                value=int(st.session_state.notifications), 
                step=10,
                help="Average count of digital push alerts received daily."
            )
            
            st.markdown('<div class="form-section-title">D. Lifestyle & Wellness</div>', unsafe_allow_html=True)
            stress_level = st.slider(
                "Stress Level Score", 
                min_value=1.0, 
                max_value=10.0, 
                value=float(st.session_state.stress_level), 
                step=0.5,
                help="Estimated baseline stress level: 1 = Perfectly Calm, 10 = Overwhelmed/Burnout."
            )
            
            caffeine = st.slider(
                "Caffeine Intake (Cups/Day)", 
                min_value=0, 
                max_value=10, 
                value=int(st.session_state.caffeine),
                help="Daily coffee, tea, energy drink, or supplement consumption in cups."
            )
            
            activity = st.slider(
                "Physical Activity (Minutes/Day)", 
                min_value=0, 
                max_value=180, 
                value=int(st.session_state.activity), 
                step=5,
                help="Minutes spent in intentional moderate-to-vigorous exercise daily."
            )

        st.markdown("<br>", unsafe_allow_html=True)
        submit_btn = st.form_submit_button("Analyze Mental Fatigue", use_container_width=True)

        if submit_btn:
            # 6. Analysis loading experience
            with st.spinner("AI analyzing behavioral patterns..."):
                time.sleep(1.5)
            
            # Reconstruct the feature dataframe structure
            input_df = pd.DataFrame(np.zeros((1, len(feature_cols))), columns=feature_cols)
            
            input_df['age'] = age
            input_df['daily_screen_time_hours'] = screen_time
            input_df['phone_usage_before_sleep_minutes'] = phone_usage
            input_df['sleep_duration_hours'] = sleep_duration
            input_df['sleep_quality_score'] = sleep_quality
            input_df['stress_level'] = stress_level
            input_df['caffeine_intake_cups'] = caffeine
            input_df['physical_activity_minutes'] = activity
            input_df['notifications_received_per_day'] = notifications
            
            # Hot-encode category strings
            if f'gender_{gender}' in feature_cols:
                input_df[f'gender_{gender}'] = 1.0
            if f'occupation_{occupation}' in feature_cols:
                input_df[f'occupation_{occupation}'] = 1.0
            
            # Scaler transform
            input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
            
            # Predict fatigue score (0.0 to 10.0 scale)
            ml_pred = model.predict(input_df)[0]
            ml_pred = max(0.0, min(10.0, ml_pred))  # Bound checker
            
            # 7. Cognitive Resilience Index calculation
            # Multi-factor balance score out of 100 based on lifestyle buffers
            # Represents positive health factors to counteract stress-only prediction
            resilience_score = 0
            
            # Sleep duration points (max 25)
            if sleep_duration >= 8.0: resilience_score += 25
            elif sleep_duration >= 7.0: resilience_score += 20
            elif sleep_duration >= 6.0: resilience_score += 12
            elif sleep_duration >= 5.0: resilience_score += 5
            
            # Sleep quality points (max 25)
            resilience_score += (sleep_quality / 10.0) * 25
            
            # Digital exposure points (max 20)
            if screen_time < 4.0: resilience_score += 20
            elif screen_time < 6.0: resilience_score += 15
            elif screen_time < 8.0: resilience_score += 10
            elif screen_time < 11.0: resilience_score += 5
            
            # Physical activity points (max 15)
            if activity >= 60: resilience_score += 15
            elif activity >= 30: resilience_score += 10
            elif activity >= 15: resilience_score += 5
            
            # Phone usage before sleep (max 15)
            if phone_usage < 15: resilience_score += 15
            elif phone_usage <= 30: resilience_score += 12
            elif phone_usage <= 60: resilience_score += 7
            elif phone_usage <= 90: resilience_score += 3
            
            # Save results in session state
            st.session_state.ml_fatigue_score = ml_pred
            st.session_state.resilience_score = resilience_score
            st.session_state.user_inputs = {
                'age': age, 'gender': gender, 'occupation': occupation,
                'screen_time': screen_time, 'phone_usage': phone_usage,
                'sleep_duration': sleep_duration, 'sleep_quality': sleep_quality,
                'stress_level': stress_level, 'caffeine': caffeine,
                'activity': activity, 'notifications': notifications
            }
            
            st.session_state.analysis_completed = True
            st.success("Analysis complete! Go to the 'Analysis Dashboard' tab to view results.")
            
            # Automatically redirect using session state
            st.info("Please select the 'Analysis Dashboard' tab in the sidebar to view your detailed metrics.")

# 6. TAB 3: RESULTS & ANALYTICS DASHBOARD
elif nav_selection == "📊 Analysis Dashboard":
    if 'analysis_completed' not in st.session_state or not st.session_state.analysis_completed:
        st.warning("⚠️ No assessment data found. Please complete the 'Take Assessment' section first.")
        st.stop()
        
    inputs = st.session_state.user_inputs
    score_10 = st.session_state.ml_fatigue_score
    score_100 = score_10 * 10.0
    resilience = st.session_state.resilience_score
    
    # Determine risk level category
    if score_10 < 4.0:
        risk_class = "LOW RISK"
        badge_style = "risk-badge risk-low"
        risk_color = "#10b981"
        risk_description = "Your behavioral patterns suggest a low cognitive fatigue risk. Your neuro-cognitive recovery and digital footprint are in a sustainable equilibrium."
    elif score_10 < 7.0:
        risk_class = "MEDIUM RISK"
        badge_style = "risk-badge risk-medium"
        risk_color = "#f59e0b"
        risk_description = "Your behavioral patterns suggest a moderate cognitive fatigue risk. Minor adjustments to your sleep-digital lifestyle parameters will help prevent cognitive degradation."
    else:
        risk_class = "HIGH RISK"
        badge_style = "risk-badge risk-high"
        risk_color = "#ef4444"
        risk_description = "Your behavioral patterns suggest an elevated cognitive fatigue risk. Chronic burnout risk is present. Interventions in sleep, digital habits, and stress levels are highly recommended."

    st.markdown('<h2 style="margin-bottom: 5px;">📊 Mental Fatigue Diagnostics Dashboard</h2>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:#94a3b8; margin-bottom: 25px;">Subject profile: <b>{inputs["age"]} y/o {inputs["gender"]} ({inputs["occupation"]})</b></p>', unsafe_allow_html=True)
    
    col_score1, col_score2 = st.columns([1, 1])
    
    with col_score1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown('<h3 style="margin-top:0; color:#06b6d4;">🧠 Mental Fatigue Score</h3>', unsafe_allow_html=True)
        
        # Display large numeric metrics
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.markdown(
                f"""
                <div style="font-size: 3.5rem; font-weight: 800; color: {risk_color}; line-height:1.1; font-family:'Outfit';">
                    {score_100:.1f}%
                </div>
                <div style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 15px;">
                    Scale Score: {score_10:.2f} / 10.0
                </div>
                """, 
                unsafe_allow_html=True
            )
        with metric_col2:
            st.markdown(
                f"""
                <div style="margin-top: 10px;">
                    <span class="{badge_style}">{risk_class}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        # Gauge chart using plotly graph objects
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = score_100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Clinical Risk Gauge", 'font': {'size': 14, 'color': '#94a3b8'}},
            number = {'font': {'color': '#f8fafc', 'size': 10}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                'bar': {'color': risk_color},
                'bgcolor': "rgba(30, 41, 59, 0.5)",
                'borderwidth': 1,
                'bordercolor': "rgba(255,255,255,0.08)",
                'steps': [
                    {'range': [0, 40], 'color': 'rgba(16, 185, 129, 0.1)'},
                    {'range': [40, 70], 'color': 'rgba(245, 158, 11, 0.1)'},
                    {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.1)'}
                ],
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f8fafc'),
            margin=dict(l=20, r=20, t=10, b=10),
            height=200
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_score2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown('<h3 style="margin-top:0; color:#06b6d4;">🛡️ Cognitive Resilience Score</h3>', unsafe_allow_html=True)
        
        # Color rating for resilience
        if resilience >= 80:
            res_color = "#10b981"
            res_grade = "EXCELLENT BUFFERING"
            res_desc = "Your lifestyle behaviors provide outstanding buffers. This significantly shields your neurological health from stress-induced degradation."
        elif resilience >= 50:
            res_color = "#f59e0b"
            res_grade = "MODERATE BUFFERING"
            res_desc = "Your lifestyle habits provide moderate defense, but digital overload or sleeping inconsistencies are creating biological friction."
        else:
            res_color = "#ef4444"
            res_grade = "WEAK BUFFERING"
            res_desc = "Your lifestyle buffers are depleted. The absence of sleep restoration and extreme digital exposure leaves you highly vulnerable to stress."
            
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.markdown(
                f"""
                <div style="font-size: 3.5rem; font-weight: 800; color: {res_color}; line-height:1.1; font-family:'Outfit';">
                    {resilience} / 100
                </div>
                <div style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 15px;">
                    Buffer Capacity Index
                </div>
                """, 
                unsafe_allow_html=True
            )
        with metric_col2:
            st.markdown(
                f"""
                <div style="margin-top: 10px;">
                    <span class="risk-badge" style="background-color: {res_color}22; color: {res_color}; border: 1px solid {res_color}55;">{res_grade}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        # Explanatory resilience gauge
        fig_res = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = resilience,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Resilience Buffers", 'font': {'size': 14, 'color': '#94a3b8'}},
            number = {'font': {'color': '#f8fafc', 'size': 10}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                'bar': {'color': res_color},
                'bgcolor': "rgba(30, 41, 59, 0.5)",
                'borderwidth': 1,
                'bordercolor': "rgba(255,255,255,0.08)",
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.1)'},
                    {'range': [50, 80], 'color': 'rgba(245, 158, 11, 0.1)'},
                    {'range': [80, 100], 'color': 'rgba(16, 185, 129, 0.1)'}
                ],
            }
        ))
        fig_res.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f8fafc'),
            margin=dict(l=20, r=20, t=10, b=10),
            height=200
        )
        st.plotly_chart(fig_res, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    # Model uncertainty statement
    st.markdown(
        f"""
        <div class="info-callout" style="border-left-color: {risk_color}; margin-top: 5px; margin-bottom: 25px;">
            <h4>🔍 CLINICAL INTERPRETATION & ML MODEL DISCLOSURE</h4>
            <p>
                <b>Uncertainty Statement:</b> {risk_description}
            </p>
            <p style="font-size: 0.9rem; color: #cbd5e1; margin-bottom: 0;">
                The classification is generated via a <b>Gradient Boosting Regressor</b> (R² = 90.25%, MAE = 0.66). 
                Since this is an empirical probabilistic predictive model, it is not a clinical diagnostics tool. 
                Stress levels account for a heavy feature weight in this dataset, meaning high acute stress naturally inflates predictions. 
                However, your lifestyle buffers (visible in your Resilience Index of <b>{resilience}/100</b>) are essential physiological buffers that 
                slow or prevent long-term cognitive burnout.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Lifestyle comparative analysis
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.subheader("📊 Lifestyle Benchmark Comparison")
    st.markdown("Compare your inputs with the study dataset averages for healthy participants (fatigue score < 4.0):")
    
    # Calculate target statistics for comparison
    avg_healthy = raw_df[raw_df['mental_fatigue_score'] < 4.0].mean(numeric_only=True)
    
    comparison_metrics = [
        {"metric": "Sleep Duration (Hrs)", "user": inputs['sleep_duration'], "benchmark": avg_healthy['sleep_duration_hours']},
        {"metric": "Sleep Quality Score", "user": inputs['sleep_quality'], "benchmark": avg_healthy['sleep_quality_score']},
        {"metric": "Daily Screen Time (Hrs)", "user": inputs['screen_time'], "benchmark": avg_healthy['daily_screen_time_hours']},
        {"metric": "Phone Before Sleep (Mins)", "user": inputs['phone_usage'], "benchmark": avg_healthy['phone_usage_before_sleep_minutes']},
        {"metric": "Daily Physical Act. (Mins)", "user": inputs['activity'], "benchmark": avg_healthy['physical_activity_minutes']},
        {"metric": "Caffeine Intake (Cups)", "user": inputs['caffeine'], "benchmark": avg_healthy['caffeine_intake_cups']},
    ]
    
    comp_df = pd.DataFrame(comparison_metrics)
    
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        name='Your Profile',
        x=comp_df['metric'],
        y=comp_df['user'],
        marker_color='#06b6d4'
    ))
    fig_comp.add_trace(go.Bar(
        name='Healthy Population Benchmark',
        x=comp_df['metric'],
        y=comp_df['benchmark'],
        marker_color='rgba(255, 255, 255, 0.25)'
    ))
    fig_comp.update_layout(
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f8fafc'),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(15, 23, 42, 0.6)'),
        margin=dict(l=10, r=10, t=10, b=10),
        height=320,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Metric Value")
    )
    st.plotly_chart(fig_comp, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Recommendation Engine Output
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.subheader("💡 Dynamic Clinical Interventions")
    st.markdown("Tailored recommendations generated from your lifestyle, sleep, and digital habits:")
    
    recs = []
    
    if inputs['sleep_duration'] < 7.0:
        recs.append({
            'icon': "🛌",
            'title': "Sleep Duration Deficit Detected",
            'desc': f"Your current sleep duration of <b>{inputs['sleep_duration']} hours</b> is below the biological restoration threshold of 7.0 hours. Set a sleep window and wind down 30 minutes earlier.",
            'action': "Goal: Increase sleep duration to at least 7.5 hours."
        })
    else:
        recs.append({
            'icon': "✅",
            'title': "Adequate Sleep Duration",
            'desc': f"Your sleep duration of <b>{inputs['sleep_duration']} hours</b> is optimal and supports neurological clearance of cellular waste.",
            'action': "Maintain this baseline."
        })
        
    if inputs['sleep_quality'] < 6.0:
        recs.append({
            'icon': "🌙",
            'title': "Compromised Sleep Architecture",
            'desc': f"A sleep quality score of <b>{inputs['sleep_quality']}/10</b> indicates a lack of deep and REM sleep stages. Avoid caffeine after 2 PM, sleep in a dark room below 68°F (20°C), and keep wake times consistent.",
            'action': "Goal: Improve sleep quality to > 7.0."
        })
        
    if inputs['screen_time'] > 7.0:
        recs.append({
            'icon': "📱",
            'title': "High Digital Cognitive Load",
            'desc': f"Your screen exposure of <b>{inputs['screen_time']} hours</b> causes heavy visual fatigue and mental multitasking overhead. Practice the 20-20-20 rule to rest eye muscles.",
            'action': "Goal: Limit screen time for non-essential use."
        })
        
    if inputs['phone_usage'] > 30:
        recs.append({
            'icon': "📵",
            'title': "Pre-Sleep Blue Light Exposure",
            'desc': f"Interacting with screens for <b>{inputs['phone_usage']} minutes</b> in bed delays melatonin release and delays REM onset. Read a paper book or listen to white noise instead.",
            'action': "Goal: Zero phone usage in bed before sleep."
        })
        
    if inputs['stress_level'] > 6.0:
        recs.append({
            'icon': "🧘",
            'title': "Elevated Cortisol/Stress Profile",
            'desc': f"Your stress score of <b>{inputs['stress_level']}/10</b> indicates acute neurological load. Schedule micro-breaks of deep breathing (e.g. box breathing 4-4-4-4) to trigger parasympathetic activity.",
            'action': "Goal: Reduce stress with active coping strategies."
        })
        
    if inputs['activity'] < 30:
        recs.append({
            'icon': "🏃",
            'title': "Sedentary Routine Friction",
            'desc': f"Doing only <b>{inputs['activity']} minutes</b> of exercise limits cognitive stamina. Physical activity stimulates BDNF (Brain-Derived Neurotrophic Factor), boosting mental resilience.",
            'action': "Goal: Aim for a minimum of 30 minutes daily walking/exercise."
        })
        
    if inputs['notifications'] > 150:
        recs.append({
            'icon': "🔔",
            'title': "Attention Fragmentation Risk",
            'desc': f"Consuming <b>{inputs['notifications']} notifications/day</b> fractures focus and leads to cognitive interruption overhead. Switch off non-essential notifications and use work focus profiles.",
            'action': "Goal: Group notifications into scheduled daily batches."
        })
        
    # Render recommendations in columns
    rec_cols = st.columns(2)
    for index, rec in enumerate(recs):
        col_to_use = rec_cols[index % 2]
        with col_to_use:
            st.markdown(
                f"""
                <div class="recommendation-card">
                    <div style="display:flex; align-items:center; gap:10px;">
                        <span style="font-size:1.5rem;">{rec['icon']}</span>
                        <h4 style="margin:0; color:#06b6d4;">{rec['title']}</h4>
                    </div>
                    <p style="font-size:0.9rem; color:#cbd5e1; margin-top:8px; margin-bottom:8px;">{rec['desc']}</p>
                    <div style="font-size:0.8rem; color:#94a3b8; font-weight:600; text-transform:uppercase;">{rec['action']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
    st.markdown('</div>', unsafe_allow_html=True)
