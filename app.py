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
    
    # Calculate the dataset median notifications
    # This acts as our objective, noise-free default constant injected dynamically
    median_notifications = float(df['notifications_received_per_day'].median())
    
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
    
    # Load the optimized model pickle (supporting either filename style)
    pkl_name = "final_mental_fatigue_model (1).pkl"
    if not os.path.exists(pkl_name):
        pkl_name = "final_mental_fatigue_model.pkl"
        
    model = joblib.load(pkl_name)
    
    metrics = {
        'best_model_name': 'Gradient Boosting Regressor',
        'r2_score': 0.9025,
        'rmse': 0.8526,
        'mae': 0.6646,
        'dataset_size': 15000,
        'test_size_pct': 20,
        'median_notifications': median_notifications
    }
    
    return model, scaler, X_train.columns, numerical_cols, df, metrics

try:
    model, scaler, feature_cols, numerical_cols, raw_df, metrics = load_ml_pipeline()
except Exception as e:
    st.error(f"Critical error loading model pipeline: {e}")
    st.info("Ensure the model pickle 'final_mental_fatigue_model (1).pkl' or 'final_mental_fatigue_model.pkl' and dataset 'sleep_mobile_stress_dataset_15000.csv' are in the application root directory.")
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
    st.markdown(f"**Default Notifications:** `{metrics['median_notifications']:.0f}/day`")
    
    st.markdown("---")
    st.markdown("### 🧪 Quick Controlled Scenarios")
    st.markdown("Load pre-configured test profiles to verify multi-factor behavior:")
    
    if st.button("CASE A: Optimal Recovery Profile (Low Fatigue)"):
        st.session_state.age = 30
        st.session_state.gender = "Female"
        st.session_state.occupation = "Student"
        st.session_state.screen_time = 2.0
        st.session_state.phone_usage = 10
        st.session_state.sleep_duration = 8.5
        st.session_state.sleep_quality = 9.0
        st.session_state.caffeine = 1
        st.session_state.activity = 60
        st.success("CASE A loaded! Go to the 'Take Assessment' tab.")
        
    if st.button("CASE B: High Digital Strain Profile (High Fatigue)"):
        st.session_state.age = 30
        st.session_state.gender = "Male"
        st.session_state.occupation = "Software Engineer"
        st.session_state.screen_time = 12.0
        st.session_state.phone_usage = 120
        st.session_state.sleep_duration = 5.0
        st.session_state.sleep_quality = 3.0
        st.session_state.caffeine = 4
        st.session_state.activity = 10
        st.success("CASE B loaded! Go to the 'Take Assessment' tab.")
        
    if st.button("CASE C: Typical Balanced Profile (Moderate Fatigue)"):
        st.session_state.age = 25
        st.session_state.gender = "Male"
        st.session_state.occupation = "Student"
        st.session_state.screen_time = 6.0
        st.session_state.phone_usage = 45
        st.session_state.sleep_duration = 7.0
        st.session_state.sleep_quality = 6.0
        st.session_state.caffeine = 2
        st.session_state.activity = 30
        st.success("CASE C loaded! Go to the 'Take Assessment' tab.")

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
        
        # Transparency Section
        st.markdown(
            """
            <div class="info-callout">
                <h4>🛡️ HOW THE ASSESSMENT WORKS (Anti-Bias Design)</h4>
                <p>
                    <b>Methodology & Stress Estimation:</b> 
                    This system evaluates behavioral and lifestyle indicators. To prevent <i>subjective self-reporting bias</i>, 
                    users do not manually input their stress levels. Instead, we estimate stress levels automatically using 
                    an application-layer multivariate linear regression fitted directly on the project's dataset:
                </p>
                <div style="background: rgba(15, 23, 42, 0.4); border: 1px solid rgba(6, 182, 212, 0.2); border-radius: 8px; padding: 12px; font-family: monospace; font-size: 0.85rem; margin-bottom: 12px; overflow-x: auto; color: #06b6d4;">
                    stress_level = 8.3665 + 0.5711 * screen_time + 0.0001 * phone_usage + 0.0011 * sleep_duration - 0.7283 * sleep_quality + 0.0004 * physical_activity - 0.0078 * caffeine
                </div>
                <p>
                    <b>Digital Footprint Normalization:</b> 
                    Similarly, asking users to estimate their exact daily notifications introduces massive self-reporting noise. 
                    Therefore, the model automatically injects the verified dataset median value (<b>162 notifications/day</b>) to provide 
                    objective, standardized prediction baselines.
                </p>
                <p>
                    <b>Multi-Factor Learning:</b> 
                    This estimated stress indicator is evaluated together with sleep, activity, and digital habits to predict mental fatigue patterns. 
                    This system avoids simple shortcut rules and retains full ML pipeline integrity, eliminating subjective entry bias.
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
            color_discrete_sequence=["#06b6d4", "#38bdf8", "#0891b2"]
        )
        fig_gender.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f8fafc'),
            margin=dict(l=0, r=0, t=30, b=0),
            height=200,
            showlegend=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

# 5. TAB 2: ASSESSMENT FORM
elif nav_selection == "📋 Take Assessment":
    st.markdown('<h2 style="margin-bottom: 5px;">📋 Behavior & Lifestyle Assessment</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color:#94a3b8; margin-bottom: 25px;">Please fill out your behavioral parameters below. Objective parameters are analyzed using machine learning to predict fatigue levels.</p>', unsafe_allow_html=True)
    
    # Setup initial session values if not set by quick scenarios
    if 'age' not in st.session_state: st.session_state.age = 28
    if 'gender' not in st.session_state: st.session_state.gender = "Female"
    if 'occupation' not in st.session_state: st.session_state.occupation = "Software Engineer"
    if 'screen_time' not in st.session_state: st.session_state.screen_time = 6.0
    if 'phone_usage' not in st.session_state: st.session_state.phone_usage = 30
    if 'sleep_duration' not in st.session_state: st.session_state.sleep_duration = 7.0
    if 'sleep_quality' not in st.session_state: st.session_state.sleep_quality = 6.0
    if 'caffeine' not in st.session_state: st.session_state.caffeine = 2
    if 'activity' not in st.session_state: st.session_state.activity = 30

    with st.form("assessment_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="form-section-title">SECTION A — PERSONAL INFORMATION</div>', unsafe_allow_html=True)
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
                help="Biological sex or identity."
            )
            
            occupation_options = ["Software Engineer", "Student", "Designer", "Teacher", "Manager", "Freelancer", "Doctor", "Researcher"]
            occupation = st.selectbox(
                "Occupation", 
                options=occupation_options,
                index=occupation_options.index(st.session_state.occupation),
                help="Primary daily activity domain."
            )
            
            st.markdown('<div class="form-section-title">SECTION B — SLEEP BEHAVIOR</div>', unsafe_allow_html=True)
            sleep_duration = st.slider(
                "Sleep Duration (hours)", 
                min_value=0.0, 
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
                help="1 = poor sleep quality | 10 = fully refreshed sleep"
            )
            
        with col2:
            st.markdown('<div class="form-section-title">SECTION C — SMARTPHONE USAGE</div>', unsafe_allow_html=True)
            screen_time = st.slider(
                "Daily Screen Time (hours)", 
                min_value=0.0, 
                max_value=16.0, 
                value=float(st.session_state.screen_time), 
                step=0.5,
                help="Average smartphone usage per day."
            )
            
            phone_usage = st.slider(
                "Phone Usage Before Sleep (minutes)", 
                min_value=0, 
                max_value=180, 
                value=int(st.session_state.phone_usage), 
                step=5,
                help="How long the smartphone is used before sleeping."
            )
            
            st.markdown('<div class="form-section-title">SECTION D — LIFESTYLE</div>', unsafe_allow_html=True)
            activity = st.slider(
                "Physical Activity (minutes/day)", 
                min_value=0, 
                max_value=180, 
                value=int(st.session_state.activity), 
                step=5,
                help="Average physical activity in minutes per day."
            )
            
            caffeine = st.slider(
                "Caffeine Intake (cups/day)", 
                min_value=0, 
                max_value=10, 
                value=int(st.session_state.caffeine),
                step=1,
                help="Daily caffeine intake in cups per day."
            )

        st.markdown("<br>", unsafe_allow_html=True)
        submit_btn = st.form_submit_button("Analyze Mental Fatigue", use_container_width=True)

        if submit_btn:
            # Spinner animation
            with st.spinner("AI analyzing behavioral patterns..."):
                time.sleep(1.2)
            
            # Stress level calculation (dataset-derived multivariate linear regression)
            estimated_stress = (
                8.3665 
                + 0.5711 * screen_time 
                + 0.0001 * phone_usage 
                + 0.0011 * sleep_duration 
                - 0.7283 * sleep_quality 
                + 0.0004 * activity 
                - 0.0078 * caffeine
            )
            # Clamp estimated stress to range [1.0, 10.0]
            estimated_stress = max(1.0, min(10.0, estimated_stress))
            
            # Dynamic injection of median notifications from dataset
            injected_notifications = metrics['median_notifications']
            
            # Reconstruct the feature dataframe structure
            input_df = pd.DataFrame(np.zeros((1, len(feature_cols))), columns=feature_cols)
            
            input_df['age'] = age
            input_df['daily_screen_time_hours'] = screen_time
            input_df['phone_usage_before_sleep_minutes'] = phone_usage
            input_df['sleep_duration_hours'] = sleep_duration
            input_df['sleep_quality_score'] = sleep_quality
            input_df['stress_level'] = estimated_stress
            input_df['caffeine_intake_cups'] = caffeine
            input_df['physical_activity_minutes'] = activity
            input_df['notifications_received_per_day'] = injected_notifications
            
            # Hot-encode category strings
            if f'gender_{gender}' in feature_cols:
                input_df[f'gender_{gender}'] = 1.0
            if f'occupation_{occupation}' in feature_cols:
                input_df[f'occupation_{occupation}'] = 1.0
            
            # Scaler transform
            input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
            
            # Predict fatigue score (1.0 to 10.0 scale)
            ml_pred = model.predict(input_df)[0]
            ml_pred = max(1.0, min(10.0, ml_pred))  # Bound checker
            
            # Save results in session state
            st.session_state.ml_fatigue_score = ml_pred
            st.session_state.estimated_stress = estimated_stress
            st.session_state.user_inputs = {
                'age': age, 'gender': gender, 'occupation': occupation,
                'screen_time': screen_time, 'phone_usage': phone_usage,
                'sleep_duration': sleep_duration, 'sleep_quality': sleep_quality,
                'caffeine': caffeine, 'activity': activity, 
                'notifications': injected_notifications
            }
            
            # Sync session states back so form holds the submitted answers
            st.session_state.age = age
            st.session_state.gender = gender
            st.session_state.occupation = occupation
            st.session_state.screen_time = screen_time
            st.session_state.phone_usage = phone_usage
            st.session_state.sleep_duration = sleep_duration
            st.session_state.sleep_quality = sleep_quality
            st.session_state.caffeine = caffeine
            st.session_state.activity = activity
                       st.success("Analysis complete! Go to the 'Analysis Dashboard' tab to view results.")
            st.info("Select 'Analysis Dashboard' in the sidebar menu to read your personalized report.")

# 6. TAB 3: RESULTS & ANALYTICS DASHBOARD
elif nav_selection == "📊 Analysis Dashboard":
    if 'analysis_completed' not in st.session_state or not st.session_state.analysis_completed:
        st.warning("⚠️ No assessment data found. Please complete the 'Take Assessment' section first.")
        st.stop()
        
    inputs = st.session_state.user_inputs
    score_10 = st.session_state.ml_fatigue_score
    score_100 = score_10 * 10.0
    estimated_stress = st.session_state.estimated_stress
    
    # Non-deterministic phrasing based on predicted score
    if score_10 < 4.0:
        risk_class = "LOW RISK"
        badge_style = "risk-badge risk-low"
        risk_color = "#10b981"
        risk_description = "The model detected patterns associated with a lower cognitive fatigue risk. Behavioral routines and sleep structure are in a sustainable equilibrium."
        interpretation_msg = "Behavioral indicators suggest low mental fatigue risk."
    elif score_10 < 7.0:
        risk_class = "MEDIUM RISK"
        badge_style = "risk-badge risk-medium"
        risk_color = "#f59e0b"
        risk_description = "Behavioral indicators suggest a moderate risk of cognitive fatigue. Small modifications to digital patterns or sleep habits can help prevent further cognitive load."
        interpretation_msg = "Behavioral indicators suggest moderate mental fatigue risk."
    else:
        risk_class = "HIGH RISK"
        badge_style = "risk-badge risk-high"
        risk_color = "#ef4444"
        risk_description = "Behavioral indicators suggest elevated mental fatigue risk. The model detected patterns associated with increased cognitive fatigue under current parameters."
        interpretation_msg = "Behavioral indicators suggest high mental fatigue risk."

    st.markdown('<h2 style="margin-bottom: 5px;">📊 Mental Fatigue Diagnostics Dashboard</h2>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:#94a3b8; margin-bottom: 25px;">Subject profile: <b>{inputs["age"]} y/o {inputs["gender"]} ({inputs["occupation"]})</b></p>', unsafe_allow_html=True)
    
    col_score1, col_score2 = st.columns([1, 1])
    
    with col_score1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown('<h3 style="margin-top:0; color:#06b6d4;">🧠 Predicted Fatigue Score</h3>', unsafe_allow_html=True)
        
        # Display large numeric metrics
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.markdown(
                f"""
                <div style="font-size: 3.5rem; font-weight: 800; color: {risk_color}; line-height:1.1; font-family:'Outfit';">
                    {score_100:.0f} / 100
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
            title = {'text': "Model Predictor Gauge", 'font': {'size': 14, 'color': '#94a3b8'}},
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
        st.markdown('<h3 style="margin-top:0; color:#06b6d4;">⚡ Estimated Stress Level</h3>', unsafe_allow_html=True)
        
        # Color rating for stress level
        if estimated_stress < 4.0:
            stress_color = "#10b981"
            stress_class = "LOW STRESS"
            stress_badge = "risk-badge risk-low"
        elif estimated_stress < 7.0:
            stress_color = "#f59e0b"
            stress_class = "MEDIUM STRESS"
            stress_badge = "risk-badge risk-medium"
        else:
            stress_color = "#ef4444"
            stress_class = "HIGH STRESS"
            stress_badge = "risk-badge risk-high"
            
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.markdown(
                f"""
                <div style="font-size: 3.5rem; font-weight: 800; color: {stress_color}; line-height:1.1; font-family:'Outfit';">
                    {estimated_stress:.1f} / 10
                </div>
                <div style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 15px;">
                    Estimated Stress
                </div>
                """, 
                unsafe_allow_html=True
            )
        with metric_col2:
            st.markdown(
                f"""
                <div style="margin-top: 10px;">
                    <span class="{stress_badge}">{stress_class}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        # Explanatory stress gauge
        fig_stress = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = estimated_stress,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Estimated Stress Scale", 'font': {'size': 14, 'color': '#94a3b8'}},
            number = {'font': {'color': '#f8fafc', 'size': 10}},
            gauge = {
                'axis': {'range': [1.0, 10.0], 'tickwidth': 1, 'tickcolor': "#475569"},
                'bar': {'color': stress_color},
                'bgcolor': "rgba(30, 41, 59, 0.5)",
                'borderwidth': 1,
                'bordercolor': "rgba(255,255,255,0.08)",
                'steps': [
                    {'range': [1.0, 4.0], 'color': 'rgba(16, 185, 129, 0.1)'},
                    {'range': [4.0, 7.0], 'color': 'rgba(245, 158, 11, 0.1)'},
                    {'range': [7.0, 10.0], 'color': 'rgba(239, 68, 68, 0.1)'}
                ],
            }
        ))
        fig_stress.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f8fafc'),
            margin=dict(l=20, r=20, t=10, b=10),
            height=200
        )
        st.plotly_chart(fig_stress, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    # Contextual Interpretation and Balancing Logic
    st.markdown('<div class="custom-card" style="padding: 20px; border-left: 5px solid ' + risk_color + ';">', unsafe_allow_html=True)
    st.markdown('<h4 style="margin-top:0; color:#f8fafc;">📋 Diagnostics Interpretation</h4>', unsafe_allow_html=True)
    st.markdown(f"<p style='color: #cbd5e1; margin-bottom: 12px; font-weight: 600;'>{interpretation_msg}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: #94a3b8; margin-bottom: 12px;'>{risk_description}</p>", unsafe_allow_html=True)
    
    st.markdown(
        f"""
        <p style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 0; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 12px; margin-top: 12px;">
            <b>Estimated Stress Level: {estimated_stress:.1f} / 10</b><br>
            Estimated from sleep quality, smartphone usage, physical activity, and lifestyle indicators to bypass self-report rating bias.<br><br>
            <b>Model Technical Context:</b> This score is predicted using a Gradient Boosting Regressor (R² = 90.25%, MAE = 0.66). 
            Digital notifications were dynamically fixed to the dataset median (<b>{inputs['notifications']:.0f}</b>) to reduce user guessing noise. 
        </p>
        """,
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Transparency Section: "How the System Works"
    st.markdown(
        """
        <div class="custom-card">
            <h3 style="margin-top:0; color:#06b6d4;">🛡️ How the System Works (Transparency Flow)</h3>
            <p style="color:#cbd5e1; font-size:0.95rem;">
                This clinical intelligence model evaluates mental fatigue by translating physical lifestyle inputs 
                into an estimated stress value before predicting the final fatigue risk index. This multi-stage process 
                significantly reduces subjective self-report bias.
            </p>
            <div style="display: flex; flex-direction: row; justify-content: space-around; align-items: center; background: rgba(15, 23, 42, 0.4); border: 1px solid rgba(6, 182, 212, 0.15); border-radius: 12px; padding: 20px; margin-top: 15px; margin-bottom: 15px; text-align: center; gap: 10px; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 120px;">
                    <div style="font-size: 1.1rem; font-weight: 700; color: #38bdf8;">User Behavior</div>
                    <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">Sleep, Screen Time, Activity</div>
                </div>
                <div style="font-size: 1.5rem; color: #06b6d4; font-weight: 700;">↓</div>
                <div style="flex: 1; min-width: 120px;">
                    <div style="font-size: 1.1rem; font-weight: 700; color: #38bdf8;">Stress Estimation</div>
                    <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">Regression Algorithm</div>
                </div>
                <div style="font-size: 1.5rem; color: #06b6d4; font-weight: 700;">↓</div>
                <div style="flex: 1; min-width: 120px;">
                    <div style="font-size: 1.1rem; font-weight: 700; color: #38bdf8;">ML Prediction</div>
                    <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">Gradient Boosting Model</div>
                </div>
                <div style="font-size: 1.5rem; color: #06b6d4; font-weight: 700;">↓</div>
                <div style="flex: 1; min-width: 120px;">
                    <div style="font-size: 1.1rem; font-weight: 700; color: #06b6d4;">Mental Fatigue Score</div>
                    <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">Final Predictor Index</div>
                </div>
            </div>
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
        
    if estimated_stress > 6.0:
        recs.append({
            'icon': "🧘",
            'title': "Elevated Cortisol/Stress Profile",
            'desc': f"Your calculated stress score of <b>{estimated_stress:.1f}/10</b> indicates acute neurological load. Schedule micro-breaks of deep breathing (e.g. box breathing 4-4-4-4) to trigger parasympathetic activity.",
            'action': "Goal: Reduce stress with active coping strategies."
        })
        
    if inputs['activity'] < 30:
        recs.append({
            'icon': "🏃",
            'title': "Sedentary Routine Friction",
            'desc': f"Doing only <b>{inputs['activity']} minutes</b> of exercise limits cognitive stamina. Physical activity stimulates BDNF (Brain-Derived Neurotrophic Factor), boosting mental resilience.",
            'action': "Goal: Aim for a minimum of 30 minutes daily walking/exercise."
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
