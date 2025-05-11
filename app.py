import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
import os
import time

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
    }
    .metric-card {
        background-color: #f0f2f6;
        color:black;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .success-text {
        color: #4CAF50;
        font-weight: bold;
    }
    .warning-text {
        color: #FF9800;
        font-weight: bold;
    }
    .danger-text {
        color: #F44336;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Application title
st.markdown("<h1 class='main-header'>Customer Churn Prediction System</h1>", unsafe_allow_html=True)

# Load the saved model
@st.cache_resource
def load_model(model_path='churn_model.pkl'):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model, True
    except FileNotFoundError:
        return None, False
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, False

# Load sentiment analysis model (if needed for customer feedback)
@st.cache_resource
def load_sentiment_model():
    try:
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        sentiment_analyzer = pipeline("sentiment-analysis", model=model_name)
        return sentiment_analyzer, True
    except Exception as e:
        st.warning(f"Could not load sentiment model. Feedback analysis will be disabled. Error: {e}")
        return None, False

# Calculate risk score
def calculate_risk_score(churn_probability, sentiment_score=0.5):
    # Base risk from churn model
    base_risk = churn_probability * 80
    
    # Sentiment adjustment (neutral = 0.5)
    sentiment_factor = 20 * (1 - sentiment_score)
    
    # Calculate final risk score (0-100)
    final_score = min(100, base_risk + sentiment_factor)
    
    return final_score

# Recommend interventions based on risk profile
def recommend_interventions(risk_profile):
    interventions = []
    
    # Base interventions on risk level
    if risk_profile['risk_level'] == 'High':
        interventions.append("Immediate account review by customer success team")
        interventions.append("Personalized retention offer")
    
    if risk_profile['risk_level'] in ['High', 'Medium']:
        interventions.append("Satisfaction survey with incentive")
    
    # Add specific interventions based on top concern if available
    if 'top_concern' in risk_profile and risk_profile['top_concern']:
        concern = risk_profile['top_concern']
        
        if concern == 'price':
            interventions.append("Discount offer or custom pricing plan")
        elif concern == 'service quality':
            interventions.append("Service quality improvement check")
        elif concern == 'technical issues':
            interventions.append("Technical troubleshooting session")
        elif concern == 'customer support':
            interventions.append("Priority support channel access")
        elif concern == 'product features':
            interventions.append("Product feature education session")
    
    # Always add a general intervention
    if risk_profile['risk_level'] == 'Low':
        interventions.append("Regular check-in email")
    
    return interventions

# Extract concerns from feedback
def extract_concerns(feedback, sentiment_analyzer):
    if not sentiment_analyzer:
        return None, 0
    
    # Simple rule-based concern extraction as fallback to full zero-shot classification
    concerns = {
        "price": ["expensive", "cost", "price", "pricing", "afford", "cheap", "discount"],
        "service quality": ["quality", "poor service", "bad service", "great service", "excellent service"],
        "technical issues": ["technical", "problem", "issue", "bug", "glitch", "error", "broken"],
        "customer support": ["support", "help", "service desk", "representative", "agent", "call center"],
        "product features": ["feature", "functionality", "option", "capability", "missing"]
    }
    
    feedback_lower = feedback.lower()
    
    # Count matches for each concern
    concern_matches = {}
    for concern, keywords in concerns.items():
        matches = sum(1 for keyword in keywords if keyword in feedback_lower)
        concern_matches[concern] = matches
    
    # Get top concern
    if not concern_matches or max(concern_matches.values()) == 0:
        return "general", 0.5
    
    top_concern = max(concern_matches, key=concern_matches.get)
    # Normalize score between 0.5 and 1
    score = min(1.0, 0.5 + (concern_matches[top_concern] / 5))
    
    return top_concern, score

# Create layout
sidebar = st.sidebar
main_left, main_right = st.columns([2, 1])

# Sidebar - Configuration
sidebar.markdown("<h2 class='sub-header'>Configuration</h2>", unsafe_allow_html=True)

# Load the model
model_path = sidebar.text_input("Model Path", "churn_model.pkl")
model, model_loaded = load_model(model_path)

if model_loaded:
    sidebar.success("✅ Model loaded successfully!")
else:
    sidebar.error("❌ Model not found. Please check the path.")
    
# Load sentiment analyzer
sentiment_analyzer, sentiment_loaded = load_sentiment_model()
if sentiment_loaded:
    sidebar.success("✅ Sentiment analyzer loaded successfully!")
else:
    sidebar.warning("⚠️ Sentiment analyzer not available.")

# Get dataset columns (will need these for user input)
# This is just a fallback if we can't determine columns from the model
default_columns = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 
    'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
    'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
]

try:
    # Attempt to get feature names from the model
    if hasattr(model, 'feature_names_in_'):
        feature_columns = model.feature_names_in_
    elif hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
        # For Pipeline objects with a preprocessor
        if hasattr(model.named_steps['preprocessor'], 'feature_names_in_'):
            feature_columns = model.named_steps['preprocessor'].feature_names_in_
        else:
            feature_columns = default_columns
    else:
        feature_columns = default_columns
except:
    feature_columns = default_columns

# Sidebar - Upload customer data file
sidebar.markdown("<h2 class='sub-header'>Data Options</h2>", unsafe_allow_html=True)
uploaded_file = sidebar.file_uploader("Upload customer data CSV (optional)", type="csv")

if uploaded_file is not None:
    # Read the CSV and show dataframe
    df = pd.read_csv(uploaded_file)
    sidebar.success(f"Loaded {df.shape[0]} customer records")
    
    # Select customer from dataframe
    customer_idx = sidebar.selectbox("Select customer from uploaded data", range(len(df)))
    
    # Pre-fill form with selected customer data (will be used below)
    if customer_idx is not None:
        selected_customer = df.iloc[customer_idx]

# Main panel - Customer data input form
with main_left:
    st.markdown("<h2 class='sub-header'>Customer Data</h2>", unsafe_allow_html=True)
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Manual Input", "View Uploaded Data"])
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        
        # Dictionary to store all form inputs
        customer_data = {}
        
        # Helper function to pre-fill form fields if data exists
        def get_default(field, default=None):
            if 'selected_customer' in locals() and field in selected_customer:
                return selected_customer[field]
            return default
        
        # Dynamically create form based on expected model features
        # This is a simplified example - you may need to customize based on your model's features
        with col1:
            # Basic info
            customer_data['gender'] = st.selectbox("Gender", ["Male", "Female"], index=0 if get_default('gender') == 'Male' else 1)
            customer_data['SeniorCitizen'] = st.selectbox("Senior Citizen", [0, 1], index=get_default('SeniorCitizen', 0))
            customer_data['Partner'] = st.selectbox("Partner", ["Yes", "No"], index=0 if get_default('Partner') == 'Yes' else 1)
            customer_data['Dependents'] = st.selectbox("Dependents", ["Yes", "No"], index=0 if get_default('Dependents') == 'Yes' else 1)
            customer_data['tenure'] = st.slider("Tenure (months)", 0, 72, get_default('tenure', 24))
            customer_data['Contract'] = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], 
                                                index=["Month-to-month", "One year", "Two year"].index(get_default('Contract', "Month-to-month")) 
                                                if get_default('Contract') in ["Month-to-month", "One year", "Two year"] else 0)
            
        with col2:
            # Services
            customer_data['PhoneService'] = st.selectbox("Phone Service", ["Yes", "No"], index=0 if get_default('PhoneService') == 'Yes' else 1)
            customer_data['MultipleLines'] = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"], 
                                                    index=["Yes", "No", "No phone service"].index(get_default('MultipleLines', "No")) 
                                                    if get_default('MultipleLines') in ["Yes", "No", "No phone service"] else 1)
            customer_data['InternetService'] = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], 
                                                        index=["DSL", "Fiber optic", "No"].index(get_default('InternetService', "DSL")) 
                                                        if get_default('InternetService') in ["DSL", "Fiber optic", "No"] else 0)
            
            internet = customer_data['InternetService'] != "No"
            disabled_state = not internet
            
            customer_data['OnlineSecurity'] = st.selectbox("Online Security", ["Yes", "No", "No internet service"], 
                                                        index=0 if get_default('OnlineSecurity') == 'Yes' else 1, 
                                                        disabled=disabled_state)
            customer_data['OnlineBackup'] = st.selectbox("Online Backup", ["Yes", "No", "No internet service"], 
                                                    index=0 if get_default('OnlineBackup') == 'Yes' else 1, 
                                                    disabled=disabled_state)
            
        with col3:
            # More services and charges
            if internet:
                customer_data['DeviceProtection'] = st.selectbox("Device Protection", ["Yes", "No", "No internet service"], 
                                                    index=0 if get_default('DeviceProtection') == 'Yes' else 1)
                customer_data['TechSupport'] = st.selectbox("Tech Support", ["Yes", "No", "No internet service"], 
                                                    index=0 if get_default('TechSupport') == 'Yes' else 1)
                customer_data['StreamingTV'] = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], 
                                                    index=0 if get_default('StreamingTV') == 'Yes' else 1)
                customer_data['StreamingMovies'] = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], 
                                                    index=0 if get_default('StreamingMovies') == 'Yes' else 1)
            else:
                for service in ['DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
                    customer_data[service] = "No internet service"
            
            customer_data['PaperlessBilling'] = st.selectbox("Paperless Billing", ["Yes", "No"], 
                                                        index=0 if get_default('PaperlessBilling') == 'Yes' else 1)
            
            payment_methods = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
            default_payment = get_default('PaymentMethod', payment_methods[0])
            payment_index = payment_methods.index(default_payment) if default_payment in payment_methods else 0
            customer_data['PaymentMethod'] = st.selectbox("Payment Method", payment_methods, index=payment_index)
            
            customer_data['MonthlyCharges'] = st.number_input("Monthly Charges ($)", 0.0, 200.0, get_default('MonthlyCharges', 70.0))
            
            # Calculate a reasonable default for TotalCharges based on tenure and monthly charges
            default_total = get_default('TotalCharges', customer_data['tenure'] * customer_data['MonthlyCharges'])
            customer_data['TotalCharges'] = st.number_input("Total Charges ($)", 0.0, 10000.0, float(default_total))

        # Customer feedback section
        st.markdown("<h3>Customer Feedback (Optional)</h3>", unsafe_allow_html=True)
        customer_feedback = st.text_area("Enter customer feedback or support notes if available", 
                                        height=100, 
                                        help="Customer feedback will be analyzed for sentiment and concerns")
        
    with tab2:
        if 'df' in locals():
            st.dataframe(df, height=400)
        else:
            st.info("No data uploaded yet. Use the file uploader in the sidebar.")

# Main panel right - Make predictions
with main_right:
    st.markdown("<h2 class='sub-header'>Prediction Results</h2>", unsafe_allow_html=True)
    
    # Button to run prediction
    if st.button("Calculate Churn Risk", type="primary", disabled=not model_loaded):
        if model_loaded:
            try:
                with st.spinner("Analyzing customer data..."):
                    # Convert customer data to DataFrame
                    customer_df = pd.DataFrame([customer_data])
                    
                    # Make prediction
                    churn_probability = model.predict_proba(customer_df)[0][1]
                    
                    # Process customer feedback if available
                    sentiment_score = 0.5  # Neutral default
                    top_concern = None
                    
                    if customer_feedback and len(customer_feedback.strip()) > 0 and sentiment_analyzer:
                        # Get sentiment
                        sentiment_result = sentiment_analyzer(customer_feedback)[0]
                        
                        # Convert to consistent format (higher = more positive)
                        if sentiment_result['label'] == 'POSITIVE':
                            sentiment_score = sentiment_result['score']
                        else:
                            sentiment_score = 1 - sentiment_result['score']
                            
                        # Extract concerns
                        top_concern, concern_score = extract_concerns(customer_feedback, sentiment_analyzer)
                    
                    # Calculate risk score
                    risk_score = calculate_risk_score(churn_probability, sentiment_score)
                    
                    # Determine risk level
                    if risk_score > 75:
                        risk_level = "High"
                        risk_color = "danger-text"
                    elif risk_score > 50:
                        risk_level = "Medium"
                        risk_color = "warning-text"
                    else:
                        risk_level = "Low"
                        risk_color = "success-text"
                    
                    # Create risk profile
                    risk_profile = {
                        'churn_probability': churn_probability,
                        'sentiment_score': sentiment_score,
                        'top_concern': top_concern,
                        'risk_score': risk_score,
                        'risk_level': risk_level
                    }
                    
                    # Get recommended interventions
                    interventions = recommend_interventions(risk_profile)

                    # Display results                    
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h2>Churn Risk Assessment</h2>
                        <h3 class='{risk_color}'>Risk Level: {risk_level}</h3>
                        <p>Churn Probability: {churn_probability:.2%}</p>
                        <p>Risk Score: {risk_score:.1f}/100</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display sentiment analysis results if feedback was provided
                    if customer_feedback and len(customer_feedback.strip()) > 0:
                        sentiment_label = "Positive" if sentiment_score > 0.5 else "Negative" if sentiment_score < 0.5 else "Neutral"
                        sentiment_color = "success-text" if sentiment_score > 0.6 else "danger-text" if sentiment_score < 0.4 else ""
                        
                        st.markdown(f"""
                        <div class='metric-card' style='margin-top: 20px;'>
                            <h2>Feedback Analysis</h2>
                            <p>Sentiment: <span class='{sentiment_color}'>{sentiment_label}</span> ({sentiment_score:.2f})</p>
                            <p>Top Concern: {top_concern.title() if top_concern else 'None detected'}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Display recommended interventions
                    st.markdown("<div class='metric-card' style='margin-top: 20px;'><h2>Recommended Interventions</h2>", unsafe_allow_html=True)
                    
                    for i, intervention in enumerate(interventions):
                        st.markdown(f"{i+1}. {intervention}")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Create and display visualizations
                    st.markdown("<h3 style='margin-top: 30px;'>Visualization</h3>", unsafe_allow_html=True)
                    
                    # Create gauge chart for risk score
                    fig, ax = plt.subplots(figsize=(4, 3))
                    
                    # Define the gauge
                    gauge_min, gauge_max = 0, 100
                    gauge_ranges = [(0, 50, 'green'), (50, 75, 'orange'), (75, 100, 'red')]
                    
                    # Draw the gauge background
                    for i, (start, end, color) in enumerate(gauge_ranges):
                        ax.barh(0, end-start, left=start, height=0.5, color=color, alpha=0.3)
                    
                    # Draw the gauge value
                    ax.barh(0, 0.1, left=risk_score, height=0.5, color='black')
                    
                    # Customize the plot
                    ax.set_xlim(gauge_min, gauge_max)
                    ax.set_ylim(-0.5, 0.5)
                    ax.set_xlabel('Risk Score')
                    ax.set_yticks([])
                    ax.set_title('Customer Risk Gauge')
                    
                    # Add value labels
                    ax.text(0, -0.25, 'Low Risk', ha='left', va='center')
                    ax.text(50, -0.25, 'Medium Risk', ha='center', va='center')
                    ax.text(100, -0.25, 'High Risk', ha='right', va='center')
                    ax.text(risk_score, 0.25, f'{risk_score:.1f}', ha='center', va='center', fontweight='bold')
                    
                    st.pyplot(fig)

            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.exception(e)
    else:
        st.info("Enter customer data and click 'Calculate Churn Risk' to get prediction results.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center;'>
    <p>Customer Churn Prediction System | Made with Streamlit</p>
</div>
""", unsafe_allow_html=True)

# Add some code to handle missing model file more gracefully
if not model_loaded:
    st.warning("""
    📝 **No model file found at the specified path.**
    
    If this is your first time running the app, make sure your model file (.pkl) is in the correct location. 
    
    Expected model location: `{}` (in the same directory as this app)
    
    If you're sure the path is correct but still seeing this message, check that:
    1. Your model was saved using `pickle`
    2. The model is compatible with the current Python environment
    3. You have permissions to read the file
    """.format(os.path.abspath(model_path)))