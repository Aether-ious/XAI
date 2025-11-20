import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import sys
import os
import matplotlib.pyplot as plt
import requests

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.explainer import ModelExplainer

# Config
DATA_FILENAME = "german_credit_data.csv"
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

# Define correct column names for the raw dataset
COLUMNS = [
    "checkin_acc", "duration", "credit_history", "purpose", "amount",
    "saving_acc", "present_emp_since", "installment_rate", "personal_status",
    "other_debtors", "residing_since", "property", "age", "inst_plans",
    "housing", "num_credits", "job", "dependents", "telephone", "foreign_worker",
    "target" # We rename status to target later
]

st.set_page_config(page_title="GlassBox XAI", layout="wide")

def get_or_download_data():
    """
    Robust Data Loader:
    1. Checks if data exists locally.
    2. If not, downloads it from UCI Repository.
    """
    # Check current directory for data folder
    data_dir = os.path.join(os.path.dirname(__file__), "../data")
    os.makedirs(data_dir, exist_ok=True)
    
    file_path = os.path.join(data_dir, DATA_FILENAME)
    
    if not os.path.exists(file_path):
        with st.spinner(f"📉 Data not found. Downloading fresh copy from UCI..."):
            try:
                response = requests.get(DATA_URL)
                # Save it temporarily
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                
                # Re-save with headers for easier reading
                df = pd.read_csv(file_path, sep=' ', names=COLUMNS)
                # Map Target (1=Good, 2=Bad) -> (0=Good, 1=Bad)
                df['target'] = df['target'].map({1: 0, 2: 1})
                df.to_csv(file_path, index=False)
                st.success("✅ Data downloaded and cached successfully!")
            except Exception as e:
                st.error(f"Failed to download data: {e}")
                return None
    
    return pd.read_csv(file_path)

@st.cache_resource
def load_system():
    # 1. Get Data (Auto-download if missing)
    df = get_or_download_data()
    if df is None:
        return None, None, None

    # 2. Quick Preprocessing (Drop non-numeric for the Demo Model)
    # In a real plugin, this would handle categorical encoding
    target = 'target'
    df_numeric = df.select_dtypes(include=[np.number]).fillna(0)
    
    X = df_numeric.drop(columns=[target], errors='ignore')
    y = df_numeric[target] if target in df_numeric.columns else pd.Series([0]*len(X))
    
    # 3. Train Proxy Model (In-Memory XGBoost)
    # We train a fresh model right here so the dashboard always works
    model = xgb.XGBClassifier(max_depth=4, n_estimators=50, eval_metric='logloss')
    model.fit(X, y)
    
    # 4. Init Explainer
    explainer = ModelExplainer(model, X)
    
    return model, X, explainer

# --- UI LAYOUT ---
st.title("🔍 GlassBox: Model Explainability Dashboard")
st.markdown("Debugging the **Black Box** using SHAP and Counterfactuals.")

model, X, explainer = load_system()

if model:
    # --- TAB 1: GLOBAL EXPLAINABILITY ---
    tab1, tab2, tab3 = st.tabs(["🌍 Global Importance", "👤 Local Explanation", "🔮 What-If Simulator"])
    
    with tab1:
        st.subheader("What drives the model decisions overall?")
        st.markdown("These features have the highest impact on Credit Risk.")
        importance_df = explainer.get_global_importance()
        st.bar_chart(importance_df.set_index('col_name'))

    # --- TAB 2: LOCAL EXPLAINABILITY ---
    with tab2:
        st.subheader("Why was a specific applicant rejected?")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            customer_id = st.number_input("Select Customer ID", 0, len(X)-1, 10)
            customer_row = X.iloc[[customer_id]]
            
            # Show Prediction
            prob = model.predict_proba(customer_row)[:, 1][0]
            st.metric(label="Default Probability", value=f"{prob:.4f}", 
                     delta="High Risk" if prob > 0.5 else "Low Risk", delta_color="inverse")
        
        with col2:
            st.write("### Decision Path (Waterfall)")
            fig = explainer.plot_local_explanation(customer_row)
            st.pyplot(fig)
            st.info("RED bars push risk UP (Bad). BLUE bars push risk DOWN (Good).")

    # --- TAB 3: COUNTERFACTUALS ---
    with tab3:
        st.subheader("Simulation: How to change the outcome?")
        
        feature = st.selectbox("Choose Feature to adjust", X.columns)
        
        current_val = X.iloc[customer_id][feature]
        min_val = float(X[feature].min())
        max_val = float(X[feature].max())
        
        st.write(f"Customer {customer_id} Current Value: **{current_val}**")
        
        # Range Slider
        start, end = st.slider(f"Simulate {feature} range", min_val, max_val, (min_val, max_val))
        range_vals = np.linspace(start, end, 20)
        
        # Run Simulation
        sim_df = explainer.simulate_counterfactual(X.iloc[[customer_id]], feature, range_vals)
        
        st.line_chart(sim_df.set_index(feature))
        st.caption(f"Impact on Risk Probability as {feature} changes.")