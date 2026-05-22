import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

# Setting up the page configuration
st.set_page_config(page_title="FraudOps AI Dashboard", page_icon="🛡️", layout="wide")

# ==========================================
# 1. CACHING DATA & MODELS FOR PERFORMANCE
# ==========================================
@st.cache_resource
def load_model():
    return joblib.load('xgb_fraud_model.pkl')

@st.cache_data
def load_data():
    return pd.read_csv('dashboard_data.csv')

model = load_model()
df = load_data()

# Separating features for prediction
feature_cols = [col for col in df.columns if col != 'isFraud']

# ==========================================
# 2. SIDEBAR NAVIGATION & FILTERS
# ==========================================
st.sidebar.title("🛡️ FraudOps System")
st.sidebar.markdown("Real-time ML Fraud Detection")

page = st.sidebar.radio("Navigation:", ["1. Overview", "2. Transaction Explorer", "3. SHAP Explainer"])

st.sidebar.markdown("---")
st.sidebar.header("Global Filters")
# Filter applied across pages
min_amt = st.sidebar.slider("Minimum Transaction Amt ($)", int(df['TransactionAmt'].min()), int(df['TransactionAmt'].max()), 0)
filtered_df = df[df['TransactionAmt'] >= min_amt]

# ==========================================
# PAGE 1: OVERVIEW
# ==========================================
if page == "1. Overview":
    st.title("📈 System Overview & KPIs")
    
    # Calculating KPIs
    total_tx = len(filtered_df)
    total_fraud = filtered_df['isFraud'].sum()
    detection_rate = (total_fraud / total_tx) * 100 if total_tx > 0 else 0
    avg_fraud_amt = filtered_df[filtered_df['isFraud'] == 1]['TransactionAmt'].mean() if total_fraud > 0 else 0

    # Displaying Top Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{total_tx:,}")
    col2.metric("Total Fraud Cases", f"{total_fraud:,}")
    col3.metric("Detection Rate", f"{detection_rate:.2f}%")
    col4.metric("Avg Fraud Amount", f"${avg_fraud_amt:.2f}")

    st.markdown("---")
    
    # Plotly Interactive Charts
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.subheader("Fraud vs Legitimate Proportion")
        fig_pie = px.pie(filtered_df, names='isFraud', hole=0.4, color_discrete_sequence=['#2ecc71', '#e74c3c'])
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_chart2:
        st.subheader("Transaction Amount Distribution")
        fig_hist = px.histogram(filtered_df, x='TransactionAmt', color='isFraud', log_y=True, nbins=50,
                                color_discrete_sequence=['#2ecc71', '#e74c3c'])
        st.plotly_chart(fig_hist, use_container_width=True)

# ==========================================
# PAGE 2: TRANSACTION EXPLORER
# ==========================================
elif page == "2. Transaction Explorer":
    st.title("🔍 Transaction Explorer")
    st.markdown("Search and filter individual transactions. Risk scores are calculated live.")
    
    # Search Bar
    search_id = st.text_input("Enter Transaction ID to search:")
    
    display_df = filtered_df.copy()
    
    # Simulating Live Risk Scores on the fly
    display_df['Live_Risk_Score'] = model.predict_proba(display_df[feature_cols])[:, 1]
    display_df['Risk_Tier'] = np.where(display_df['Live_Risk_Score'] >= 0.75, '🔴 Critical Risk', 
                                       np.where(display_df['Live_Risk_Score'] >= 0.40, '🟡 Suspicious', '🟢 Clear'))
    
    if search_id:
        try:
            display_df = display_df[display_df['TransactionID'] == float(search_id)]
        except ValueError:
            st.warning("Please enter a valid numeric ID.")
            
    # Displaying the dataframe
    st.dataframe(display_df[['TransactionID', 'TransactionAmt', 'Live_Risk_Score', 'Risk_Tier', 'isFraud']], 
                 use_container_width=True)

# ==========================================
# PAGE 3: SHAP EXPLAINER
# ==========================================
elif page == "3. SHAP Explainer":
    st.title("🧠 AI Explainer (SHAP)")
    st.markdown("Understand **why** the model made its prediction.")
    
    # Dropdown for Transaction Selection
    tx_input = st.selectbox("Select a Transaction ID to analyze:", filtered_df['TransactionID'].head(100))
    
    if st.button("Analyze Transaction"):
        with st.spinner("Generating Explanation..."):
            tx_data = filtered_df[filtered_df['TransactionID'] == tx_input][feature_cols]
            prob = model.predict_proba(tx_data)[0][1]
            
            # Plain English Explanation
            st.subheader("Plain-English Summary")
            if prob >= 0.75:
                st.error(f"🚨 **CRITICAL RISK (Score: {prob:.2f})**: The AI is highly confident this is fraud.")
                st.markdown("Factors such as highly unusual transaction amounts or risky device signatures are heavily influencing this decision.")
            elif prob >= 0.40:
                st.warning(f"⚠️ **SUSPICIOUS (Score: {prob:.2f})**: Borderline activity detected.")
                st.markdown("The transaction shares some patterns with known fraud, but also contains normal behavioral metrics. Manual review is recommended.")
            else:
                st.success(f"✅ **CLEAR (Score: {prob:.2f})**: Transaction appears legitimate.")
                st.markdown("The underlying data points strongly align with normal customer behavior.")
            
            st.markdown("---")
            st.subheader("SHAP Waterfall Plot")
            
            # SHAP Calculation
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer(tx_data)
            
            # Displaying Plotly/Matplotlib Plot
            fig, ax = plt.subplots(figsize=(8, 5))
            shap.plots.waterfall(shap_vals[0], show=False)
            st.pyplot(fig)
