# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import sys

# Add src to path for model loading
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Paths
MODEL_PATH = Path("src/model/trained_model.joblib")
DATA_PATH = Path("src/data/bank-full.csv")

# App configuration
st.set_page_config(page_title="Bank Term Deposit Predictor", layout="wide", initial_sidebar_state="expanded")

# Sample input
SAMPLE_DATA = {
    'age': 40,
    'job': 'admin.',
    'marital': 'married',
    'education': 'secondary',
    'default': 'no',
    'balance': 2000,
    'housing': 'yes',
    'loan': 'no',
    'contact': 'cellular',
    'campaign': 2,
    'pdays': 10,
    'previous': 2,
    'poutcome': 'success'
}

# Load model
@st.cache_resource
def load_model():
    try:
        model_package = joblib.load(MODEL_PATH)
        return model_package['model'], model_package['feature_engineer'], model_package['metrics'], model_package.get('X_test'), model_package.get('y_test')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Load data
@st.cache_data
def load_data():
    try:
        return pd.read_csv(DATA_PATH)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

model, feature_engineer, metrics, X_test, y_test = load_model()
data = load_data()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Section", ["Home", "EDA", "Model Insights", "Prediction"])

# Home
if page == "Home":
    st.title("Bank Term Deposit Subscription Predictor")
    st.markdown("""
    Predict whether a client will subscribe to a term deposit.
    - **EDA**: Explore dataset trends
    - **Model Insights**: Review performance and importance
    - **Prediction**: Input new client data

    **Model**: Random Forest (with SMOTE + ENN), optimized for F1-score.
    """)
    st.header("Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("F1 Score", f"{metrics['test_f1']:.3f}")
    col2.metric("Recall", f"{metrics['test_recall']:.3f}")
    col3.metric("Precision", f"{metrics['test_precision']:.3f}")
    col4.metric("Model Type", type(model.named_steps['classifier']).__name__)

# EDA
elif page == "EDA":
    st.header("Exploratory Data Analysis")
    data_clean = data.drop(columns=['duration', 'day', 'month', 'contact', 'default'], errors='ignore')
    data_clean['subscription'] = data_clean['subscription'].map({'yes': '1', 'no': '0'})

    st.subheader("Subscription Distribution")
    fig = px.histogram(data_clean, x='subscription', color='subscription', barmode='group',
                       title="Subscription Distribution")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Distributions")
    feature = st.selectbox("Select Feature", ['age', 'balance', 'campaign', 'pdays', 'previous', 'job', 'marital', 'education', 'poutcome'])
    if feature in ['age', 'balance', 'campaign', 'pdays', 'previous']:
        fig = px.histogram(data_clean, x=feature, color='subscription', marginal='box', barmode='overlay',
                           title=f"{feature.capitalize()} by Subscription")
    else:
        fig = px.histogram(data_clean, x=feature, color='subscription', barmode='group',
                           title=f"{feature.capitalize()} by Subscription")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap")
    corr = data_clean[['age', 'balance', 'campaign', 'pdays', 'previous']].corr()
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale='Viridis',
                                    text=corr.values, texttemplate="%{text:.2f}"))
    st.plotly_chart(fig, use_container_width=True)

# Model Insights
elif page == "Model Insights":
    st.header("Model Insights")
    st.subheader("Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("F1 Score", f"{metrics['test_f1']:.3f}")
    col2.metric("Recall", f"{metrics['test_recall']:.3f}")
    col3.metric("Precision", f"{metrics['test_precision']:.3f}")
    if X_test is not None and y_test is not None:
        y_pred = model.predict(X_test)
        col4.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
    else:
        col4.write("X_test / y_test not available")

    st.subheader("Feature Importance")
    if hasattr(model.named_steps['classifier'], 'feature_importances_'):
        importances = model.named_steps['classifier'].feature_importances_
        feature_names = model.named_steps['preprocessing'].get_feature_names_out()
        imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
        fig = px.bar(imp_df.head(10), x='Importance', y='Feature', orientation='h', title="Top 10 Important Features")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Confusion Matrix")
    if X_test is not None and y_test is not None:
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

    st.subheader("Key Insights")
    st.markdown("""
    - Campaign and previous outcomes significantly influence subscription.
    - Clients with positive history and fewer contacts often subscribe.
    """)

# Prediction
elif page == "Prediction":
    st.header("Predict Subscription Probability")
    use_sample = st.checkbox("Use Sample Data", value=True)

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", 18, 100, SAMPLE_DATA['age'] if use_sample else 30)
            job = st.selectbox("Job", ["admin.", "technician", "services", "management", "blue-collar", "retired", "unemployed"],
                               index=["admin.", "technician", "services", "management", "blue-collar", "retired", "unemployed"].index(SAMPLE_DATA['job']) if use_sample else 0)
            marital = st.selectbox("Marital", ["married", "single", "divorced"],
                                   index=["married", "single", "divorced"].index(SAMPLE_DATA['marital']) if use_sample else 0)
        with col2:
            education = st.selectbox("Education", ["secondary", "primary", "tertiary", "unknown"],
                                     index=["secondary", "primary", "tertiary", "unknown"].index(SAMPLE_DATA['education']) if use_sample else 0)
            balance = st.number_input("Balance", -10000, 1000000, SAMPLE_DATA['balance'] if use_sample else 0)
            housing = st.radio("Housing Loan", ["yes", "no"], index=0 if SAMPLE_DATA['housing'] == 'yes' else 1 if use_sample else 0)
        with col3:
            loan = st.radio("Personal Loan", ["yes", "no"], index=0 if SAMPLE_DATA['loan'] == 'yes' else 1 if use_sample else 0)
            campaign = st.number_input("Campaign Contacts", 1, 50, SAMPLE_DATA['campaign'] if use_sample else 1)
            pdays = st.number_input("Days Since Last Contact", -1, 1000, SAMPLE_DATA['pdays'] if use_sample else -1)
            previous = st.number_input("Previous Contacts", 0, 100, SAMPLE_DATA['previous'] if use_sample else 0)
            poutcome = st.selectbox("Previous Outcome", ["success", "failure", "unknown", "other"],
                                    index=["success", "failure", "unknown", "other"].index(SAMPLE_DATA['poutcome']) if use_sample else 0)
        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame([{
            'age': age, 'job': job, 'marital': marital, 'education': education,
            'balance': balance, 'housing': housing, 'loan': loan,
            'campaign': campaign, 'pdays': pdays, 'previous': previous, 'poutcome': poutcome
        }])

        try:
            st.info("Running prediction...")
            input_processed = feature_engineer.transform(input_df)
            prob = model.predict_proba(input_processed)[0][1]
            decision = "Subscribe" if prob > 0.5 else "Not Subscribe"

            st.subheader("Prediction Result")
            col1, col2 = st.columns([1, 3])
            col1.metric("Decision", decision, delta=f"{prob*100:.1f}%")
            col2.progress(prob)
            st.caption(f"Confidence: {prob*100:.1f}% (Threshold: 50%)")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
