# # streamlit_app.py
# import streamlit as st
# import pandas as pd
# import joblib
# import plotly.express as px
# import plotly.graph_objects as go
# from pathlib import Path
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# import os
# import sys

# # Add src to path for model loading
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# # Paths
# MODEL_PATH = Path("src/model/trained_model.joblib")
# DATA_PATH = Path("src/data/bank-full.csv")

# # App configuration
# st.set_page_config(page_title="Bank Term Deposit Predictor", layout="wide", initial_sidebar_state="expanded")

# # Sample input
# SAMPLE_DATA = {
#     'age': 40,
#     'job': 'admin.',
#     'marital': 'married',
#     'education': 'secondary',
#     'default': 'no',
#     'balance': 2000,
#     'housing': 'yes',
#     'loan': 'no',
#     'contact': 'cellular',
#     'campaign': 2,
#     'pdays': 10,
#     'previous': 2,
#     'poutcome': 'success'
# }

# # Load model
# @st.cache_resource
# def load_model():
#     try:
#         model_package = joblib.load(MODEL_PATH)
#         return model_package['model'], model_package['feature_engineer'], model_package['metrics'], model_package.get('X_test'), model_package.get('y_test')
#     except Exception as e:
#         st.error(f"Error loading model: {e}")
#         st.stop()

# # Load data
# @st.cache_data
# def load_data():
#     try:
#         return pd.read_csv(DATA_PATH)
#     except Exception as e:
#         st.error(f"Error loading data: {e}")
#         st.stop()

# model, feature_engineer, metrics, X_test, y_test = load_model()
# data = load_data()

# # Sidebar
# st.sidebar.title("Navigation")
# page = st.sidebar.radio("Select Section", ["Home", "EDA", "Model Insights", "Prediction"])

# # Home
# if page == "Home":
#     st.title("Bank Term Deposit Subscription Predictor")
#     st.markdown("""
#     Predict whether a client will subscribe to a term deposit.
#     - **EDA**: Explore dataset trends
#     - **Model Insights**: Review performance and importance
#     - **Prediction**: Input new client data

#     **Model**: Random Forest (with SMOTE + ENN), optimized for F1-score.
#     """)
#     st.header("Model Performance")
#     col1, col2, col3, col4 = st.columns(4)
#     col1.metric("F1 Score", f"{metrics['test_f1']:.3f}")
#     col2.metric("Recall", f"{metrics['test_recall']:.3f}")
#     col3.metric("Precision", f"{metrics['test_precision']:.3f}")
#     col4.metric("Model Type", type(model.named_steps['classifier']).__name__)

# # EDA
# elif page == "EDA":
#     st.header("Exploratory Data Analysis")
#     data_clean = data.drop(columns=['duration', 'day', 'month', 'contact', 'default'], errors='ignore')
#     data_clean['subscription'] = data_clean['subscription'].map({'yes': '1', 'no': '0'})

#     st.subheader("Subscription Distribution")
#     fig = px.histogram(data_clean, x='subscription', color='subscription', barmode='group',
#                        title="Subscription Distribution")
#     st.plotly_chart(fig, use_container_width=True)

#     st.subheader("Feature Distributions")
#     feature = st.selectbox("Select Feature", ['age', 'balance', 'campaign', 'pdays', 'previous', 'job', 'marital', 'education', 'poutcome'])
#     if feature in ['age', 'balance', 'campaign', 'pdays', 'previous']:
#         fig = px.histogram(data_clean, x=feature, color='subscription', marginal='box', barmode='overlay',
#                            title=f"{feature.capitalize()} by Subscription")
#     else:
#         fig = px.histogram(data_clean, x=feature, color='subscription', barmode='group',
#                            title=f"{feature.capitalize()} by Subscription")
#     st.plotly_chart(fig, use_container_width=True)

#     st.subheader("Correlation Heatmap")
#     corr = data_clean[['age', 'balance', 'campaign', 'pdays', 'previous']].corr()
#     fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale='Viridis',
#                                     text=corr.values, texttemplate="%{text:.2f}"))
#     st.plotly_chart(fig, use_container_width=True)

# # Model Insights
# elif page == "Model Insights":
#     st.header("Model Insights")
#     st.subheader("Performance Metrics")
#     col1, col2, col3, col4 = st.columns(4)
#     col1.metric("F1 Score", f"{metrics['test_f1']:.3f}")
#     col2.metric("Recall", f"{metrics['test_recall']:.3f}")
#     col3.metric("Precision", f"{metrics['test_precision']:.3f}")
#     if X_test is not None and y_test is not None:
#         y_pred = model.predict(X_test)
#         col4.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
#     else:
#         col4.write("X_test / y_test not available")

#     st.subheader("Feature Importance")
#     if hasattr(model.named_steps['classifier'], 'feature_importances_'):
#         try:
#             feature_names = model.named_steps['preprocessing'].get_feature_names_out()
#         except Exception as e:
#             st.error(f"Error getting feature names: {e}")
#             feature_names = None

#         if feature_names is not None:
#             importances = model.named_steps['classifier'].feature_importances_
#             imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
#             fig = px.bar(imp_df.head(10), x='Importance', y='Feature', orientation='h', title="Top 10 Important Features")
#             st.plotly_chart(fig, use_container_width=True)
#         else:
#             st.write("Feature names unavailable, cannot show feature importances.")

#     st.subheader("Confusion Matrix")
#     if X_test is not None and y_test is not None:
#         cm = confusion_matrix(y_test, y_pred)
#         fig, ax = plt.subplots()
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
#         ax.set_title("Confusion Matrix")
#         st.pyplot(fig)

#     st.subheader("Key Insights")
#     st.markdown("""
#     - Campaign and previous outcomes significantly influence subscription.
#     - Clients with positive history and fewer contacts often subscribe.
#     """)


# # Prediction
# elif page == "Prediction":
#     st.header("Predict Subscription Probability")
#     use_sample = st.checkbox("Use Sample Data", value=True)

#     with st.form("prediction_form"):
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             age = st.number_input("Age", 18, 100, SAMPLE_DATA['age'] if use_sample else 30)
#             job = st.selectbox("Job", ["admin.", "technician", "services", "management", "blue-collar", "retired", "unemployed"],
#                                index=["admin.", "technician", "services", "management", "blue-collar", "retired", "unemployed"].index(SAMPLE_DATA['job']) if use_sample else 0)
#             marital = st.selectbox("Marital", ["married", "single", "divorced"],
#                                    index=["married", "single", "divorced"].index(SAMPLE_DATA['marital']) if use_sample else 0)
#         with col2:
#             education = st.selectbox("Education", ["secondary", "primary", "tertiary", "unknown"],
#                                      index=["secondary", "primary", "tertiary", "unknown"].index(SAMPLE_DATA['education']) if use_sample else 0)
#             balance = st.number_input("Balance", -10000, 1000000, SAMPLE_DATA['balance'] if use_sample else 0)
#             housing = st.radio("Housing Loan", ["yes", "no"], index=0 if SAMPLE_DATA['housing'] == 'yes' else 1 if use_sample else 0)
#         with col3:
#             loan = st.radio("Personal Loan", ["yes", "no"], index=0 if SAMPLE_DATA['loan'] == 'yes' else 1 if use_sample else 0)
#             campaign = st.number_input("Campaign Contacts", 1, 50, SAMPLE_DATA['campaign'] if use_sample else 1)
#             pdays = st.number_input("Days Since Last Contact", -1, 1000, SAMPLE_DATA['pdays'] if use_sample else -1)
#             previous = st.number_input("Previous Contacts", 0, 100, SAMPLE_DATA['previous'] if use_sample else 0)
#             poutcome = st.selectbox("Previous Outcome", ["success", "failure", "unknown", "other"],
#                                     index=["success", "failure", "unknown", "other"].index(SAMPLE_DATA['poutcome']) if use_sample else 0)
#         submitted = st.form_submit_button("Predict")

#     if submitted:
#         input_df = pd.DataFrame([{
#             'age': age, 'job': job, 'marital': marital, 'education': education,
#             'balance': balance, 'housing': housing, 'loan': loan,
#             'campaign': campaign, 'pdays': pdays, 'previous': previous, 'poutcome': poutcome
#         }])

#         try:
#             st.info("Running prediction...")
#             input_processed = feature_engineer.transform(input_df)
#             prob = model.predict_proba(input_processed)[0][1]
#             decision = "Subscribe" if prob > 0.5 else "Not Subscribe"

#             st.subheader("Prediction Result")
#             col1, col2 = st.columns([1, 3])
#             col1.metric("Decision", decision, delta=f"{prob*100:.1f}%")
#             col2.progress(prob)
#             st.caption(f"Confidence: {prob*100:.1f}% (Threshold: 50%)")

#         except Exception as e:
#             st.error(f"Prediction failed: {e}")






import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, roc_auc_score, 
                            precision_recall_curve, average_precision_score)
import os
import sys
from collections import defaultdict
import numpy as np
from datetime import datetime
import json

# Add src to path for model loading
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Paths
MODEL_PATH = Path("src/model/trained_model.joblib")
DATA_PATH = Path("src/data/bank-full.csv")
HISTORY_PATH = Path("prediction_history.json")

# App configuration
st.set_page_config(
    page_title="Bank Term Deposit Predictor", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="🏦"
)

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

# Load model with enhanced error handling
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    try:
        model_package = joblib.load(MODEL_PATH)
        return {
            'model': model_package['model'],
            'feature_engineer': model_package['feature_engineer'],
            'metrics': model_package['metrics'],
            'X_test': model_package.get('X_test'),
            'y_test': model_package.get('y_test'),
            'threshold': model_package.get('optimal_threshold', 0.5)
        }
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# Load data with caching
@st.cache_data(show_spinner="Loading data...")
def load_data():
    try:
        data = pd.read_csv(DATA_PATH)
        # Basic data cleaning
        data_clean = data.drop(columns=['duration', 'day', 'month', 'contact', 'default'], errors='ignore')
        data_clean['subscription'] = data_clean['subscription'].map({'yes': '1', 'no': '0'})
        return data_clean
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

# Load prediction history from file
def load_history():
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH, 'r') as f:
            return json.load(f)
    return {
        'predictions': [],
        'probabilities': [],
        'actuals': [],
        'inputs': [],
        'timestamps': []
    }

# Save prediction history to file
def save_history(history):
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history, f)

# Initialize session state
def init_session_state():
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = load_history()
    
    if 'current_prediction' not in st.session_state:
        st.session_state.current_prediction = None
    
    if 'show_details' not in st.session_state:
        st.session_state.show_details = False

# Enhanced performance metrics calculation
def calculate_metrics(y_true, y_pred, y_probs=None):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
    }
    
    if y_probs is not None:
        metrics.update({
            'roc_auc': roc_auc_score(y_true, y_probs),
            'avg_precision': average_precision_score(y_true, y_probs)
        })
    
    return metrics

# Enhanced confusion matrix visualization
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['No', 'Yes'],
        y=['No', 'Yes'],
        text_auto=True,
        color_continuous_scale='Blues'
    )
    fig.update_layout(
        title=title,
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        coloraxis_showscale=False
    )
    return fig

# ROC Curve visualization
def plot_roc_curve(y_true, y_probs):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name='ROC Curve',
        line=dict(color='royalblue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(color='gray', dash='dash')
    ))
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=600, height=600
    )
    return fig

# Precision-Recall Curve visualization
def plot_precision_recall_curve(y_true, y_probs):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode='lines',
        name='Precision-Recall Curve',
        line=dict(color='firebrick', width=2)
    ))
    fig.update_layout(
        title='Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision',
        yaxis=dict(range=[0, 1.05]),
        xaxis=dict(range=[0, 1.05]),
        width=600, height=600
    )
    return fig

# Feature distribution visualization
def plot_feature_distribution(data, feature, target='subscription'):
    if data[feature].dtype in ['int64', 'float64']:
        fig = px.histogram(
            data, 
            x=feature, 
            color=target, 
            marginal='box',
            barmode='overlay',
            title=f"{feature.capitalize()} Distribution by Subscription"
        )
    else:
        fig = px.histogram(
            data, 
            x=feature, 
            color=target, 
            barmode='group',
            title=f"{feature.capitalize()} Distribution by Subscription"
        )
    return fig

# Load model and data
model_data = load_model()
data = load_data()
init_session_state()

# Sidebar for navigation and real-time monitoring
st.sidebar.title("Navigation & Monitoring")
page = st.sidebar.radio(
    "Select Section", 
    ["Home", "EDA", "Model Insights", "Prediction", "Performance History"]
)

# Real-time performance monitoring in sidebar
st.sidebar.subheader("Real-Time Model Performance")
history = st.session_state.prediction_history

if history['actuals'] and any(a is not None for a in history['actuals']):
    valid_indices = [i for i, a in enumerate(history['actuals']) if a is not None]
    actuals = [history['actuals'][i] for i in valid_indices]
    predictions = [history['predictions'][i] for i in valid_indices]
    probabilities = [history['probabilities'][i] for i in valid_indices]
    
    metrics = calculate_metrics(actuals, predictions, probabilities)
    
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    col2.metric("F1 Score", f"{metrics['f1']:.3f}")
    col1.metric("Precision", f"{metrics['precision']:.3f}")
    col2.metric("Recall", f"{metrics['recall']:.3f}")
    
    if 'roc_auc' in metrics:
        st.sidebar.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
    
    # Mini confusion matrix
    st.sidebar.plotly_chart(
        plot_confusion_matrix(actuals, predictions, "Recent Performance"),
        use_container_width=True
    )
else:
    st.sidebar.info("No performance data yet. Make predictions and provide actual outcomes.")

# Home Page
if page == "Home":
    st.title("🏦 Bank Term Deposit Subscription Predictor")
    st.markdown("""
    **Predict whether a client will subscribe to a term deposit**  
    - 📊 **EDA**: Explore dataset trends and distributions
    - 🔍 **Model Insights**: Review model performance and feature importance
    - 🔮 **Prediction**: Input new client data for real-time predictions
    - 📈 **Performance History**: Track model performance over time
    
    **Model Details**:  
    - Algorithm: Random Forest (with SMOTE + ENN)  
    - Optimized for: F1-score  
    - Threshold: {:.2f} (optimized for business needs)
    """.format(model_data['threshold']))
    
    st.header("Model Performance Summary")
    cols = st.columns(4)
    cols[0].metric("F1 Score", f"{model_data['metrics']['test_f1']:.3f}")
    cols[1].metric("Recall", f"{model_data['metrics']['test_recall']:.3f}")
    cols[2].metric("Precision", f"{model_data['metrics']['test_precision']:.3f}")
    cols[3].metric("ROC AUC", f"{model_data['metrics'].get('test_roc_auc', 'N/A')}")
    
    if model_data['X_test'] is not None and model_data['y_test'] is not None:
        with st.expander("Test Set Performance Details"):
            y_pred_test = model_data['model'].predict(model_data['X_test'])
            y_probs_test = model_data['model'].predict_proba(model_data['X_test'])[:, 1]
            
            tab1, tab2 = st.tabs(["Confusion Matrix", "ROC Curve"])
            with tab1:
                st.plotly_chart(
                    plot_confusion_matrix(model_data['y_test'], y_pred_test, "Test Set Confusion Matrix"),
                    use_container_width=True
                )
            with tab2:
                st.plotly_chart(
                    plot_roc_curve(model_data['y_test'], y_probs_test),
                    use_container_width=True
                )

# EDA Page
elif page == "EDA":
    st.header("📊 Exploratory Data Analysis")
    
    st.subheader("Subscription Distribution")
    fig = px.pie(
        data, 
        names='subscription', 
        title="Subscription Distribution (0=No, 1=Yes)",
        hole=0.4
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Feature Distributions")
    feature = st.selectbox(
        "Select Feature to Explore", 
        ['age', 'balance', 'campaign', 'pdays', 'previous', 'job', 'marital', 'education', 'poutcome']
    )
    
    st.plotly_chart(
        plot_feature_distribution(data, feature),
        use_container_width=True
    )
    
    st.subheader("Correlation Analysis")
    numeric_features = ['age', 'balance', 'campaign', 'pdays', 'previous']
    corr = data[numeric_features].corr()
    
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale='Viridis',
            text=corr.values.round(2),
            texttemplate="%{text}",
            zmin=-1,
            zmax=1
        )
    )
    fig.update_layout(title="Feature Correlation Heatmap")
    st.plotly_chart(fig, use_container_width=True)

# Model Insights Page
elif page == "Model Insights":
    st.header("🔍 Model Insights")
    
    st.subheader("Performance Metrics")
    cols = st.columns(4)
    cols[0].metric("F1 Score", f"{model_data['metrics']['test_f1']:.3f}")
    cols[1].metric("Recall", f"{model_data['metrics']['test_recall']:.3f}")
    cols[2].metric("Precision", f"{model_data['metrics']['test_precision']:.3f}")
    cols[3].metric("ROC AUC", f"{model_data['metrics'].get('test_roc_auc', 'N/A')}")
    
    if model_data['X_test'] is not None and model_data['y_test'] is not None:
        y_pred = model_data['model'].predict(model_data['X_test'])
        y_probs = model_data['model'].predict_proba(model_data['X_test'])[:, 1]
        
        tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "ROC Curve", "Precision-Recall"])
        with tab1:
            st.plotly_chart(
                plot_confusion_matrix(model_data['y_test'], y_pred),
                use_container_width=True
            )
        with tab2:
            st.plotly_chart(
                plot_roc_curve(model_data['y_test'], y_probs),
                use_container_width=True
            )
        with tab3:
            st.plotly_chart(
                plot_precision_recall_curve(model_data['y_test'], y_probs),
                use_container_width=True
            )
    
    st.subheader("Feature Importance")
    if hasattr(model_data['model'].named_steps['classifier'], 'feature_importances_'):
        try:
            feature_names = model_data['model'].named_steps['preprocessing'].get_feature_names_out()
            importances = model_data['model'].named_steps['classifier'].feature_importances_
            
            imp_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(
                imp_df.head(20),
                x='Importance',
                y='Feature',
                orientation='h',
                title="Top 20 Important Features",
                color='Importance',
                color_continuous_scale='Bluered'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("View All Feature Importances"):
                st.dataframe(imp_df)
        except Exception as e:
            st.error(f"Could not display feature importances: {str(e)}")
    else:
        st.warning("Feature importances not available for this model type")
    
    st.subheader("Key Insights")
    st.markdown("""
    - **Campaign contacts** have significant negative impact - fewer contacts tend to result in better conversion
    - **Positive previous outcomes** strongly predict subscription
    - **Balance tiers** show clear patterns in subscription likelihood
    - **Age groups** have different response patterns
    """)

# Prediction Page
elif page == "Prediction":
    st.header("🔮 Predict Subscription Probability")
    use_sample = st.checkbox("Use Sample Data", value=True)
    
    with st.form("prediction_form"):
        cols = st.columns(3)
        
        with cols[0]:
            st.subheader("Demographics")
            age = st.number_input("Age", 18, 100, SAMPLE_DATA['age'] if use_sample else 30)
            job = st.selectbox(
                "Job", 
                ["admin.", "technician", "services", "management", "blue-collar", "retired", "unemployed"],
                index=0 if not use_sample else ["admin.", "technician", "services", "management", "blue-collar", "retired", "unemployed"].index(SAMPLE_DATA['job'])
            )
            marital = st.selectbox(
                "Marital Status", 
                ["married", "single", "divorced"],
                index=0 if not use_sample else ["married", "single", "divorced"].index(SAMPLE_DATA['marital'])
            )
            education = st.selectbox(
                "Education Level", 
                ["secondary", "primary", "tertiary", "unknown"],
                index=0 if not use_sample else ["secondary", "primary", "tertiary", "unknown"].index(SAMPLE_DATA['education'])
            )
        
        with cols[1]:
            st.subheader("Financial Information")
            balance = st.number_input("Account Balance", -10000, 1000000, SAMPLE_DATA['balance'] if use_sample else 0)
            housing = st.radio(
                "Housing Loan", 
                ["yes", "no"],
                index=0 if not use_sample else (0 if SAMPLE_DATA['housing'] == 'yes' else 1),
                horizontal=True
            )
            loan = st.radio(
                "Personal Loan", 
                ["yes", "no"],
                index=0 if not use_sample else (0 if SAMPLE_DATA['loan'] == 'yes' else 1),
                horizontal=True
            )
        
        with cols[2]:
            st.subheader("Campaign Details")
            campaign = st.number_input("Number of Contacts", 1, 50, SAMPLE_DATA['campaign'] if use_sample else 1)
            pdays = st.number_input("Days Since Last Contact", -1, 1000, SAMPLE_DATA['pdays'] if use_sample else -1)
            previous = st.number_input("Previous Contacts", 0, 100, SAMPLE_DATA['previous'] if use_sample else 0)
            poutcome = st.selectbox(
                "Previous Outcome", 
                ["success", "failure", "unknown", "other"],
                index=0 if not use_sample else ["success", "failure", "unknown", "other"].index(SAMPLE_DATA['poutcome'])
            )
        
        submitted = st.form_submit_button("Predict Subscription")

    if submitted:
        input_data = {
            'age': age,
            'job': job,
            'marital': marital,
            'education': education,
            'balance': balance,
            'housing': housing,
            'loan': loan,
            'campaign': campaign,
            'pdays': pdays,
            'previous': previous,
            'poutcome': poutcome
        }
        
        input_df = pd.DataFrame([input_data])
        
        try:
            with st.spinner("Processing prediction..."):
                # Feature engineering
                input_processed = model_data['feature_engineer'].transform(input_df)
                
                # Prediction
                prob = model_data['model'].predict_proba(input_processed)[0][1]
                prediction = 1 if prob > model_data['threshold'] else 0
                decision = "SUBSCRIBE" if prediction == 1 else "NOT SUBSCRIBE"
                confidence = prob * 100
                
                # Store prediction
                prediction_id = len(history['predictions'])
                history['predictions'].append(prediction)
                history['probabilities'].append(prob)
                history['inputs'].append(input_data)
                history['timestamps'].append(datetime.now().isoformat())
                st.session_state.current_prediction = prediction_id
                
                # Display results
                st.success("Prediction completed successfully!")
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric(
                        "Prediction", 
                        decision,
                        delta=f"{confidence:.1f}%",
                        delta_color="normal"
                    )
                
                with col2:
                    st.progress(prob)
                    st.caption(f"Confidence: {confidence:.1f}% (Threshold: {model_data['threshold']*100:.1f}%)")
                
                # Probability distribution visualization
                fig = go.Figure()
                fig.add_vline(
                    x=model_data['threshold'], 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text=f"Threshold: {model_data['threshold']:.2f}",
                    annotation_position="top right"
                )
                fig.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=prob,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Subscription Probability"},
                    gauge={
                        'axis': {'range': [0, 1]},
                        'steps': [
                            {'range': [0, model_data['threshold']], 'color': "lightgray"},
                            {'range': [model_data['threshold'], 1], 'color': "lightblue"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': model_data['threshold']
                        }
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
                
                # Actual outcome input
                with st.expander("Provide Actual Outcome for Model Monitoring"):
                    actual = st.radio(
                        f"Did this client actually subscribe? (Prediction #{prediction_id})",
                        ["Unknown", "Yes", "No"],
                        index=0,
                        key=f"actual_{prediction_id}"
                    )
                    
                    if actual != "Unknown":
                        history['actuals'].append(1 if actual == "Yes" else 0)
                        st.success("Thank you for providing feedback! This helps improve the model.")
                    else:
                        history['actuals'].append(None)
                
                # Save history
                save_history(history)
                
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.exception(e)

# Performance History Page
elif page == "Performance History":
    st.header("📈 Performance History")
    
    if not history['predictions']:
        st.info("No prediction history available yet.")
    else:
        # Create dataframe from history
        history_df = pd.DataFrame({
            'Timestamp': history['timestamps'],
            'Prediction': history['predictions'],
            'Probability': history['probabilities'],
            'Actual': history['actuals']
        })
        history_df['Correct'] = history_df.apply(
            lambda x: x['Prediction'] == x['Actual'] if pd.notna(x['Actual']) else None,
            axis=1
        )
        history_df['Timestamp'] = pd.to_datetime(history_df['Timestamp'])
        
        # Filter only predictions with actual outcomes
        valid_history = history_df.dropna(subset=['Actual'])
        
        if not valid_history.empty:
            st.subheader("Performance Over Time")
            
            # Calculate cumulative metrics
            valid_history['Cumulative Accuracy'] = valid_history['Correct'].expanding().mean()
            valid_history['Cumulative F1'] = valid_history.apply(
                lambda x: f1_score(
                    valid_history.loc[:x.name, 'Actual'],
                    valid_history.loc[:x.name, 'Prediction']
                ),
                axis=1
            )
            
            fig = px.line(
                valid_history,
                x='Timestamp',
                y=['Cumulative Accuracy', 'Cumulative F1'],
                title="Model Performance Over Time",
                labels={'value': 'Score', 'variable': 'Metric'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show recent predictions with actuals
            st.subheader("Recent Predictions with Outcomes")
            st.dataframe(
                valid_history.sort_values('Timestamp', ascending=False).head(10),
                column_config={
                    'Timestamp': st.column_config.DatetimeColumn("Time"),
                    'Prediction': st.column_config.NumberColumn("Prediction", format="%d"),
                    'Probability': st.column_config.NumberColumn("Probability", format="%.2f"),
                    'Actual': st.column_config.NumberColumn("Actual", format="%d"),
                    'Correct': st.column_config.CheckboxColumn("Correct")
                }
            )
        else:
            st.info("No predictions with actual outcomes recorded yet.")
        
        # Show all prediction history
        with st.expander("View Full Prediction History"):
            st.dataframe(
                history_df.sort_values('Timestamp', ascending=False),
                column_config={
                    'Timestamp': st.column_config.DatetimeColumn("Time"),
                    'Prediction': st.column_config.NumberColumn("Prediction", format="%d"),
                    'Probability': st.column_config.NumberColumn("Probability", format="%.2f"),
                    'Actual': st.column_config.NumberColumn("Actual", format="%d"),
                    'Correct': st.column_config.CheckboxColumn("Correct")
                },
                use_container_width=True
            )
        
        # Option to clear history
        if st.button("Clear Prediction History", type="secondary"):
            st.session_state.prediction_history = {
                'predictions': [],
                'probabilities': [],
                'actuals': [],
                'inputs': [],
                'timestamps': []
            }
            save_history(st.session_state.prediction_history)
            st.rerun()