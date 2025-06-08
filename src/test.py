import sys
import os
import pandas as pd
import joblib
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now you can import from preprocess
from preprocess import get_preprocessor, FeatureEngineer

# Path configuration
MODEL_PATH = os.path.join("src/model", "trained_model.joblib")
TEST_DATA_PATH = os.path.join("src/data", "bank.csv")

def load_data(test_path):
    """Load and preprocess test data."""
    data = pd.read_csv(test_path)
    data = data.drop(['duration', 'day', 'month', 'contact', 'default'], axis=1, errors='ignore')
    X_test = data.drop('subscription', axis=1)
    y_test = data['subscription'].map({'yes': 1, 'no': 0})
    return X_test, y_test

def evaluate_model(model, feature_engineer, X_test, y_test):
    """Generate evaluation metrics and plots."""
    # Preprocess test data
    X_test_processed = feature_engineer.transform(X_test)
    
    # Predictions
    y_pred = model.predict(X_test_processed)
    y_proba = model.predict_proba(X_test_processed)[:, 1]  # Probabilities for ROC-AUC
    
    # Metrics
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No', 'Yes']))
    
    print("\nROC-AUC Score:", roc_auc_score(y_test, y_proba))
    
    # Confusion Matrix Plot
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("results/confusion_matrix.png")
    plt.show()

if __name__ == "__main__":
    # Load model and test data
    # Load model data - now handles dictionary format
    loaded_data = joblib.load(MODEL_PATH)
    model = loaded_data['model']
    feature_engineer = loaded_data['feature_engineer']
    

     # Load test data
    X_test, y_test = load_data(TEST_DATA_PATH)
    
    # Evaluate
    evaluate_model(model, feature_engineer, X_test, y_test)