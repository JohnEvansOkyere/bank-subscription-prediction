import os
import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours  
from preprocess import get_preprocessor, FeatureEngineer
from sklearn.metrics import make_scorer, f1_score, recall_score, precision_score

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'data', 'bank-full.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'trained_model.joblib')

# models directory 
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Load and preprocess data
data = pd.read_csv(CSV_PATH)
data = data.drop(['duration', 'day', 'month', 'contact', 'default'], axis=1)
X = data.drop('subscription', axis=1)
y = data['subscription'].map({'yes': 1, 'no': 0})

# Feature engineering
feature_engineer = FeatureEngineer()
X_fe = feature_engineer.fit_transform(X)

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_fe, y, 
    test_size=0.2, 
    stratify=y, 
    random_state=42
)

# RESAMPLER IMPLEMENTATION
smote = SMOTE(sampling_strategy=0.3, k_neighbors=5)
enn = EditedNearestNeighbours(sampling_strategy='majority')  
resampler = SMOTEENN(smote=smote, enn=enn) 

# Class weights
class_weights = {0: 1, 1: len(y_train[y_train==0])//len(y_train[y_train==1])}

# Enhanced pipeline
pipeline = ImbPipeline(steps=[
    ('preprocessing', get_preprocessor()),
    ('resample', resampler),
    ('classifier', RandomForestClassifier(
        random_state=42,
        class_weight='balanced',
        bootstrap=True
    ))
])

# Simplified parameter grid (focus on key parameters)
param_grid = {
    'classifier__n_estimators': [100, 150],
    'classifier__max_depth': [10, 15],
    'classifier__min_samples_leaf': [3, 5],
    'resample__smote__k_neighbors': [3, 5]
}

# Scoring metrics
scoring = {
    'f1': make_scorer(f1_score),
    'recall': make_scorer(recall_score),
    'precision': make_scorer(precision_score)
}

# Try-except block for robust error handling and debugging
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,  
    scoring=scoring,
    refit='f1',  
    n_jobs=-1,
    verbose=2,
    error_score='raise'  
)

try:
    grid_search.fit(X_train, y_train)
    
    # Save best model with metadata
    best_model = {
        'model': grid_search.best_estimator_,
        'feature_engineer': feature_engineer,
        'metrics': {
            'test_f1': f1_score(y_test, grid_search.predict(X_test)),
            'test_recall': recall_score(y_test, grid_search.predict(X_test)),
            'test_precision': precision_score(y_test, grid_search.predict(X_test))
        },
        'best_params': grid_search.best_params_
    }
    
    joblib.dump(best_model, MODEL_PATH)
    print(f"✅ Model training complete. Saved to: {MODEL_PATH}")
    print(f"Best parameters: {grid_search.best_params_}")
    
except Exception as e:
    print(f"❌ Training failed: {str(e)}")
    
    print("\nDebug Info:")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train value counts:\n{y_train.value_counts()}")
    print(f"Pipeline steps: {[step[0] for step in pipeline.steps]}")