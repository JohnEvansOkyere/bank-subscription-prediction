
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

# Custom Transformers
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Store input feature names
        if hasattr(X, 'columns'):
            self.feature_names_in_ = np.array(X.columns)
        else:
            self.feature_names_in_ = np.array([f"feature_{i}" for i in range(X.shape[1])])
        
        # Define output feature names
        self.feature_names_out_ = np.array([
            'was_contacted', 
            'engagement_score',
            'age_group_young', 'age_group_mid', 
            'age_group_senior', 'age_group_retired',
            'Balance_Tier_negative', 'Balance_Tier_low',
            'Balance_Tier_medium', 'Balance_Tier_high'
        ])
        return self
        
    def transform(self, X):
        X = X.copy()
        X["was_contacted"] = (X["pdays"] != -1).astype(int)
        X["engagement_score"] = X["previous"] / (X["pdays"].replace(-1, 999) + 1)
        X["age_group"] = pd.cut(X["age"], bins=[18, 30, 45, 60, 100], 
                              labels=["young", "mid", "senior", "retired"])
        X["Balance_Tier"] = pd.cut(X["balance"], 
                                 bins=[-float("inf"), 0, 1000, 5000, float("inf")],
                                 labels=["negative", "low", "medium", "high"])
        return X.drop(["pdays", "previous", "age", "balance"], axis=1)
    
    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            # Validate that input_features matches what we saw in fit
            if len(input_features) != len(self.feature_names_in_):
                raise ValueError(f"input_features has length {len(input_features)} but expected {len(self.feature_names_in_)}")
        return self.feature_names_out_

        
class BinaryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names_in_ = None
        self.feature_names_out_ = None
    
    def fit(self, X, y=None):
        # Store input feature names
        if hasattr(X, 'columns'):
            self.feature_names_in_ = np.array(X.columns)
        else:
            self.feature_names_in_ = np.array([f"feature_{i}" for i in range(X.shape[1])])
        
        # Output features are same as input for binary encoding
        self.feature_names_out_ = self.feature_names_in_
        return self
        
    def transform(self, X):
        # Convert yes/no to 1/0
        if hasattr(X, 'replace'):
            return X.replace({'yes': 1, 'no': 0})
        return X
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            if self.feature_names_out_ is None:
                raise ValueError("Need to fit transformer first")
            return self.feature_names_out_
        return np.asarray(input_features, dtype=object)


class UnknownHandler(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
        self.feature_names_in_ = None
        self.feature_names_out_ = None
        
    def fit(self, X, y=None):
        if hasattr(X, 'columns'):
            self.feature_names_in_ = np.array(X.columns)
        else:
            self.feature_names_in_ = np.array([f"feature_{i}" for i in range(X.shape[1])])
        self.feature_names_out_ = self.feature_names_in_
        return self
        
    def transform(self, X):
        X = X.copy()
        if self.columns:
            for col in self.columns:
                X[col] = X[col].replace('unknown', np.nan)
        return X
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            if self.feature_names_out_ is None:
                raise ValueError("Need to fit transformer first")
            return self.feature_names_out_
        return np.asarray(input_features, dtype=object)

def get_preprocessor():
    """Return the ColumnTransformer with all preprocessing steps."""
    numeric_features = ['campaign', 'engagement_score']
    categorical_features = ['job', 'marital', 'education', 'poutcome', 'age_group', 'Balance_Tier']
    binary_features = ['housing', 'loan', 'was_contacted']

    # New: Handle unknown values before other processing
    unknown_handler = UnknownHandler(columns=['job', 'education', 'poutcome'])

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('unknown_handler', unknown_handler),  # Add unknown handler first
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Will now handle np.nan from unknown
        ('encoder', OneHotEncoder(handle_unknown='ignore'))  # Handles any remaining unknowns
    ])

    binary_transformer = Pipeline(steps=[
        ('encoder', BinaryEncoder())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('bin', binary_transformer, binary_features)
    ])

    return preprocessor

