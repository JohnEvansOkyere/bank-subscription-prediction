
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

# Custom Transformers
class FeatureEngineer(BaseEstimator, TransformerMixin):
    # Create new features from existing ones and drop originals
    def fit(self, X, y=None):
        # Store input feature names
        if hasattr(X, 'columns'):
            self.feature_names_in_ = np.array(X.columns)
        else:
            self.feature_names_in_ = np.array([f"feature_{i}" for i in range(X.shape[1])])
        
        # Define output feature names for compatibility with pipelines
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
        # Flag if contacted (pdays != -1)
        X["was_contacted"] = (X["pdays"] != -1).astype(int)
        # Calculate engagement score: previous contacts divided by pdays+1 (with -1 replaced by 999)
        X["engagement_score"] = X["previous"] / (X["pdays"].replace(-1, 999) + 1)
        # Categorize age into groups
        X["age_group"] = pd.cut(X["age"], bins=[18, 30, 45, 60, 100], 
                              labels=["young", "mid", "senior", "retired"])
        # Categorize balance into tiers
        X["Balance_Tier"] = pd.cut(X["balance"], 
                                 bins=[-float("inf"), 0, 1000, 5000, float("inf")],
                                 labels=["negative", "low", "medium", "high"])
        # Drop original columns used for engineered features
        return X.drop(["pdays", "previous", "age", "balance"], axis=1)
    
    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            # Validate input feature length matches fitted data
            if len(input_features) != len(self.feature_names_in_):
                raise ValueError(f"input_features has length {len(input_features)} but expected {len(self.feature_names_in_)}")
        return self.feature_names_out_

        
class BinaryEncoder(BaseEstimator, TransformerMixin):
    # Encode binary 'yes'/'no' string features as 1/0 integers
    def __init__(self):
        self.feature_names_in_ = None
        self.feature_names_out_ = None
    
    def fit(self, X, y=None):
        # Store input feature names
        if hasattr(X, 'columns'):
            self.feature_names_in_ = np.array(X.columns)
        else:
            self.feature_names_in_ = np.array([f"feature_{i}" for i in range(X.shape[1])])
        
        # Output features same as input features for binary encoding
        self.feature_names_out_ = self.feature_names_in_
        return self
        
    def transform(self, X):
        # Convert 'yes' to 1 and 'no' to 0
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
    # Replace 'unknown' entries with np.nan in specified categorical columns to enable imputation
    def __init__(self, columns=None):
        self.columns = columns
        self.feature_names_in_ = None
        self.feature_names_out_ = None
        
    def fit(self, X, y=None):
        # Store input feature names
        if hasattr(X, 'columns'):
            self.feature_names_in_ = np.array(X.columns)
        else:
            self.feature_names_in_ = np.array([f"feature_{i}" for i in range(X.shape[1])])
        self.feature_names_out_ = self.feature_names_in_
        return self
        
    def transform(self, X):
        X = X.copy()
        # Replace 'unknown' with np.nan for the specified columns
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
    # Define feature groups
    numeric_features = ['campaign', 'engagement_score']
    categorical_features = ['job', 'marital', 'education', 'poutcome', 'age_group', 'Balance_Tier']
    binary_features = ['housing', 'loan', 'was_contacted']

    # Custom transformer to handle 'unknown' values in categorical features before imputation
    unknown_handler = UnknownHandler(columns=['job', 'education', 'poutcome'])

    # Pipeline for numeric features: median imputation and standard scaling
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline for categorical features:
    # - Replace 'unknown' with NaN
    # - Impute missing values with most frequent category
    # - One-hot encode categories, ignoring unknown categories at transform time
    categorical_transformer = Pipeline(steps=[
        ('unknown_handler', unknown_handler),  
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Pipeline for binary features: encode 'yes'/'no' to 1/0
    binary_transformer = Pipeline(steps=[
        ('encoder', BinaryEncoder())
    ])

    # Combine all transformers into a single ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('bin', binary_transformer, binary_features)
    ])

    return preprocessor
