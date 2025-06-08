
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

        # Store feature names that will be output after transform
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
        if self.feature_names_out_ is None:
            raise ValueError("Transformer has not been fitted yet")
        return np.array(self.feature_names_out_)

class BinaryEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.replace({'yes': 1, 'no': 0})
        return X.infer_objects(copy=False)

class UnknownHandler(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        if self.columns:
            for col in self.columns:
                # Replace 'unknown' with np.nan so SimpleImputer can handle it
                X[col] = X[col].replace('unknown', np.nan)
        return X
    

    def get_feature_names_out(self, input_features=None):
        # Return input feature names unchanged
        return np.array(input_features) if input_features is not None else np.array(X.columns)
    

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

