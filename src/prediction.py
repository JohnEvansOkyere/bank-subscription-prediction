# Import necessary libraries
import pandas as pd
import joblib
import sys

# Function to load the trained model and feature engineer from a joblib file
def load_model(model_path):
    try:
        data = joblib.load(model_path)
        model = data['model']                     # Extract trained model
        feature_engineer = data['feature_engineer']  # Extract preprocessing pipeline
        return model, feature_engineer
    except Exception as e:
        print(f"Error loading model: {e}")        # Display error if model fails to load
        sys.exit(1)                               # Exit program on failure

# Function to preprocess raw input data using the saved feature engineering pipeline
def prepare_input_data(raw_data, feature_engineer):
    try:
        processed_data = feature_engineer.transform(raw_data)  # Apply preprocessing
        return processed_data
    except Exception as e:
        print(f"Error during feature engineering: {e}")        # Display error if transformation fails
        sys.exit(1)

# Function to make predictions and return both label and probability
def predict_new_data(model_path, new_data):
    model, feature_engineer = load_model(model_path)           # Load model and transformer
    new_data_fe = prepare_input_data(new_data, feature_engineer)  # Preprocess input
    
    prediction = model.predict(new_data_fe)                    # Predict class label
    probability = model.predict_proba(new_data_fe)[:, 1]       # Probability of class 1 (yes)
    
    return prediction, probability

# Entry point for script execution
if __name__ == "__main__":
    # Create new input data as a DataFrame
    # Ensure columns match what your model expects BEFORE feature engineering
    new_data = pd.DataFrame([{
        'age': 40,
        'job': 'admin.',
        'marital': 'married',
        'education': 'secondary',
        'default': 'no',        # even if dropped in train, may still exist
        'balance': 2000,
        'housing': 'yes',
        'loan': 'no',
        'contact': 'cellular',  # dropped in train, but harmless
        'day': 5,               # dropped
        'month': 'may',         # dropped
        'duration': 120,        # dropped
        'campaign': 2,
        'pdays': 10,
        'previous': 2,
        'poutcome': 'success'
    }])
    
    model_path = "src/model/trained_model.joblib"              # Path to saved model
    prediction, prob = predict_new_data(model_path, new_data)  # Get prediction and probability

    # Display prediction results
    print(f"Prediction: {'Subscribed' if prediction[0] == 1 else 'Not Subscribed'}")
    print(f"Probability of Subscription: {prob[0]:.2f}")
