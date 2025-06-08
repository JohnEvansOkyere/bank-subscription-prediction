import pandas as pd
import joblib
import sys

def load_model(model_path):
    try:
        model, feature_engineer = joblib.load(model_path)
        return model, feature_engineer
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def prepare_input_data(raw_data, feature_engineer):
    try:
        processed_data = feature_engineer.transform(raw_data)
        return processed_data
    except Exception as e:
        print(f"Error during feature engineering: {e}")
        sys.exit(1)

def predict_new_data(model_path, new_data):
    model, feature_engineer = load_model(model_path)
    new_data_fe = prepare_input_data(new_data, feature_engineer)
    
    prediction = model.predict(new_data_fe)
    probability = model.predict_proba(new_data_fe)[:, 1]  # Probability of class 1 (yes)
    
    return prediction, probability

# Example usage
if __name__ == "__main__":
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
    
    model_path = "src/model/trained_model.joblib"
    prediction, prob = predict_new_data(model_path, new_data)

    print(f"Prediction: {'Subscribed' if prediction[0] == 1 else 'Not Subscribed'}")
    print(f"Probability of Subscription: {prob[0]:.2f}")
