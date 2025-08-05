from flask import Flask, request, jsonify
import pandas as pd
import joblib
import yaml

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_model(model_path):
    try:
        return joblib.load(model_path)
    except Exception as e:
        raise ValueError(f"Failed to load the model: {str(e)}")

def preprocess_input(data):
    try:
        # Perform any necessary preprocessing here
        return pd.DataFrame([data])
    except Exception as e:
        raise ValueError(f"Failed to preprocess input data: {str(e)}")

def make_prediction(model, input_data):
    try:
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)[:, 1]
        return int(prediction[0]), float(prediction_proba[0])
    except Exception as e:
        raise ValueError(f"Failed to make prediction: {str(e)}")

def create_response(prediction, probability):
    return {'prediction': prediction, 'probability': probability}

# Load configuration
config = load_config()

# Load the trained model
model = load_model(config['model']['save_path'])

# Initialize Flask app
app = Flask(__name__)

# Define routes
@app.route('/')
def home():
    return "AI-Based Threat Intelligence Platform"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = preprocess_input(data)
        prediction, probability = make_prediction(model, input_data)
        response = create_response(prediction, probability)
        return jsonify(response)
    except ValueError as ve:
        return str(ve), 400
    except Exception as e:
        return str(e), 500

# Run the app
if __name__ == '__main__':
    app.run(host=config['server']['host'], port=config['server']['port'])

