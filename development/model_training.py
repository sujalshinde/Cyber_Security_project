# model_training.py

from sklearn.ensemble import RandomForestClassifier
from config_loader import load_config
import joblib

config = load_config()

model_type = config['model']['type']
model_params = config['model']['parameters']

# Choose model
if model_type == 'RandomForest':
    model = RandomForestClassifier(**model_params)

# Train model
model.fit(X_train, y_train)

# Save model
joblib.dump(model, '../models/saved_model.pkl')

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import yaml


def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config()


# Load data
data_path = config['data']['path']
data = pd.read_csv(f'{data_path}/data.csv')

# Data preprocessing
fillna_method = config['data']['preprocessing']['fillna']
data.fillna(method=fillna_method, inplace=True)

# Feature and target split
X = data.drop('target', axis=1)
y = data['target']

# Train-test split
test_size = config['data']['split']['test_size']
random_state = config['data']['split']['random_state']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Scaling
scaling_method = config['data']['preprocessing']['scaling']
if scaling_method == 'StandardScaler':
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


# Model parameters from config
model_type = config['model']['type']
model_params = config['model']['parameters']

# Initialize model
if model_type == 'RandomForest':
    model = RandomForestClassifier(**model_params)
else:
    raise ValueError(f"Model type '{model_type}' is not supported.")

# Train model
model.fit(X_train, y_train)


# Predictions on test set
y_pred = model.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)


# Save the trained model
model_save_path = config['model']['save_path']
joblib.dump(model, model_save_path)
print(f"Model saved to {model_save_path}")

























