import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
import yaml
import os

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config()

# Load processed data
processed_data_path = config['data']['processed_path']
X_test = pd.read_csv(os.path.join(processed_data_path, 'X_test.csv'))
y_test = pd.read_csv(os.path.join(processed_data_path, 'y_test.csv'))

# Load the trained model
model_save_path = config['model']['save_path']
model = joblib.load(model_save_path)

# Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Assuming binary classification for ROC AUC

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC AUC: {roc_auc}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Save evaluation metrics (optional)
evaluation_results = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'roc_auc': roc_auc,
    'confusion_matrix': conf_matrix.tolist(),
    'classification_report': class_report
}

evaluation_results_path = config['evaluation']['results_path']
if not os.path.exists(evaluation_results_path):
    os.makedirs(evaluation_results_path)

with open(os.path.join(evaluation_results_path, 'evaluation_results.yaml'), 'w') as file:
    yaml.dump(evaluation_results, file)

