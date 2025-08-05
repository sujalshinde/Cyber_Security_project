import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import yaml
import os

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config()

# Load data
data_path = config['data']['path']
data = pd.read_csv(os.path.join(data_path, 'data.csv'))

# Create new features
data['login_success_ratio'] = data['successful_logins'] / (data['successful_logins'] + data['failed_logins'] + 1)
data['bytes_ratio'] = data['bytes_sent'] / (data['bytes_sent'] + data['bytes_received'] + 1)

# Handle infinity values
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Handle missing values
imputer = SimpleImputer(strategy=config['data']['preprocessing']['fillna'])
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Encode categorical variables
categorical_features = config['data']['categorical_features']
encoder = OneHotEncoder(drop='first', sparse=False)
encoded_categorical = pd.DataFrame(encoder.fit_transform(data_imputed[categorical_features]), columns=encoder.get_feature_names_out(categorical_features))

# Drop original categorical columns and concatenate encoded ones
data_imputed.drop(categorical_features, axis=1, inplace=True)
data_final = pd.concat([data_imputed, encoded_categorical], axis=1)

# Scale numerical features
numerical_features = data_final.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
data_final[numerical_features] = scaler.fit_transform(data_final[numerical_features])

# Split data into features and target
X = data_final.drop('target', axis=1)  # Replace 'target' with the actual target column name
y = data_final['target']

# Save processed data (optional)
processed_data_path = config['data']['processed_path']
if not os.path.exists(processed_data_path):
    os.makedirs(processed_data_path)

X.to_csv(os.path.join(processed_data_path, 'X_processed.csv'), index=False)
y.to_csv(os.path.join(processed_data_path, 'y_processed.csv'), index=False)




