# data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from config_loader import load_config

config = load_config()

data_path = config['data']['path']
fillna_method = config['data']['preprocessing']['fillna']
scaling_method = config['data']['preprocessing']['scaling']
test_size = config['data']['split']['test_size']
random_state = config['data']['split']['random_state']

# Load data
data = pd.read_csv(f'{data_path}/data.csv')

# Data preprocessing
data.fillna(method=fillna_method, inplace=True)

# Feature and target split
X = data.drop('target', axis=1)
y = data['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Scaling
if scaling_method == 'StandardScaler':
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


