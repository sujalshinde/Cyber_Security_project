import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the network logs data
data = pd.read_csv('network_logs.csv')

# Feature engineering (assuming you have extracted relevant features)
# For simplicity, let's assume you've already done feature engineering and have features ready for modeling.

# Define features and target variable
X = data.drop('action', axis=1)  # Assuming 'action' is the target variable
y = data['action']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, predictions))
