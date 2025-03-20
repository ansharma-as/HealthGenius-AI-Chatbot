import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Load the datasets
train_data = pd.read_csv("training_dataset.csv")  # Replace with the actual training dataset file name
test_data = pd.read_csv("testing_dataset.csv")    # Replace with the actual testing dataset file name

# Preprocess the datasets
X_train = train_data.iloc[:, :-1]  # Symptoms in training data
y_train = train_data.iloc[:, -1]   # Diseases in training data
X_test = test_data.iloc[:, :-1]    # Symptoms in testing data
y_test = test_data.iloc[:, -1]     # Diseases in testing data

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model
model_path = 'new_disease_prediction.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"Model saved as '{model_path}'")
