import os
# Required Libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, 
    recall_score, precision_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier

# Load Data
data_0 = pd.read_excel("model/Crop_recommendation.xlsx")
data = data_0.copy()

# Data Cleaning
data.replace(["?", " ?", "? "], np.nan, inplace=True)

# Ensure numeric columns are properly processed
numeric_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')  # Convert to numeric, set errors as NaN
    data.loc[data[col] == 0, col] = data[col].mean()  # Replace 0 with mean only for numeric columns

# Fill missing values for numeric columns only
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Label Encoding for Crop Names
if 'label' in data.columns:
    Lc = LabelEncoder()
    data['label'] = Lc.fit_transform(data['label'])
    Crop_Mappings = dict(zip(Lc.classes_, Lc.transform(Lc.classes_)))
    print('Crop_Mappings: \n', Crop_Mappings)

# Features and Target
x = data.drop('label', axis=1)
y = data['label']

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

# Random Forest Classifier
RF_classifier = RandomForestClassifier(n_estimators=900, criterion='gini', random_state=0)
RF_classifier.fit(x_train, y_train)

# Model Evaluation on Test Set
y_pred_RF = RF_classifier.predict(x_test)
print("RF Test Accuracy:", accuracy_score(y_test, y_pred_RF))
print("RF Test F1 score:", f1_score(y_test, y_pred_RF, average='weighted'))
print("RF Test Precision score:", precision_score(y_test, y_pred_RF, average='weighted'))
print("RF Test Recall score:", recall_score(y_test, y_pred_RF, average='weighted'))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_RF))
print("Classification Report:\n", classification_report(y_test, y_pred_RF))

# Check and create the directory if it doesn't exist
os.makedirs("model", exist_ok=True)

print("✅ Saving model to:", os.path.abspath("model/RF_classifier.pkl"))
print("✅ Saving mappings to:", os.path.abspath("model/Crop_Mappings.pkl"))

# Save Model and Mappings
# Save the classifier model
with open("model/RF_classifier.pkl", "wb") as model_file:
    pickle.dump(RF_classifier, model_file)

# Save the label mappings
with open("model/Crop_Mappings.pkl", "wb") as mappings_file:
    pickle.dump(Crop_Mappings, mappings_file)

print("✅ Model and mappings have been successfully saved!")
