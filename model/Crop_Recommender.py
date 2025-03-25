# Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, 
    recall_score, precision_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load Data
data_0 = pd.read_excel("model/Crop_recommendation.xlsx")
data = data_0.copy()

# Data Cleaning
data.replace("?", np.nan, inplace=True)
data.replace(" ?", np.nan, inplace=True)
data.replace("? ", np.nan, inplace=True)

# Replace zero values with the mean for numeric columns
for col in ['N', 'P', 'K', 'temperature', 'humidity', 'ph']:
    data[col].replace(0, data.loc[data[col] != 0, col].mean(), inplace=True)

# Fill missing values
data.fillna(data.mean(), inplace=True)

# Label Encoding
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

# Model Evaluation on Train Set
y_train_pred_RF = RF_classifier.predict(x_train)
print("RF Train Accuracy:", accuracy_score(y_train, y_train_pred_RF))
print("RF Train F1 score:", f1_score(y_train, y_train_pred_RF, average='weighted'))
print("RF Train Precision score:", precision_score(y_train, y_train_pred_RF, average='weighted'))
print("RF Train Recall score:", recall_score(y_train, y_train_pred_RF, average='weighted'))

# Prediction Function
def predict_crop(N, P, K, temperature, humidity, pH, rainfall):
    input_values = np.array([[N, P, K, temperature, humidity, pH, rainfall]])
    prediction = RF_classifier.predict(input_values)
    return prediction[0]

# Example Prediction
N, P, K, tem, humidity, ph, rainfall = 21, 26, 27, 27.003155, 47.675254, 5.699587, 95.851183
pred = predict_crop(N, P, K, tem, humidity, ph, rainfall)

# Crop Prediction
crop_names = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
    6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon",
    11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil",
    16: "Blackgram", 17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
    20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

if pred in crop_names:
    print(f"{crop_names[pred]} is the best crop to be cultivated right there.")
else:
    print("Sorry, we could not determine the best crop to be cultivated with the provided data.")

# Save Model and Mappings
with open("model/RF_classifier.pkl", "wb") as model_file:
    pickle.dump(RF_classifier, model_file)

with open("model/Crop_Mappings.pkl", "wb") as mapping_file:
    pickle.dump(Crop_Mappings, mapping_file)

print("Model and mappings have been successfully saved!")
