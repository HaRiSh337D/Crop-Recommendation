from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import sklearn
import os
import pickle
import warnings

app = Flask(__name__)

loaded_model = None  # Define loaded_model outside of any function


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    global loaded_model  # Reference the global variable within the function

    if loaded_model is None:
        # Load the model if it's not already loaded
        loaded_model = pickle.load(open("RF_classifier.pkl", 'rb'))

        print(type(loaded_model))
        if hasattr(loaded_model, 'predict'):
            print("Model has predict method.")
        else:
            print("Model does not have predict method.")

    N = int(request.form['Nitrogen'])
    P = int(request.form['Phosporus'])
    K = int(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['pH'])
    rainfall = float(request.form['Rainfall'])

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    if hasattr(loaded_model, 'predict'):
        # Check if the model has predict method
        prediction = loaded_model.predict(single_pred)

        crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                     8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                     14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                     19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

        if prediction[0] in crop_dict:
            crop = crop_dict[prediction[0]]
            result = "{} is the best crop to be cultivated right there".format(crop)
        else:
            result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    else:
        result = "Error: Loaded model does not have a predict method."

    return render_template('home.html', prediction=result)


if __name__ == '__main__':
    app.run(host="0.0.0.0")



## .\venv\scripts\Activate
## python Deployment.py