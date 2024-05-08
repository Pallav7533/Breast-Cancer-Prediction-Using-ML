import pickle
import numpy as np
from flask import Flask, request, render_template
import os

app = Flask(__name__)


with open("lr.pkl", "rb") as file:
    lr_model = pickle.load(file)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    
    features = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean',
        'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se',
        'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
        'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]

   
    feature_values = [float(request.form.get(feature, 0)) for feature in features]

    features_array = np.array(feature_values).reshape(1, -1)

    
    prediction = lr_model.predict(features_array)[0]
    prediction_prob = lr_model.predict_proba(features_array)[0][1]

    prediction_text = "Malignant" if prediction == 1 else "Benign"
    prediction_confidence = round(prediction_prob * 100, 2)

    return render_template("index.html", prediction=prediction_text, confidence=prediction_confidence)


if __name__ == "__main__":
    app.run(debug=True)
