from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model, scaler, and threshold
model = load_model("diabetes_model.h5")
scaler = joblib.load("scaler.pkl")
optimal_threshold = joblib.load("optimal_threshold.pkl")

# Initialize Flask app
app = Flask(__name__)

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract input values from form
        features = [float(x) for x in request.form.values()]
        final_features = np.array(features).reshape(1, -1)

        # Define feature names
        feature_names = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]
        final_features_df = pd.DataFrame(final_features, columns=feature_names)

        # Feature Engineering
        final_features_df["Glucose_BMI_Ratio"] = final_features_df["Glucose"] / final_features_df["BMI"].replace(0, np.nan)
        final_features_df["Age_Glucose_Ratio"] = final_features_df["Age"] / final_features_df["Glucose"].replace(0, np.nan)
        final_features_df["BloodPressure_BMI_Ratio"] = final_features_df["BloodPressure"] / final_features_df["BMI"].replace(0, np.nan)
        final_features_df["Insulin_Glucose_Ratio"] = final_features_df["Insulin"] / final_features_df["Glucose"].replace(0, np.nan)

        # Handle division by zero
        final_features_df.fillna(0, inplace=True)

        # Scale features
        final_features_scaled = scaler.transform(final_features_df)

        # Make prediction
        prediction_prob = model.predict(final_features_scaled)[0][0]
        
        # Print raw probability for debugging
        print(f"Raw Prediction Probability: {prediction_prob}")        
        print(f"Optimal Threshold Used: {optimal_threshold}")


        # Define a borderline range
        borderline_range = 0.1
        lower_threshold = optimal_threshold - borderline_range
        upper_threshold = optimal_threshold + borderline_range

        # Determine the result and confidence
        if lower_threshold <= prediction_prob <= upper_threshold:
            result = "Borderline (Uncertain)"
            confidence = 0.5  # Set a lower confidence for borderline cases
        else:
            result = "Diabetic" if prediction_prob > optimal_threshold else "Not Diabetic"
            confidence = max(prediction_prob, 1 - prediction_prob)

        # Output message
        output_text = f"Prediction: {result} (Confidence: {confidence:.2%})"

        return render_template("index.html", prediction_text=output_text)

    except Exception as e:
        print("Error:", e)
        return render_template("index.html", prediction_text="Error in processing data!")

# Run app
if __name__ == "__main__":
    app.run(debug=True)