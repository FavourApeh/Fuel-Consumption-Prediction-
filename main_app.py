from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load saved models and preprocessing artifacts
lr_model = joblib.load("linear_regression_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

# Numeric columns used in scaling
numeric_columns = ['Engine Size(L)', 'Cylinders', 'CO2 Emissions(g/km)']

@app.route("/")
def home():
    """
    Welcome endpoint to confirm that the API is running.
    """
    return jsonify({"message": "Fuel Consumption Prediction API is running!"})


def validate_input(data):
    """
    Validate input payload to ensure all required fields are present and valid.
    """
    required_fields = ['Engine Size(L)', 'Cylinders', 'CO2 Emissions(g/km)', 'Vehicle Class', 'Fuel Type', 'Transmission']
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise ValueError(f"Missing fields in input: {missing_fields}")
    return True


def preprocess_input(input_data):
    """
    Preprocess input data for prediction.
    """
    # Create a DataFrame for input
    input_df = pd.DataFrame([input_data])

    # Encode categorical features
    encoded_array = encoder.transform(input_df[['Vehicle Class', 'Fuel Type', 'Transmission']])
    encoded_columns = encoder.get_feature_names_out(['Vehicle Class', 'Fuel Type', 'Transmission'])
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_columns)

    # Combine numeric and encoded features
    input_features = pd.concat([input_df[numeric_columns], encoded_df], axis=1)

    # Scale numeric features
    input_features[numeric_columns] = scaler.transform(input_features[numeric_columns])
    return input_features


@app.route('/predict_lr', methods=['POST'])
def predict_lr():
    """
    Predict fuel consumption using Linear Regression.
    """
    try:
        input_data = request.json
        validate_input(input_data)
        input_features = preprocess_input(input_data)
        prediction = lr_model.predict(input_features)[0]
        return jsonify({"Linear Regression Prediction": prediction})
    except Exception as e:
        logging.error(f"Error during Linear Regression prediction: {e}")
        return jsonify({"error": str(e)}), 400


@app.route('/predict_rf', methods=['POST'])
def predict_rf():
    """
    Predict fuel consumption using Random Forest Regressor.
    """
    try:
        input_data = request.json
        validate_input(input_data)
        input_features = preprocess_input(input_data)
        prediction = rf_model.predict(input_features)[0]
        return jsonify({"Random Forest Prediction": prediction})
    except Exception as e:
        logging.error(f"Error during Random Forest prediction: {e}")
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
