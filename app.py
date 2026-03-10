from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# ----------------------------
# Load ML model and scaler
# ----------------------------
try:
    model = pickle.load(open("diabetes_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
except Exception as e:
    print("Error loading model or scaler:", e)

# ----------------------------
# # Home route (GET)
# # ----------------------------
# @app.route("/", methods=["GET"])
# def home():
#     return jsonify({
#         "message": "Welcome to the Diabetes Prediction API! Use POST /predict with features to get prediction."
#     })


# ----------------------------
# Prediction route (POST)
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if "features" not in data:
            return jsonify({"message": "Missing 'features' in request body"}), 400

        features = np.array(data["features"]).reshape(1, -1)

        # Feature scaling
        features_scaled = scaler.transform(features)

        # Model prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "risk_percentage": round(float(probability * 100), 2)
        })

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"message": "Prediction failed"}), 500


# ----------------------------
# Run Flask
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5001)  # Flask port 5001, Node backend port 5000