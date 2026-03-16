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
    print("Model and scaler loaded successfully")
except Exception as e:
    print("Error loading model or scaler:", e)


# ----------------------------
# Health check route
# ----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "DiaPredict ML API is running"
    })


# ----------------------------
# Prediction route
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "features" not in data:
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
# Local development only
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)