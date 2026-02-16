import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# ===============================
# Load Trained Models
# ===============================
try:
    fault_type_model = joblib.load("fault_type_classifier.joblib")
    fault_location_model = joblib.load("fault_location_regressor.joblib")
    scaler = joblib.load("scaler.joblib")
    label_encoder = joblib.load("label_encoder.joblib")
    print("✅ All models loaded successfully!")
except Exception as e:
    print("❌ Error loading models:", e)
    fault_type_model = None


# ===============================
# Home Page
# ===============================
@app.route("/")
def home():
    return render_template("index.html")


# ===============================
# Prediction Route
# ===============================
@app.route("/predict", methods=["POST"])
def predict():

    if fault_type_model is None:
        return render_template("index.html",
                               error_message="Model files not found. Please check joblib files.")

    try:
        # Get user input
        voltage = float(request.form["voltage_kv"])
        conductor_csa = float(request.form["conductor_csa"])
        cable_length = float(request.form["cable_length"])
        fault_resistance = float(request.form["fault_resistance"])

        # Create dataframe (must match training feature order)
        input_df = pd.DataFrame([[
            voltage,
            conductor_csa,
            cable_length,
            fault_resistance,
            0.0  # Line Delay placeholder
        ]],
        columns=[
            "Voltage_kV_Numeric",
            "Conductor CSA/mmsq",
            "Cable Length/km",
            "Fault resistance/Ohms",
            "Line Delay/ms"
        ])

        # Scale input
        scaled_input = scaler.transform(input_df)

        # Predict Fault Type
        encoded_fault = fault_type_model.predict(scaled_input)
        fault_name = label_encoder.inverse_transform(encoded_fault)[0]

        # Predict Fault Location
        fault_location = fault_location_model.predict(scaled_input)[0]

        # Output Formatting
        if fault_name.lower() == "no fault":
            result_type = "✅ System Normal (No Fault Detected)"
            result_location = "-"
        else:
            result_type = f"⚠ Fault Type: {fault_name}"
            result_location = f"{fault_location:.2f} km"

        return render_template("index.html",
                               predicted_fault_type=result_type,
                               predicted_fault_location=result_location)

    except ValueError:
        return render_template("index.html",
                               error_message="Please enter valid numeric values.")

    except Exception as e:
        return render_template("index.html",
                               error_message=f"Prediction Error: {str(e)}")


# ===============================
# Run Application
# ===============================
if __name__ == "__main__":
    app.run(debug=True)
