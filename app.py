import math
import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Feature specifications in the exact order required by the model
FEATURE_FIELDS = [
    {
        "name": "N",
        "label": "Nitrogen (N)",
        "unit": "ratio / kg/ha",
        "placeholder": "e.g., 90",
        "step": "any",
    },
    {
        "name": "P",
        "label": "Phosphorus (P)",
        "unit": "ratio / kg/ha",
        "placeholder": "e.g., 42",
        "step": "any",
    },
    {
        "name": "K",
        "label": "Potassium (K)",
        "unit": "ratio / kg/ha",
        "placeholder": "e.g., 43",
        "step": "any",
    },
    {
        "name": "temperature",
        "label": "Temperature",
        "unit": "°C",
        "placeholder": "e.g., 20.88",
        "step": "any",
    },
    {
        "name": "humidity",
        "label": "Relative Humidity",
        "unit": "%",
        "placeholder": "e.g., 82.00",
        "step": "any",
    },
    {
        "name": "ph",
        "label": "Soil pH",
        "unit": "pH scale (0-14)",
        "placeholder": "e.g., 6.50",
        "step": "any",
    },
    {
        "name": "rainfall",
        "label": "Rainfall",
        "unit": "mm",
        "placeholder": "e.g., 202.94",
        "step": "any",
    },
]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "crop_recommender.joblib")


def load_model():
    """Load the serialized machine-learning model safely."""
    if not os.path.exists(MODEL_PATH):
        return None, (
            f"Trained model file not found at '{MODEL_PATH}'. "
            "Please ensure 'train_model.py' has been run to generate the model."
        )
    try:
        model = joblib.load(MODEL_PATH)
        return model, None
    except Exception as exc:
        return None, f"An error occurred while loading the model: {str(exc)}"


@app.route("/", methods=["GET", "POST"])
@app.route("/predict", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    # Initialize empty form data dictionary to preserve user input on postback
    form_data = {field["name"]: "" for field in FEATURE_FIELDS}

    model, model_err = load_model()

    if request.method == "POST":
        if model_err:
            error = model_err
            return render_template(
                "index.html",
                fields=FEATURE_FIELDS,
                prediction=None,
                error=error,
                form_data=form_data,
            )

        parsed_values = []
        validation_failed = False

        for field in FEATURE_FIELDS:
            field_name = field["name"]
            field_label = field["label"]
            raw_value = request.form.get(field_name, "").strip()
            form_data[field_name] = raw_value

            if raw_value == "":
                error = f"Missing input for '{field_label}'. Please provide all seven parameters."
                validation_failed = True
                break

            try:
                numeric_val = float(raw_value)
                if math.isnan(numeric_val) or math.isinf(numeric_val):
                    error = (
                        f"Invalid numeric value for '{field_label}'. Value must be a finite number."
                    )
                    validation_failed = True
                    break
                parsed_values.append(numeric_val)
            except ValueError:
                error = f"'{field_label}' must be a valid number. Received: '{raw_value}'."
                validation_failed = True
                break

        if not validation_failed:
            try:
                # Construct DataFrame with exact column names and feature order
                input_df = pd.DataFrame(
                    [parsed_values],
                    columns=[f["name"] for f in FEATURE_FIELDS],
                )
                pred = model.predict(input_df)[0]
                prediction = str(pred)
            except Exception as exc:
                error = f"Model prediction error: {str(exc)}"

    return render_template(
        "index.html",
        fields=FEATURE_FIELDS,
        prediction=prediction,
        error=error,
        form_data=form_data,
    )


if __name__ == "__main__":
    # Debug mode is strictly disabled for production security
    app.run(host="127.0.0.1", port=5000, debug=False)
