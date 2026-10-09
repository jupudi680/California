from flask import Flask, request
import pandas as pd
import joblib

# Load model
obj = joblib.load("california.joblib")

model = obj["model"]
columns = obj["columns"]

app = Flask(__name__)


@app.route('/')
def Main():
    return "Welcome to California Housing Prediction API"


@app.route('/predict')
def predict():

    input_data = []

    for i in columns:

        val = request.args.get(i)

        if val is None:
            return f"Missing value for {i}"

        input_data.append(float(val))

    # Convert input into DataFrame
    input_df = pd.DataFrame(
        [input_data],
        columns=columns
    )

    # Prediction
    prediction = model.predict(input_df)

    return {
        "prediction": float(prediction[0])
    }


if __name__ == "__main__":
    app.run(debug=True)