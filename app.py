import joblib
import numpy as np
from flask import Flask, request, render_template

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model, encoders, and scaler
model = joblib.load("churn_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
scaler = joblib.load("scaler.pkl")

# Define expected input features (must match form fields in `index.html`)
FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
    'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
]

@app.route('/')
def home():
    """Render the main input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle form submission, process input, and return prediction."""
    try:
        input_data = request.form  # Get user inputs from the form
        processed_input = []

        # Process inputs: Encode categorical & convert numerical features
        for feature in FEATURES:
            value = input_data.get(feature)
            if value is None or value.strip() == "":
                return render_template('result.html', prediction="Invalid input. Please fill all fields.")

            if feature in label_encoders:
                try:
                    value = label_encoders[feature].transform([value])[0]
                except ValueError:
                    return render_template('result.html', prediction=f"Invalid value for {feature}.")
            else:
                value = float(value)  # Convert numeric values

            processed_input.append(value)

        print("Processed Input:", processed_input)  # Debugging log

        # Convert to NumPy array and scale
        input_array = np.array(processed_input).reshape(1, -1)
        input_array = scaler.transform(input_array)

        # Make prediction
        prediction = model.predict(input_array)[0]
        result = "Churn" if prediction == 1 else "No Churn"

        return render_template('result.html', prediction=result)

    except Exception as e:
        return render_template('result.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
