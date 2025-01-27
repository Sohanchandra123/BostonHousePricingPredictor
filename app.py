import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open('regression_model.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')  # Ensure 'home.html' exists in the templates folder

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print("Input data:", data)
    
    # Convert input to numpy array and reshape based on the model's input dimensions
    input_data = np.array(list(data.values())).reshape(1, -1)
    
    # Apply scaling
    new_data = scalar.transform(input_data)
    
    # Predict using the loaded model
    output = model.predict(new_data)
    print("Prediction output:", output[0])
    
    return jsonify(output=output[0].tolist())

if __name__ == "__main__":
    app.run(debug=True)
    print("Starting Flask application...")