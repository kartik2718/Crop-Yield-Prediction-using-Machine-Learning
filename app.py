from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib

# Load the Decision Tree Regressor model
dtr = joblib.load('dtr.pkl')

# Load the preprocessor
preprocessor = joblib.load('preprocessor.pkl')

# Create Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        Area = request.form['Area']
        Item = request.form['Item']
        Year = request.form['Year']
        average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
        pesticides_tonnes = request.form['pesticides_tonnes']
        avg_temp = request.form['avg_temp']

        features = np.array([[Area, Item, Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp]])
        transformed_features = preprocessor.transform(features)
        predicted_value = dtr.predict(transformed_features).reshape(1, -1)

        return render_template('index.html', predicted_value=predicted_value)

# Python main
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
