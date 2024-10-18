from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the dataset 
dataset = pd.read_csv("C:\\Users\\HP\\Tennis\\Tennis_dataset.csv")

# Encode categorical data (Outlook, Temp, Humidity, Wind) to numerical values
label_encoders = {}
categorical_columns = ['Outlook', 'Temp', 'Humidity', 'Wind', 'Tennis']
for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    dataset[col] = label_encoders[col].fit_transform(dataset[col])

# Define features (X) and target (y)
X = dataset.drop(columns=['Tennis', 'Day'])
y = dataset['Tennis']

# Create a Decision Tree Classifier
clf = DecisionTreeClassifier()
clf.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    outlook = request.form['outlook']
    temp = request.form['temp']
    humidity = request.form['humidity']
    wind = request.form['wind']

    # Encode user input
    outlook_encoded = label_encoders['Outlook'].transform([outlook])[0]
    temp_encoded = label_encoders['Temp'].transform([temp])[0]
    humidity_encoded = label_encoders['Humidity'].transform([humidity])[0]
    wind_encoded = label_encoders['Wind'].transform([wind])[0]

    user_input = [[outlook_encoded, temp_encoded, humidity_encoded, wind_encoded]]

    # Make predictions
    prediction = clf.predict(user_input)[0]
    predicted_result = label_encoders['Tennis'].inverse_transform([prediction])[0]

    return jsonify({'result': predicted_result})

if __name__ == '__main__':
    app.run(debug=True)
