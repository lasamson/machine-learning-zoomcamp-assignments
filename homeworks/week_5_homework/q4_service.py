import pickle

from flask import Flask
from flask import request
from flask import jsonify

# File names
model_file = 'model1.bin'
dv_file = 'dv.bin'

# Load model
with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

# Load DictVectorizer
with open(dv_file, 'rb') as f_in:
    dv = pickle.load(f_in)

# Create Flask app
app = Flask('subscription')

# Function to make soft prediction, given an example
def subscription_proba(client):
    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0, 1]
    return y_pred

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()
    y_pred = subscription_proba(client)

    result = {
        'subscription_probability': float(y_pred),
        'subscription': bool(y_pred >= 0.5)
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)