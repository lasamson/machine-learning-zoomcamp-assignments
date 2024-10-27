import pickle

# File names
model_file = 'model1.bin'
dv_file = 'dv.bin'

# Load model
with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

# Load DictVectorizer
with open(dv_file, 'rb') as f_in:
    dv = pickle.load(f_in)

# Transform example to get feature matrix
client = {"job": "management", "duration": 400, "poutcome": "success"}
X = dv.transform([client])

# Use model to predict on this example
y_pred = model.predict_proba(X)[0, 1]

# Output P(Y=1|X)
print(f'Probability that client gets subscription: {y_pred}')