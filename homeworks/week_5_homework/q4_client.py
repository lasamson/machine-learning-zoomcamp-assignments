import requests

# URL
url = 'http://localhost:9696/predict'

# New client
client = {"job": "student", "duration": 280, "poutcome": "failure"}

# Get the body of the response as a JSON object
response = requests.post(url, json=client).json()
print(response)

# Output the decision
if response['subscription']:
    print('The client is likely to subscribe.')
else:
    print('The client is not likely to subscribe.')