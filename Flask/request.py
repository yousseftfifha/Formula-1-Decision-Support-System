import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url, json={'Code': 0, 'Nationality': 4, 'points': 170, 'age': 30})

print(r.json())
