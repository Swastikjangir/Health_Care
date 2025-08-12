import requests
import json

# Test the prediction endpoint
url = "http://localhost:8000/predict/health-risk"
data = {
    "patient_data": {
        "age": 35,
        "blood_pressure": 130,
        "glucose": 110,
        "bmi": 28,
        "gender": "male",
        "smoking": "never_smoker",
        "alcohol": "minimal",
        "sleep_hours": 7,
        "stress_level": "medium",
        "exercise_frequency": "weekly"
    },
    "model_type": "random_forest"
}

try:
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")
