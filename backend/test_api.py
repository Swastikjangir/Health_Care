#!/usr/bin/env python3
"""
Test script to debug the backend API endpoint
"""

import requests
import json

# Test the health prediction endpoint
def test_health_prediction():
    url = "https://health-care-aec6.onrender.com/predict/health-risk"
    
    # Sample patient data
    test_data = {
        "patient_data": {
            "age": 35.0,
            "gender": "male",
            "blood_pressure": 120.0,
            "glucose": 100.0,
            "bmi": 25.0,
            "cholesterol": 180.0,
            "creatinine": 1.0,
            "smoking": "no",
            "alcohol": "no",
            "sleep_hours": 7.0,
            "stress_level": "low",
            "exercise_frequency": "moderate"
        },
        "model_type": "random_forest"
    }
    
    try:
        print("Testing health prediction endpoint...")
        print(f"URL: {url}")
        print(f"Data: {json.dumps(test_data, indent=2)}")
        
        response = requests.post(url, json=test_data)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            print("✅ Success!")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        else:
            print("❌ Error!")
            print(f"Response Text: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

# Test the health check endpoint
def test_health_check():
    url = "https://health-care-aec6.onrender.com/health"
    
    try:
        print("\nTesting health check endpoint...")
        response = requests.get(url)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    print("🔍 Backend API Debug Test")
    print("=" * 50)
    
    test_health_check()
    test_health_prediction()
