from src.recommend import HealthRecommendationEngine

# Test the recommendation engine
engine = HealthRecommendationEngine()

# Test data
patient_data = {
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
}

try:
    # Test health assessment
    health_assessment = engine.assess_health_parameters(patient_data)
    print("Health Assessment:", health_assessment)
    
    # Test recommendations
    recommendations = engine.generate_recommendations(patient_data, health_assessment)
    print("Recommendations:", recommendations)
    
    # Test personalized plan
    personalized_plan = engine.create_personalized_plan(patient_data)
    print("Personalized Plan:", personalized_plan)
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
