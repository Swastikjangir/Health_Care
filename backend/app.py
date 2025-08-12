from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import json
import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import DataPreprocessor
from classification_models import ClassificationModels
from clustering import PatientClustering
from recommend import HealthRecommendationEngine

app = FastAPI(
    title="Smart Health Risk Prediction & Recommendation System",
    description="AI-powered health risk assessment and personalized recommendations",
    version="1.0.0"
)

# Add CORS middleware
# FRONTEND_ORIGINS can be a comma-separated list of origins, e.g. "https://your-app.vercel.app,https://*.vercel.app"
frontend_origins_env = os.getenv("FRONTEND_ORIGINS", "*")
allow_origins = [o.strip() for o in frontend_origins_env.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins if allow_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
preprocessor = DataPreprocessor()
classifier = ClassificationModels()
clustering = PatientClustering()
recommendation_engine = HealthRecommendationEngine()

# Pydantic models for request/response
class PatientData(BaseModel):
    age: Optional[float] = None
    gender: Optional[str] = None
    blood_pressure: Optional[float] = None
    glucose: Optional[float] = None
    bmi: Optional[float] = None
    cholesterol: Optional[float] = None
    creatinine: Optional[float] = None
    smoking: Optional[str] = None
    alcohol: Optional[str] = None
    sleep_hours: Optional[float] = None
    stress_level: Optional[str] = None
    exercise_frequency: Optional[str] = None

class PredictionRequest(BaseModel):
    patient_data: PatientData
    model_type: Optional[str] = "random_forest"

class HealthAssessmentResponse(BaseModel):
    risk_level: str
    risk_score: float
    predictions: Dict[str, Any]
    recommendations: Dict[str, Any]
    cluster_info: Optional[Dict[str, Any]] = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Smart Health Risk Prediction & Recommendation System",
        "version": "1.0.0",
        "status": "active"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "components": ["preprocessor", "classifier", "clustering", "recommendations"]}

@app.post("/predict/health-risk", response_model=HealthAssessmentResponse)
async def predict_health_risk(request: PredictionRequest):
    """Predict health risk and generate recommendations"""
    try:
        # Convert patient data to dictionary
        patient_dict = request.patient_data.dict()
        
        # Remove None values and convert to proper types
        cleaned_patient_dict = {}
        for k, v in patient_dict.items():
            if v is not None and v != "":
                # Convert numeric fields to float
                if k in ['age', 'blood_pressure', 'glucose', 'bmi', 'cholesterol', 'creatinine', 'sleep_hours']:
                    try:
                        cleaned_patient_dict[k] = float(v)
                    except (ValueError, TypeError):
                        cleaned_patient_dict[k] = 0.0
                else:
                    cleaned_patient_dict[k] = str(v)
        
        if not cleaned_patient_dict:
            raise HTTPException(status_code=400, detail="No valid patient data provided")
        
        # Ensure required fields have default values
        if 'age' not in cleaned_patient_dict:
            cleaned_patient_dict['age'] = 30.0
        if 'blood_pressure' not in cleaned_patient_dict:
            cleaned_patient_dict['blood_pressure'] = 120.0
        if 'glucose' not in cleaned_patient_dict:
            cleaned_patient_dict['glucose'] = 100.0
        if 'bmi' not in cleaned_patient_dict:
            cleaned_patient_dict['bmi'] = 25.0
        
        # Generate health assessment and recommendations
        personalized_plan = recommendation_engine.create_personalized_plan(cleaned_patient_dict)
        
        # Create response
        response = HealthAssessmentResponse(
            risk_level=personalized_plan['patient_summary']['overall_risk_level'],
            risk_score=personalized_plan['patient_summary']['overall_risk_score'],
            predictions={
                'health_assessment': personalized_plan['patient_summary']['health_assessment'],
                'next_steps': personalized_plan['next_steps'],
                'follow_up_schedule': personalized_plan['follow_up_schedule']
            },
            recommendations=personalized_plan['recommendations']
        )
        
        return response
        
    except Exception as e:
        import traceback
        print(f"Error in predict_health_risk: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/cluster/patients")
async def cluster_patients(patient_data: List[PatientData]):
    """Cluster patients based on health parameters"""
    try:
        if len(patient_data) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 patients for clustering")
        
        # Convert to DataFrame
        df_patients = pd.DataFrame([p.dict() for p in patient_data])
        
        # Basic preprocessing
        numeric_features = ['age', 'blood_pressure', 'glucose', 'bmi', 'cholesterol', 'creatinine']
        available_features = [f for f in numeric_features if f in df_patients.columns]
        
        if len(available_features) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 numeric features for clustering")
        
        # Select features and handle missing values
        X = df_patients[available_features].fillna(df_patients[available_features].mean())
        
        # Initialize clustering
        clustering.initialize_models()
        
        # Find optimal number of clusters
        optimal_clusters = clustering.find_optimal_clusters(X.values)
        n_clusters = optimal_clusters['best_silhouette_k'] if optimal_clusters else 3
        
        # Train clustering models
        results, X_scaled, X_pca = clustering.train_clustering_models(X.values, n_clusters)
        
        # Analyze clusters
        cluster_analysis = clustering.analyze_clusters(X_scaled, results['kmeans']['labels'], available_features)
        
        # Get risk assessment for clusters
        risk_assessment = clustering.get_risk_assessment(cluster_analysis, available_features)
        
        # Create cluster profiles
        cluster_profiles = clustering.create_cluster_profiles(cluster_analysis, available_features)
        
        return {
            "n_clusters": n_clusters,
            "optimal_clusters": optimal_clusters,
            "cluster_analysis": cluster_analysis,
            "risk_assessment": risk_assessment,
            "cluster_profiles": cluster_profiles,
            "best_model": "kmeans",
            "silhouette_score": results['kmeans']['silhouette']
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in clustering: {str(e)}")

@app.post("/upload/dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload dataset for training models"""
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Read the uploaded file
        df = pd.read_csv(file.file)
        
        # Basic dataset info
        dataset_info = {
            "filename": file.filename,
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "sample_data": df.head().to_dict('records')
        }
        
        return {
            "message": "Dataset uploaded successfully",
            "dataset_info": dataset_info
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading dataset: {str(e)}")

@app.get("/models/available")
async def get_available_models():
    """Get list of available ML models"""
    return {
        "classification_models": [
            "logistic_regression",
            "decision_tree", 
            "random_forest",
            "gradient_boosting",
            "svm",
            "naive_bayes",
            "knn",
            "xgboost"
        ],
        "clustering_models": [
            "kmeans",
            "hierarchical",
            "dbscan",
            "gaussian_mixture"
        ]
    }

@app.get("/features/health-parameters")
async def get_health_parameters():
    """Get list of health parameters the system can analyze"""
    return {
        "basic_parameters": [
            "age", "gender", "height", "weight", "bmi"
        ],
        "vital_signs": [
            "blood_pressure", "heart_rate", "temperature", "respiratory_rate"
        ],
        "lab_values": [
            "glucose", "cholesterol", "triglycerides", "creatinine", "hemoglobin"
        ],
        "lifestyle_factors": [
            "smoking", "alcohol", "exercise_frequency", "sleep_hours", "stress_level"
        ],
        "medical_history": [
            "family_history", "previous_diseases", "medications", "allergies"
        ]
    }

@app.get("/recommendations/templates")
async def get_recommendation_templates():
    """Get recommendation templates for different health conditions"""
    return {
        "diabetes": {
            "diet": recommendation_engine.diet_recommendations['diabetes'],
            "exercise": recommendation_engine.exercise_recommendations['diabetes']
        },
        "heart_disease": {
            "diet": recommendation_engine.diet_recommendations['heart_disease'],
            "exercise": recommendation_engine.exercise_recommendations['heart_disease']
        },
        "general_health": {
            "diet": recommendation_engine.diet_recommendations['general_health'],
            "exercise": recommendation_engine.exercise_recommendations['general_fitness']
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
