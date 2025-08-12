import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import joblib
import os

class ClassificationModels:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.model_path = "data/models/"
        
        # Create models directory if it doesn't exist
        os.makedirs(self.model_path, exist_ok=True)
        
    def initialize_models(self):
        """Initialize all classification models"""
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'decision_tree': DecisionTreeClassifier(random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'naive_bayes': GaussianNB(),
            'knn': KNeighborsClassifier(n_neighbors=5),
            'xgboost': xgb.XGBClassifier(random_state=42)
        }
        
    def train_models(self, X, y, test_size=0.2, random_state=42):
        """Train all models and find the best one"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            try:
                # Train the model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'auc': auc,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
                
                print(f"{name}: Accuracy = {accuracy:.4f}, AUC = {auc:.4f}")
                
                # Update best model
                if accuracy > self.best_score:
                    self.best_score = accuracy
                    self.best_model = model
                    
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
        
        return results, X_test, y_test
    
    def hyperparameter_tuning(self, X, y, model_name='random_forest'):
        """Perform hyperparameter tuning for a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Define parameter grids for different models
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            },
            'logistic_regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'svm': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        }
        
        if model_name in param_grids:
            grid_search = GridSearchCV(
                model, param_grids[model_name], 
                cv=5, scoring='accuracy', n_jobs=-1
            )
            grid_search.fit(X, y)
            
            print(f"Best parameters for {model_name}: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
            
            # Update the model with best parameters
            self.models[model_name] = grid_search.best_estimator_
            
            return grid_search.best_estimator_
        
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """Evaluate a specific model"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            'accuracy': accuracy,
            'auc': auc,
            'classification_report': report,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"\n{model_name} Evaluation:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print(f"\nConfusion Matrix:")
        print(cm)
        
        return results
    
    def predict_proba(self, X, model_name=None):
        """Get probability predictions"""
        if model_name:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            model = self.models[model_name]
        else:
            model = self.best_model
            
        if model is None:
            raise ValueError("No model available for prediction")
            
        return model.predict_proba(X)
    
    def predict(self, X, model_name=None):
        """Make predictions"""
        if model_name:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            model = self.models[model_name]
        else:
            model = self.best_model
            
        if model is None:
            raise ValueError("No model available for prediction")
            
        return model.predict(X)
    
    def save_model(self, model, filename):
        """Save a trained model"""
        filepath = os.path.join(self.model_path, filename)
        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filename):
        """Load a trained model"""
        filepath = os.path.join(self.model_path, filename)
        if os.path.exists(filepath):
            model = joblib.load(filepath)
            print(f"Model loaded from {filepath}")
            return model
        else:
            print(f"Model file {filepath} not found")
            return None
    
    def get_feature_importance(self, model_name='random_forest'):
        """Get feature importance for tree-based models"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
            
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        else:
            print(f"Model {model_name} doesn't support feature importance")
            return None
    
    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation on all models"""
        cv_results = {}
        
        for name, model in self.models.items():
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
                cv_results[name] = {
                    'mean_score': scores.mean(),
                    'std_score': scores.std(),
                    'scores': scores
                }
                print(f"{name}: CV Score = {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            except Exception as e:
                print(f"Error in cross-validation for {name}: {str(e)}")
                continue
        
        return cv_results
