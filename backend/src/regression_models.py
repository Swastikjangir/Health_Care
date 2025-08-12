import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import os

class RegressionModels:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.scaler = StandardScaler()
        self.model_path = "data/models/"
        
        # Create models directory if it doesn't exist
        os.makedirs(self.model_path, exist_ok=True)
        
    def initialize_models(self):
        """Initialize all regression models"""
        self.models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(random_state=42),
            'lasso': Lasso(random_state=42),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(random_state=42),
            'svr': SVR(),
            'mlp': MLPRegressor(random_state=42, max_iter=1000),
            'xgboost': xgb.XGBRegressor(random_state=42)
        }
        
    def train_models(self, X, y, test_size=0.2, random_state=42):
        """Train all models and find the best one"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            try:
                # Train the model
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results[name] = {
                    'model': model,
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'y_pred': y_pred
                }
                
                print(f"{name}: R² = {r2:.4f}, RMSE = {rmse:.4f}")
                
                # Update best model (based on R² score)
                if r2 > self.best_score:
                    self.best_score = r2
                    self.best_model = model
                    
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
        
        return results, X_test_scaled, y_test
    
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
            'ridge': {
                'alpha': [0.1, 1, 10, 100]
            },
            'lasso': {
                'alpha': [0.1, 1, 10, 100]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9]
            }
        }
        
        if model_name in param_grids:
            grid_search = GridSearchCV(
                model, param_grids[model_name], 
                cv=5, scoring='r2', n_jobs=-1
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
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'y_pred': y_pred
        }
        
        print(f"\n{model_name} Evaluation:")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        return results
    
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
        
        # Scale features if scaler is fitted
        if hasattr(self.scaler, 'mean_'):
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
            
        return model.predict(X_scaled)
    
    def get_feature_importance(self, model_name='random_forest', feature_names=None):
        """Get feature importance for tree-based models"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
            
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            if feature_names is None:
                feature_names = [f'Feature_{i}' for i in range(len(importances))]
            
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return feature_importance_df
        else:
            print(f"Model {model_name} doesn't support feature importance")
            return None
    
    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation on all models"""
        cv_results = {}
        
        for name, model in self.models.items():
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
                cv_results[name] = {
                    'mean_score': scores.mean(),
                    'std_score': scores.std(),
                    'scores': scores
                }
                print(f"{name}: CV R² Score = {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            except Exception as e:
                print(f"Error in cross-validation for {name}: {str(e)}")
                continue
        
        return cv_results
    
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
    
    def create_prediction_interval(self, X, model, confidence=0.95):
        """Create prediction intervals for regression"""
        predictions = model.predict(X)
        
        # For tree-based models, we can use the variance of predictions
        if hasattr(model, 'estimators_'):
            # Bootstrap predictions for ensemble methods
            bootstrap_predictions = []
            for estimator in model.estimators_:
                bootstrap_predictions.append(estimator.predict(X))
            
            bootstrap_predictions = np.array(bootstrap_predictions)
            
            # Calculate percentiles for confidence interval
            alpha = 1 - confidence
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            lower_bound = np.percentile(bootstrap_predictions, lower_percentile, axis=0)
            upper_bound = np.percentile(bootstrap_predictions, upper_percentile, axis=0)
            
            return {
                'predictions': predictions,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'confidence': confidence
            }
        else:
            # For linear models, use standard error
            print("Prediction intervals only available for tree-based models")
            return None
    
    def analyze_residuals(self, y_true, y_pred):
        """Analyze model residuals"""
        residuals = y_true - y_pred
        
        residual_analysis = {
            'residuals': residuals,
            'mean_residual': np.mean(residuals),
            'std_residual': np.std(residuals),
            'min_residual': np.min(residuals),
            'max_residual': np.max(residuals),
            'residual_range': np.max(residuals) - np.min(residuals)
        }
        
        # Check for patterns in residuals
        residual_analysis['normality_check'] = np.abs(np.mean(residuals)) < 0.1 * np.std(residuals)
        
        return residual_analysis
