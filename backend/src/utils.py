import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class HealthDataVisualizer:
    """Class for creating health data visualizations"""
    
    def __init__(self):
        self.color_palette = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
        
    def create_risk_dashboard(self, patient_data: Dict[str, Any], 
                            health_assessment: Dict[str, Any]) -> go.Figure:
        """Create a comprehensive risk dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Overall Risk Score', 'Risk Factors', 'Health Parameters', 'Recommendations'),
            specs=[[{"type": "indicator"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Overall Risk Score (Gauge Chart)
        risk_score = health_assessment.get('overall_risk_score', 0)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=risk_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Risk Score (%)"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ),
            row=1, col=1
        )
        
        # Risk Factors Bar Chart
        risk_factors = []
        risk_scores = []
        for condition, assessment in health_assessment.get('health_assessment', {}).items():
            risk_factors.append(condition.replace('_', ' ').title())
            risk_scores.append(assessment.get('score', 0))
        
        fig.add_trace(
            go.Bar(
                x=risk_factors,
                y=risk_scores,
                marker_color=['red' if score >= 2 else 'orange' if score >= 1 else 'green' 
                            for score in risk_scores],
                name="Risk Scores"
            ),
            row=1, col=2
        )
        
        # Health Parameters Scatter Plot
        numeric_params = {k: v for k, v in patient_data.items() 
                         if isinstance(v, (int, float)) and v is not None}
        
        if len(numeric_params) >= 2:
            param_names = list(numeric_params.keys())
            param_values = list(numeric_params.values())
            
            fig.add_trace(
                go.Scatter(
                    x=param_names,
                    y=param_values,
                    mode='markers+lines',
                    marker=dict(size=10, color='blue'),
                    name="Health Parameters"
                ),
                row=2, col=1
            )
        
        # Recommendations Summary
        recommendations = health_assessment.get('recommendations', {})
        rec_categories = list(recommendations.keys())
        rec_counts = [len(recs) if isinstance(recs, list) else 0 
                     for recs in recommendations.values()]
        
        fig.add_trace(
            go.Bar(
                x=rec_categories,
                y=rec_counts,
                marker_color='lightblue',
                name="Recommendation Count"
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Health Risk Assessment Dashboard",
            height=800,
            showlegend=False
        )
        
        return fig
    
    def create_clustering_visualization(self, X_pca: np.ndarray, labels: np.ndarray, 
                                      cluster_analysis: Dict[str, Any]) -> go.Figure:
        """Create clustering visualization"""
        # Create DataFrame for visualization
        df_viz = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'Cluster': labels
        })
        
        # Create scatter plot
        fig = px.scatter(
            df_viz, 
            x='PC1', 
            y='PC2', 
            color='Cluster',
            title='Patient Clustering Results',
            labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        
        # Add cluster centers if available
        if 'cluster_centers' in cluster_analysis:
            centers = cluster_analysis['cluster_centers']
            fig.add_trace(
                go.Scatter(
                    x=centers[:, 0],
                    y=centers[:, 1],
                    mode='markers',
                    marker=dict(symbol='x', size=15, color='black'),
                    name='Cluster Centers'
                )
            )
        
        fig.update_layout(
            width=800,
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_feature_importance_chart(self, feature_importance: pd.DataFrame) -> go.Figure:
        """Create feature importance visualization"""
        fig = px.bar(
            feature_importance,
            x='importance',
            y='feature',
            orientation='h',
            title='Feature Importance Analysis',
            labels={'importance': 'Importance Score', 'feature': 'Features'},
            color='importance',
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            width=800,
            height=600,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    
    def create_correlation_heatmap(self, data: pd.DataFrame, 
                                 method: str = 'pearson') -> go.Figure:
        """Create correlation heatmap"""
        # Calculate correlation matrix
        corr_matrix = data.corr(method=method)
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            title=f'Feature Correlation Matrix ({method.title()})',
            labels=dict(x='Features', y='Features', color='Correlation'),
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        
        # Add correlation values as text
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                fig.add_annotation(
                    x=i, y=j,
                    text=f"{corr_matrix.iloc[i, j]:.2f}",
                    showarrow=False,
                    font=dict(color="black" if abs(corr_matrix.iloc[i, j]) < 0.5 else "white")
                )
        
        fig.update_layout(
            width=800,
            height=600
        )
        
        return fig

class DataProcessor:
    """Class for data processing utilities"""
    
    @staticmethod
    def validate_health_data(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate health data for required fields and ranges"""
        errors = []
        
        # Required fields
        required_fields = ['age']
        for field in required_fields:
            if field not in data or data[field] is None:
                errors.append(f"Missing required field: {field}")
        
        # Age validation
        if 'age' in data and data['age'] is not None:
            if not (0 <= data['age'] <= 150):
                errors.append("Age must be between 0 and 150")
        
        # Blood pressure validation
        if 'blood_pressure' in data and data['blood_pressure'] is not None:
            if not (50 <= data['blood_pressure'] <= 300):
                errors.append("Blood pressure must be between 50 and 300 mmHg")
        
        # Glucose validation
        if 'glucose' in data and data['glucose'] is not None:
            if not (20 <= data['glucose'] <= 1000):
                errors.append("Glucose must be between 20 and 1000 mg/dL")
        
        # BMI validation
        if 'bmi' in data and data['bmi'] is not None:
            if not (10 <= data['bmi'] <= 100):
                errors.append("BMI must be between 10 and 100")
        
        # Cholesterol validation
        if 'cholesterol' in data and data['cholesterol'] is not None:
            if not (50 <= data['cholesterol'] <= 1000):
                errors.append("Cholesterol must be between 50 and 1000 mg/dL")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def normalize_health_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize health data to standard ranges"""
        normalized = data.copy()
        
        # Normalize age to 0-1 range (assuming max age is 120)
        if 'age' in normalized and normalized['age'] is not None:
            normalized['age_normalized'] = normalized['age'] / 120
        
        # Normalize blood pressure (assuming normal range 90-140)
        if 'blood_pressure' in normalized and normalized['blood_pressure'] is not None:
            normalized['bp_normalized'] = (normalized['blood_pressure'] - 90) / (140 - 90)
        
        # Normalize glucose (assuming normal range 70-140)
        if 'glucose' in normalized and normalized['glucose'] is not None:
            normalized['glucose_normalized'] = (normalized['glucose'] - 70) / (140 - 70)
        
        # Normalize BMI (assuming normal range 18.5-30)
        if 'bmi' in normalized and normalized['bmi'] is not None:
            normalized['bmi_normalized'] = (normalized['bmi'] - 18.5) / (30 - 18.5)
        
        return normalized
    
    @staticmethod
    def calculate_health_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate additional health metrics"""
        metrics = {}
        
        # Calculate BMI if height and weight are provided
        if 'height' in data and 'weight' in data:
            if data['height'] and data['weight']:
                height_m = data['height'] / 100  # Convert cm to meters
                bmi = data['weight'] / (height_m ** 2)
                metrics['calculated_bmi'] = round(bmi, 2)
                
                # BMI category
                if bmi < 18.5:
                    metrics['bmi_category'] = 'Underweight'
                elif bmi < 25:
                    metrics['bmi_category'] = 'Normal weight'
                elif bmi < 30:
                    metrics['bmi_category'] = 'Overweight'
                else:
                    metrics['bmi_category'] = 'Obese'
        
        # Calculate blood pressure category
        if 'blood_pressure' in data and data['blood_pressure']:
            bp = data['blood_pressure']
            if bp < 90:
                metrics['bp_category'] = 'Low'
            elif bp < 120:
                metrics['bp_category'] = 'Normal'
            elif bp < 140:
                metrics['bp_category'] = 'Elevated'
            else:
                metrics['bp_category'] = 'High'
        
        # Calculate glucose category
        if 'glucose' in data and data['glucose']:
            glucose = data['glucose']
            if glucose < 70:
                metrics['glucose_category'] = 'Low'
            elif glucose < 100:
                metrics['glucose_category'] = 'Normal'
            elif glucose < 126:
                metrics['glucose_category'] = 'Prediabetes'
            else:
                metrics['glucose_category'] = 'Diabetes'
        
        return metrics

class ReportGenerator:
    """Class for generating health reports"""
    
    @staticmethod
    def generate_health_report(patient_data: Dict[str, Any], 
                             health_assessment: Dict[str, Any],
                             recommendations: Dict[str, Any]) -> str:
        """Generate a comprehensive health report"""
        report = []
        report.append("=" * 60)
        report.append("HEALTH RISK ASSESSMENT REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Patient Information
        report.append("PATIENT INFORMATION:")
        report.append("-" * 20)
        for key, value in patient_data.items():
            if value is not None:
                report.append(f"{key.replace('_', ' ').title()}: {value}")
        report.append("")
        
        # Health Assessment
        report.append("HEALTH ASSESSMENT:")
        report.append("-" * 20)
        report.append(f"Overall Risk Level: {health_assessment.get('overall_risk_level', 'Unknown')}")
        report.append(f"Risk Score: {health_assessment.get('overall_risk_score', 0):.1f}%")
        report.append("")
        
        # Individual Risk Factors
        report.append("RISK FACTORS:")
        report.append("-" * 20)
        for condition, assessment in health_assessment.get('health_assessment', {}).items():
            risk_level = assessment.get('risk', 'unknown')
            report.append(f"• {condition.replace('_', ' ').title()}: {risk_level.upper()} Risk")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        report.append("-" * 20)
        for category, recs in recommendations.items():
            if isinstance(recs, list) and recs:
                report.append(f"{category.replace('_', ' ').title()}:")
                for rec in recs:
                    report.append(f"  - {rec}")
                report.append("")
        
        # Next Steps
        if 'next_steps' in health_assessment:
            report.append("NEXT STEPS:")
            report.append("-" * 20)
            for step in health_assessment['next_steps']:
                report.append(f"• {step}")
            report.append("")
        
        # Follow-up Schedule
        if 'follow_up_schedule' in health_assessment:
            report.append("FOLLOW-UP SCHEDULE:")
            report.append("-" * 20)
            for period, timing in health_assessment['follow_up_schedule'].items():
                report.append(f"• {period.replace('_', ' ').title()}: {timing}")
        
        report.append("")
        report.append("=" * 60)
        report.append("Report generated on: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
        report.append("=" * 60)
        
        return "\n".join(report)
    
    @staticmethod
    def save_report_to_file(report_content: str, filename: str = "health_report.txt"):
        """Save health report to file"""
        try:
            with open(filename, 'w') as f:
                f.write(report_content)
            print(f"Report saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving report: {str(e)}")
            return False

class ModelEvaluator:
    """Class for evaluating ML model performance"""
    
    @staticmethod
    def calculate_model_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                              y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate comprehensive model evaluation metrics"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix, classification_report
        )
        
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # ROC AUC if probabilities are available
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, average='weighted')
            except:
                metrics['roc_auc'] = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Additional metrics
        metrics['total_samples'] = len(y_true)
        metrics['positive_samples'] = np.sum(y_true == 1)
        metrics['negative_samples'] = np.sum(y_true == 0)
        
        return metrics
    
    @staticmethod
    def create_model_comparison_chart(model_results: Dict[str, Dict[str, float]]) -> go.Figure:
        """Create a comparison chart for multiple models"""
        model_names = list(model_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig = go.Figure()
        
        for metric in metrics:
            values = [model_results[model].get(metric, 0) for model in model_names]
            fig.add_trace(go.Bar(
                name=metric.replace('_', ' ').title(),
                x=model_names,
                y=values,
                text=[f'{v:.3f}' for v in values],
                textposition='auto'
            ))
        
        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Models',
            yaxis_title='Score',
            barmode='group',
            height=600
        )
        
        return fig
