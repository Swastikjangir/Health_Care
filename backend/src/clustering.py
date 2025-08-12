import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class PatientClustering:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.scaler = StandardScaler()
        self.pca = None
        
    def initialize_models(self):
        """Initialize clustering models"""
        self.models = {
            'kmeans': KMeans(random_state=42),
            'hierarchical': AgglomerativeClustering(),
            'dbscan': DBSCAN(),
            'gaussian_mixture': GaussianMixture(random_state=42)
        }
        
    def preprocess_data(self, X):
        """Preprocess data for clustering"""
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA for visualization (optional)
        if X_scaled.shape[1] > 2:
            self.pca = PCA(n_components=2)
            X_pca = self.pca.fit_transform(X_scaled)
        else:
            X_pca = X_scaled
            
        return X_scaled, X_pca
    
    def find_optimal_clusters(self, X, max_clusters=10, method='kmeans'):
        """Find optimal number of clusters using elbow method and silhouette analysis"""
        if method == 'kmeans':
            inertias = []
            silhouette_scores = []
            k_range = range(2, max_clusters + 1)
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(X)
                inertias.append(kmeans.inertia_)
                
                if k > 1:
                    silhouette_scores.append(silhouette_score(X, kmeans.labels_))
                else:
                    silhouette_scores.append(0)
            
            # Find elbow point
            elbow_k = self._find_elbow_point(k_range, inertias)
            
            # Find best silhouette score
            best_silhouette_k = k_range[np.argmax(silhouette_scores)]
            
            print(f"Elbow method suggests {elbow_k} clusters")
            print(f"Silhouette analysis suggests {best_silhouette_k} clusters")
            
            return {
                'elbow_k': elbow_k,
                'best_silhouette_k': best_silhouette_k,
                'inertias': inertias,
                'silhouette_scores': silhouette_scores,
                'k_range': list(k_range)
            }
        
        return None
    
    def _find_elbow_point(self, k_range, inertias):
        """Find elbow point using second derivative method"""
        if len(inertias) < 3:
            return k_range[0]
        
        # Calculate second derivative
        second_derivative = np.diff(np.diff(inertias))
        
        # Find the point with maximum second derivative
        elbow_idx = np.argmax(second_derivative) + 1
        
        return k_range[elbow_idx]
    
    def train_clustering_models(self, X, n_clusters=3):
        """Train different clustering models"""
        X_scaled, X_pca = self.preprocess_data(X)
        
        results = {}
        
        # K-Means
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans_labels = kmeans.fit_predict(X_scaled)
            
            results['kmeans'] = {
                'model': kmeans,
                'labels': kmeans_labels,
                'silhouette': silhouette_score(X_scaled, kmeans_labels),
                'calinski_harabasz': calinski_harabasz_score(X_scaled, kmeans_labels),
                'davies_bouldin': davies_bouldin_score(X_scaled, kmeans_labels)
            }
            
            print(f"K-Means - Silhouette: {results['kmeans']['silhouette']:.4f}")
        except Exception as e:
            print(f"Error in K-Means: {str(e)}")
        
        # Hierarchical Clustering
        try:
            hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
            hierarchical_labels = hierarchical.fit_predict(X_scaled)
            
            results['hierarchical'] = {
                'model': hierarchical,
                'labels': hierarchical_labels,
                'silhouette': silhouette_score(X_scaled, hierarchical_labels),
                'calinski_harabasz': calinski_harabasz_score(X_scaled, hierarchical_labels),
                'davies_bouldin': davies_bouldin_score(X_scaled, hierarchical_labels)
            }
            
            print(f"Hierarchical - Silhouette: {results['hierarchical']['silhouette']:.4f}")
        except Exception as e:
            print(f"Error in Hierarchical: {str(e)}")
        
        # Gaussian Mixture
        try:
            gmm = GaussianMixture(n_components=n_clusters, random_state=42)
            gmm_labels = gmm.fit_predict(X_scaled)
            
            results['gaussian_mixture'] = {
                'model': gmm,
                'labels': gmm_labels,
                'silhouette': silhouette_score(X_scaled, gmm_labels),
                'calinski_harabasz': calinski_harabasz_score(X_scaled, gmm_labels),
                'davies_bouldin': davies_bouldin_score(X_scaled, gmm_labels)
            }
            
            print(f"Gaussian Mixture - Silhouette: {results['gaussian_mixture']['silhouette']:.4f}")
        except Exception as e:
            print(f"Error in Gaussian Mixture: {str(e)}")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['silhouette'])
        self.best_model = results[best_model_name]['model']
        self.best_score = results[best_model_name]['silhouette']
        
        print(f"\nBest clustering model: {best_model_name} with silhouette score: {self.best_score:.4f}")
        
        return results, X_scaled, X_pca
    
    def analyze_clusters(self, X, labels, feature_names=None):
        """Analyze the characteristics of each cluster"""
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        
        # Create DataFrame with cluster labels
        df_clusters = pd.DataFrame(X, columns=feature_names)
        df_clusters['Cluster'] = labels
        
        # Analyze each cluster
        cluster_analysis = {}
        
        for cluster_id in sorted(set(labels)):
            cluster_data = df_clusters[df_clusters['Cluster'] == cluster_id]
            
            cluster_analysis[cluster_id] = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(df_clusters) * 100,
                'mean_values': cluster_data[feature_names].mean().to_dict(),
                'std_values': cluster_data[feature_names].std().to_dict(),
                'min_values': cluster_data[feature_names].min().to_dict(),
                'max_values': cluster_data[feature_names].max().to_dict()
            }
        
        return cluster_analysis
    
    def visualize_clusters(self, X_pca, labels, method='plotly'):
        """Visualize clustering results"""
        if method == 'plotly':
            # Create interactive plot with Plotly
            df_viz = pd.DataFrame({
                'PC1': X_pca[:, 0],
                'PC2': X_pca[:, 1],
                'Cluster': labels
            })
            
            fig = px.scatter(
                df_viz, x='PC1', y='PC2', color='Cluster',
                title='Patient Clustering Results',
                labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            
            fig.update_layout(
                width=800,
                height=600,
                showlegend=True
            )
            
            return fig
        
        elif method == 'matplotlib':
            # Create matplotlib plot
            plt.figure(figsize=(10, 8))
            
            unique_labels = set(labels)
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
            
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    col = 'k'
                
                class_member_mask = (labels == k)
                xy = X_pca[class_member_mask]
                plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, 
                        markeredgecolor='k', markersize=6, label=f'Cluster {k}')
            
            plt.title('Patient Clustering Results')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            return plt.gcf()
    
    def create_cluster_profiles(self, cluster_analysis, feature_names):
        """Create human-readable cluster profiles"""
        profiles = {}
        
        for cluster_id, analysis in cluster_analysis.items():
            profile = f"Cluster {cluster_id} Profile:\n"
            profile += f"Size: {analysis['size']} patients ({analysis['percentage']:.1f}%)\n\n"
            
            profile += "Key Characteristics:\n"
            for feature in feature_names:
                mean_val = analysis['mean_values'][feature]
                std_val = analysis['std_values'][feature]
                profile += f"- {feature}: {mean_val:.2f} ± {std_val:.2f}\n"
            
            profiles[cluster_id] = profile
        
        return profiles
    
    def get_risk_assessment(self, cluster_analysis, risk_features=None):
        """Assess risk level for each cluster"""
        if risk_features is None:
            risk_features = ['glucose', 'blood_pressure', 'bmi', 'age']
        
        risk_assessment = {}
        
        for cluster_id, analysis in cluster_analysis.items():
            risk_score = 0
            
            for feature in risk_features:
                if feature in analysis['mean_values']:
                    mean_val = analysis['mean_values'][feature]
                    
                    # Simple risk scoring (can be customized)
                    if feature == 'glucose':
                        if mean_val > 140: risk_score += 3
                        elif mean_val > 100: risk_score += 2
                        elif mean_val > 70: risk_score += 1
                    elif feature == 'blood_pressure':
                        if mean_val > 140: risk_score += 3
                        elif mean_val > 120: risk_score += 2
                        elif mean_val > 90: risk_score += 1
                    elif feature == 'bmi':
                        if mean_val > 30: risk_score += 3
                        elif mean_val > 25: risk_score += 2
                        elif mean_val > 18.5: risk_score += 1
                    elif feature == 'age':
                        if mean_val > 65: risk_score += 3
                        elif mean_val > 45: risk_score += 2
                        elif mean_val > 25: risk_score += 1
            
            # Determine risk level
            if risk_score >= 8:
                risk_level = "High Risk"
            elif risk_score >= 5:
                risk_level = "Medium Risk"
            else:
                risk_level = "Low Risk"
            
            risk_assessment[cluster_id] = {
                'risk_score': risk_score,
                'risk_level': risk_level,
                'recommendations': self._get_risk_recommendations(risk_level)
            }
        
        return risk_assessment
    
    def _get_risk_recommendations(self, risk_level):
        """Get recommendations based on risk level"""
        recommendations = {
            "High Risk": [
                "Immediate medical consultation required",
                "Frequent health monitoring (weekly)",
                "Lifestyle modifications mandatory",
                "Consider medication if prescribed"
            ],
            "Medium Risk": [
                "Regular health check-ups (monthly)",
                "Moderate lifestyle changes",
                "Monitor key health indicators",
                "Consult healthcare provider"
            ],
            "Low Risk": [
                "Annual health check-ups",
                "Maintain healthy lifestyle",
                "Preventive measures",
                "Regular exercise and balanced diet"
            ]
        }
        
        return recommendations.get(risk_level, [])
    
    def predict_cluster(self, X_new):
        """Predict cluster for new data"""
        if self.best_model is None:
            raise ValueError("No trained clustering model available")
        
        # Scale the new data
        X_new_scaled = self.scaler.transform(X_new)
        
        # Predict cluster
        if hasattr(self.best_model, 'predict'):
            cluster = self.best_model.predict(X_new_scaled)
        else:
            # For models without predict method (like DBSCAN)
            cluster = self.best_model.fit_predict(X_new_scaled)
        
        return cluster
