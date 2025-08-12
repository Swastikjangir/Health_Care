import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='mean')
        self.feature_selector = None
        self.selected_features = None
        
    def clean_data(self, df):
        """Clean the dataset by handling missing values and outliers"""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        # Impute numeric columns with mean
        if len(numeric_columns) > 0:
            df[numeric_columns] = self.imputer.fit_transform(df[numeric_columns])
        
        # Impute categorical columns with mode
        for col in categorical_columns:
            df[col] = df[col].fillna(df[col].mode()[0])
            
        return df
    
    def handle_outliers(self, df, columns, method='iqr'):
        """Handle outliers using IQR method"""
        df_clean = df.copy()
        
        for col in columns:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing them
                df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, df_clean[col])
                df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])
                
        return df_clean
    
    def encode_categorical(self, df, categorical_columns):
        """Encode categorical variables using Label Encoding"""
        df_encoded = df.copy()
        
        for col in categorical_columns:
            if col in df_encoded.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col])
                else:
                    # Handle unseen categories
                    df_encoded[col] = self.label_encoders[col].transform(df_encoded[col])
                    
        return df_encoded
    
    def scale_features(self, df, columns, method='standard'):
        """Scale numerical features"""
        df_scaled = df.copy()
        
        if method == 'standard':
            df_scaled[columns] = self.scaler.fit_transform(df_scaled[columns])
        elif method == 'minmax':
            minmax_scaler = MinMaxScaler()
            df_scaled[columns] = minmax_scaler.fit_transform(df_scaled[columns])
            
        return df_scaled
    
    def select_features(self, X, y, method='kbest', k=10):
        """Select the most important features"""
        if method == 'kbest':
            self.feature_selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
            X_selected = self.feature_selector.fit_transform(X, y)
            self.selected_features = X.columns[self.feature_selector.get_support()].tolist()
            
        elif method == 'rfe':
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            self.feature_selector = RFE(estimator=estimator, n_features_to_select=k)
            X_selected = self.feature_selector.fit_transform(X, y)
            self.selected_features = X.columns[self.feature_selector.support_].tolist()
            
        return X_selected
    
    def get_feature_importance(self, X, y):
        """Get feature importance scores"""
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def preprocess_pipeline(self, df, target_column=None, categorical_columns=None, 
                          numeric_columns=None, handle_outliers=True):
        """Complete preprocessing pipeline"""
        # Clean data
        df_clean = self.clean_data(df)
        
        # Handle outliers for numeric columns
        if handle_outliers and numeric_columns:
            df_clean = self.handle_outliers(df_clean, numeric_columns)
        
        # Encode categorical variables
        if categorical_columns:
            df_clean = self.encode_categorical(df_clean, categorical_columns)
        
        # Scale numeric features
        if numeric_columns:
            df_clean = self.scale_features(df_clean, numeric_columns)
        
        return df_clean
    
    def transform_new_data(self, df, categorical_columns=None, numeric_columns=None):
        """Transform new data using fitted preprocessors"""
        df_transformed = df.copy()
        
        # Encode categorical variables
        if categorical_columns:
            for col in categorical_columns:
                if col in df_transformed.columns and col in self.label_encoders:
                    # Handle unseen categories by assigning a new label
                    unseen_mask = ~df_transformed[col].isin(self.label_encoders[col].classes_)
                    if unseen_mask.any():
                        df_transformed.loc[unseen_mask, col] = self.label_encoders[col].classes_[0]
                    df_transformed[col] = self.label_encoders[col].transform(df_transformed[col])
        
        # Scale numeric features
        if numeric_columns:
            df_transformed[numeric_columns] = self.scaler.transform(df_transformed[numeric_columns])
        
        # Select features if feature selector is fitted
        if self.feature_selector and self.selected_features:
            df_transformed = df_transformed[self.selected_features]
        
        return df_transformed
