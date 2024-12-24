# Weather Data Predictive Analysis
# Author: Data Science Team
# Last Updated: December 2024

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                           roc_auc_score, roc_curve, precision_score, recall_score)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from imblearn.over_sampling import SMOTE

class WeatherAnalysis:
    def __init__(self, data_path):
        """
        Initialize the Weather Analysis class.
        
        Args:
            data_path (str): Path to the input CSV file
        """
        self.df = pd.read_csv(data_path)
        self.prepare_data()
        
    def prepare_data(self):
        """Prepare the dataset for analysis by handling numeric conversions."""
        # Convert global radiation to numeric
        self.df['global_rad_num'] = pd.to_numeric(self.df['global_radiation'], errors='coerce')
        # Select numeric columns
        self.df_numeric = self.df.select_dtypes(include=['number'])
        
    def train_random_forest(self):
        """
        Train and evaluate a Random Forest model for extreme rain prediction.
        Uses SMOTE for handling class imbalance.
        """
        # Prepare features and target
        X = self.df_numeric.copy().dropna(axis=1)
        y = self.df_numeric['extreme_rain']
        X = X.drop(columns=["extreme_rain"], errors="ignore")
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        # Train model with optimal parameters
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.rf_model.fit(X_train_resampled, y_train_resampled)
        
        # Evaluate model
        y_val_pred = self.rf_model.predict(X_val)
        print("\nValidation Set Results:")
        print('Accuracy:', accuracy_score(y_val, y_val_pred))
        print('Classification Report:\n', classification_report(y_val, y_val_pred))
        
    def perform_clustering(self):
        """
        Perform K-means clustering on the weather data and visualize results.
        """
        # Prepare data for clustering
        X_clustering = self.df_numeric.copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clustering)
        
        # Find optimal number of clusters
        optimal_k = 3
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Dimensionality reduction for visualization
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        # Visualize clusters
        plt.figure(figsize=(10, 8))
        for cluster in range(optimal_k):
            mask = cluster_labels == cluster
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'Cluster {cluster}')
        
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('Weather Pattern Clusters')
        plt.legend()
        plt.show()
        
    def forecast_temperature(self):
        """
        Generate temperature forecasts using Exponential Smoothing.
        """
        # Prepare time series data
        temp_data = pd.DataFrame({
            'date': pd.to_datetime(self.df['date']),
            'max_temp': self.df['max_temp']
        }).set_index('date').sort_index()
        
        # Apply Exponential Smoothing
        model = ExponentialSmoothing(
            temp_data['max_temp'].fillna(method='ffill'),
            trend='add',
            seasonal='add',
            seasonal_periods=12
        )
        model_fit = model.fit()
        
        # Generate forecast
        forecast_days = 365
        forecast = model_fit.forecast(steps=forecast_days)
        
        # Create forecast DataFrame
        forecast_dates = pd.date_range(
            start=temp_data.index[-1] + pd.Timedelta(days=1),
            periods=forecast_days,
            freq='D'
        )
        self.forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'temperature_forecast': forecast
        })

def main():
    """Main execution function."""
    # Initialize analysis
    analysis = WeatherAnalysis("Mera_CleanData.csv")
    
    # Perform analysis steps
    analysis.train_random_forest()
    analysis.perform_clustering()
    analysis.forecast_temperature()
    
    # Save results
    analysis.forecast_df.to_csv('temperature_forecast.csv', index=False)
    print("Analysis complete. Results saved to CSV.")

if __name__ == "__main__":
    main()
