# ---------------------------
# Data Manipulation and Analysis
# ---------------------------

import pandas as pd
import numpy as np

# ---------------------------
# Machine Learning and Preprocessing
# ---------------------------

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error, r2_score, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split

# ---------------------------
# Deep Learning Framework
# ---------------------------

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------
# Visualization Libraries
# ---------------------------

import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------
# Load the dataset
# ---------------------------

file_path = "/content/Mera_CleanData.csv"
data = pd.read_csv(file_path)

# Remove the 'extreme_weather' column as it's not needed for clustering
data = data.drop('extreme_weather', axis=1)

# ---------------------------
# Select relevant features for clustering
# ---------------------------

features = ['extreme_rain', 'heatwave', 'high_wind']
data_clustering = data[features].dropna()

# ---------------------------
# Standardize features for clustering
# ---------------------------

scaler = StandardScaler()
scaled_features = scaler.fit_transform(data_clustering)

# ---------------------------
# Apply K-Means Clustering
# ---------------------------

kmeans = KMeans(n_clusters=3, random_state=42)  # Adjust n_clusters based on your requirements
kmeans_labels = kmeans.fit_predict(scaled_features)

# Calculate the silhouette score to evaluate the clustering quality
silhouette_kmeans = silhouette_score(scaled_features, kmeans_labels)

# Add the cluster labels to the dataset
data_clustering['Cluster'] = kmeans_labels

# ---------------------------
# Visualization: Extreme Rain vs High Wind
# ---------------------------

fig1 = px.scatter(
    data_clustering,
    x='extreme_rain',
    y='high_wind',
    color='Cluster',
    title="K-Means Clustering: Extreme Rain vs High Wind",
    labels={'Cluster': 'Cluster'},
    template="plotly_dark"
)

fig1.update_layout(
    xaxis_title="Extreme Rain",
    yaxis_title="High Wind",
    legend_title="Cluster",
    height=600,
    width=800
)
fig1.show()

# ---------------------------
# Visualization: Heatwave vs High Wind
# ---------------------------

fig2 = px.scatter(
    data_clustering,
    x='heatwave',
    y='high_wind',
    color='Cluster',
    title="K-Means Clustering: Heatwave vs High Wind",
    labels={'Cluster': 'Cluster'},
    template="plotly_dark"
)

fig2.update_layout(
    xaxis_title="Heatwave",
    yaxis_title="High Wind",
    legend_title="Cluster",
    height=600,
    width=800
)
fig2.show()

# ---------------------------
# 3D Scatter Plot for Extreme Rain, Heatwave, and High Wind
# ---------------------------

fig3 = px.scatter_3d(
    data_clustering,
    x='extreme_rain',
    y='heatwave',
    z='high_wind',
    color='Cluster',
    title="K-Means Clustering: 3D Visualization",
    labels={'Cluster': 'Cluster'},
    template="plotly_dark"
)

fig3.update_layout(
    scene=dict(
        xaxis_title="Extreme Rain",
        yaxis_title="Heatwave",
        zaxis_title="High Wind"
    ),
    height=700,
    width=900
)
fig3.show()

# ---------------------------
# Cluster Averages for Further Insights
# ---------------------------

print(f"K-Means Silhouette Score: {silhouette_kmeans}")
cluster_means = data_clustering.groupby('Cluster').mean()
print("Cluster Means:\n", cluster_means)

# ---------------------------
# Create 'Extreme Weather' feature
# ---------------------------

rain_weight = 0.333
heatwave_weight = 0.333
high_wind_weight = 0.333

# Calculate a weighted sum for extreme weather conditions
data['Extreme_Weather'] = (
    data['extreme_rain'] * rain_weight +
    data['heatwave'] * heatwave_weight +
    data['high_wind'] * high_wind_weight
)

# Normalize the 'Extreme_Weather' feature
data['Extreme_Weather'] = data['Extreme_Weather'] / data['Extreme_Weather'].max()

# ---------------------------
# Define the features and target for the model
# ---------------------------

features = [
    'max_temp', 'min_temp', 'temp_range', 'rain', 'pressure_cbl', 'wind_speed',
    'max_10minute_wind', 'dir_10minute_wind', 'max_gust', 'sun', 'global_radiation',
    'soil', 'potential_evap', 'evap', 'smd_combined',
    'heatwave', 'high_wind'
]

target = 'Extreme_Weather'

# ---------------------------
# Handle missing or invalid values in features
# ---------------------------

# Replace empty strings with NaN and convert the columns to numeric
data[features] = data[features].replace(r'^\s*$', np.nan, regex=True)
data[features] = data[features].astype(float)
data[features] = data[features].fillna(data[features].mean())  # Fill NaN values with column mean

# Normalize features using MinMaxScaler
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

# ---------------------------
# Create sequences for LSTM model
# ---------------------------

def create_sequences(data, target_col, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:i + seq_length][features].values)
        y.append(data.iloc[i + seq_length][target_col])
    return np.array(X), np.array(y)

seq_length = 90
X, y = create_sequences(data, target, seq_length)

# ---------------------------
# Train-test split
# ---------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ---------------------------
# Build and Compile LSTM Model
# ---------------------------

model = Sequential([
    LSTM(128, activation='relu', return_sequences=True, input_shape=(seq_length, len(features))),
    Dropout(0.3),
    LSTM(64, activation='relu'),
    Dense(1, activation='linear')  # Use linear activation for regression
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# ---------------------------
# Early stopping callback
# ---------------------------

early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# ---------------------------
# Train the model
# ---------------------------

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
    verbose=1
)

# ---------------------------
# Evaluate the model
# ---------------------------

y_pred = model.predict(X_test).flatten()

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Classification Metrics (using a threshold)
threshold = 0.5
y_pred_binary = (y_pred > threshold).astype(int)
y_test_binary = (y_test > threshold).astype(int)

accuracy = accuracy_score(y_test_binary, y_pred_binary)
precision = precision_score(y_test_binary, y_pred_binary, zero_division=1)
f1 = f1_score(y_test_binary, y_pred_binary, zero_division=1)

# ---------------------------
# Plot training history
# ---------------------------

fig_history = go.Figure()
fig_history.add_trace(go.Scatter(y=history.history['loss'], mode='lines', name='Training Loss'))
fig_history.add_trace(go.Scatter(y=history.history['val_loss'], mode='lines', name='Validation Loss'))
fig_history.update_layout(title="LSTM Training and Validation Loss",
                          xaxis_title="Epochs",
                          yaxis_title="Loss",
                          template="plotly_dark")
fig_history.show()

# ---------------------------
# Plot predictions vs actual
# ---------------------------

fig_predictions = go.Figure()
fig_predictions.add_trace(go.Scatter(y=y_test, mode='lines', name='Actual'))
fig_predictions.add_trace(go.Scatter(y=y_pred, mode='lines', name='Predicted'))
fig_predictions.update_layout(title="Predicted vs Actual Extreme Weather",
                               xaxis_title="Samples",
                               yaxis_title="Extreme Weather",
                               template="plotly_dark")
fig_predictions.show()

# ---------------------------
# Model Performance Metrics
# ---------------------------

print(f"""
LSTM Model Performance:
------------------------
Accuracy: {accuracy:.4f}
Precision: {precision:.4f}
F1-Score: {f1:.4f}
Mean Absolute Error (MAE): {mae:.4f}
Root Mean Squared Error (RMSE): {mse:.4f}
R² Score: {r2:.4f}
""")
