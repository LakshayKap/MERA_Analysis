# ---------------------------
# Import Libraries
# ---------------------------

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------
# Load the dataset
# ---------------------------

file_path = "Mera_CleanData.csv"
data = pd.read_csv(file_path)

# ---------------------------
# Validate Columns
# ---------------------------

# Original features list
features = [
    'max_temp', 'min_temp', 'temp_range', 'rain', 'pressure_cbl', 'wind_speed',
    'max_10minute_wind', 'dir_10minute_wind', 'max_gust', 'sun', 'global_radiation',
    'soil', 'potential_evap', 'evap', 'smd_combined',
    'heatwave', 'high_wind'
]

# Check for missing columns
existing_features = [feature for feature in features if feature in data.columns]
missing_features = [feature for feature in features if feature not in data.columns]

if missing_features:
    print(f"Warning: The following features are missing and will be excluded: {missing_features}")

# Update features to only include existing columns
features = existing_features

# ---------------------------
# Handle missing or invalid values
# ---------------------------

data[features] = data[features].replace(r'^\s*$', np.nan, regex=True)
data[features] = data[features].astype(float)
data[features] = data[features].fillna(data[features].mean())

# Normalize the features
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

# ---------------------------
# Create Extreme Weather Feature
# ---------------------------

rain_weight = 0.333
heatwave_weight = 0.333
high_wind_weight = 0.333

data['Extreme_Weather'] = (
    data['extreme_rain'] * rain_weight +
    data['heatwave'] * heatwave_weight +
    data['high_wind'] * high_wind_weight
)
data['Extreme_Weather'] = data['Extreme_Weather'] / data['Extreme_Weather'].max()

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
X, y = create_sequences(data, 'Extreme_Weather', seq_length)

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
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Early stopping callback
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
Root Mean Squared Error (RMSE): {mse**0.5:.4f}
R² Score: {r2:.4f}
""")
