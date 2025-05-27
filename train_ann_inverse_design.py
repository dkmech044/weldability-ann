
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load dataset (adjust path as needed)
df = pd.read_csv("final_ann_dataset_material_specific_thickness.csv")

# Input features
X = df[[
    "Material_Flag", "beta1 (deg)", "T_Interface1 (K)", "T_Interface2 (K)",
    "Clearance_OK", "Vc12 (m/s)", "Vc23 (m/s)"
]]

# Output features
Y = df[[
    "V1 (m/s)", "b (mm)", "d12 (mm)", "d23 (mm)", "h1 (mm)", "h2 (mm)", "h3 (mm)"
]]

# Normalize
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y)

# Split
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

# ANN model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(Y.shape[1], activation='linear')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
history = model.fit(X_train, Y_train, epochs=150, batch_size=32, validation_split=0.1, verbose=1)

# Predict
Y_pred_scaled = model.predict(X_test)
Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)
Y_true = scaler_Y.inverse_transform(Y_test)

# Save model and scalers
model.save("weldability_inverse_ann_model.h5")

import joblib
joblib.dump(scaler_X, "scaler_X.pkl")
joblib.dump(scaler_Y, "scaler_Y.pkl")

# Training plot
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("ANN Training Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("ann_training_loss.png")
plt.show()

# Output CSV
comparison_df = pd.DataFrame(Y_true, columns=[f"True_{col}" for col in Y.columns])
for i, col in enumerate(Y.columns):
    comparison_df[f"Pred_{col}"] = Y_pred[:, i]
comparison_df.to_csv("ann_prediction_comparison.csv", index=False)
