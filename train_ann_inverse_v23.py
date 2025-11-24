import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ---------------------------
# 1) Load dataset
# ---------------------------
df = pd.read_csv("ann_dataset_full_v23.csv")
print("Full dataset shape:", df.shape)

# Use only *weldable* rows for inverse design
df_w = df[df["Weldable"] == 1].copy()
print("Weldable rows:", df_w.shape[0])

if df_w.shape[0] == 0:
    raise RuntimeError(
        "No Weldable == 1 rows in ann_dataset_full_v23.csv.\n"
        "Relax weldability criteria / f_heat in the forward model and regenerate the dataset."
    )

# ---------------------------
# 2) Define X (inputs) and Y (outputs)
# ---------------------------
X_cols = [
    "Material_Flag",
    "beta1_deg",
    "T_Interface1",
    "T_Interface2",
    "Clearance_OK",
    "Vc12",
    "Vimpact23",
]

Y_cols = [
    "V1 (m/s)",
    "w_top (mm)",
    "w_bot (mm)",
    "d12 (mm)",
    "d23 (mm)",
    "h1 (mm)",
    "h2 (mm)",
    "h3 (mm)",
]

X = df_w[X_cols].values.astype(np.float32)
Y = df_w[Y_cols].values.astype(np.float32)

print("X shape:", X.shape)
print("Y shape:", Y.shape)

# ---------------------------
# 3) Train/validation split
# ---------------------------
X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# ---------------------------
# 4) Scaling
# ---------------------------
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train_s = scaler_X.fit_transform(X_train)
X_val_s   = scaler_X.transform(X_val)

Y_train_s = scaler_Y.fit_transform(Y_train)
Y_val_s   = scaler_Y.transform(Y_val)

# Save scalers for later use
joblib.dump(scaler_X, "scaler_X_v23.pkl")
joblib.dump(scaler_Y, "scaler_Y_v23.pkl")
print("Saved scaler_X_v23.pkl and scaler_Y_v23.pkl")

# ---------------------------
# 5) Define ANN model
# ---------------------------
input_dim  = X_train_s.shape[1]
output_dim = Y_train_s.shape[1]

model = Sequential([
    Dense(128, activation="relu", input_shape=(input_dim,)),
    Dense(128, activation="relu"),
    Dense(64,  activation="relu"),
    Dense(32,  activation="relu"),
    Dense(output_dim, activation="linear"),
])

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="mse",
    metrics=["mae"],
)

model.summary()

# ---------------------------
# 6) Train with early stopping
# ---------------------------
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=50,
        restore_best_weights=True,
        verbose=1,
    ),
    ModelCheckpoint(
        "weldability_inverse_ann_model_v23.h5",
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
]

history = model.fit(
    X_train_s, Y_train_s,
    validation_data=(X_val_s, Y_val_s),
    epochs=1000,
    batch_size=128,
    callbacks=callbacks,
    verbose=2,
)

# Final validation metrics
val_loss, val_mae = model.evaluate(X_val_s, Y_val_s, verbose=0)
print(f"Final validation loss = {val_loss:.4e}, MAE = {val_mae:.4e}")

print("Saved weldability_inverse_ann_model_v23.h5")
