
import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model and scalers
model = load_model("weldability_inverse_ann_model.h5")
scaler_X = joblib.load("scaler_X.pkl")
scaler_Y = joblib.load("scaler_Y.pkl")

st.title("Inverse Weldability Design Tool")
st.markdown("Use the sliders and options below to input your desired weldability conditions and retrieve optimal design parameters.")

# User Inputs
material_flag = st.selectbox("Material Combination", ["Al–Fe–Fe (0)", "Al–Cu–Cu (1)"])
material_flag = 0 if "Fe" in material_flag else 1

beta1 = st.slider("Desired β₁ (deg)", 4.0, 20.0, 12.0)
T_interface1 = st.slider("Interface 1 Temp (°C)", 300.0, 1600.0, 900.0)
T_interface2 = st.slider("Interface 2 Temp (°C)", 300.0, 1600.0, 1000.0)
clearance_ok = st.checkbox("Clearance OK?", value=True)
Vc12 = st.slider("Vc₁₂ (m/s)", 600.0, 1800.0, 1000.0)
Vc23 = st.slider("Vc₂₃ (m/s)", 600.0, 1800.0, 1000.0)

if st.button("🔍 Predict Optimal Design Parameters"):
    input_features = np.array([[material_flag, beta1, T_interface1, T_interface2,
                                int(clearance_ok), Vc12, Vc23]])
    input_scaled = scaler_X.transform(input_features)
    output_scaled = model.predict(input_scaled)
    output = scaler_Y.inverse_transform(output_scaled)

    st.subheader("📐 Predicted Parameters:")
    labels = ["V₁ (m/s)", "Die Width b (mm)", "Standoff d₁₂ (mm)", "Standoff d₂₃ (mm)",
              "Thickness h₁ (mm)", "Thickness h₂ (mm)", "Thickness h₃ (mm)"]
    for name, val in zip(labels, output[0]):
        st.write(f"**{name}**: {val:.2f}")
