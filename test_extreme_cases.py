import pandas as pd
import joblib

# 1. Load the "Brain" and "Lens"
print("--- Loading Trained ANN and Scaler ---")
model = joblib.load('student_ann_model.pkl')
scaler = joblib.load('scaler.pkl')

# 2. Define the Mapping
# This maps YOUR column names to the NAMES THE ANN WAS TRAINED ON
column_mapping = {
    'material_flag': 'Material_Flag',
    'V1_mps': 'V1 (m/s)',
    'w_top_mm': 'w_top (mm)',
    'w_bot_mm': 'w_bot (mm)',
    'd12_mm': 'd12 (mm)',
    'd23_mm': 'd23 (mm)',
    'h1_mm': 'h1 (mm)',
    'h2_mm': 'h2 (mm)',
    'h3_mm': 'h3 (mm)'
}

# The exact order the model expects
expected_features = ['Material_Flag', 'V1 (m/s)', 'w_top (mm)', 'w_bot (mm)', 
                     'd12 (mm)', 'd23 (mm)', 'h1 (mm)', 'h2 (mm)', 'h3 (mm)']

# 3. Load your Extreme Cases
filename = 'extreme_cases.csv'
df_extreme = pd.read_csv(filename)
print(f"--- Loaded {len(df_extreme)} cases from {filename} ---")

# 4. Prepare the data for the ANN
# Create a copy with only the columns we need, renamed to match training
X_extreme = df_extreme[list(column_mapping.keys())].rename(columns=column_mapping)

# Ensure the columns are in the EXACT order the scaler expects
X_extreme = X_extreme[expected_features]

# 5. Scale and Predict
X_scaled = scaler.transform(X_extreme)
predictions = model.predict(X_scaled)

# 6. Merge results back to the original dataframe for viewing
df_extreme['ANN_Weldable_Pred'] = predictions[:, 0]
df_extreme['ANN_CrackOK_Pred'] = predictions[:, 1]

# Save and Display
output_file = 'extreme_results_final.csv'
df_extreme.to_csv(output_file, index=False)

print(f"\n--- PREDICTION COMPLETE ---")
print(f"Results saved to {output_file}")

# Show top 5 rows of the result
print("\nPreview of Results:")
print(df_extreme[['material_flag', 'V1_mps', 'regime', 'ANN_Weldable_Pred', 'ANN_CrackOK_Pred']].head())