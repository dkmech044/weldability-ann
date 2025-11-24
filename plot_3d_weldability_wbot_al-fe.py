import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestClassifier
# NOTE: Make sure 'ann_dataset_full_v23.csv' is in the current directory

# --- Load Data and Train Model ---
df = pd.read_csv('ann_dataset_full_v23.csv')
features = ['Material_Flag', 'V1 (m/s)', 'w_top (mm)', 'w_bot (mm)', 
            'd12 (mm)', 'd23 (mm)', 'h1 (mm)', 'h2 (mm)', 'h3 (mm)']
X = df[features]
y = df['Weldable']

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X, y)

# --- Define Grid and Fixed Parameters for Al–Fe–Fe ---
V1_range = np.linspace(600, 1800, 20)
d23_range = np.linspace(0.5, 4.0, 20)
wtop_range = np.linspace(6.0, 14.0, 20)

fixed_params_al_fefe = {
    'Material_Flag': 0,
    'd12 (mm)': 3.0,
    'h1 (mm)': 1.5,
    'h2 (mm)': 0.8,
    'h3 (mm)': 0.8,
}
wbot_slices = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

# --- Generate Prediction Grid and Plot Slices ---
results_list = []
V1_grid, d23_grid, wtop_grid = np.meshgrid(V1_range, d23_range, wtop_range, indexing='ij')
V1_flat, d23_flat, wtop_flat = V1_grid.flatten(), d23_grid.flatten(), wtop_grid.flatten()

for wbot_val in wbot_slices:
    # 1. Prepare Prediction DataFrame
    predict_df = pd.DataFrame({'V1 (m/s)': V1_flat, 'd23 (mm)': d23_flat, 'w_top (mm)': wtop_flat})
    for col, val in fixed_params_al_fefe.items():
        predict_df[col] = val
    predict_df['w_bot (mm)'] = wbot_val
    predict_df = predict_df[features]
    
    # 2. Predict Weldability
    Weldable_pred = rf_model.predict(predict_df)
    slice_result = predict_df.copy()
    slice_result['Weldable_Predicted'] = Weldable_pred
    results_list.append(slice_result)
    
    # 3. Plotting the 3D Scatter
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    weldable_points = slice_result[slice_result['Weldable_Predicted'] == 1]
    non_weldable_points = slice_result[slice_result['Weldable_Predicted'] == 0]
    
    # Plot non-weldable points (red 'x') and weldable points (green 'o')
    ax.scatter(non_weldable_points['V1 (m/s)'], non_weldable_points['d23 (mm)'], 
               non_weldable_points['w_top (mm)'], color='red', marker='x', label='Not Weldable', s=10, alpha=0.5)
    ax.scatter(weldable_points['V1 (m/s)'], weldable_points['d23 (mm)'], 
               weldable_points['w_top (mm)'], color='green', marker='o', label='Weldable', s=10, alpha=0.5)

    # Set labels and title
    ax.set_xlabel(r'Flyer Velocity $V_1$ (m/s)')
    ax.set_ylabel(r'Standoff $d_{23}$ (mm)')
    ax.set_zlabel(r'Top Die Width $w_{top}$ (mm)')
    title_latex = (r'Al–Fe–Fe Weldability Map ($M=0$) for Fixed $w_{bot} = ' + 
                   f'{wbot_val:.1f}$ mm')
    ax.set_title(title_latex, pad=20)
    ax.legend(loc='lower left')
    plt.show() # Use plt.show() when running interactively
    # In a notebook, use fig.savefig(f'AlFeFe_Map_wbot_{wbot_val:.1f}mm.png')

# 4. Save aggregated results to CSV
final_df_al_fefe = pd.concat(results_list, ignore_index=True)
output_csv_filename = 'AlFeFe_Weldability_Slices_V1_d23_wtop.csv'
final_df_al_fefe.to_csv(output_csv_filename, index=False)
print(f"Aggregated data saved to: {output_csv_filename}")