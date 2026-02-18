import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 1. Load the Dataset
# Ensure 'ann_dataset_full_v23.csv' is in the same directory
filename = 'ann_dataset_full_v23.csv'
print(f"--- Loading {filename} ---")
df = pd.read_csv(filename)

# 2. Define Features (Inputs) and Targets (Outputs)
# We use the parameters to predict if it is Weldable AND if the Crack status is OK
features = ['Material_Flag', 'V1 (m/s)', 'w_top (mm)', 'w_bot (mm)', 
            'd12 (mm)', 'd23 (mm)', 'h1 (mm)', 'h2 (mm)', 'h3 (mm)']
targets = ['Weldable', 'Crack_OK']

X = df[features]
y = df[targets]

# 3. Split the Data (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Scaling (Extremely important for Neural Networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Initialize and Train the Forward ANN
# We use two hidden layers (128 and 64 neurons)
print("--- Training the Student (Forward ANN) ---")
student_ann = MLPClassifier(
    hidden_layer_sizes=(128, 64), 
    activation='relu', 
    solver='adam', 
    max_iter=1000, 
    random_state=42,
    verbose=False
)

student_ann.fit(X_train_scaled, y_train)
print("--- Training Complete ---\n")

# 6. Evaluation
y_pred = student_ann.predict(X_test_scaled)

# Print Detailed Reports for both outputs
for i, target_name in enumerate(targets):
    print(f"Performance Report for: {target_name}")
    print(classification_report(y_test.iloc[:, i], y_pred[:, i]))
    print("-" * 30)

# 7. Visualization of Results
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for i, target_name in enumerate(targets):
    cm = confusion_matrix(y_test.iloc[:, i], y_pred[:, i])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
    axes[i].set_title(f'Confusion Matrix: {target_name}')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# 8. Save the model and scaler for future use (The "Brain" of the Student)
joblib.dump(student_ann, 'student_ann_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model and Scaler saved as .pkl files.")