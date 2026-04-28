# Hybrid Physics-Informed Neural Network Framework for Inverse Design of Multi-layer Impact Welding

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow/Keras](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Inverse%20Design-success)

## Overview
This repository contains the code, models, and datasets for a **Hybrid Physics-Informed Neural Network (PINN) Framework** dedicated to the inverse design of multi-layer, high-velocity impact welding processes (such as Vaporizing Foil Actuator Welding and Magnetic Pulse Welding). 

By leveraging both forward analytical data engines and inverse AI models, this framework predicts optimal process parameters and geometric configurations to achieve targeted weldability windows in complex dissimilar joints (e.g., Al-Fe, Al-Cu, and 3-layer trimetal configurations). 

## Key Features
* **Forward Modeling & Data Generation:** Physics-guided analytical engines to simulate multi-layer impact welding kinematics.
* **Inverse Design AI:** Artificial Neural Networks (ANNs) trained to back-calculate optimal initial geometries and process parameters from desired impact velocity and angle targets.
* **Uncertainty Quantification (UQ):** Monte Carlo (MC) Dropout networks to assess the confidence and robustness of the predicted inverse designs.
* **Trimetal Case Studies:** Specific modules addressing the complex dynamics of 3-layer impact welding setups.
* **Stress Testing:** Evaluation of extreme edge-case process conditions to ensure robust process execution.

---

## Repository Structure

### 📊 Datasets & Scalers
* `ann_dataset_full_v23.csv`, `final_ann_dataset_material_specific_thickness.csv` - Core training and validation datasets.
* `extreme_cases.csv`, `extreme_results_final.csv`, `AppendixB_OOD_StressTest_ForwardOutputs.csv` - Data for Out-Of-Distribution (OOD) stress testing and boundary condition evaluations.
* `scaler.pkl`, `scaler_X.pkl`, `scaler_Y.pkl`, `scaler_weldable.pkl` - Serialized data pre-processing and normalization scales.
* `feature_cols_weldable (1).json` - Feature column configurations for data pipelines.

### 🧠 Core Models & Networks
* `weldability_inverse_ann_model.h5` - The primary trained inverse ANN model.
* `student_ann_model.pkl`, `student_ann_model_weldable.pkl` - Serialized lightweight student models for rapid inference.

### ⚙️ Scripts & Modules
**1. Forward Modeling & Simulation**
* `forward_model_full_v23.py` / `forward_model_full_v23_trimetal.py` - Physics-based forward data generation for 2-layer and 3-layer configurations.

**2. Inverse Design & Training Pipelines**
* `train_ann_inverse_v23.py` / `train_ann_inverse_design.py` - Training scripts for the core inverse design framework.
* `train_ann_inverse_geom2proc_v23.py` - Model training for mapping geometry to process parameters.
* `train_student_ann.py` / `train_student_ann_REWRITTEN.py` - Knowledge distillation / student network training.

**3. Inference & Prediction**
* `inverse_design_app.py` - Main application script for running inverse design predictions.
* `predict_design.py` / `predict_design_v23.py` - Scripts for predicting optimal designs based on user-defined weldability targets.
* `predict_from_geom_v23.py` - Forward prediction based on initial setup geometry.

**4. Uncertainty Quantification (UQ) & Robustness**
* `train_mc_dropout.py` / `mc_dropout_infer.py` - Training and inference using MC Dropout for uncertainty mapping.
* `uq_utils.py` - Utility functions for statistical analysis.
* `vfaw_ann_sensitivity_uq.py` - Sensitivity analysis for VFAW specific parameters.

**5. Visualization & Plotting**
* `plot_3d_weldability_wbot_al-fe.py` - 3D visualizations of the weldability window for Al-Fe dissimilar joints.
* `plot_uq_map.py` - Visual mapping of model uncertainties.
* `make_appendixB_robustness_plots.py` - Generates robustness and stress test plots for publications/reports.

**6. Case Studies & Edge Cases**
* `run_trimetal_case_study.py` / `run_trimetal_case_study_UPDATED.py` - Execution scripts for 3-plate (Al-Fe-Fe / Al-Cu-Cu) setup case studies.
* `test_extreme_cases.py` - Automated evaluation of the framework against extreme processing limits.
* `augment_dataset.py` - Utility to synthesize and augment sparse experimental data points.

---

## 🚀 Getting Started

### Prerequisites
Ensure you have Python 3.8+ installed. The main dependencies are:
```bash
pip install numpy pandas scikit-learn tensorflow keras matplotlib
