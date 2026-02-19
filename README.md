# VFAW HPINN – UQ Add-on (MC Dropout + Noise Augmentation)

This folder contains *drop-in* scripts to address the reviewer request:
> "Enhance ML robustness with uncertainty quantification"

## What you already have (Appendix A)
Appendix A perturbs material parameters within bounded ranges representative of batch scatter/temper dependence
and propagates them through the weldability-margin metric. This is already Monte Carlo–style uncertainty propagation
(using variance-based methods such as Saltelli/Sobol sampling).

## What you add here
1) **Noise-aware augmentation**: `augment_dataset.py`
2) **Dropout-trained regressor**: `train_mc_dropout.py`
3) **MC Dropout UQ at inference**: `mc_dropout_infer.py`
4) **Uncertainty maps**: `plot_uq_map.py`

## Quick start
1. Edit `X_COLS` and `Y_COLS` in each script to match your CSV column names.
2. Edit NOISE_CFG in `augment_dataset.py` (or keep default = no noise).
3. Run:

```bash
python augment_dataset.py --in data/base.csv --out data/aug.csv --k 5
python train_mc_dropout.py --data data/aug.csv --outdir artifacts_uq --dropout 0.15
python mc_dropout_infer.py --artifacts artifacts_uq --cases data/cases.csv --out data/uq_predictions.csv --T 50
python plot_uq_map.py --uq data/uq_predictions.csv --x <feature1> --y <feature2> --z sigma_<target> --out fig_uq.png
```

## Streamlit integration (idea)
In your Streamlit app, after you get a prediction, also compute `(mu, sigma)` with MC-dropout.
If `sigma` is above a threshold, display "low-confidence" and trigger your existing analytical feasibility check.
