"""
Utilities for uncertainty-aware HPINN (MC Dropout) workflows.

These scripts are designed to be dropped into your existing repo.
You only need to:
  1) point CSV_PATH to your dataset,
  2) set X_COLS and Y_COLS to match your column names,
  3) (optional) tune NOISE_CFG.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

@dataclass
class NoiseSpec:
    mode: str = "relative"   # "relative" or "absolute"
    sigma: float = 0.01      # relative fraction (e.g., 0.01 = 1%) OR absolute value
    clip: Optional[Tuple[float, float]] = None  # hard bounds after noise (min, max)

def apply_noise(x: np.ndarray, spec: NoiseSpec, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if spec.mode == "relative":
        eps = rng.normal(loc=0.0, scale=spec.sigma, size=x.shape)
        y = x * (1.0 + eps)
    elif spec.mode == "absolute":
        eps = rng.normal(loc=0.0, scale=spec.sigma, size=x.shape)
        y = x + eps
    else:
        raise ValueError(f"Unknown noise mode: {spec.mode}")

    if spec.clip is not None:
        y = np.clip(y, spec.clip[0], spec.clip[1])
    return y

def mc_dropout_predict(model, x: np.ndarray, T: int = 50, batch_size: int = 2048):
    """
    Monte Carlo Dropout inference:
    returns mean and std across T stochastic forward passes.

    NOTE: Works only if the model contains Dropout layers and you call model(x, training=True).
    """
    import tensorflow as tf

    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
    preds = []
    # batch the sampling to avoid GPU/CPU memory spikes for large x
    n = x.shape[0]
    for t in range(T):
        out = []
        for i in range(0, n, batch_size):
            xb = x_tf[i:i+batch_size]
            yb = model(xb, training=True)  # dropout ON
            out.append(yb)
        preds.append(tf.concat(out, axis=0))
    preds = tf.stack(preds, axis=0)  # [T, N, D]
    mean = tf.reduce_mean(preds, axis=0).numpy()
    std  = tf.math.reduce_std(preds, axis=0).numpy()
    return mean, std
