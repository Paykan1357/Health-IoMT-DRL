# ============================================================
# data_utils.py
# Core preprocessing utilities for IoMT Health Monitoring.
#
# State definition (for this project):
#   s_t = Flattened window of (window_length x 10 features)
#   Features: HR, SpO2, SBP, DBP, Temp, RR, Activity, Stress, Battery, Congestion
#
# Action space (discrete):
#   a_t in {0, 1, 2, 3}
#   (Normal Route, High Priority Alert, Reduce Sampling, Reroute)
#
# Suitable for DQN, PPO, and other discrete-action DRL agents.
# ============================================================

import numpy as np
import torch

# ============================================================
# Constants
# ============================================================
FLOAT = torch.float32
EPS = 1e-8

# Action dimension (discrete: 4 actions)
ACTION_DIM = 4

# Feature configuration
HEALTH_FEATURES = [
    "heart_rate",
    "blood_oxygen",
    "blood_pressure_systolic",
    "blood_pressure_diastolic",
    "body_temperature",
    "respiratory_rate",
    "activity_level",
    "stress_level",
]
SIMULATED_FEATURES = ["battery_level", "network_congestion"]
ALL_FEATURES = HEALTH_FEATURES + SIMULATED_FEATURES  # 10 features
NUM_FEATURES = len(ALL_FEATURES)


# ============================================================
#                 DATA CLEANING (Unchanged)
# ============================================================
def clean_timeseries(data: np.ndarray):
    """
    Clean raw time-series data.
    
    Operations:
    - Replace NaN / Inf
    - Forward-fill then backward-fill
    - Clip extreme outliers (1st–99th percentile)

    Parameters
    ----------
    data : np.ndarray
        Shape (T, F)

    Returns
    -------
    cleaned : np.ndarray
    """
    data = data.copy()

    # Replace inf with nan
    data[~np.isfinite(data)] = np.nan

    # Forward fill
    for t in range(1, data.shape[0]):
        nan_mask = np.isnan(data[t])
        data[t, nan_mask] = data[t - 1, nan_mask]

    # Backward fill (if first rows had NaNs)
    for t in range(data.shape[0] - 2, -1, -1):
        nan_mask = np.isnan(data[t])
        data[t, nan_mask] = data[t + 1, nan_mask]

    # Robust clipping (1st–99th percentile)
    lower = np.percentile(data, 1, axis=0)
    upper = np.percentile(data, 99, axis=0)
    data = np.clip(data, lower, upper)

    return data


# ============================================================
#         TRAINING STATISTICS (STANDARDIZATION)
# ============================================================
def compute_standardization_stats(data: np.ndarray):
    """
    Compute mean and std from TRAIN data only.
    
    Parameters
    ----------
    data : np.ndarray
        Shape (T, F)

    Returns
    -------
    stats : dict
        {"mean": (F,), "std": (F,)}
    """
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std = np.maximum(std, EPS)
    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
    }


def standardize(data: np.ndarray, stats: dict):
    """
    Apply standardization using precomputed TRAIN stats.

    Parameters
    ----------
    data : np.ndarray
        Shape (..., F)
    stats : dict

    Returns
    -------
    standardized : np.ndarray
    """
    return (data - stats["mean"]) / (stats["std"] + EPS)


# ============================================================
#        TIME-SERIES WINDOW NORMALIZATION
# ============================================================
def normalize_window_last_step(window: np.ndarray):
    """
    Normalize time-series window by its last step.
    Useful for detecting relative changes in vital signs.

    Parameters
    ----------
    window : np.ndarray
        Shape (W, F)

    Returns
    -------
    normalized : np.ndarray
    """
    reference = window[-1] + EPS
    return window / reference


def normalize_window_minmax(window: np.ndarray):
    """
    Min-max normalization across time window.

    Parameters
    ----------
    window : np.ndarray
        Shape (W, F)

    Returns
    -------
    normalized : np.ndarray
    """
    min_v = window.min(axis=0, keepdims=True)
    max_v = window.max(axis=0, keepdims=True)
    return (window - min_v) / (max_v - min_v + EPS)


# ============================================================
#        Time-Series Transformations
# ============================================================
def compute_differences(data: np.ndarray):
    """
    Compute first-order temporal differences.
    Useful for detecting sudden changes in vital signs.

    Parameters
    ----------
    data : np.ndarray
        Shape (T, F)

    Returns
    -------
    diffs : np.ndarray
        Shape (T, F)
    """
    diffs = np.diff(data, axis=0)
    pad = np.zeros((1, data.shape[1]), dtype=data.dtype)
    return np.vstack([pad, diffs])


# ============================================================
#        Tensor Reshaping (for Neural Networks)
# ============================================================
def to_rnn_format(window: np.ndarray):
    """
    RNN-compatible format.

    Parameters
    ----------
    window : np.ndarray
        Shape (window_length, features)

    Returns
    -------
    Same shape (window_length, features)
    """
    return window


def to_cnn_format(window: np.ndarray):
    """
    CNN-compatible format.

    From: (window_length, features)
    To:   (features, 1, window_length)

    Parameters
    ----------
    window : np.ndarray
        Shape (window_length, features)

    Returns
    -------
    cnn_input : np.ndarray
        Shape (features, 1, window_length)
    """
    return np.transpose(window, (1, 0))[..., None]


def to_cnn_rnn_format(window: np.ndarray):
    """
    Hybrid CNN-RNN format.

    From: (window_length, features)
    To:   (features, window_length)

    Parameters
    ----------
    window : np.ndarray
        Shape (window_length, features)

    Returns
    -------
    hybrid_input : np.ndarray
        Shape (features, window_length)
    """
    return np.transpose(window, (1, 0))


# ============================================================
#        Health-Event Detection (Optional)
# ============================================================
def detect_health_event_from_state(state: np.ndarray, thresholds: dict = None):
    """
    Simple rule-based health event detection from a single state.
    Useful if health_event label is not available.

    Args:
        state: Array of 8 health features (or 10 including simulated)
        thresholds: Dict of thresholds for each feature

    Returns:
        health_event: integer 0-3
    """
    if thresholds is None:
        thresholds = {
            "heart_rate": (60, 100, 120, 150),
            "blood_oxygen": (95, 90, 85),
            "blood_pressure_systolic": (90, 140, 180),
            "blood_pressure_diastolic": (60, 90, 110),
            "body_temperature": (36.0, 37.5, 38.5),
            "respiratory_rate": (12, 20, 30),
        }

    score = 0
    # HR
    hr = state[0]
    if hr < 60 or hr > 100:
        score += 1
    if 100 <= hr < 120:
        score += 2
    if hr >= 150:
        score += 3

    # SpO2
    spo2 = state[1]
    if spo2 < 95:
        score += 1
    if spo2 < 90:
        score += 2
    if spo2 < 85:
        score += 3

    # SBP
    sbp = state[2]
    if sbp < 90 or sbp > 140:
        score += 1
    if 140 <= sbp < 180:
        score += 2
    if sbp >= 180:
        score += 3

    # DBP
    dbp = state[3]
    if dbp < 60 or dbp > 90:
        score += 1
    if 90 <= dbp < 110:
        score += 2
    if dbp >= 110:
        score += 3

    # Temp
    temp = state[4]
    if temp < 36.0 or temp > 37.5:
        score += 1
    if 37.5 <= temp < 38.5:
        score += 2
    if temp >= 38.5:
        score += 3

    # RR
    rr = state[5]
    if rr < 12 or rr > 20:
        score += 1
    if 20 <= rr < 30:
        score += 2
    if rr >= 30:
        score += 3

    if score <= 2:
        return 0
    elif score <= 4:
        return 1
    elif score <= 6:
        return 2
    else:
        return 3


# ============================================================
#       STATE ASSEMBLY (for this specific project)
# ============================================================
def build_state_from_window(window: np.ndarray):
    """
    Build the state vector from a sliding window.
    
    For this project, state is simply the flattened window.
    
    Parameters
    ----------
    window : np.ndarray
        Shape (window_length, num_features)
    
    Returns
    -------
    state : np.ndarray
        Flattened state vector of shape (window_length * num_features,)
    """
    return window.flatten().astype(np.float32)
