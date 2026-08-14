# ============================================================
# health_data_utils.py
# Data inspection and preprocessing utilities for
# IoMT Health Monitoring Systems.
#
# Designed for use with:
#   DQN, PPO, and other discrete-action DRL algorithms.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EPS = 1e-8


# ============================================================
#            Visualization Utilities (Health-Specific)
# ============================================================
def plot_health_features(
    data: np.ndarray,
    feature_names: list,
    time_index=None,
    num_cols: int = 2,
    plot_diff: bool = False,
):
    """
    Plot health-related time-series features.

    Parameters
    ----------
    data : np.ndarray
        Shape (T, F)
    feature_names : list
        Names of features (length F)
    time_index : array-like or None
        Optional time index for plotting
    num_cols : int
        Number of subplot columns
    plot_diff : bool
        If True, plot first-order differences
    """
    if plot_diff:
        data = compute_temporal_differences(data)
        title_suffix = " (Δ)"
    else:
        title_suffix = ""

    T, F = data.shape
    num_rows = int(np.ceil(F / num_cols))

    fig, axes = plt.subplots(
        num_rows, num_cols,
        figsize=(4 * num_cols, 3 * num_rows),
        squeeze=False
    )

    for i in range(F):
        r, c = divmod(i, num_cols)
        x = time_index if time_index is not None else np.arange(T)
        axes[r, c].plot(x, data[:, i])
        axes[r, c].set_title(feature_names[i] + title_suffix)
        axes[r, c].grid(alpha=0.3)

    # Hide empty subplots
    for j in range(F, num_rows * num_cols):
        r, c = divmod(j, num_cols)
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.show()


# ============================================================
#        Time-Series Transformations
# ============================================================
def compute_temporal_differences(data: np.ndarray):
    """
    Compute first-order temporal differences.
    Useful for detecting rapid changes in vital signs.

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
    zero_pad = np.zeros((1, data.shape[1]), dtype=data.dtype)
    return np.concatenate([zero_pad, diffs], axis=0)


def normalize_by_last_health_state(window: np.ndarray):
    """
    Normalize a sliding window by its last step.
    Useful for making the agent focus on relative changes.

    Parameters
    ----------
    window : np.ndarray
        Shape (window_length, features)

    Returns
    -------
    normalized : np.ndarray
    """
    ref = window[-1] + EPS
    return window / ref


def min_max_scale_window(window: np.ndarray):
    """
    Min-max scale features within a sliding window.

    Parameters
    ----------
    window : np.ndarray
        Shape (window_length, features)

    Returns
    -------
    scaled : np.ndarray
    """
    min_v = window.min(axis=0, keepdims=True)
    max_v = window.max(axis=0, keepdims=True)
    return (window - min_v) / (max_v - min_v + EPS)


# ============================================================
#          Feature Selection (Health-Specific)
# ============================================================
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

ALL_FEATURES = HEALTH_FEATURES + SIMULATED_FEATURES


def select_health_features(data: np.ndarray, feature_indices=None):
    """
    Select a subset of features.

    Parameters
    ----------
    data : np.ndarray
        Shape (..., features)
    feature_indices : list or None
        Indices of features to keep

    Returns
    -------
    filtered : np.ndarray
    """
    if feature_indices is None:
        return data
    return data[..., feature_indices]


# ============================================================
#       Tensor Reshaping for Networks
# ============================================================
def to_rnn_format(window: np.ndarray):
    """
    RNN-compatible format.

    (window_length, features)
    """
    return window


def to_cnn_format(window: np.ndarray):
    """
    CNN-compatible format.

    From:
        (window_length, features)
    To:
        (features, 1, window_length)
    """
    return np.transpose(window, (1, 0))[..., None]


def to_cnn_rnn_format(window: np.ndarray):
    """
    Hybrid CNN-RNN format.

    From:
        (window_length, features)
    To:
        (features, window_length)
    """
    return np.transpose(window, (1, 0))


# ============================================================
#       Health-Event Detection (Optional)
# ============================================================
def detect_health_event_from_state(state: np.ndarray, thresholds: dict = None):
    """
    Simple rule-based health event detection from a single state.

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
