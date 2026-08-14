# ============================================================
# health_env.py
# Health Monitoring Environment for IoMT Networks
# Discrete Action Deep Reinforcement Learning (DQN / PPO)
# با رویکرد ترکیبی: Supervised Predictor + RL
# ============================================================

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class HealthEnvironment:
    """
    Environment for IoMT Health Monitoring with Supervised Predictor.
    
    State features (11 total per time step):
    - 8 health features: HR, SpO2, SBP, DBP, Temp, RR, Activity, Stress
    - 1 predicted probability of critical event (from LSTM)
    - 2 simulated features: Battery Level, Network Congestion
    """

    REQUIRED_FEATURES = [
        "heart_rate",
        "blood_oxygen",
        "blood_pressure_systolic",
        "blood_pressure_diastolic",
        "body_temperature",
        "respiratory_rate",
        "activity_level",
        "stress_level",
        "health_event"
    ]

    ACTION_NAMES = {
        0: "Normal Route",
        1: "High Priority Alert",
        2: "Reduce Sampling Rate",
        3: "Reroute via Backup"
    }

    def __init__(
        self,
        data_dir,
        filename,
        window_length=5,
        split="train",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
        predictor_path="./models/predictor.pth",
        curriculum_epochs=200,
    ):
        self.data_dir = data_dir
        self.filename = filename
        self.window_length = window_length
        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.curriculum_epochs = curriculum_epochs
        self.episode_count = 0
        self.np_rng = np.random.default_rng(seed)

        # Simulated state variables
        self.battery_level = 100.0
        self.network_congestion = 0.3
        self.current_step = 0
        self.current_patient_id = None
        self.patient_data = None
        self.patient_ids = None
        self.patient_weights = None
        self.last_prediction = 0.0

        # ============================================================
        # Load Supervised Predictor (LSTM)
        # ============================================================
        self.predictor = self._load_predictor(predictor_path)
        self.predictor.eval()

        self._load_and_validate_data()
        self._apply_split()
        self._build_patient_episodes()
        self.reset()

    # ============================================================
    # Data Loading
    # ============================================================
    def _load_and_validate_data(self):
        file_path = os.path.join(self.data_dir, self.filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset not found: {file_path}")

        df = pd.read_csv(file_path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        else:
            raise KeyError("timestamp column missing")

        missing = [f for f in self.REQUIRED_FEATURES if f not in df.columns]
        if missing:
            raise KeyError(f"Missing features: {missing}")

        self.feature_names = self.REQUIRED_FEATURES
        self.feature_indices = {name: i for i, name in enumerate(self.feature_names)}
        self.num_health_features = 8  # Excluding health_event
        self.num_features = 11  # 8 health + 1 prediction + 2 simulated

        self.df = df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
        self.all_patient_ids = self.df["patient_id"].unique().tolist()

    def _apply_split(self):
        n = len(self.all_patient_ids)
        indices = np.arange(n)
        self.np_rng.shuffle(indices)
        n_train = int(self.train_ratio * n)
        n_val = int(self.val_ratio * n)

        train_ids = [self.all_patient_ids[i] for i in indices[:n_train]]
        val_ids = [self.all_patient_ids[i] for i in indices[n_train:n_train+n_val]]
        test_ids = [self.all_patient_ids[i] for i in indices[n_train+n_val:]]

        if self.split == "train":
            self.patient_ids = train_ids
        elif self.split == "val":
            self.patient_ids = val_ids
        elif self.split == "test":
            self.patient_ids = test_ids
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

        print(f"[{self.split.upper()}] Patients (initial): {len(self.patient_ids)}")

    def _build_patient_episodes(self):
        """Build episodes and compute weights for Curriculum Learning."""
        self.patient_data_cache = {}
        valid_patient_ids = []
        critical_counts = []

        for pid in self.patient_ids:
            patient_df = self.df[self.df["patient_id"] == pid].copy()
            patient_df = patient_df.sort_values("timestamp").reset_index(drop=True)
            data = patient_df[self.REQUIRED_FEATURES].to_numpy(dtype=np.float32)

            if len(data) > self.window_length:
                health_events = data[:, self.feature_indices["health_event"]]
                critical_count = np.sum(health_events >= 2)

                self.patient_data_cache[pid] = {
                    "data": data,
                    "length": len(data),
                    "timestamps": patient_df["timestamp"].tolist(),
                    "critical_count": critical_count,
                }
                valid_patient_ids.append(pid)
                critical_counts.append(critical_count + 1)

        self.patient_ids = valid_patient_ids
        if len(self.patient_ids) == 0:
            raise ValueError("No patients with sufficient data.")

        self.patient_weights = np.array(critical_counts, dtype=np.float32)
        self.patient_weights /= self.patient_weights.sum()

        print(f"[{self.split.upper()}] Valid Patients: {len(self.patient_ids)}")
        print(f"[{self.split.upper()}] Total critical events: {sum(critical_counts)-len(self.patient_ids)}")

    # ============================================================
    # Supervised Predictor
    # ============================================================
    def _load_predictor(self, path):
        """Load the trained LSTM predictor."""
        from supervised_predictor import HealthEventPredictor
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = HealthEventPredictor(input_dim=8)
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
            print(f"[INFO] Predictor loaded from {path}")
        else:
            print(f"[WARNING] Predictor not found at {path}. Using zero predictions.")
        model.to(device)
        model.eval()
        return model

    def _predict_health_event(self, window_features):
        """
        Predict probability of critical event (class 2 or 3) from 5-step window.
        
        Args:
            window_features: (window_length, 8) - 8 health features
        
        Returns:
            critical_prob: float between 0 and 1
        """
        if self.predictor is None:
            return 0.0
        
        device = next(self.predictor.parameters()).device
        with torch.no_grad():
            x = torch.tensor(window_features, dtype=torch.float32).unsqueeze(0).to(device)
            logits = self.predictor(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            critical_prob = probs[2] + probs[3]  # classes 2 and 3
        return float(critical_prob)

    # ============================================================
    # Core Environment
    # ============================================================
    def reset(self, episode=None):
        """Reset with Curriculum Learning: oversample critical patients early."""
        if episode is not None:
            self.episode_count = episode
        else:
            self.episode_count += 1

        progress = min(1.0, self.episode_count / self.curriculum_epochs)
        oversample_factor = 1.0 + 4.0 * (1.0 - progress)

        weights = self.patient_weights.copy()
        for i, pid in enumerate(self.patient_ids):
            if self.patient_data_cache[pid].get("critical_count", 0) > 0:
                weights[i] *= oversample_factor
        weights = weights / weights.sum()

        self.current_patient_id = self.np_rng.choice(self.patient_ids, p=weights)

        patient_info = self.patient_data_cache[self.current_patient_id]
        self.patient_data = patient_info["data"]
        self.total_steps = len(self.patient_data) - self.window_length
        if self.total_steps <= 0:
            raise ValueError(f"Patient {self.current_patient_id} has insufficient data.")

        self.battery_level = 100.0
        self.network_congestion = 0.3
        self.current_step = 0
        self.last_prediction = 0.0

        return self.get_observation(self.current_step)

    def get_observation(self, step):
        """
        Build observation window with 11 features per step:
        - 8 health features
        - 1 predicted critical probability
        - 2 simulated features (battery, congestion)
        """
        # Get health data window (including health_event)
        health_window = self.patient_data[step: step + self.window_length].copy()
        # Extract 8 health features (excluding health_event)
        health_features = np.delete(health_window, self.feature_indices["health_event"], axis=1)

        # Predict critical probability from health features
        predicted_prob = self._predict_health_event(health_features)
        self.last_prediction = predicted_prob

        # Simulate battery and congestion
        bat_levels = self._simulate_battery_window(step)
        cong_levels = self._simulate_congestion_window(step)

        # Combine: 8 health + 1 prediction + 1 battery + 1 congestion = 11 features per step
        obs = np.column_stack([
            health_features,                       # (window_length, 8)
            np.full((self.window_length, 1), predicted_prob, dtype=np.float32),  # (window_length, 1)
            bat_levels,                            # (window_length, 1)
            cong_levels                            # (window_length, 1)
        ])
        return obs.astype(np.float32)

    def get_current_observation(self, step):
        """Return single step observation (for health_event extraction)."""
        if step >= len(self.patient_data):
            return None
        return self.patient_data[step].copy()

    def _simulate_battery_window(self, step):
        bat_levels = np.zeros(self.window_length, dtype=np.float32)
        bat = self.battery_level
        for i in range(self.window_length):
            step_idx = step + i
            if step_idx >= len(self.patient_data):
                break
            activity = self.patient_data[step_idx, self.feature_indices["activity_level"]]
            drain_rate = 0.01 + 0.02 * activity
            bat = max(0, bat - drain_rate)
            bat_levels[i] = bat
        self.battery_level = bat_levels[-1]
        return bat_levels

    def _simulate_congestion_window(self, step):
        cong_levels = np.zeros(self.window_length, dtype=np.float32)
        cong = self.network_congestion
        for i in range(self.window_length):
            drift = 0.02 * (0.4 - cong)
            noise = self.np_rng.normal(0, 0.05)
            cong = np.clip(cong + drift + noise, 0.0, 1.0)
            cong_levels[i] = cong
        self.network_congestion = cong_levels[-1]
        return cong_levels

    def step(self):
        if self.current_step >= self.total_steps - 1:
            done = True
            obs = self.get_observation(self.current_step)
            next_obs = obs
        else:
            done = False
            obs = self.get_observation(self.current_step)
            self.current_step += 1
            next_obs = self.get_observation(self.current_step)
        return obs, next_obs, done

    # ============================================================
    # Utility Methods
    # ============================================================
    def get_patient_ids(self):
        return self.patient_ids.copy()

    def get_total_episodes(self):
        return len(self.patient_ids)

    def compute_mean_std(self, feature_type="health"):
        """Compute mean/std for normalization (excluding prediction and simulated)."""
        all_data = []
        for pid in self.patient_ids:
            data = self.patient_data_cache[pid]["data"]
            health_data = np.delete(data, self.feature_indices["health_event"], axis=1)
            all_data.append(health_data)
        combined = np.vstack(all_data)
        mean = np.mean(combined, axis=0)
        std = np.std(combined, axis=0)
        std[std < 1e-8] = 1.0
        # For the full 11-feature state: add mean/std for prediction (0.5, 0.3) and simulated (0.5, 0.3)
        mean = np.concatenate([mean, [0.5, 0.5, 0.5]])
        std = np.concatenate([std, [0.3, 0.3, 0.3]])
        return mean, std

    # ============================================================
    # Torch Dataset
    # ============================================================
    def get_dataset(self):
        return HealthDataset(self)


class HealthDataset(Dataset):
    def __init__(self, env: HealthEnvironment):
        self.env = env
        self.patient_ids = env.patient_ids
        self.cache = env.patient_data_cache
        self.window_length = env.window_length
        self.samples = []
        for pid in self.patient_ids:
            data = self.cache[pid]["data"]
            n_steps = len(data) - self.window_length
            for step in range(n_steps):
                self.samples.append((pid, step))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid, step = self.samples[idx]
        data = self.cache[pid]["data"]
        obs = data[step: step + self.window_length]
        next_obs = data[step + 1: step + self.window_length + 1]
        return obs.astype(np.float32), next_obs.astype(np.float32)
