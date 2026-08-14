# ============================================================
# health_controller.py
# IoMT Health Monitoring Environment for Discrete Action DRL
# DQN / PPO Implementation
# با رویکرد ترکیبی: Supervised Predictor + RL
# ============================================================

import gym
import numpy as np
from gym import spaces


class HealthEnvironment(gym.Env):
    """
    IoMT Health Monitoring Environment with Discrete Actions.
    
    Action Space (4 discrete):
    0: Normal Route
    1: High Priority Alert
    2: Reduce Sampling Rate
    3: Reroute via Backup
    
    State Space (11 features per time step):
    - 8 health features: HR, SpO2, SBP, DBP, Temp, RR, Activity, Stress
    - 1 predicted critical probability (from LSTM)
    - 2 simulated features: Battery Level, Network Congestion
    
    State dimension: window_length * 11
    """
    def __init__(self, config, health_data):
        super().__init__()

        self.health_data = health_data
        self.window_length = getattr(config, "window_length", 5)
        self.delta_t = getattr(config, "delta_t", 1.0)

        self.action_space = spaces.Discrete(4)
        self.action_names = {
            0: "Normal Route",
            1: "High Priority Alert",
            2: "Reduce Sampling Rate",
            3: "Reroute via Backup"
        }

        # ============================================================
        # State Space: 11 features per time step
        # 8 health + 1 prediction + 2 simulated = 11
        # ============================================================
        self.num_health_features = 9   # 8 health + 1 prediction
        self.num_simulated_features = 2  # battery + congestion
        self.num_features = self.num_health_features + self.num_simulated_features  # 11
        
        obs_dim = self.window_length * self.num_features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Feature map for health_data (original 9 features including health_event)
        self.feature_map = {
            "heart_rate": 0,
            "blood_oxygen": 1,
            "blood_pressure_systolic": 2,
            "blood_pressure_diastolic": 3,
            "body_temperature": 4,
            "respiratory_rate": 5,
            "activity_level": 6,
            "stress_level": 7,
            "health_event": 8
        }

        # Reward weights
        self.lambda_med = getattr(config, "lambda_med", 1.0)
        self.lambda_net = getattr(config, "lambda_net", 0.5)
        self.lambda_bat = getattr(config, "lambda_bat", 0.3)
        self.lambda_pred = getattr(config, "lambda_pred", 0.5)  # وزن برای هم‌سویی با پیش‌بینی

        self.reset()

    def reset(self, episode=None):
        """
        Reset environment.
        
        Args:
            episode: Optional episode number for curriculum learning.
        """
        self.current_step = 0
        self.battery_level = 100.0
        self.network_congestion = 0.3
        self.previous_action = 0
        
        self.health_data.reset(episode=episode)
        self.current_patient_id = self.health_data.current_patient_id
        
        return self._get_state()

    def _get_state(self):
        """Get flattened state vector from health_data observation."""
        obs = self.health_data.get_observation(self.current_step)
        state = obs.flatten().astype(np.float32)
        return state

    def step(self, action):
        action = int(np.clip(action, 0, 3))
        
        # Get current health_event for reward calculation
        current_health_event = self._get_current_health_event()
        
        # Get prediction probability from health_data
        predicted_prob = self.health_data.last_prediction
        
        # Compute reward
        reward = self._compute_reward(action, current_health_event, predicted_prob)

        # Step the underlying health data environment
        next_obs, _, done = self.health_data.step()
        
        # Update simulated variables
        self._update_simulated_variables(action)

        self.previous_action = action

        info = {
            "health_event": current_health_event,
            "action": action,
            "action_name": self.action_names[action],
            "battery_level": self.battery_level,
            "network_congestion": self.network_congestion,
            "predicted_prob": predicted_prob,
        }

        next_state = self._get_state()
        self.current_step += 1

        return next_state, float(reward), done, info

    def _get_current_health_event(self):
        """Get the current health_event value from the health_data."""
        obs = self.health_data.get_current_observation(self.current_step)
        if obs is None:
            return 0
        health_event = int(obs[self.feature_map["health_event"]])
        return health_event

    # ============================================================
# health_controller.py – بخش پاداش (اصلاح‌شده)
# ============================================================

    def _compute_reward(self, action, health_event, predicted_prob):
        """
        پاداش فقط بر اساس عملکرد شبکه:
        - تأخیر (Latency)
        - مصرف انرژی (Energy)
        - Packet Delivery Ratio (PDR)
        - مدیریت باتری
        """
    
        # ============================================================
        # 1. محاسبه‌ی تأخیر بر اساس اقدام و ازدحام
        # ============================================================
        if action == 1:  # High Priority Alert
            latency = 5 + 10 * self.network_congestion
        elif action == 3:  # Reroute
            if self.network_congestion > 0.7:
                latency = 8 + 5 * self.network_congestion
            else:
                latency = 15 + 10 * self.network_congestion
        elif action == 2:  # Reduce Sampling
            latency = 7 + 8 * self.network_congestion
        else:  # Normal Route
            latency = 10 + 15 * self.network_congestion
    
        # ============================================================
        # 2. محاسبه‌ی مصرف انرژی بر اساس اقدام
        # ============================================================
        if action == 1:
            energy = 2.5
        elif action == 3:
            energy = 1.8
        elif action == 2:
            energy = 0.5
        else:
            energy = 1.0
    
        # ============================================================
        # 3. محاسبه‌ی PDR بر اساس اقدام و ازدحام
        # ============================================================
        if action == 1:
            pdr = 0.99 - 0.05 * self.network_congestion
        elif action == 3:
            if self.network_congestion > 0.7:
                pdr = 0.98 - 0.02 * self.network_congestion
            else:
                pdr = 0.92 - 0.05 * self.network_congestion
        elif action == 2:
            pdr = 0.95 - 0.08 * self.network_congestion
        else:
            pdr = 0.90 - 0.10 * self.network_congestion
        pdr = max(0.5, min(1.0, pdr))
    
        # ============================================================
        # 4. پاداش نهایی (ترکیبی با وزن‌های مناسب)
        # ============================================================
        # هدف: کم کردن تأخیر و انرژی، زیاد کردن PDR
        r_latency = -latency / 10.0          # هرچه تأخیر کمتر، پاداش بیشتر
        r_energy = -energy / 2.0             # هرچه انرژی کمتر، پاداش بیشتر
        r_pdr = pdr * 5.0                    # هرچه PDR بیشتر، پاداش بیشتر
        r_battery = 2.0 if self.battery_level > 30 else -5.0  # مدیریت باتری
    
        # ✅ پاداش هم‌سویی با پیش‌بینی (اما نه به‌عنوان هدف اصلی)
        # اگر پیش‌بینی بحران است و اقدام ۱ انتخاب شده، پاداش کوچک
        # اگر پیش‌بینی بحران است و اقدام ۱ انتخاب نشده، جریمه کوچک
        if predicted_prob >= 0.5:
            if action == 1:
                r_alignment = 15.0 * predicted_prob
            else:
                r_alignment = -30.0   # جریمه برای نادیده گرفتن بحران
        else:
            if action == 1:
                r_alignment = -10.0    # جریمه برای هشدار اشتباه
            else:
                r_alignment = 2.0
    
        # ============================================================
        # پاداش نهایی (وزن‌های تنظیم‌شده)
        # ============================================================
        reward = (
            1.0 * r_latency +
            0.3 * r_energy +
            1.5 * r_pdr +
            0.5 * r_battery +
            0.3 * r_alignment
        )
    
        reward = np.clip(reward, -20.0, 20.0)
        return float(reward)

    def _update_simulated_variables(self, action):
        """Update battery and congestion based on action."""
        if action == 1:
            drain_rate = 0.15
        elif action == 2:
            drain_rate = 0.02
        else:
            drain_rate = 0.08
        self.battery_level = max(0.0, self.battery_level - drain_rate * self.delta_t)

        if action == 3:
            congestion_change = -0.05
        elif action == 1:
            congestion_change = 0.03
        elif action == 2:
            congestion_change = -0.02
        else:
            congestion_change = 0.01
        self.network_congestion = np.clip(
            self.network_congestion + congestion_change + np.random.normal(0, 0.01),
            0.0, 1.0
        )

    def render(self, mode='human'):
        if mode == 'human':
            print(f"Step: {self.current_step}, Battery: {self.battery_level:.1f}%, Congestion: {self.network_congestion:.2f}")
        return None

    def close(self):
        pass
