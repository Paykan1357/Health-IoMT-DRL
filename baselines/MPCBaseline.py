
# ============================================================
# mpc_baseline.py
# Model Predictive Control Baseline for Health Monitoring
# با ۴ اقدام گسسته و افق پیش‌بینی کوتاه
# ============================================================

import numpy as np
from itertools import product

class MPCBaseline:
    """
    MPC for discrete actions in IoMT health monitoring.
    At each step, it evaluates all possible action sequences of length H
    and selects the first action of the sequence that minimizes a cost function.
    The cost is the negative of cumulative reward (or a custom cost).
    """

    def __init__(self, horizon=5, delta_t=1.0):
        self.H = horizon
        self.delta_t = delta_t
        self.action_space = [0, 1, 2, 3]

        # Parameters for simulation (must match environment)
        self.battery_drain_rates = {0: 0.08, 1: 0.15, 2: 0.02, 3: 0.08}
        self.congestion_changes = {0: 0.01, 1: 0.03, 2: -0.02, 3: -0.05}
        self.congestion_noise_std = 0.01

        # Reward weights (same as in health_controller.py)
        self.lambda_latency = 1.0
        self.lambda_energy = 0.3
        self.lambda_pdr = 1.5
        self.lambda_battery = 0.5
        self.lambda_alignment = 0.3

    def solve(self, predicted_prob, battery_level, network_congestion):
        """
        حل MPC با جستجوی تمام دنباله‌های اقدام به طول H.
        
        ورودی:
            predicted_prob: float [0,1] – احتمال بحران از LSTM
            battery_level: float [0,100] – درصد باتری
            network_congestion: float [0,1] – ازدحام شبکه
        
        خروجی:
            best_action: int در {0,1,2,3}
        """
        best_action = 0
        best_cost = float('inf')

        # تولید تمام دنباله‌های ممکن (4^H)
        for action_seq in product(self.action_space, repeat=self.H):
            cost = self._evaluate_sequence(action_seq, predicted_prob, battery_level, network_congestion)
            if cost < best_cost:
                best_cost = cost
                best_action = action_seq[0]  # فقط اولین اقدام را برمی‌گردانیم

        return best_action

    def _evaluate_sequence(self, action_seq, pred_prob, battery, congestion):
        """
        شبیه‌سازی یک دنباله از اقدامات و محاسبه هزینه (منفی جمع پاداش‌ها).
        """
        total_reward = 0.0
        bat = battery
        cong = congestion

        for t, action in enumerate(action_seq):
            # محاسبه پاداش گام جاری با استفاده از تابع مشابه محیط
            reward = self._compute_step_reward(action, pred_prob, bat, cong)
            total_reward += reward

            # به‌روزرسانی باتری و ازدحام برای گام بعدی
            bat = self._update_battery(bat, action)
            cong = self._update_congestion(cong, action)

            # (اختیاری) می‌توانیم در صورت اتمام افق، یک هزینه نهایی برای وضعیت نهایی اضافه کنیم
            # مثلاً اگر باتری خیلی کم باشد یا ازدحام بالا باشد، جریمه می‌دهیم.

        # هزینه = منفی جمع پاداش‌ها (چون می‌خواهیم کمینه شود)
        # همچنین می‌توانیم یک ترم پایانی اضافه کنیم
        terminal_penalty = self._terminal_cost(bat, cong)
        return -total_reward + terminal_penalty

    def _compute_step_reward(self, action, pred_prob, battery, congestion):
        """
        محاسبه پاداش برای یک گام مشخص – دقیقاً مشابه تابع _compute_reward در health_controller.py.
        """
        # تأخیر (Latency)
        if action == 1:
            latency = 5 + 10 * congestion
        elif action == 3:
            if congestion > 0.7:
                latency = 8 + 5 * congestion
            else:
                latency = 15 + 10 * congestion
        elif action == 2:
            latency = 7 + 8 * congestion
        else:
            latency = 10 + 15 * congestion

        # انرژی (Energy)
        if action == 1:
            energy = 2.5
        elif action == 3:
            energy = 1.8
        elif action == 2:
            energy = 0.5
        else:
            energy = 1.0

        # PDR
        if action == 1:
            pdr = 0.99 - 0.05 * congestion
        elif action == 3:
            if congestion > 0.7:
                pdr = 0.98 - 0.02 * congestion
            else:
                pdr = 0.92 - 0.05 * congestion
        elif action == 2:
            pdr = 0.95 - 0.08 * congestion
        else:
            pdr = 0.90 - 0.10 * congestion
        pdr = max(0.5, min(1.0, pdr))

        # اجزای پاداش
        r_latency = -latency / 10.0
        r_energy = -energy / 2.0
        r_pdr = pdr * 5.0
        r_battery = 2.0 if battery > 30 else -5.0

        # هم‌سویی با پیش‌بینی (alignment)
        if pred_prob >= 0.5:
            if action == 1:
                r_alignment = 15.0 * pred_prob
            else:
                r_alignment = -30.0
        else:
            if action == 1:
                r_alignment = -10.0
            else:
                r_alignment = 2.0

        # پاداش نهایی
        reward = (
            self.lambda_latency * r_latency +
            self.lambda_energy * r_energy +
            self.lambda_pdr * r_pdr +
            self.lambda_battery * r_battery +
            self.lambda_alignment * r_alignment
        )
        return np.clip(reward, -20.0, 20.0)

    def _update_battery(self, battery, action):
        """به‌روزرسانی باتری بر اساس اقدام (همانند health_controller)."""
        drain_rate = self.battery_drain_rates.get(action, 0.08)
        new_battery = max(0.0, battery - drain_rate * self.delta_t)
        return new_battery

    def _update_congestion(self, congestion, action):
        """به‌روزرسانی ازدحام شبکه بر اساس اقدام (با نویز تصادفی)."""
        change = self.congestion_changes.get(action, 0.01)
        noise = np.random.normal(0, self.congestion_noise_std)
        new_cong = np.clip(congestion + change + noise, 0.0, 1.0)
        return new_cong

    def _terminal_cost(self, battery, congestion):
        """
        هزینه پایانی برای وضعیت نهایی بعد از افق.
        هدف: تشویق به حفظ باتری و جلوگیری از ازدحام زیاد.
        """
        bat_penalty = 0.0
        if battery < 20:
            bat_penalty = 50.0 * (20 - battery) / 20.0
        cong_penalty = 0.0
        if congestion > 0.8:
            cong_penalty = 50.0 * (congestion - 0.8) / 0.2
        return bat_penalty + cong_penalty
