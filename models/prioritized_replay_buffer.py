# ============================================================
# prioritized_replay_buffer.py
# Prioritized Experience Replay with Oversampling for Critical Events
# ============================================================

import random
import numpy as np
import torch


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay with oversampling of critical events.
    """
    
    def __init__(
        self,
        capacity=100000,
        device=None,
        alpha=0.6,
        beta=0.4,
        beta_increment=0.001,
        critical_oversample=3.0,
    ):
        """
        Args:
            capacity: Maximum number of transitions
            device: PyTorch device
            alpha: Prioritization exponent (0 = uniform, 1 = full priority)
            beta: Importance sampling exponent (starts low, increases to 1)
            beta_increment: Amount to increase beta each step
            critical_oversample: Oversampling factor for critical events
        """
        self.capacity = capacity
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.critical_oversample = critical_oversample
        self.epsilon = 1e-6
        
        self.buffer = []
        self.priorities = []
        self.pos = 0
        self.size = 0
    
    def __len__(self):
        return self.size
    
    def add(self, obs, action, reward, done, next_obs, health_event=0):
        obs = np.asarray(obs, dtype=np.float32).flatten()
        next_obs = np.asarray(next_obs, dtype=np.float32).flatten()
        action = int(action)
        
        # محاسبه اولویت بر اساس health_event
        if health_event >= 2:
            max_priority = self.critical_oversample * 2.0  # ~6.0
        elif health_event == 1:
            max_priority = 2.0
        else:
            max_priority = 1.0
        
        if self.size < self.capacity:
            self.buffer.append((obs, action, reward, done, next_obs))
            self.priorities.append(max_priority)
            self.size += 1
        else:
            self.buffer[self.pos] = (obs, action, reward, done, next_obs)
            self.priorities[self.pos] = max_priority
            self.pos = (self.pos + 1) % self.capacity
        
        # افزایش beta به‌مرور
        self.beta = min(1.0, self.beta + self.beta_increment)
    
    def sample_batch(self, batch_size):
        if self.size < batch_size:
            return None
        
        # محاسبه احتمالات بر اساس اولویت‌ها
        priorities = np.array(self.priorities[:self.size], dtype=np.float32)
        probs = priorities ** self.alpha
        probs = probs / probs.sum()
        
        # نمونه‌برداری
        idxs = np.random.choice(self.size, batch_size, p=probs, replace=True)
        
        # استخراج نمونه‌ها
        obs_batch = []
        act_batch = []
        rew_batch = []
        done_batch = []
        next_obs_batch = []
        
        for idx in idxs:
            obs, action, reward, done, next_obs = self.buffer[idx]
            obs_batch.append(obs)
            act_batch.append(action)
            rew_batch.append(reward)
            done_batch.append(done)
            next_obs_batch.append(next_obs)
        
        # محاسبه وزن‌های importance sampling
        weights = (self.size * probs[idxs]) ** (-self.beta)
        weights = weights / weights.max()
        
        # تبدیل به تنسور
        obs = torch.tensor(np.stack(obs_batch), dtype=torch.float32, device=self.device)
        actions = torch.tensor(act_batch, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rew_batch, dtype=torch.float32, device=self.device).unsqueeze(-1)
        dones = torch.tensor(done_batch, dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_obs = torch.tensor(np.stack(next_obs_batch), dtype=torch.float32, device=self.device)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(-1)
        
        return obs, actions, rewards, dones, next_obs, weights, idxs
    
    def update_priorities(self, idxs, td_errors):
        """
        به‌روزرسانی اولویت‌ها بر اساس TD-error
        """
        for idx, td_error in zip(idxs, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.priorities[idx] = max(priority, 1.0)
    
    def is_ready(self, batch_size):
        return self.size >= batch_size
    
    def clear(self):
        self.buffer = []
        self.priorities = []
        self.pos = 0
        self.size = 0
