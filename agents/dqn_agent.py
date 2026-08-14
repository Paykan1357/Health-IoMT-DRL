# ============================================================
# dqn_agent.py (اصلاح‌شده)
# با Prioritized Replay + Double DQN
# ============================================================

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from prioritized_replay_buffer import PrioritizedReplayBuffer


# ============================================================
# Dueling Q-Network (اختیاری)
# ============================================================
class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, hidden_layers=2):
        super().__init__()
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.shared = nn.Sequential(*layers)
        self.value = nn.Linear(hidden_dim, 1)
        self.advantage = nn.Linear(hidden_dim, action_dim)
        nn.init.uniform_(self.value.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.advantage.weight, -3e-3, 3e-3)
    
    def forward(self, state):
        x = self.shared(state)
        v = self.value(x)
        a = self.advantage(x)
        return v + a - a.mean(dim=-1, keepdim=True)


class StandardQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, hidden_layers=2):
        super().__init__()
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dim, action_dim)
        nn.init.uniform_(self.output.weight, -3e-3, 3e-3)
    
    def forward(self, state):
        return self.output(self.network(state))


class DQNAgent:
    def __init__(self, env, args=None):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        
        # Hyperparameters
        self.batch_size = getattr(args, "batch_size", 128)
        self.gamma = getattr(args, "gamma", 0.99)
        self.tau = getattr(args, "tau", 0.005)
        self.lr = getattr(args, "lr", 1e-4)
        self.buffer_capacity = getattr(args, "buffer_size", 200000)
        
        self.epsilon = getattr(args, "epsilon_start", 1.0)
        self.epsilon_min = getattr(args, "epsilon_min", 0.01)
        self.epsilon_decay = getattr(args, "epsilon_decay", 0.998)
        self.epsilon_steps = getattr(args, "epsilon_steps", 30000)
        
        self.use_dueling = getattr(args, "use_dueling", True)
        self.use_double_dqn = getattr(args, "use_double_dqn", True)
        self.total_steps = 0
        
        # Networks
        if self.use_dueling:
            self.q_network = DuelingQNetwork(self.state_dim, self.action_dim).to(self.device)
            self.target_network = DuelingQNetwork(self.state_dim, self.action_dim).to(self.device)
        else:
            self.q_network = StandardQNetwork(self.state_dim, self.action_dim).to(self.device)
            self.target_network = StandardQNetwork(self.state_dim, self.action_dim).to(self.device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)
        
        # ✅ Prioritized Replay Buffer
        # در __init__

        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=self.buffer_capacity,
            device=self.device,
            alpha=0.6,
            beta=0.4,
            beta_increment=0.001,
            critical_oversample=3.0,   # ✅ اضافه شد
        )
        
        self.checkpoint_dir = getattr(args, "checkpoint_dir", "./checkpoints_DQN")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        print(f"[DQN] Initialized (Dueling={self.use_dueling}, Double={self.use_double_dqn})")
    
    def select_action(self, state, deterministic=False):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        if not deterministic and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        
        with torch.no_grad():
            q_values = self.q_network(state)
            return q_values.argmax(dim=-1).item()
    
    def store_transition(self, state, action, reward, next_state, done, health_event=0):
        self.replay_buffer.add(state, action, reward, done, next_state, health_event)
        self.total_steps += 1
        self._update_epsilon()
    
    def _update_epsilon(self):
        if self.epsilon_steps > 0:
            self.epsilon = max(self.epsilon_min, self.epsilon - (1.0 - self.epsilon_min) / self.epsilon_steps)
        else:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update(self):
        if not self.replay_buffer.is_ready(self.batch_size):
            return None
        
        # Sample batch with priorities
        batch = self.replay_buffer.sample_batch(self.batch_size)
        if batch is None:
            return None
        
        states, actions, rewards, dones, next_states, weights, idxs = batch
        
        # ---------- TD Target ----------
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: Q-network selects action, target-network evaluates
                next_actions = self.q_network(next_states).argmax(dim=-1, keepdim=True)
                next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(-1)
            else:
                next_q_values = self.target_network(next_states).max(dim=-1)[0]
            
            target = rewards.squeeze(-1) + self.gamma * (1 - dones.squeeze(-1)) * next_q_values
        
        # ---------- Current Q-values ----------
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        # ---------- Weighted Loss ----------
        td_errors = (current_q_values - target).detach().cpu().numpy()
        loss = (weights.squeeze(-1) * F.mse_loss(current_q_values, target, reduction='none')).mean()
        
        # ---------- Optimize ----------
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        
        # Update priorities
        self.replay_buffer.update_priorities(idxs, td_errors)
        
        # Soft update target
        self._soft_update_target()
        
        return {
            "loss": loss.item(),
            "q_value_mean": current_q_values.mean().item(),
            "epsilon": self.epsilon,
        }
    
    def _soft_update_target(self):
        for target_param, source_param in zip(
            self.target_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(self.tau * source_param.data + (1 - self.tau) * target_param.data)
    
    def hard_update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_models(self, episode=None):
        suffix = f"_ep{episode}" if episode is not None else ""
        torch.save(self.q_network.state_dict(), f"{self.checkpoint_dir}/q_network{suffix}.pth")
        torch.save(self.target_network.state_dict(), f"{self.checkpoint_dir}/target_network{suffix}.pth")
        torch.save(self.optimizer.state_dict(), f"{self.checkpoint_dir}/optimizer{suffix}.pth")
        print(f"[DQN] Models saved to {self.checkpoint_dir}")
    
    def load_models(self, episode=None):
        suffix = f"_ep{episode}" if episode is not None else ""
        q_path = f"{self.checkpoint_dir}/q_network{suffix}.pth"
        if not os.path.exists(q_path):
            return
        self.q_network.load_state_dict(torch.load(q_path, map_location=self.device))
        self.target_network.load_state_dict(torch.load(f"{self.checkpoint_dir}/target_network{suffix}.pth", map_location=self.device))
        self.optimizer.load_state_dict(torch.load(f"{self.checkpoint_dir}/optimizer{suffix}.pth", map_location=self.device))
        print(f"[DQN] Models loaded from {self.checkpoint_dir}")
    
    def reset(self):
        pass
    
    def set_eval(self):
        self.q_network.eval()
    
    def set_train(self):
        self.q_network.train()
