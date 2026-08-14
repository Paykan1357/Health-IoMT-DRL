# ============================================================
# ppo_agent.py (اصلاح‌شده)
# با Hyperparameters تنظیم‌شده برای پایداری بیشتر
# ============================================================

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class DiscreteActor(nn.Module):
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
    
    def get_distribution(self, state):
        return Categorical(logits=self.forward(state))
    
    def get_action(self, state, deterministic=False):
        dist = self.get_distribution(state)
        if deterministic:
            action = dist.probs.argmax(dim=-1)
            log_prob = dist.log_prob(action)
        else:
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action, log_prob
    
    def evaluate_actions(self, state, actions):
        dist = self.get_distribution(state)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, hidden_layers=2):
        super().__init__()
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dim, 1)
        nn.init.uniform_(self.output.weight, -3e-3, 3e-3)
    
    def forward(self, state):
        return self.output(self.network(state))


class PPOAgent:
    def __init__(self, env, args=None):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.window_length = getattr(args, "window_length", 5)
        self.num_features = getattr(args, "num_features", 11)
        
        # PPO Hyperparameters (تنظیم‌شده)
        self.gamma = getattr(args, "gamma", 0.99)
        self.gae_lambda = getattr(args, "gae_lambda", 0.98)
        self.epsilon_clip = getattr(args, "epsilon_clip", 0.15)   # کاهش از 0.2
        self.entropy_coef = getattr(args, "entropy_coef", 0.1)    # افزایش از 0.05
        self.value_loss_coef = getattr(args, "value_loss_coef", 0.5)
        self.max_grad_norm = getattr(args, "max_grad_norm", 0.5)
        
        self.lr_actor = getattr(args, "lr_actor", 5e-5)    # کاهش
        self.lr_critic = getattr(args, "lr_critic", 1e-4)  # کاهش
        
        self.ppo_epochs = getattr(args, "ppo_epochs", 20)   # افزایش
        self.batch_size = getattr(args, "batch_size", 32)   # کاهش
        
        # Networks
        self.actor = DiscreteActor(self.state_dim, self.action_dim).to(self.device)
        self.critic = Critic(self.state_dim).to(self.device)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        
        self.checkpoint_dir = getattr(args, "checkpoint_dir", "./checkpoints_PPO")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
        print(f"[PPO] Initialized (Epochs={self.ppo_epochs}, LR_actor={self.lr_actor})")
    
    # ------------------------------------------------------------
    # Action Selection
    # ------------------------------------------------------------
    def select_action(self, state, deterministic=False):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            action, log_prob = self.actor.get_action(state, deterministic=deterministic)
            value = self.critic(state)
        
        if not deterministic:
            self.states.append(state.cpu().numpy().flatten())
            self.actions.append(action.item())
            self.log_probs.append(log_prob.item())
            self.values.append(value.item())
        
        return action.item()
    
    def store_reward(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)
    
    # ------------------------------------------------------------
    # Update (با Weighted Entropy)
    # ------------------------------------------------------------
    def update(self):
        if len(self.states) < self.batch_size:
            return None
        
        # GAE Computation
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)
        
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        with torch.no_grad():
            last_state = torch.tensor(self.states[-1], dtype=torch.float32, device=self.device).unsqueeze(0)
            last_value = self.critic(last_state).item()
        
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value * (1 - dones[t])
            else:
                next_value = values[t + 1] * (1 - dones[t])
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states = torch.tensor(np.array(self.states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(self.actions), dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(np.array(self.log_probs), dtype=torch.float32, device=self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # Weighted Entropy for Critical States
        health_event_idx = (self.window_length - 1) * self.num_features + 8
        health_events = states[:, health_event_idx].long().cpu().numpy()
        entropy_weights = np.ones(len(health_events), dtype=np.float32)
        for i, he in enumerate(health_events):
            if he >= 2:
                entropy_weights[i] = 5.0
            elif he == 1:
                entropy_weights[i] = 2.0
        entropy_weights = torch.tensor(entropy_weights, dtype=torch.float32, device=self.device)
        
        # PPO Epochs
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        n_updates = 0
        
        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(len(states))
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_entropy_weights = entropy_weights[batch_indices]
                
                # Actor Loss
                log_probs, entropy = self.actor.evaluate_actions(batch_states, batch_actions)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * batch_advantages
                weighted_entropy = (entropy * batch_entropy_weights).mean()
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * weighted_entropy
                
                # Critic Loss
                values = self.critic(batch_states).squeeze(-1)
                critic_loss = F.mse_loss(values, batch_returns)
                
                # Total Loss
                total_loss = actor_loss + self.value_loss_coef * critic_loss
                
                # Optimize
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += weighted_entropy.item()
                n_updates += 1
        
        # Clear buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
        return {
            "actor_loss": total_actor_loss / n_updates if n_updates > 0 else 0,
            "critic_loss": total_critic_loss / n_updates if n_updates > 0 else 0,
            "entropy": total_entropy / n_updates if n_updates > 0 else 0,
        }
    
    # ------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------
    def save_models(self, episode=None):
        suffix = f"_ep{episode}" if episode is not None else ""
        torch.save(self.actor.state_dict(), f"{self.checkpoint_dir}/actor{suffix}.pth")
        torch.save(self.critic.state_dict(), f"{self.checkpoint_dir}/critic{suffix}.pth")
        print(f"[PPO] Models saved to {self.checkpoint_dir}")
    
    def load_models(self, episode=None):
        suffix = f"_ep{episode}" if episode is not None else ""
        actor_path = f"{self.checkpoint_dir}/actor{suffix}.pth"
        if os.path.exists(actor_path):
            self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
            self.critic.load_state_dict(torch.load(f"{self.checkpoint_dir}/critic{suffix}.pth", map_location=self.device))
            print(f"[PPO] Models loaded from {self.checkpoint_dir}")
    
    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def set_eval(self):
        self.actor.eval()
        self.critic.eval()
    
    def set_train(self):
        self.actor.train()
        self.critic.train()
