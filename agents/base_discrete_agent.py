# ============================================================
# base_discrete_agent.py
# Base class for Discrete-Action DRL Agents (DQN, PPO, etc.)
#
# Provides common functionality:
#   - Replay buffer (for DQN)
#   - Target network soft/hard updates
#   - Checkpoint saving/loading
#   - Device management
#
# Designed for IoMT Health Monitoring with discrete actions.
# ============================================================

import numpy as np
import torch
from copy import deepcopy
from replay_buffer import ReplayBuffer


class BaseDiscreteAgent:
    """
    Base class for discrete-action DRL agents.
    
    Attributes:
        name (str): Agent identifier
        env (gym.Env): Health monitoring environment
        device (torch.device): CUDA or CPU
        state_dim (int): Dimension of flattened state vector
        action_dim (int): Number of discrete actions (default: 4)
        batch_size (int): Training batch size
        buffer_size (int): Replay buffer capacity
        gamma (float): Discount factor
        tau (float): Soft update coefficient
        buffer (ReplayBuffer): Experience replay buffer (for off-policy agents)
    """
    
    def __init__(self, name, env, seed, args):
        self.name = name
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set random seeds for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Environment-derived dimensions
        self.state_dim = env.observation_space.shape[0]  # Flattened state vector (50)
        self.action_dim = env.action_space.n if hasattr(env.action_space, 'n') else 4  # Discrete actions

        # Hyperparameters
        self.batch_size = getattr(args, "batch_size", 64)
        self.buffer_size = getattr(args, "buffer_size", 1000000)
        self.gamma = getattr(args, "gamma", 0.99)
        self.tau = getattr(args, "tau", 0.005)
        
        # Learning rates (can be overridden by subclasses)
        self.learning_rate = getattr(args, "learning_rate", 3e-4)

        # Replay buffer (for off-policy agents like DQN)
        self.buffer = ReplayBuffer(capacity=self.buffer_size, device=self.device)

        # Placeholders to be set by subclasses
        self.policy_net = None       # Q-network (DQN) or Actor (PPO)
        self.target_net = None       # Target network (DQN) or Value network (PPO)
        self.optimizer = None
        
        # For PPO: additional networks
        self.value_net = None
        self.value_optimizer = None

    def set_networks(self, policy_net, target_net=None, value_net=None):
        """
        Set the networks and initialize optimizers.
        Must be called by subclasses after creating the networks.
        
        Args:
            policy_net: Main policy network (Q-network or Actor)
            target_net: Target network (for DQN) or None
            value_net: Value network (for PPO) or None
        """
        self.policy_net = policy_net.to(self.device)
        if target_net is not None:
            self.target_net = target_net.to(self.device)
        if value_net is not None:
            self.value_net = value_net.to(self.device)
        
        # Initialize optimizers
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        if self.value_net is not None:
            self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.learning_rate)

    def copy_params_to_target(self):
        """
        Copy parameters from policy network to target network (hard update).
        Used for DQN target network initialization.
        """
        if self.target_net is not None and self.policy_net is not None:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def soft_update_target(self):
        """
        Perform Polyak averaging for target network (soft update).
        target = tau * source + (1 - tau) * target
        Used for DQN target network updates.
        """
        if self.target_net is not None and self.policy_net is not None:
            for target_param, source_param in zip(
                self.target_net.parameters(), self.policy_net.parameters()
            ):
                target_param.data.copy_(
                    self.tau * source_param.data + (1.0 - self.tau) * target_param.data
                )

    def store_transition(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay buffer.
        (Only used by off-policy agents like DQN)
        
        Args:
            state: Current state observation
            action: Action taken (integer)
            reward: Reward received
            next_state: Next state observation
            done: Episode termination flag
        """
        self.buffer.push(state, action, reward, next_state, done)

    def sample_batch(self):
        """
        Sample a random batch of transitions from the replay buffer.
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        return self.buffer.sample(self.batch_size)

    def save_checkpoint(self, filepath):
        """
        Save agent checkpoint.
        
        Args:
            filepath: Path to save the checkpoint
        """
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict() if self.policy_net else None,
            'target_net_state_dict': self.target_net.state_dict() if self.target_net else None,
            'value_net_state_dict': self.value_net.state_dict() if self.value_net else None,
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'value_optimizer_state_dict': self.value_optimizer.state_dict() if self.value_optimizer else None,
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath):
        """
        Load agent checkpoint.
        
        Args:
            filepath: Path to load the checkpoint from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        if self.policy_net and checkpoint['policy_net_state_dict']:
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        if self.target_net and checkpoint['target_net_state_dict']:
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        if self.value_net and checkpoint['value_net_state_dict']:
            self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        if self.optimizer and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.value_optimizer and checkpoint['value_optimizer_state_dict']:
            self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        
        print(f"Checkpoint loaded from {filepath}")
