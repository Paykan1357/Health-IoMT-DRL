# ============================================================
# discrete_actor.py
# Actor Network for PPO (Proximal Policy Optimization)
# with Discrete Actions.
#
# This network takes a state vector as input and outputs
# logits (or a categorical distribution) over discrete actions.
#
# Designed for IoMT Health Monitoring with:
#   - State dimension: 50 (flattened 5x10 window)
#   - Action dimension: 4 (discrete)
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class DiscreteActor(nn.Module):
    """
    Actor network for PPO with discrete actions.
    Outputs a probability distribution over actions (0, 1, 2, 3).
    
    Args:
        state_dim: Dimension of the flattened state vector.
        action_dim: Number of discrete actions (default: 4).
        hidden_dim: Number of neurons in hidden layers.
        hidden_layers: Number of hidden layers (default: 2).
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 4,
        hidden_dim: int = 256,
        hidden_layers: int = 2,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build MLP layers
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        self.network = nn.Sequential(*layers)
        
        # Output layer: logits for each action
        self.output_layer = nn.Linear(hidden_dim, action_dim)
        
        # Initialize output layer with small weights
        nn.init.uniform_(self.output_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.output_layer.bias, -3e-3, 3e-3)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            
        Returns:
            Logits tensor of shape (batch_size, action_dim)
        """
        x = self.network(state)
        logits = self.output_layer(x)
        return logits
    
    def get_action_logits(self, state: torch.Tensor) -> torch.Tensor:
        """Alias for forward()."""
        return self.forward(state)
    
    def get_action_probs(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get action probabilities (softmax of logits).
        
        Args:
            state: State tensor (batch_size, state_dim)
            
        Returns:
            Probabilities tensor (batch_size, action_dim)
        """
        logits = self.forward(state)
        return F.softmax(logits, dim=-1)
    
    def get_distribution(self, state: torch.Tensor) -> Categorical:
        """
        Get a categorical distribution over actions.
        
        Args:
            state: State tensor (batch_size, state_dim)
            
        Returns:
            Categorical distribution object
        """
        logits = self.forward(state)
        return Categorical(logits=logits)
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Sample an action from the policy.
        
        Args:
            state: State tensor (batch_size, state_dim)
            deterministic: If True, return greedy action (argmax).
                          If False, sample from the distribution.
            
        Returns:
            Action indices (batch_size,)
        """
        dist = self.get_distribution(state)
        if deterministic:
            return dist.probs.argmax(dim=-1)
        else:
            return dist.sample()
    
    def get_action_and_logprob(self, state: torch.Tensor) -> tuple:
        """
        Sample an action and return its log probability.
        Useful for PPO training.
        
        Args:
            state: State tensor (batch_size, state_dim)
            
        Returns:
            action: Action indices (batch_size,)
            log_prob: Log probability of the chosen action (batch_size,)
        """
        dist = self.get_distribution(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob
    
    def evaluate_actions(self, state: torch.Tensor, actions: torch.Tensor) -> tuple:
        """
        Evaluate the log probabilities of given actions.
        Used for PPO's policy loss computation.
        
        Args:
            state: State tensor (batch_size, state_dim)
            actions: Action indices (batch_size,)
            
        Returns:
            log_probs: Log probabilities of the actions (batch_size,)
            entropy: Entropy of the distribution (batch_size,)
        """
        dist = self.get_distribution(state)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy
    
    def freeze(self):
        """Freeze network parameters."""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """Unfreeze network parameters."""
        for param in self.parameters():
            param.requires_grad = True
