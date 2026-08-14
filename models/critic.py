
# ============================================================
# critic.py
# Critic (Q-Network) for Discrete Action Space (DQN).
#
# This network takes a state vector as input and outputs
# Q-values for each discrete action (0, 1, 2, 3).
#
# Designed for IoMT Health Monitoring with:
#   - State dimension: 50 (flattened 5x10 window)
#   - Action dimension: 4 (discrete)
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    """
    Q-Network for DQN (discrete actions).
    Maps state -> Q-values for each action.
    
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
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Output layer: Q-values for each action
        self.output_layer = nn.Linear(hidden_dim, action_dim)
        
        # Initialize output layer with small weights for stability
        nn.init.uniform_(self.output_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.output_layer.bias, -3e-3, 3e-3)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            
        Returns:
            Q-values tensor of shape (batch_size, action_dim)
        """
        x = self.shared_layers(state)
        q_values = self.output_layer(x)
        return q_values
    
    def get_q_value(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Get Q-value for a specific action (useful for training).
        
        Args:
            state: State tensor (batch_size, state_dim)
            action: Action tensor (batch_size,) with discrete indices
            
        Returns:
            Q-values for the selected actions (batch_size, 1)
        """
        q_values = self.forward(state)
        return q_values.gather(1, action.unsqueeze(-1)).squeeze(-1)
    
    def get_max_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get the action with the highest Q-value (greedy policy).
        
        Args:
            state: State tensor (batch_size, state_dim)
            
        Returns:
            Action indices (batch_size,)
        """
        q_values = self.forward(state)
        return q_values.argmax(dim=-1)
    
    def freeze(self):
        """Freeze network parameters (useful for target networks)."""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """Unfreeze network parameters."""
        for param in self.parameters():
            param.requires_grad = True


# ============================================================
# Optional: Dueling Critic (improved DQN variant)
# ============================================================

class DuelingCritic(nn.Module):
    """
    Dueling Q-Network architecture.
    Separates state-value (V) and advantage (A) streams.
    
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
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
        
        # Shared feature extractor
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Value stream (scalar)
        self.value_layer = nn.Linear(hidden_dim, 1)
        
        # Advantage stream (action_dim)
        self.advantage_layer = nn.Linear(hidden_dim, action_dim)
        
        # Initialize output layers
        nn.init.uniform_(self.value_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.value_layer.bias, -3e-3, 3e-3)
        nn.init.uniform_(self.advantage_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.advantage_layer.bias, -3e-3, 3e-3)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: State tensor (batch_size, state_dim)
            
        Returns:
            Q-values tensor (batch_size, action_dim)
        """
        x = self.shared_layers(state)
        value = self.value_layer(x)  # (batch, 1)
        advantage = self.advantage_layer(x)  # (batch, action_dim)
        
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_values
    
    def get_q_value(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q_values = self.forward(state)
        return q_values.gather(1, action.unsqueeze(-1)).squeeze(-1)
    
    def get_max_action(self, state: torch.Tensor) -> torch.Tensor:
        q_values = self.forward(state)
        return q_values.argmax(dim=-1)
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
