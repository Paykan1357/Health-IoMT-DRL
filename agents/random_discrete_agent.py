# ============================================================
# random_discrete_agent.py
# Random Agent for IoMT Health Monitoring (Discrete Actions).
#
# This agent serves as a baseline by selecting random discrete actions
# uniformly from {0, 1, 2, 3}.
#
# It is useful for:
# - Establishing a lower performance bound
# - Validating environment functionality
# - Comparing against learned policies (DQN, PPO)
#
# Action space (4 discrete):
#   0: Normal Route
#   1: High Priority Alert
#   2: Reduce Sampling Rate
#   3: Reroute via Backup
# ============================================================

import numpy as np
from base_discrete_agent import BaseDiscreteAgent


class RandomDiscreteAgent(BaseDiscreteAgent):
    """
    Random action baseline agent for discrete-action environments.
    
    Selects actions uniformly at random from {0, 1, 2, 3}.
    """
    
    def __init__(self, name, env, seed=42):
        """
        Args:
            name: Agent identifier
            env: IoMT Health environment
            seed: Random seed for reproducibility
        """
        super().__init__(name, env, seed)
        
        # Set random seed
        np.random.seed(seed)
        
        self.action_dim = 4  # Discrete actions: 0, 1, 2, 3

    def act(self, obs, exploration=True):
        """
        Generate a random action.
        
        Args:
            obs: Current observation (ignored for random agent)
            exploration: Whether to use exploration (ignored for random agent)
            
        Returns:
            action: Integer action in {0, 1, 2, 3}
        """
        return self.random_action()
    
    def random_action(self):
        """
        Generate uniformly random discrete action.
        
        Returns:
            action: Integer in {0, 1, 2, 3}
        """
        return np.random.randint(0, self.action_dim)
    
    def evaluate(self, env, exploration=False, render=False):
        """
        Evaluate the random agent over one episode.
        
        Args:
            env: IoMT Health environment
            exploration: Ignored for random agent
            render: Whether to render (if supported)
            
        Returns:
            total_reward: Cumulative reward
            step_infos: List of step information dicts
            final_metrics: Dictionary of final performance metrics
        """
        return super().evaluate(env, exploration=False, render=render)
