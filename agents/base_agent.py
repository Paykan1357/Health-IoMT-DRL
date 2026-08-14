# ============================================================
# base_agent.py
# Base Agent for IoMT Health Monitoring with Discrete Actions.
#
# This module provides the foundational class for DRL agents
# in IoMT Health Monitoring systems.
# 
# Action space: Discrete (0-3)
#   0: Normal Route
#   1: High Priority Alert
#   2: Reduce Sampling Rate
#   3: Reroute via Backup
# ============================================================

import numpy as np
from collections import defaultdict


class BaseAgent:
    """
    Base class for all IoMT Health Monitoring DRL agents.
    
    Attributes:
        name (str): Agent identifier
        env (gym.Env): Health monitoring environment
        seed (int): Random seed for reproducibility
        training (bool): Training mode flag
        action_dim (int): Number of discrete actions (default: 4)
    """
    
    def __init__(self, name, env, seed=0, action_dim=4):
        self.name = name
        self.env = env
        self.seed = seed
        self.training = True
        self.action_dim = action_dim
        
        # Set random seeds for reproducibility
        np.random.seed(seed)
        
        # Action names for logging
        self.action_names = {
            0: "Normal Route",
            1: "High Priority Alert",
            2: "Reduce Sampling Rate",
            3: "Reroute via Backup"
        }

    def set_eval(self):
        """Set agent to evaluation mode (no exploration)."""
        self.training = False

    def set_train(self):
        """Set agent to training mode (with exploration)."""
        self.training = True

    def act(self, obs, exploration=False):
        """
        Select an action given the current observation.
        
        Args:
            obs: Current state observation (flattened window)
            exploration: Whether to add exploration noise
            
        Returns:
            action: Integer (0-3) representing the chosen action
        """
        raise NotImplementedError("Subclass must implement act()")

    def evaluate(self, env, exploration=False, render=False):
        """
        Evaluate agent in the given environment.
        
        Args:
            env: Health monitoring environment
            exploration: Whether to use exploration during evaluation
            render: Whether to render (if supported)
        
        Returns:
            total_reward (float): Cumulative reward over episode
            step_infos (list): List of step information dicts
            final_metrics (dict): Final performance metrics
        """
        self.set_eval()
        obs = env.reset()
        done = False
        
        rewards = []
        step_infos = []
        actions_taken = []
        
        while not done:
            # Select action
            action = self.act(obs, exploration=exploration)
            actions_taken.append(action)
            
            # Step environment
            next_obs, reward, done, info = env.step(action)
            
            # Store step data
            rewards.append(reward)
            step_infos.append(info)
            
            # Update observation
            obs = next_obs
        
        # Compute episode metrics
        total_reward = sum(rewards)
        final_metrics = self._aggregate_metrics(step_infos, actions_taken)
        
        return total_reward, step_infos, final_metrics
    
    def _aggregate_metrics(self, step_infos, actions_taken):
        """
        Aggregate step-wise metrics into episode-level metrics.
        
        Args:
            step_infos: List of info dicts from each step
            actions_taken: List of actions taken during the episode
        
        Returns:
            metrics: Dict of episode-level metrics
        """
        if not step_infos:
            return {}
        
        n_steps = len(step_infos)
        
        # Extract health events and actions
        health_events = [info.get("health_event", 0) for info in step_infos]
        actions = actions_taken
        
        # --- Health-related metrics ---
        # Detection Accuracy: % of critical/emergency events correctly responded with action 1
        critical_events = [h for h in health_events if h >= 2]
        correct_responses = sum(
            1 for h, a in zip(health_events, actions)
            if h >= 2 and a == 1
        )
        detection_accuracy = correct_responses / (len(critical_events) + 1e-8)
        
        # False Alarm Rate: % of normal events incorrectly marked as alert (action 1)
        normal_events = [h for h in health_events if h == 0]
        false_alarms = sum(
            1 for h, a in zip(health_events, actions)
            if h == 0 and a == 1
        )
        false_alarm_rate = false_alarms / (len(normal_events) + 1e-8)
        
        # --- Network metrics ---
        avg_congestion = np.mean([info.get("congestion", 0) for info in step_infos])
        avg_latency = np.mean([info.get("latency", 0) for info in step_infos])
        
        # --- Battery metrics ---
        avg_battery = np.mean([info.get("battery_level", 0) for info in step_infos])
        min_battery = np.min([info.get("battery_level", 100) for info in step_infos])
        
        # --- Action distribution ---
        action_counts = defaultdict(int)
        for a in actions:
            action_counts[a] += 1
        action_distribution = dict(action_counts)
        
        # --- Reward metrics ---
        total_reward = sum([info.get("reward", 0) for info in step_infos])
        avg_reward = total_reward / (n_steps + 1e-8)
        
        metrics = {
            "detection_accuracy": detection_accuracy,
            "false_alarm_rate": false_alarm_rate,
            "avg_congestion": avg_congestion,
            "avg_latency": avg_latency,
            "avg_battery": avg_battery,
            "min_battery": min_battery,
            "action_distribution": action_distribution,
            "total_reward": total_reward,
            "avg_reward": avg_reward,
            "num_steps": n_steps,
        }
        
        return metrics
