# ============================================================
# health_agents_evaluator.py
# Agents Evaluator for IoMT Health Monitoring.
#
# Evaluates DRL agents (DQN, PPO) on metrics including:
# - Detection Accuracy (Critical/Emergency events)
# - False Alarm Rate (Normal events misclassified as Alert)
# - Average Latency (simulated)
# - Energy Consumption (simulated)
# - Packet Delivery Ratio (PDR)
# - Average Reward
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class HealthAgentsEvaluator:
    """
    Evaluate and compare multiple DRL agents on IoMT Health Monitoring environment.
    """
    
    def __init__(self, env, agents_list):
        """
        Args:
            env: IoMT Health environment
            agents_list: List of agent instances to evaluate
        """
        self.env = env
        self.agents_list = agents_list
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    def evaluate_all(
        self,
        num_episodes=10,
        plot_metrics=True,
        plot_health_flows=True,
        save_folder=None,
    ):
        """
        Evaluate all agents and generate comparison plots.
        
        Args:
            num_episodes: Number of episodes to evaluate per agent
            plot_metrics: Whether to generate metric comparison plots
            plot_health_flows: Whether to plot health monitoring time series
            save_folder: Optional folder to save figures
            
        Returns:
            results_df: DataFrame with agent performance metrics
        """
        agent_results = []
        
        for agent_idx, agent in enumerate(self.agents_list):
            print(f"Evaluating {agent.name}... ({num_episodes} episodes)")
            
            # Store all episode step infos for plotting
            all_step_infos = []
            total_rewards = []
            
            for episode in range(num_episodes):
                obs = self.env.reset()
                done = False
                episode_reward = 0
                episode_infos = []
                
                while not done:
                    # Select action
                    action = agent.act(obs, exploration=False)
                    
                    # Step environment
                    next_obs, reward, done, info = self.env.step(action)
                    
                    # Store step data
                    episode_infos.append(info)
                    episode_reward += reward
                    
                    # Update observation
                    obs = next_obs
                
                total_rewards.append(episode_reward)
                all_step_infos.extend(episode_infos)
            
            # Compute metrics
            metrics = self._compute_metrics(all_step_infos)
            metrics['avg_reward'] = np.mean(total_rewards)
            metrics['std_reward'] = np.std(total_rewards)
            metrics['agent'] = agent.name
            
            agent_results.append(metrics)
            
            # Save step infos for plotting
            agent.step_infos = all_step_infos
        
        # Create results DataFrame
        results_df = pd.DataFrame(agent_results).set_index('agent')
        
        print("\n=== Agent Performance Metrics ===")
        print(results_df.round(3))
        print("\n")
        
        # Generate plots
        if plot_metrics:
            self._plot_metrics_comparison(results_df, save_folder)
        
        if plot_health_flows:
            self._plot_health_flows(save_folder)
        
        return results_df
    
    def _compute_metrics(self, step_infos):
        """
        Compute episode-level metrics from step infos.
        
        Args:
            step_infos: List of info dicts from each step
        
        Returns:
            metrics: Dict of episode-level metrics
        """
        if not step_infos:
            return {}
        
        n_steps = len(step_infos)
        
        # Extract data
        health_events = [info.get('health_event', 0) for info in step_infos]
        actions = [info.get('action', 0) for info in step_infos]
        battery_levels = [info.get('battery_level', 50) for info in step_infos]
        congestion = [info.get('network_congestion', 0.3) for info in step_infos]
        
        # Critical/Emergency events (health_event >= 2)
        critical_events = [h for h in health_events if h >= 2]
        
        # --- Detection Accuracy ---
        correct_responses = sum(
            1 for h, a in zip(health_events, actions)
            if h >= 2 and a == 1
        )
        detection_accuracy = correct_responses / (len(critical_events) + 1e-8)
        
        # --- False Alarm Rate ---
        normal_events = [h for h in health_events if h == 0]
        false_alarms = sum(
            1 for h, a in zip(health_events, actions)
            if h == 0 and a == 1
        )
        false_alarm_rate = false_alarms / (len(normal_events) + 1e-8)
        
        # --- Simulated Latency ---
        latencies = []
        for action, cong in zip(actions, congestion):
            if action == 1:  # High Priority Alert
                latency = 5 + 10 * cong
            elif action == 3:  # Reroute
                if cong > 0.7:
                    latency = 8 + 5 * cong
                else:
                    latency = 15 + 10 * cong
            elif action == 2:  # Reduce Sampling
                latency = 7 + 8 * cong
            else:  # Normal Route
                latency = 10 + 15 * cong
            latencies.append(latency)
        
        avg_latency = np.mean(latencies)
        
        # --- Simulated Energy Consumption ---
        energy_consumptions = []
        for action, battery in zip(actions, battery_levels):
            base = 1.0
            if action == 1:  # High Priority Alert
                energy = base * 2.5
            elif action == 3:  # Reroute
                energy = base * 1.8
            elif action == 2:  # Reduce Sampling
                energy = base * 0.5
            else:  # Normal Route
                energy = base * 1.0
            
            # Scale with battery level
            if battery < 30:
                energy *= 0.8
            energy_consumptions.append(energy)
        
        avg_energy = np.mean(energy_consumptions)
        total_energy = np.sum(energy_consumptions)
        
        # --- Packet Delivery Ratio (PDR) ---
        # Simulate PDR based on congestion and action
        pdr_values = []
        for action, cong in zip(actions, congestion):
            if action == 1:  # High Priority Alert
                pdr = 0.99 - 0.05 * cong
            elif action == 3:  # Reroute
                if cong > 0.7:
                    pdr = 0.98 - 0.02 * cong
                else:
                    pdr = 0.92 - 0.05 * cong
            elif action == 2:  # Reduce Sampling
                pdr = 0.95 - 0.08 * cong
            else:  # Normal Route
                pdr = 0.90 - 0.10 * cong
            pdr_values.append(max(0.5, min(1.0, pdr)))
        
        avg_pdr = np.mean(pdr_values)
        
        # --- Action Distribution ---
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for a in actions:
            action_counts[a] = action_counts.get(a, 0) + 1
        
        # Normalize action distribution
        action_distribution = {k: v / n_steps for k, v in action_counts.items()}
        
        return {
            'detection_accuracy': detection_accuracy,
            'false_alarm_rate': false_alarm_rate,
            'avg_latency': avg_latency,
            'avg_energy': avg_energy,
            'total_energy': total_energy,
            'avg_pdr': avg_pdr,
            'action_distribution': action_distribution,
            'avg_battery': np.mean(battery_levels),
            'min_battery': np.min(battery_levels),
            'avg_congestion': np.mean(congestion),
        }
    
    def _plot_metrics_comparison(self, results_df, save_folder=None):
        """
        Create bar charts comparing agent performance across metrics.
        """
        metrics_to_plot = [
            ('detection_accuracy', 'Detection Accuracy', 0, 1),
            ('false_alarm_rate', 'False Alarm Rate', 0, 1),
            ('avg_latency', 'Avg Latency (ms)', None, None),
            ('avg_energy', 'Avg Energy (norm.)', None, None),
            ('avg_pdr', 'Packet Delivery Ratio', 0, 1),
            ('avg_reward', 'Avg Reward', None, None),
        ]
        
        n_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        
        for idx, (metric, label, y_min, y_max) in enumerate(metrics_to_plot):
            if metric not in results_df.columns:
                continue
            
            values = results_df[metric].values
            agents = results_df.index.values
            
            bars = axes[idx].bar(agents, values, color=self.colors[:len(agents)])
            axes[idx].set_title(label)
            axes[idx].set_ylabel('Value')
            axes[idx].tick_params(axis='x', rotation=45)
            
            if y_min is not None:
                axes[idx].set_ylim(y_min, y_max)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                axes[idx].text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.02 * (y_max - y_min if y_max else 1),
                    f'{val:.3f}',
                    ha='center', va='bottom', fontsize=8
                )
            axes[idx].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(len(metrics_to_plot), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_folder:
            plt.savefig(f"{save_folder}/health_metrics_comparison.png", dpi=150, bbox_inches="tight")
        
        plt.show()
    
    def _plot_health_flows(self, save_folder=None):
        """
        Plot health monitoring time series for each agent.
        """
        for agent in self.agents_list:
            if not hasattr(agent, 'step_infos'):
                continue
            
            step_infos = agent.step_infos
            n_steps = len(step_infos)
            steps = range(n_steps)
            
            # Extract data
            health_events = [info.get('health_event', 0) for info in step_infos]
            actions = [info.get('action', 0) for info in step_infos]
            action_names = [info.get('action_name', 'Unknown') for info in step_infos]
            battery_levels = [info.get('battery_level', 50) for info in step_infos]
            congestion = [info.get('network_congestion', 0.3) for info in step_infos]
            
            # Simulate latency for plotting
            latencies = []
            for action, cong in zip(actions, congestion):
                if action == 1:
                    latency = 5 + 10 * cong
                elif action == 3:
                    latency = 8 + 5 * cong if cong > 0.7 else 15 + 10 * cong
                elif action == 2:
                    latency = 7 + 8 * cong
                else:
                    latency = 10 + 15 * cong
                latencies.append(latency)
            
            fig, axes = plt.subplots(3, 2, figsize=(14, 10))
            fig.suptitle(f"Health Monitoring Metrics - {agent.name}", fontsize=14)
            
            # Plot 1: Health Event over time
            ax = axes[0, 0]
            ax.step(steps, health_events, where='post', color='red', linewidth=2)
            ax.axhline(y=2, color='orange', linestyle='--', label='Critical Threshold')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Health Event Level')
            ax.set_title('Health Event Level (0=Normal, 1=Warning, 2=Critical, 3=Emergency)')
            ax.set_ylim(-0.5, 3.5)
            ax.set_yticks([0, 1, 2, 3])
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 2: Actions taken
            ax = axes[0, 1]
            colors = {'Normal Route': 'green', 'High Priority Alert': 'red', 
                      'Reduce Sampling Rate': 'blue', 'Reroute via Backup': 'orange'}
            action_colors = [colors.get(name, 'gray') for name in action_names]
            ax.scatter(steps, actions, c=action_colors, s=20, alpha=0.7)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Action')
            ax.set_title('Actions Taken')
            ax.set_yticks([0, 1, 2, 3])
            ax.set_yticklabels(['Normal', 'Alert', 'Reduce', 'Reroute'])
            ax.grid(True, alpha=0.3)
            
            # Plot 3: Latency
            ax = axes[1, 0]
            ax.plot(steps, latencies, color='purple', linewidth=1.5)
            ax.fill_between(steps, 0, latencies, alpha=0.3, color='purple')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Latency (ms)')
            ax.set_title('Simulated Network Latency')
            ax.grid(True, alpha=0.3)
            
            # Plot 4: Battery Level
            ax = axes[1, 1]
            ax.plot(steps, battery_levels, color='blue', linewidth=1.5)
            ax.fill_between(steps, 0, battery_levels, alpha=0.3, color='blue')
            ax.axhline(y=30, color='red', linestyle='--', label='Low Battery Threshold')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Battery Level (%)')
            ax.set_title('Battery Level')
            ax.set_ylim(0, 105)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 5: Network Congestion
            ax = axes[2, 0]
            ax.plot(steps, congestion, color='orange', linewidth=1.5)
            ax.fill_between(steps, 0, congestion, alpha=0.3, color='orange')
            ax.axhline(y=0.7, color='red', linestyle='--', label='High Congestion Threshold')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Congestion Level')
            ax.set_title('Network Congestion')
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 6: Action Distribution
            ax = axes[2, 1]
            action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
            for a in actions:
                action_counts[a] = action_counts.get(a, 0) + 1
            
            labels = ['Normal', 'Alert', 'Reduce', 'Reroute']
            counts = [action_counts[i] for i in range(4)]
            ax.bar(labels, counts, color=['green', 'red', 'blue', 'orange'])
            ax.set_xlabel('Action Type')
            ax.set_ylabel('Count')
            ax.set_title('Action Distribution')
            for i, v in enumerate(counts):
                ax.text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_folder:
                plt.savefig(f"{save_folder}/health_flows_{agent.name}.png", dpi=150, bbox_inches="tight")
            
            plt.show()
    
    def plot_accuracy_comparison(self, save_folder=None):
        """
        Plot Detection Accuracy comparison across agents.
        """
        agents = []
        accuracies = []
        false_alarm_rates = []
        
        for agent in self.agents_list:
            if hasattr(agent, 'step_infos'):
                metrics = self._compute_metrics(agent.step_infos)
                agents.append(agent.name)
                accuracies.append(metrics.get('detection_accuracy', 0))
                false_alarm_rates.append(metrics.get('false_alarm_rate', 0))
        
        if not agents:
            print("No data available.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Detection Accuracy
        ax = axes[0]
        bars = ax.bar(agents, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_xlabel('Agent')
        ax.set_ylabel('Detection Accuracy')
        ax.set_title('Critical/Event Detection Accuracy')
        ax.set_ylim(0, 1.05)
        for bar, val in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2%}', ha='center', va='bottom', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # False Alarm Rate
        ax = axes[1]
        bars = ax.bar(agents, false_alarm_rates, color=['#d62728', '#ff7f0e', '#2ca02c'])
        ax.set_xlabel('Agent')
        ax.set_ylabel('False Alarm Rate')
        ax.set_title('False Alarm Rate')
        ax.set_ylim(0, max(0.5, max(false_alarm_rates) * 1.2))
        for bar, val in zip(bars, false_alarm_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2%}', ha='center', va='bottom', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_folder:
            plt.savefig(f"{save_folder}/accuracy_comparison.png", dpi=150, bbox_inches="tight")
        
        plt.show()
