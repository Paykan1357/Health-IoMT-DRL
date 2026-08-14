
# ============================================================
# health_control_utilities.py
# Utility Functions for IoMT Health Monitoring
# Discrete Action Deep Reinforcement Learning (DQN / PPO)
# ============================================================

import numpy as np
from collections import defaultdict

EPS = 1e-8

# ============================================================
#               SAFETY & FALLBACK ACTIONS
# ============================================================

def safe_fallback_action():
    """
    Safe fallback action:
    Always choose Normal Route (action 0).
    This ensures data delivery without alert fatigue or resource waste.
    """
    return 0  # action 0 = Normal Route


def emergency_action():
    """
    Emergency override action:
    Always choose High Priority Alert (action 1).
    Used when health_event == 3 (Emergency) regardless of policy.
    """
    return 1  # action 1 = High Priority Alert


# ============================================================
#               HEALTH EVENT DETECTION UTILITIES
# ============================================================

def detect_health_event(heart_rate, blood_oxygen, systolic_bp, diastolic_bp, temperature, respiratory_rate):
    """
    Simple rule-based health event detection.
    Used as an alternative to the dataset's health_event label.
    
    Returns:
        0 = Normal
        1 = Warning
        2 = Critical
        3 = Emergency
    """
    # Define thresholds
    HR_NORMAL = (60, 100)
    HR_WARNING = (100, 120)
    HR_CRITICAL = (120, 150)
    
    SPO2_NORMAL = 95
    SPO2_WARNING = 90
    SPO2_CRITICAL = 85
    
    SBP_NORMAL = (90, 140)
    SBP_WARNING = (140, 180)
    SBP_CRITICAL = 180
    
    DBP_NORMAL = (60, 90)
    DBP_WARNING = (90, 110)
    DBP_CRITICAL = 110
    
    TEMP_NORMAL = (36.0, 37.5)
    TEMP_WARNING = (37.5, 38.5)
    TEMP_CRITICAL = 38.5
    
    RR_NORMAL = (12, 20)
    RR_WARNING = (20, 30)
    RR_CRITICAL = 30
    
    # Score each feature
    score = 0
    
    # Heart Rate
    if heart_rate < HR_NORMAL[0] or heart_rate > HR_NORMAL[1]:
        score += 1
    if HR_WARNING[0] <= heart_rate < HR_WARNING[1]:
        score += 2
    if heart_rate >= HR_CRITICAL[0]:
        score += 3
    
    # Blood Oxygen
    if blood_oxygen < SPO2_NORMAL:
        score += 1
    if blood_oxygen < SPO2_WARNING:
        score += 2
    if blood_oxygen < SPO2_CRITICAL:
        score += 3
    
    # Systolic BP
    if systolic_bp < SBP_NORMAL[0] or systolic_bp > SBP_NORMAL[1]:
        score += 1
    if SBP_WARNING[0] <= systolic_bp < SBP_WARNING[1]:
        score += 2
    if systolic_bp >= SBP_CRITICAL:
        score += 3
    
    # Diastolic BP
    if diastolic_bp < DBP_NORMAL[0] or diastolic_bp > DBP_NORMAL[1]:
        score += 1
    if DBP_WARNING[0] <= diastolic_bp < DBP_WARNING[1]:
        score += 2
    if diastolic_bp >= DBP_CRITICAL:
        score += 3
    
    # Temperature
    if temperature < TEMP_NORMAL[0] or temperature > TEMP_NORMAL[1]:
        score += 1
    if TEMP_WARNING[0] <= temperature < TEMP_WARNING[1]:
        score += 2
    if temperature >= TEMP_CRITICAL:
        score += 3
    
    # Respiratory Rate
    if respiratory_rate < RR_NORMAL[0] or respiratory_rate > RR_NORMAL[1]:
        score += 1
    if RR_WARNING[0] <= respiratory_rate < RR_WARNING[1]:
        score += 2
    if respiratory_rate >= RR_CRITICAL:
        score += 3
    
    # Map score to health_event
    if score <= 2:
        return 0  # Normal
    elif score <= 4:
        return 1  # Warning
    elif score <= 6:
        return 2  # Critical
    else:
        return 3  # Emergency


# ============================================================
#               REWARD COMPONENTS
# ============================================================

def medical_reward(health_event, action):
    """
    Compute medical reward based on health_event and chosen action.
    
    Args:
        health_event: 0=Normal, 1=Warning, 2=Critical, 3=Emergency
        action: 0=Normal Route, 1=High Priority Alert, 2=Reduce Sampling, 3=Reroute
    
    Returns:
        float: Medical reward component
    """
    if health_event == 0:  # Normal
        if action in [0, 2]:
            return 5.0
        elif action == 1:
            return -20.0  # False alarm
        else:  # action == 3
            return -5.0
    
    elif health_event == 1:  # Warning
        if action in [0, 1]:
            return 2.0
        else:  # action 2 or 3
            return -10.0
    
    elif health_event == 2:  # Critical
        if action == 1:
            return 100.0
        else:
            return -200.0
    
    elif health_event == 3:  # Emergency
        if action == 1:
            return 200.0
        else:
            return -300.0
    
    else:
        return 0.0


def network_reward(congestion, action):
    """
    Compute network reward based on congestion and chosen action.
    
    Args:
        congestion: float in [0, 1]
        action: 0=Normal Route, 3=Reroute
    
    Returns:
        float: Network reward component
    """
    if congestion > 0.7 and action == 3:
        return 10.0  # Good: rerouting during congestion
    elif congestion < 0.3 and action == 3:
        return -5.0  # Bad: unnecessary reroute
    else:
        return 0.0


def battery_reward(battery_level, action):
    """
    Compute battery reward based on battery level and chosen action.
    
    Args:
        battery_level: float in [0, 100]
        action: 0=Normal Route, 2=Reduce Sampling
    
    Returns:
        float: Battery reward component
    """
    if battery_level < 30.0 and action == 2:
        return 15.0  # Good: saving energy when low
    elif battery_level > 70.0 and action == 2:
        return 0.0  # Neutral: no immediate benefit
    else:
        return 0.0


def total_reward(health_event, action, congestion, battery_level):
    """
    Compute total reward as sum of all components.
    
    Args:
        health_event: 0=Normal, 1=Warning, 2=Critical, 3=Emergency
        action: 0-3
        congestion: float in [0, 1]
        battery_level: float in [0, 100]
    
    Returns:
        float: Total reward
    """
    r_med = medical_reward(health_event, action)
    r_net = network_reward(congestion, action)
    r_bat = battery_reward(battery_level, action)
    return r_med + r_net + r_bat


# ============================================================
#               ACTION PENALTIES & CONSTRAINTS
# ============================================================

def action_switching_penalty(prev_action, curr_action):
    """
    Penalize frequent action switching (reduces instability).
    
    Args:
        prev_action: Integer 0-3
        curr_action: Integer 0-3
    
    Returns:
        float: Penalty value (0 if same action, -1 if different)
    """
    if prev_action == curr_action:
        return 0.0
    else:
        return -1.0  # Small penalty for switching


def resource_usage_penalty(action, battery_level):
    """
    Penalize resource-intensive actions when resources are scarce.
    
    Args:
        action: Integer 0-3
        battery_level: float in [0, 100]
    
    Returns:
        float: Penalty value
    """
    if action == 1 and battery_level < 20.0:
        # High Priority Alert consumes more energy
        return -5.0
    elif action == 3 and battery_level < 20.0:
        # Reroute requires additional processing
        return -3.0
    else:
        return 0.0


# ============================================================
#               EPISODE-LEVEL METRICS
# ============================================================

def compute_episode_metrics(log):
    """
    Compute evaluation metrics over one episode (patient).
    
    Args:
        log: Dictionary with episode history containing:
            - actions: list of actions taken
            - health_events: list of health_event values
            - rewards: list of rewards received
            - battery_levels: list of battery levels
            - congestions: list of congestion values
            - latencies: list of simulated latencies (optional)
            - energy_consumptions: list of energy consumptions (optional)
    
    Returns:
        metrics: Dictionary of computed metrics
    """
    metrics = {}
    
    # Health-related metrics
    n_steps = len(log["health_events"])
    critical_events = [h for h in log["health_events"] if h >= 2]
    
    # Detection Accuracy: % of critical events correctly responded with action 1
    correct_responses = sum(
        1 for h, a in zip(log["health_events"], log["actions"])
        if h >= 2 and a == 1
    )
    metrics["detection_accuracy"] = (
        correct_responses / (len(critical_events) + EPS)
        if critical_events else 1.0
    )
    
    # False Alarm Rate: % of normal events incorrectly marked as alert (action 1)
    normal_events = [h for h in log["health_events"] if h == 0]
    false_alarms = sum(
        1 for h, a in zip(log["health_events"], log["actions"])
        if h == 0 and a == 1
    )
    metrics["false_alarm_rate"] = (
        false_alarms / (len(normal_events) + EPS)
        if normal_events else 0.0
    )
    
    # Action distribution
    action_counts = defaultdict(int)
    for a in log["actions"]:
        action_counts[a] += 1
    metrics["action_distribution"] = dict(action_counts)
    
    # Cumulative reward
    metrics["total_reward"] = np.sum(log["rewards"])
    metrics["avg_reward"] = np.mean(log["rewards"])
    
    # Battery health
    metrics["avg_battery"] = np.mean(log["battery_levels"])
    metrics["min_battery"] = np.min(log["battery_levels"])
    
    # Network congestion
    metrics["avg_congestion"] = np.mean(log["congestions"])
    metrics["max_congestion"] = np.max(log["congestions"])
    
    # Optional metrics (if available)
    if "latencies" in log:
        metrics["avg_latency"] = np.mean(log["latencies"])
        metrics["max_latency"] = np.max(log["latencies"])
    
    if "energy_consumptions" in log:
        metrics["total_energy"] = np.sum(log["energy_consumptions"])
        metrics["avg_energy"] = np.mean(log["energy_consumptions"])
    
    return metrics


def print_episode_summary(metrics, episode_num):
    """
    Print a formatted summary of episode metrics.
    """
    print(f"\n{'='*50}")
    print(f"Episode {episode_num} Summary")
    print(f"{'='*50}")
    print(f"Total Reward        : {metrics['total_reward']:.2f}")
    print(f"Avg Reward          : {metrics['avg_reward']:.2f}")
    print(f"Detection Accuracy  : {metrics['detection_accuracy']:.2%}")
    print(f"False Alarm Rate    : {metrics['false_alarm_rate']:.2%}")
    print(f"Avg Battery Level   : {metrics['avg_battery']:.1f}%")
    print(f"Min Battery Level   : {metrics['min_battery']:.1f}%")
    print(f"Avg Congestion      : {metrics['avg_congestion']:.3f}")
    print(f"Action Distribution : {metrics['action_distribution']}")
    if "avg_latency" in metrics:
        print(f"Avg Latency         : {metrics['avg_latency']:.2f} ms")
        print(f"Max Latency         : {metrics['max_latency']:.2f} ms")
    print(f"{'='*50}\n")


# ============================================================
#               RULE-BASED BASELINE
# ============================================================

def rule_based_controller(health_event, congestion, battery_level, previous_action):
    """
    Simple heuristic baseline controller.
    
    Decision logic:
    1. If health_event >= 2 (Critical/Emergency) -> High Priority Alert (action 1)
    2. Else if battery_level < 30 -> Reduce Sampling (action 2)
    3. Else if congestion > 0.7 -> Reroute (action 3)
    4. Else -> Normal Route (action 0)
    
    Args:
        health_event: 0-3
        congestion: float in [0, 1]
        battery_level: float in [0, 100]
        previous_action: Integer 0-3 (for hysteresis)
    
    Returns:
        action: Integer 0-3
    """
    # Emergency override: always prioritize health
    if health_event >= 2:
        return 1
    
    # Battery-saving mode
    if battery_level < 30:
        return 2
    
    # Network congestion management
    if congestion > 0.7:
        return 3
    
    # Default: normal route
    return 0


# ============================================================
#               MPC BASELINE (SIMPLIFIED)
# ============================================================

def mpc_controller(health_event, congestion, battery_level, lookahead=5):
    """
    Simplified Model Predictive Control baseline.
    
    This simulates a lookahead optimization by evaluating all possible
    actions over a horizon and choosing the one with highest expected
    total reward.
    
    In a real MPC, this would use a system model. Here we use a
    simplified heuristic model.
    
    Args:
        health_event: 0-3
        congestion: float in [0, 1]
        battery_level: float in [0, 100]
        lookahead: Number of future steps to consider
    
    Returns:
        action: Integer 0-3
    """
    # Simulate future states (simplified)
    # In practice, you would use a learned or analytical model
    
    best_action = 0
    best_score = -float('inf')
    
    for action in range(4):
        # Estimate immediate reward
        immediate = total_reward(health_event, action, congestion, battery_level)
        
        # Estimate future reward (simplified)
        # Assume health_event, congestion, and battery evolve in a simple way
        future_battery = battery_level - 2.0 if action == 1 else battery_level - 0.5
        future_battery = max(0, future_battery)
        
        future_congestion = congestion + 0.1 if action == 0 else congestion - 0.1
        future_congestion = np.clip(future_congestion, 0, 1)
        
        # Assume health_event remains same or improves
        future_health = health_event
        
        # Estimate total score over horizon
        # This is a heuristic; in practice you'd use a model
        score = immediate + 0.9 * (future_health == 0) * 10 + 0.9 * (future_battery > 50) * 5
        
        if score > best_score:
            best_score = score
            best_action = action
    
    return best_action


# ============================================================
#               SIMULATION HELPERS (Latency & Energy)
# ============================================================

def simulate_latency(action, congestion, base_latency=10.0):
    """
    Simulate network latency based on action and congestion.
    
    Args:
        action: Integer 0-3
        congestion: float in [0, 1]
        base_latency: Base latency in ms
    
    Returns:
        latency: Simulated latency in ms
    """
    if action == 1:  # High Priority Alert
        # Prioritized traffic = lower latency
        latency_factor = 0.5 + 0.3 * congestion
    elif action == 3:  # Reroute via Backup
        # Rerouting may have overhead but avoids congestion
        if congestion > 0.7:
            latency_factor = 0.8 + 0.1 * congestion
        else:
            latency_factor = 1.2 + 0.2 * congestion  # Slight overhead
    elif action == 2:  # Reduce Sampling Rate
        # Less data = lower latency
        latency_factor = 0.7 + 0.2 * congestion
    else:  # Normal Route
        latency_factor = 1.0 + 0.5 * congestion
    
    return base_latency * latency_factor + np.random.normal(0, 0.5)


def simulate_energy_consumption(action, battery_level):
    """
    Simulate energy consumption based on action.
    
    Args:
        action: Integer 0-3
        battery_level: float in [0, 100]
    
    Returns:
        energy: Simulated energy consumption (arbitrary units)
    """
    base_consumption = 1.0
    
    if action == 1:  # High Priority Alert (most energy)
        consumption = base_consumption * 2.5
    elif action == 3:  # Reroute
        consumption = base_consumption * 1.8
    elif action == 2:  # Reduce Sampling (least energy)
        consumption = base_consumption * 0.5
    else:  # Normal Route
        consumption = base_consumption * 1.0
    
    # Scale based on battery level (higher battery = more aggressive)
    if battery_level < 30:
        consumption *= 0.8  # Energy-saving mode
    
    return consumption


# ============================================================
#               EXAMPLE USAGE
# ============================================================
if __name__ == "__main__":
    # Test the utilities
    print("Testing Health Control Utilities...")
    
    # Test reward function
    print(f"Medical Reward (Normal, Action 0): {medical_reward(0, 0)}")
    print(f"Medical Reward (Emergency, Action 1): {medical_reward(3, 1)}")
    print(f"Medical Reward (Emergency, Action 0): {medical_reward(3, 0)}")
    
    # Test rule-based controller
    action = rule_based_controller(health_event=2, congestion=0.5, battery_level=50, previous_action=0)
    print(f"Rule-Based Controller (Critical): {action}")
    
    # Test episode metrics
    log = {
        "health_events": [0, 0, 1, 2, 3, 0, 1],
        "actions": [0, 0, 0, 1, 1, 0, 1],
        "rewards": [5, 5, 2, 100, 200, 5, 2],
        "battery_levels": [90, 85, 80, 75, 70, 65, 60],
        "congestions": [0.3, 0.4, 0.5, 0.6, 0.7, 0.5, 0.4],
    }
    metrics = compute_episode_metrics(log)
    print_episode_summary(metrics, 1)
