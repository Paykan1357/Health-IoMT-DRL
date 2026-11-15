

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class DQNNetwork(nn.Module):
    def __init__(self, input_dim=55, num_actions=4):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),       
            nn.Linear(256, 256), nn.ReLU()              
        )
        self.value = nn.Linear(256, 1)                  
        self.advantage = nn.Linear(256, num_actions)    

    def forward(self, x):
        h = self.shared(x)
        v = self.value(h)
        a = self.advantage(h)
        return v + (a - a.mean(dim=1, keepdim=True))
    def forward(self, x):
        h = self.shared(x)
        v = self.value(h)
        a = self.advantage(h)
        return v + (a - a.mean(dim=1, keepdim=True))

def load_physionet_data_with_lstm(dataset_dir, lstm_path, window_length=5, max_patients=200):

    import torch
    from supervised_predictor import load_predictor

    print(f"[INFO] Loading PhysioNet data and LSTM predictor...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm = load_predictor(lstm_path, device=device)
    lstm.eval()

    FEATURE_BOUNDS = {
        'HR':     (40.0, 180.0),
        'SpO2':   (85.0, 100.0),
        'SBP':    (70.0, 200.0),
        'DBP':    (40.0, 130.0),
        'Temp':   (95.0, 105.0),   
        'RR':     (8.0,  40.0),
        'Activity': (0.0, 1.0),
        'Stress':   (0.0, 1.0),
    }
    features = list(FEATURE_BOUNDS.keys())

    def minmax(col_name, values):
        lo, hi = FEATURE_BOUNDS[col_name]
        return np.clip((values - lo) / (hi - lo + 1e-8), 0.0, 1.0)

    all_states = []
    p_crit_values = []
    file_list = [f for f in os.listdir(dataset_dir) if f.endswith('.txt')][:max_patients]
    print(f"[INFO] Processing {len(file_list)} patient files...")

    for filename in file_list:
        filepath = os.path.join(dataset_dir, filename)
        try:
            df = pd.read_csv(filepath)
            df_pivot = df.pivot_table(index='Time', columns='Parameter',
                                       values='Value', aggfunc='first')
            df_pivot = df_pivot.ffill().bfill()

            raw = pd.DataFrame(index=df_pivot.index)
            raw['HR']       = df_pivot.get('HR',     80.0)
            raw['SpO2']     = df_pivot.get('SaO2',   97.0)
            raw['SBP']      = df_pivot.get('SysABP', 120.0)
            raw['DBP']      = df_pivot.get('DiasABP',80.0)
            raw['Temp']     = df_pivot.get('Temp',   98.6)
            raw['RR']       = df_pivot.get('RespRate', 16.0)
            raw['Activity'] = 0.5
            raw['Stress']   = 0.3
            raw = raw.fillna(raw.mean()).fillna(0)

            norm = pd.DataFrame(index=raw.index)
            for f in features:
                norm[f] = minmax(f, raw[f].values)

            if len(norm) < window_length:
                continue

            for i in range(len(norm) - window_length + 1):
                window = norm.iloc[i:i+window_length].values.astype(np.float32)

                
                with torch.no_grad():
                    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
                    logits = lstm(x)
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                    p_crit_value = float(probs[2] + probs[3])

                p_crit     = np.full((window_length, 1), p_crit_value, dtype=np.float32)
                battery    = np.full((window_length, 1), 1.0, dtype=np.float32)
                congestion = np.full((window_length, 1), 0.3, dtype=np.float32)

                state = np.concatenate([window, p_crit, battery, congestion], axis=1).flatten()
                all_states.append(state)
                p_crit_values.append(p_crit_value)
        except Exception:
            continue

    p_arr = np.array(p_crit_values)
    print(f"[INFO] Extracted {len(all_states)} windows.")
    print(f"[INFO] p_crit distribution: mean={p_arr.mean():.3f}, "
          f"std={p_arr.std():.3f}, "
          f"% >= 0.5: {100*np.mean(p_arr >= 0.5):.1f}%")

    return np.array(all_states), p_arr


def load_pretrained_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQNNetwork()
    
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    print(f"[INFO] Pre-trained DQN model loaded from {model_path}")
    return model


def evaluate_zero_shot(model, states):
    device = next(model.parameters()).device
    states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        q_values = model(states_tensor)
        actions = torch.argmax(q_values, dim=1).cpu().numpy()
    
    unique, counts = np.unique(actions, return_counts=True)
    action_dist = dict(zip(unique, counts))
    
    print("\n" + "="*60)
    print(f"ZERO-SHOT EVALUATION ON PHYSIONET 2012 (Total Samples: {len(actions)})")
    print("="*60)
    action_names = {0: "Normal", 1: "Alert", 2: "Reduce", 3: "Reroute"}
    
    for a in range(4):
        pct = (action_dist.get(a, 0) / len(actions)) * 100
        print(f"  Action {a} ({action_names[a]:7s}): {pct:5.2f}%")
        
    print("="*60)
    print("[INFO] A diverse action distribution indicates the model has learned")
    print("       transferable logic and is not overfitted to the training data.")
    print("="*60)


    if __name__ == "__main__":
    DATASET_DIR = r"C:\Users\a\Desktop\A Hybrid Learning Framework for Adaptive Decision-Making in SDN-Based Healthcare IoT Systems\set-a"
    PRETRAINED_MODEL_PATH = "./checkpoints_DQN/seed_1/q_network_epbest.pth"
    LSTM_PATH = "./models/predictor.pth"

    states, p_crits = load_physionet_data_with_lstm(DATASET_DIR, LSTM_PATH, max_patients=200)

    if len(states) == 0:
        print("[ERROR] No valid data.")
    else:
        model = load_pretrained_model(PRETRAINED_MODEL_PATH)
        evaluate_zero_shot(model, states)
