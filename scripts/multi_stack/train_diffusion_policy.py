
import os
import sys
import torch
import numpy as np
import csv
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import argparse

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from diffusion.model import DiffusionMLP, DiffusionTCN, FlowMatchingTCN
from diffusion.ddpm import DDPMScheduler
from diffusion.flow_matching import FlowMatchingScheduler

class AWEDataset(Dataset):
    def __init__(self, csv_file, horizon=10, normalize=True):
        self.data = []
        self.headers = []
        self.horizon = horizon
        
        # Load CSV
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            self.headers = next(reader)
            for row in reader:
                # Convert to float
                try:
                    self.data.append([float(x) for x in row])
                except ValueError:
                    continue
        
        self.data = np.array(self.data)
        
        # Define column indices (based on header names)
        # t,P_ref,P_real,T_s_mean,T_sep,T_c_out,T_ref,I_mean,v_lye_mean,v_c,HTO,H2_rate,T_s_1,T_s_2,T_s_3,T_s_4...
        self.col_map = {name: i for i, name in enumerate(self.headers)}
        
        # --- Construct Features ---
        n_samples = len(self.data)
        
        # Pre-allocate arrays
        # Cond: 13 + 1 + N + 9
        # T_s_in(1), T_s_vec(4), T_sep(1), T_c_out(1), n_H2_an_vec(4), n_liq(1), n_gas(1), T_ref(1), P_ref(N), I_prev(4), v_lye_prev(4), v_c_prev(1)
        self.cond_dim = 1 + 4 + 1 + 1 + 4 + 1 + 1 + 1 + self.horizon + 4 + 4 + 1
        self.cond_data = np.zeros((n_samples, self.cond_dim))
        
        # Action: I(4), v_lye(4), v_c(1)
        self.action_dim = 4 + 4 + 1
        self.action_data = np.zeros((n_samples, self.action_dim))
        
        # Helper to get column data
        def get_col(name):
            return self.data[:, self.col_map[name]]
            
        # 1. Fill Condition Data
        # T_s_in
        if 'T_s_in' in self.col_map:
            self.cond_data[:, 0] = get_col('T_s_in')
        else:
            self.cond_data[:, 0] = get_col('T_sep') # Proxy
        
        # T_s_vec (4 stacks)
        if 'T_s_1' in self.col_map:
            self.cond_data[:, 1] = get_col('T_s_1')
            self.cond_data[:, 2] = get_col('T_s_2')
            self.cond_data[:, 3] = get_col('T_s_3')
            self.cond_data[:, 4] = get_col('T_s_4')
        else:
             raise ValueError("Dataset missing T_s columns (T_s_1..4)")
        
        # T_sep
        self.cond_data[:, 5] = get_col('T_sep')
        
        # T_c_out
        self.cond_data[:, 6] = get_col('T_c_out')
        
        # n_H2_an_vec (4 stacks)
        if 'n_H2_an_1' in self.col_map:
            self.cond_data[:, 7] = get_col('n_H2_an_1')
            self.cond_data[:, 8] = get_col('n_H2_an_2')
            self.cond_data[:, 9] = get_col('n_H2_an_3')
            self.cond_data[:, 10] = get_col('n_H2_an_4')
        else:
             self.cond_data[:, 7:11] = 0.0 # Default to 0 if missing (less critical, or raise error?)
             # Let's keep the default 0.0 behavior for robustness, or raise error if critical.
             # Given user instruction "Unified to latest standard", I should probably enforce it.
             # But n_H2_an might be optional in some contexts? 
             # Let's assume critical for now based on user intent.
             # Actually, previous code had `else: self.cond_data[:, 7:11] = 0.0`.
             # I'll stick to that default if missing, but remove the `elif` for `_vec`.
             pass
        
        # n_liq
        if 'n_liq' in self.col_map:
            self.cond_data[:, 11] = get_col('n_liq')
        else:
            self.cond_data[:, 11] = 0.0
        
        # n_gas
        if 'n_gas' in self.col_map:
            self.cond_data[:, 12] = get_col('n_gas')
        else:
            self.cond_data[:, 12] = get_col('HTO') # Proxy
        
        # T_ref
        self.cond_data[:, 13] = get_col('T_ref')
        
        # P_ref(N) - Future horizon
        # Prefer P_ref_future_0...9 columns if available
        if 'P_ref_future_0' in self.col_map:
            for k in range(self.horizon):
                col_name = f'P_ref_future_{k}'
                if col_name in self.col_map:
                    self.cond_data[:, 14 + k] = get_col(col_name)
        else:
            # Fallback to shifting P_ref column
            p_ref_col = get_col('P_ref')
            for i in range(n_samples):
                end_idx = min(i + self.horizon, n_samples)
                steps_available = end_idx - i
                self.cond_data[i, 14 : 14 + steps_available] = p_ref_col[i : end_idx]
                if steps_available < self.horizon:
                    self.cond_data[i, 14 + steps_available : 14 + self.horizon] = p_ref_col[-1]
        
        # Previous Controls
        base_idx = 14 + self.horizon
        
        # I_prev (4)
        if 'I_prev_1' in self.col_map:
            self.cond_data[:, base_idx] = get_col('I_prev_1')
            self.cond_data[:, base_idx+1] = get_col('I_prev_2')
            self.cond_data[:, base_idx+2] = get_col('I_prev_3')
            self.cond_data[:, base_idx+3] = get_col('I_prev_4')
        else:
             raise ValueError("Dataset missing I_prev columns (I_prev_1..4)")
            
        # v_lye_prev (4)
        if 'v_lye_prev_1' in self.col_map:
            self.cond_data[:, base_idx+4] = get_col('v_lye_prev_1')
            self.cond_data[:, base_idx+5] = get_col('v_lye_prev_2')
            self.cond_data[:, base_idx+6] = get_col('v_lye_prev_3')
            self.cond_data[:, base_idx+7] = get_col('v_lye_prev_4')
        else:
             raise ValueError("Dataset missing v_lye_prev columns")
            
        # v_c_prev (1)
        if 'v_c_prev' in self.col_map:
            self.cond_data[:, base_idx+8] = get_col('v_c_prev')
        else:
             # Fallback if v_c_prev is missing but v_c exists (less critical)
             if 'v_c' in self.col_map:
                 v_c = get_col('v_c')
                 v_c_prev = np.roll(v_c, 1)
                 v_c_prev[0] = v_c[0]
                 self.cond_data[:, base_idx+8] = v_c_prev
             else:
                 raise ValueError("Dataset missing v_c_prev or v_c columns")
        
        # 2. Fill Action Data (Sequence)
        # Action dim: 9. Sequence length: Horizon.
        # Shape: (n_samples, action_dim, horizon) for TCN/FlowMatching
        
        self.action_seq_data = np.zeros((n_samples, 9, self.horizon))
        
        # We need to reconstruct the plan from the dataset columns
        # Columns format: plan_step_{k}_I_{i}, plan_step_{k}_v_lye_{i}, plan_step_{k}_v_c
        
        # Check if plan columns exist
        if 'plan_step_0_I_1' in self.col_map:
            for k in range(self.horizon):
                # I (4)
                for i in range(4):
                    col_name = f'plan_step_{k}_I_{i+1}'
                    if col_name in self.col_map:
                        self.action_seq_data[:, i, k] = get_col(col_name)
                
                # v_lye (4)
                for i in range(4):
                    col_name = f'plan_step_{k}_v_lye_{i+1}'
                    if col_name in self.col_map:
                        self.action_seq_data[:, 4+i, k] = get_col(col_name)
                        
                # v_c (1)
                col_name = f'plan_step_{k}_v_c'
                if col_name in self.col_map:
                    self.action_seq_data[:, 8, k] = get_col(col_name)
        else:
            # Fallback for old datasets (just repeat single action or shift?)
            # For strict training, maybe raise error or warn.
            # Let's fallback to repeating the single step action for now to avoid breaking old data tests immediately
            print("Warning: Plan columns not found. Using single step action repeated.")
            # I (4)
            if 'I_1' in self.col_map:
                self.action_seq_data[:, 0, :] = get_col('I_1')[:, None]
                self.action_seq_data[:, 1, :] = get_col('I_2')[:, None]
                self.action_seq_data[:, 2, :] = get_col('I_3')[:, None]
                self.action_seq_data[:, 3, :] = get_col('I_4')[:, None]
            
            # v_lye (4)
            if 'v_lye_1' in self.col_map:
                self.action_seq_data[:, 4, :] = get_col('v_lye_1')[:, None]
                self.action_seq_data[:, 5, :] = get_col('v_lye_2')[:, None]
                self.action_seq_data[:, 6, :] = get_col('v_lye_3')[:, None]
                self.action_seq_data[:, 7, :] = get_col('v_lye_4')[:, None]
                
            # v_c (1)
            if 'v_c' in self.col_map:
                self.action_seq_data[:, 8, :] = get_col('v_c')[:, None]

        # For MLP, we might still want flattened or single step. 
        # But user asked for "actions sequence".
        # If model is MLP, we might need to flatten or just predict first step?
        # Usually Diffusion Policy predicts sequence.
        
        self.action_dim = 9 # Base dimension
        
        self.normalize = normalize
        if self.normalize:
            # Min-Max Normalization
            self.cond_min = self.cond_data.min(axis=0)
            self.cond_max = self.cond_data.max(axis=0)
            
            # Handle constant columns (max == min) to avoid div/0
            diff = self.cond_max - self.cond_min
            diff[diff < 1e-6] = 1.0 # Prevent division by zero
            
            # Action normalization (global min/max across horizon)
            # Flatten to (N*Horizon, 9) to find min/max
            flat_actions = self.action_seq_data.transpose(0, 2, 1).reshape(-1, 9)
            self.action_min = flat_actions.min(axis=0)
            self.action_max = flat_actions.max(axis=0)
            
            act_diff = self.action_max - self.action_min
            act_diff[act_diff < 1e-6] = 1.0
            
            self.cond_data = (self.cond_data - self.cond_min) / diff
            
            # Normalize actions to [-1, 1] for diffusion
            # Expand dims for broadcasting: (1, 9, 1)
            act_min_b = self.action_min[None, :, None]
            act_diff_b = act_diff[None, :, None]
            
            self.action_seq_data = 2 * ((self.action_seq_data - act_min_b) / act_diff_b) - 1
            
            # Save normalization stats
            stats_path = os.path.join(r'd:\Projects\AWE\output\multi_stack', 'diffusion_stats.npz')
            np.savez(stats_path, 
                     cond_min=self.cond_min, cond_max=self.cond_max,
                     action_min=self.action_min, action_max=self.action_max)
            print(f"Stats saved to {stats_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cond = torch.FloatTensor(self.cond_data[idx])
        action = torch.FloatTensor(self.action_seq_data[idx]) # (9, Horizon)
        return cond, action

def train():
    parser = argparse.ArgumentParser(description='Train Diffusion Policy')
    parser.add_argument('--model_type', type=str, default='tcn', choices=['mlp', 'tcn', 'flow_matching'], help='Model type: mlp, tcn, or flow_matching')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    args = parser.parse_args()

    # Configuration
    # Find latest CSV
    output_dir = r"d:\Projects\AWE\output\multi_stack"
    csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv') and 'nmpc_data' in f]
    if not csv_files:
        print("No CSV data found!")
        return
    latest_csv = max([os.path.join(output_dir, f) for f in csv_files], key=os.path.getctime)
    print(f"Using dataset: {latest_csv}")
    
    csv_path = latest_csv
    batch_size = 64
    num_epochs = args.epochs
    lr = 1e-4
    horizon = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Data
    if not os.path.exists(csv_path):
        print(f"Error: Data file not found at {csv_path}")
        return

    dataset = AWEDataset(csv_path, horizon=horizon)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Model
    action_dim = dataset.action_dim # 9
    obs_dim = dataset.cond_dim      # 33 approx
    
    print(f"Observation Dim: {obs_dim}, Action Dim: {action_dim}")
    
    if args.model_type == 'mlp':
        model = DiffusionMLP(action_dim=action_dim, obs_dim=obs_dim).to(device)
        print("Using ConditionalDiffusionMLP model")
    elif args.model_type == 'tcn':
        model = DiffusionTCN(action_dim=action_dim, obs_dim=obs_dim, horizon=horizon).to(device)
        print("Using TCNDiffusion model")
    elif args.model_type == 'flow_matching':
        # Flow Matching uses the same architecture as TCN Diffusion
        model = FlowMatchingTCN(action_dim=action_dim, obs_dim=obs_dim, horizon=horizon).to(device)
        print("Using FlowMatchingTCN model for Flow Matching")
    
    if args.model_type == 'flow_matching':
        scheduler = FlowMatchingScheduler(device=device)
    else:
        scheduler = DDPMScheduler(num_timesteps=100, device=device)
        
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training Loop
    print("Starting training...")
    model.train()
    
    try:
        for epoch in range(num_epochs):
            epoch_loss = 0
            for cond, action in dataloader:
                cond = cond.to(device)
                action = action.to(device) # x_start
                
                if args.model_type == 'flow_matching':
                    # Flow Matching Loss
                    loss = scheduler.compute_loss(model, action, cond)
                else:
                    # DDPM Loss
                    # Sample timesteps
                    t = torch.randint(0, scheduler.num_timesteps, (cond.shape[0],), device=device).long()
                    # Compute loss
                    loss = scheduler.p_losses(model, action, t, cond)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.6f}")
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        # Save model
        model_filename = f'diffusion_policy_model_{args.model_type}.pth'
        torch.save(model.state_dict(), os.path.join(r'd:\Projects\AWE\output\multi_stack', model_filename))
        print(f"Model saved to output/multi_stack/{model_filename}")

if __name__ == "__main__":
    train()
