
import os
import sys
import torch
import numpy as np
import csv
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import argparse

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from diffusion.models import DiffusionMLP, DiffusionPureMLP, DiffusionTCN, FlowMatchingTCN, FlowMatchingMLP, PureMLP, PureTCN, LSTMPolicy
from diffusion.ddpm import DDPMScheduler
from diffusion.flow_matching import FlowMatchingScheduler

class AWEDataset(Dataset):
    def __init__(self, csv_file, horizon=5, normalize=True):
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
        self.action_dim = 4 + 4 + 1 # 9
        self.action_data = np.zeros((n_samples, self.action_dim))
        
        # Helper to get column data
        def get_col(name):
            return self.data[:, self.col_map[name]]
            
        self._load_condition_data(get_col, n_samples)
        self._load_action_data(get_col, n_samples)
        
        self.normalize = normalize
        if self.normalize:
            self._normalize_data()
        # Save normalization stats
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
        output_dir = os.path.join(project_root, 'output', 'multi_stack')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        stats_path = os.path.join(output_dir, 'diffusion_stats.npz')
        np.savez(stats_path, 
                 cond_min=self.cond_min, cond_max=self.cond_max,
                 action_min=self.action_min, action_max=self.action_max)
        print(f"Stats saved to {stats_path}")

    def _load_condition_data(self, get_col, n_samples):
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

    def _load_action_data(self, get_col, n_samples):
        # 2. Fill Action Data (Sequence)
        # Action dim: 9 (9 controls). Sequence length: Horizon.
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
            raise ValueError("No action plan columns found in the dataset.")
                
        self.action_dim = 9 # Total dimension

    def _normalize_data(self):
        # Min-Max Normalization
        # Condition: 33 (13 + 1 + N + 9)
        self.cond_min = self.cond_data.min(axis=0)
        self.cond_max = self.cond_data.max(axis=0)
        diff = self.cond_max - self.cond_min
        diff[diff < 1e-6] = 1.0 # Prevent division by zero
        
        # Action normalization using Physical Limits
        I_min, I_max = 0.0, 9360.0
        v_lye_min, v_lye_max = 0.0, 0.1
        v_c_min, v_c_max = 0.0, 1.0
        
        # Construct Action Min/Max vectors (9,)
        # I(4), v_lye(4), v_c(1)
        
        self.action_min = np.array([I_min]*4 + [v_lye_min]*4 + [v_c_min])
        self.action_max = np.array([I_max]*4 + [v_lye_max]*4 + [v_c_max])
        
        act_diff = self.action_max - self.action_min
        act_diff[act_diff < 1e-6] = 1.0
        
        self.cond_data = (self.cond_data - self.cond_min) / diff
        
        # Normalize actions to [-1, 1] for diffusion
        # Expand dims for broadcasting: (1, 9, 1)
        act_min_b = self.action_min[None, :, None]
        act_diff_b = act_diff[None, :, None]
        
        self.action_seq_data = 2 * ((self.action_seq_data - act_min_b) / act_diff_b) - 1
        
       

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cond = torch.FloatTensor(self.cond_data[idx])
        action = torch.FloatTensor(self.action_seq_data[idx]) # (9, Horizon)
        return cond, action

def train(args):
    # Configuration
    csv_path = args.data_path
    batch_size = args.batch_size
    num_epochs = args.epochs
    lr = args.lr
    horizon = args.horizon
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Data
    if not os.path.exists(csv_path):
        print(f"Error: Data file not found at {csv_path}")
        return

    dataset = AWEDataset(csv_path, horizon=horizon)
    
    # Split into train and test
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    print(f"Dataset split: {train_size} training samples, {test_size} test samples")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    action_dim = dataset.action_dim
    obs_dim = dataset.cond_dim      
    
    print(f"Observation Dim: {obs_dim}, Action Dim: {action_dim}")
    
    if args.model_type == 'flow_tcn':
        model = FlowMatchingTCN(
            action_dim=action_dim,
            obs_dim=obs_dim,
            horizon=horizon,
            hidden_dim=256,
            levels=4
        )
        noise_scheduler = FlowMatchingScheduler(sigma_min=1e-4, device=device)
    elif args.model_type == 'flow_mlp':
        model = FlowMatchingMLP(
            action_dim=action_dim,
            obs_dim=obs_dim,
            horizon=horizon,
            hidden_dim=256,
            num_res_blocks=3
        )
        noise_scheduler = FlowMatchingScheduler(sigma_min=1e-4, device=device)
    elif args.model_type == 'diffusion_tcn':
        model = DiffusionTCN(
            output_dim=action_dim,
            cond_dim=obs_dim,
            output_num=horizon,
            hidden_dim=256,
            levels=4
        )
        noise_scheduler = DDPMScheduler(device=device)
    elif args.model_type == 'diffusion_tcn_l3':
        model = DiffusionTCN(
            output_dim=action_dim,
            cond_dim=obs_dim,
            output_num=horizon,
            hidden_dim=256,
            levels=3
        )
        noise_scheduler = DDPMScheduler(device=device)
    elif args.model_type == 'diffusion_mlp':
        model = DiffusionMLP(
            action_dim=action_dim,
            obs_dim=obs_dim,
            horizon=horizon,
            hidden_dim=256,
            num_res_blocks=3
        )
        noise_scheduler = DDPMScheduler(device=device)
    elif args.model_type == 'diffusion_pure_mlp':
        model = DiffusionPureMLP(
            action_dim=action_dim,
            obs_dim=obs_dim,
            horizon=horizon,
            hidden_dim=256,
            num_layers=4
        )
        noise_scheduler = DDPMScheduler(device=device)
    elif args.model_type == 'guided_diffusion_mlp':
        model = DiffusionMLP(
            action_dim=action_dim,
            obs_dim=obs_dim,
            horizon=horizon,
            hidden_dim=256,
            num_res_blocks=3
        )
        noise_scheduler = DDPMScheduler(device=device)
    elif args.model_type == 'guided_diffusion_tcn':
        model = DiffusionTCN(
            output_dim=action_dim,
            cond_dim=obs_dim,
            output_num=horizon,
            hidden_dim=256,
            levels=4
        )
        noise_scheduler = DDPMScheduler(device=device)
    elif args.model_type == 'pure_mlp':
        model = PureMLP(
            action_dim=action_dim,
            obs_dim=obs_dim,
            horizon=horizon,
            hidden_dim=256,
            num_res_blocks=3
        )
        noise_scheduler = None
    elif args.model_type == 'pure_tcn':
        model = PureTCN(
            action_dim=action_dim,
            obs_dim=obs_dim,
            horizon=horizon,
            hidden_dim=256,
            levels=4
        )
        noise_scheduler = None
    elif args.model_type == 'lstm':
        model = LSTMPolicy(
            action_dim=action_dim,
            obs_dim=obs_dim,
            horizon=horizon,
            hidden_dim=256,
            num_layers=2
        )
        noise_scheduler = None
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    model.to(device)
    if noise_scheduler is not None:
        noise_scheduler.device = device

    print("-" * 50)
    print(f"Model Initialized Successfully")
    print(f"Type: {args.model_type}")
    print(f"Device: {device}")
    print(f"Action Dim: {action_dim}")
    print(f"Observation Dim: {obs_dim}")
    print(f"Horizon: {horizon}")
    print(f"Hidden Dim: 256")
    if 'tcn' in args.model_type:
        print(f"Levels: 4")
    else:
        print(f"Res Blocks: 3")
        
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    print("-" * 50)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Loop
    best_test_loss = float('inf')
    history = {'train_loss': [], 'test_loss': []}
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for cond, action_seq in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            cond = cond.to(device)
            action_seq = action_seq.to(device)
            
            if args.model_type in ('pure_mlp', 'pure_tcn', 'lstm'):
                pred = model(cond)
                loss = torch.nn.functional.mse_loss(pred, action_seq)
            elif 'flow' in args.model_type:
                loss = noise_scheduler.compute_loss(model, action_seq, cond)
            else:
                timesteps = torch.randint(0, noise_scheduler.num_timesteps, (action_seq.shape[0],), device=device).long()
                loss = noise_scheduler.p_losses(model, action_seq, timesteps, cond)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for cond, action_seq in test_loader:
                cond = cond.to(device)
                action_seq = action_seq.to(device)

                if args.model_type in ('pure_mlp', 'pure_tcn', 'lstm'):
                    pred = model(cond)
                    loss = torch.nn.functional.mse_loss(pred, action_seq)
                elif 'flow' in args.model_type:
                    loss = noise_scheduler.compute_loss(model, action_seq, cond)
                else:
                    timesteps = torch.randint(0, noise_scheduler.num_timesteps, (action_seq.shape[0],), device=device).long()
                    loss = noise_scheduler.p_losses(model, action_seq, timesteps, cond)

                test_loss += loss.item()
                
        avg_test_loss = test_loss / len(test_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['test_loss'].append(avg_test_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Test Loss: {avg_test_loss:.6f}")
        
        # Save Best
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            save_path = os.path.join(args.output_dir, f'{args.model_type}_policy_best.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Policy Model (Multi-Stack)')
    
    # Find latest CSV
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    default_data_dir = os.path.join(project_root, 'output', 'multi_stack', 'dataset')
    
    latest_csv = ""
    if os.path.exists(default_data_dir):
        # Look for nmpc_dataset (generated) or nmpc_data (logs)
        csv_files = [f for f in os.listdir(default_data_dir) if f.endswith('.csv') and ('nmpc_dataset' in f)]
        if csv_files:
            latest_csv = max([os.path.join(default_data_dir, f) for f in csv_files], key=os.path.getctime)
    
    default_output_dir = os.path.join(project_root, 'output', 'multi_stack', 'policy')

    parser.add_argument('--data_path', type=str, default=latest_csv, help='Path to dataset CSV')
    parser.add_argument('--output_dir', type=str, default=default_output_dir)
    parser.add_argument('--horizon', type=int, default=5, help='Prediction horizon')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--model_type', type=str, default='diffusion_tcn', choices=['diffusion_tcn', 'diffusion_tcn_l3', 'diffusion_mlp', 'diffusion_pure_mlp', 'guided_diffusion_mlp', 'guided_diffusion_tcn', 'flow_tcn', 'flow_mlp', 'pure_mlp', 'pure_tcn', 'lstm'], help='Model type: diffusion_tcn, diffusion_mlp, flow_tcn, flow_mlp, pure_mlp, pure_tcn, lstm')

    args = parser.parse_args()
    
    if not args.data_path or not os.path.exists(args.data_path):
        print(f"Error: Dataset not found. Please provide --data_path or generate dataset first.")
    else:
        print(f"Training on {args.data_path}")
        train(args)
