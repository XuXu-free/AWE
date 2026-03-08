
import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from diffusion.models import DiffusionMLP, DiffusionTCN, FlowMatchingTCN, FlowMatchingMLP
from diffusion.ddpm import DDPMScheduler
from diffusion.flow_matching import FlowMatchingScheduler

class SingleStackDataset(Dataset):
    def __init__(self, csv_file, horizon=5, normalize=True):
        self.horizon = horizon
        
        # Load CSV using pandas for easier handling
        df = pd.read_csv(csv_file)
        
        # --- Preprocessing for Previous Actions ---
        # CSV has 'v_c_prev', but might miss 'I_prev' and 'v_lye_prev'
        # We construct them by shifting current actions
        if 'I_prev' not in df.columns:
            df['I_prev'] = df['I'].shift(1).fillna(0.0)
        if 'v_lye_prev' not in df.columns:
            df['v_lye_prev'] = df['v_lye'].shift(1).fillna(df['v_lye'].iloc[0] if len(df) > 0 else 0.0)
        if 'v_c_prev' not in df.columns: # Should exist, but just in case
            df['v_c_prev'] = df['v_c'].shift(1).fillna(0.0)
            
        self.data = df
        n_samples = len(df)
        
        # --- Construct Features ---
        
        # Condition Vector Structure:
        # 1. System State (6): T_s_in, T_s, T_sep, T_c_out, n_liq, n_gas
        # 2. Reference (1 + N): T_ref, P_ref_future_0...N-1
        # 3. Prev Actions (3): I_prev, v_lye_prev, v_c_prev
        
        self.cond_dim = 6 + 1 + self.horizon + 3
        self.cond_data = np.zeros((n_samples, self.cond_dim))
        
        # 1. System State
        self.cond_data[:, 0] = df['T_s_in'].values
        self.cond_data[:, 1] = df['T_s'].values
        self.cond_data[:, 2] = df['T_sep'].values
        self.cond_data[:, 3] = df['T_c_out'].values
        self.cond_data[:, 4] = df['n_liq'].values
        self.cond_data[:, 5] = df['n_gas'].values
        
        # 2. Reference
        self.cond_data[:, 6] = df['T_ref'].values
        
        # P_ref_future
        for k in range(self.horizon):
            col_name = f'P_ref_future_{k}'
            if col_name in df.columns:
                self.cond_data[:, 7 + k] = df[col_name].values
            else:
                # Fallback if specific future column missing (though generate_dataset should produce them)
                # Use P_ref and shift
                print(f"Warning: {col_name} missing, using shifted P_ref")
                self.cond_data[:, 7 + k] = df['P_ref'].shift(-k).ffill().values

        # 3. Prev Actions
        base_idx = 7 + self.horizon
        self.cond_data[:, base_idx] = df['I_prev'].values
        self.cond_data[:, base_idx + 1] = df['v_lye_prev'].values
        self.cond_data[:, base_idx + 2] = df['v_c_prev'].values
        
        # --- Construct Action Sequence (Target) ---
        # Action Dim per step: 3 (controls)
        # I, v_lye, v_c
        
        self.action_dim = 3
        self.action_seq_data = np.zeros((n_samples, self.action_dim, self.horizon))
        
        for k in range(self.horizon):
            # Controls
            col_I = f'plan_step_{k}_I'
            col_v_lye = f'plan_step_{k}_v_lye'
            col_v_c = f'plan_step_{k}_v_c'
            
            if col_I in df.columns:
                self.action_seq_data[:, 0, k] = df[col_I].values
                self.action_seq_data[:, 1, k] = df[col_v_lye].values
                self.action_seq_data[:, 2, k] = df[col_v_c].values
        
        # Normalization
        self.normalize = normalize
        if self.normalize:
            self._normalize_data()
            
            # Save stats
            output_dir = os.path.dirname(csv_file)
            stats_path = os.path.join(output_dir, 'diffusion_stats.npz')
            np.savez(stats_path, 
                     cond_min=self.cond_min, cond_max=self.cond_max,
                     action_min=self.action_min, action_max=self.action_max)
            print(f"Stats saved to {stats_path}")

    def _normalize_data(self):
        # Condition normalization
        self.cond_min = np.min(self.cond_data, axis=0)
        self.cond_max = np.max(self.cond_data, axis=0)
        
        # Avoid div by zero
        self.cond_max[self.cond_max == self.cond_min] += 1.0
        
        self.cond_data = (self.cond_data - self.cond_min) / (self.cond_max - self.cond_min) * 2 - 1
        
        # Action normalization (Global min/max across horizon for each dimension)
        # Shape: (N, 7, Horizon) -> reshape to (N*Horizon, 7) to find min/max
        action_flat = self.action_seq_data.transpose(0, 2, 1).reshape(-1, self.action_dim)
        self.action_min = np.min(action_flat, axis=0)
        self.action_max = np.max(action_flat, axis=0)
        
        self.action_max[self.action_max == self.action_min] += 1.0
        
        # Reshape min/max for broadcasting: (7, 1)
        action_min_bc = self.action_min.reshape(1, -1, 1)
        action_max_bc = self.action_max.reshape(1, -1, 1)
        
        self.action_seq_data = (self.action_seq_data - action_min_bc) / (action_max_bc - action_min_bc) * 2 - 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cond = torch.FloatTensor(self.cond_data[idx])
        action_seq = torch.FloatTensor(self.action_seq_data[idx])
        return cond, action_seq

def train(args):
    # 1. Dataset
    dataset = SingleStackDataset(args.data_path, horizon=args.horizon)
    
    # Split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Train size: {train_size}, Val size: {val_size}")
    print(f"Condition Dim: {dataset.cond_dim}, Action Dim: {dataset.action_dim}, Horizon: {args.horizon}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. Model
    if args.model_type == 'flow_tcn':
        model = FlowMatchingTCN(
            action_dim=dataset.action_dim, # 7
            obs_dim=dataset.cond_dim, # 10+N
            horizon=args.horizon,
            hidden_dim=256,
            levels=4
        )
        noise_scheduler = FlowMatchingScheduler(sigma_min=1e-4, device=device)
    elif args.model_type == 'flow_mlp':
        model = FlowMatchingMLP(
            action_dim=dataset.action_dim,
            obs_dim=dataset.cond_dim,
            horizon=args.horizon,
            hidden_dim=256,
            num_res_blocks=3
        )
        noise_scheduler = FlowMatchingScheduler(sigma_min=1e-4, device=device)
    elif args.model_type == 'diffusion_tcn':
        model = DiffusionTCN(
            output_dim=dataset.action_dim,
            cond_dim=dataset.cond_dim,
            output_num=args.horizon,
            hidden_dim=256,
            levels=4
        )
        noise_scheduler = DDPMScheduler(num_timesteps=100, device=device)
    elif args.model_type == 'diffusion_mlp':
        model = DiffusionMLP(
            action_dim=dataset.action_dim,
            obs_dim=dataset.cond_dim,
            horizon=args.horizon,
            hidden_dim=256,
            num_res_blocks=3
        )
        noise_scheduler = DDPMScheduler(num_timesteps=100, device=device)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    model.to(device)
    noise_scheduler.device = device # Ensure scheduler knows device if needed
    
    print("-" * 50)
    print(f"Model Initialized Successfully")
    print(f"Type: {args.model_type}")
    print(f"Device: {device}")
    print(f"Action Dim: {dataset.action_dim}")
    print(f"Observation Dim: {dataset.cond_dim}")
    print(f"Horizon: {args.horizon}")
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
    
    # 4. Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 5. Loop
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for cond, action_seq in train_loader:
            cond = cond.to(device)
            action_seq = action_seq.to(device)
            
            if 'flow' in args.model_type:
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
        val_loss = 0
        with torch.no_grad():
            for cond, action_seq in val_loader:
                cond = cond.to(device)
                action_seq = action_seq.to(device)
                
                if 'flow' in args.model_type:
                    loss = noise_scheduler.compute_loss(model, action_seq, cond)
                else:
                    timesteps = torch.randint(0, noise_scheduler.num_timesteps, (action_seq.shape[0],), device=device).long()
                    loss = noise_scheduler.p_losses(model, action_seq, timesteps, cond)
                    
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        # Save Best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(args.output_dir, f'{args.model_type}_policy_best.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

    # Plot
    plt.figure()
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, f'{args.model_type}_training_curve.png'))
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Find the latest dataset
    default_data_dir = r'c:\Users\admin\Desktop\sjtu\AWE\output\single_stack\dataset'
    # Simple logic to find latest csv
    latest_csv = ""
    if os.path.exists(default_data_dir):
        files = [f for f in os.listdir(default_data_dir) if f.endswith('.csv')]
        if files:
            files.sort(reverse=True)
            latest_csv = os.path.join(default_data_dir, files[0])
    
    # Default output dir for policy
    default_output_dir = r'c:\Users\admin\Desktop\sjtu\AWE\output\single_stack\policy'
    if not os.path.exists(default_output_dir):
        os.makedirs(default_output_dir)

    parser.add_argument('--data_path', type=str, default=latest_csv, help='Path to dataset CSV')
    parser.add_argument('--output_dir', type=str, default=default_output_dir)
    parser.add_argument('--horizon', type=int, default=5, help='Prediction horizon')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--model_type', type=str, default='diffusion_tcn', choices=['diffusion_tcn', 'diffusion_mlp', 'flow_tcn', 'flow_mlp'], help='Model type: diffusion_tcn, diffusion_mlp, flow_tcn, flow_mlp')
    
    args = parser.parse_args()
    
    if not args.data_path or not os.path.exists(args.data_path):
        print(f"Error: Dataset not found at {args.data_path}")
    else:
        print(f"Training on {args.data_path}")
        train(args)
