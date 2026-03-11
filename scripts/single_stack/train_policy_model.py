
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
        
        df = pd.read_csv(csv_file)
        
        if 'I_prev' not in df.columns:
            df['I_prev'] = df['I'].shift(1).fillna(0.0)
        if 'v_lye_prev' not in df.columns:
            df['v_lye_prev'] = df['v_lye'].shift(1).fillna(df['v_lye'].iloc[0] if len(df) > 0 else 0.0)
        if 'v_c_prev' not in df.columns:
            df['v_c_prev'] = df['v_c'].shift(1).fillna(0.0)
            
        self.data = df
        n_samples = len(df)
        
        self.col_map = {c: i for i, c in enumerate(df.columns)}
        def get_col(name):
            return df[name].values
        
        self._load_condition_data(get_col, n_samples)
        self._load_action_data(get_col, n_samples)
        
        # Normalization
        self.normalize = normalize
        if self.normalize:
            self._normalize_data()
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
            output_dir = os.path.join(project_root, 'output', 'single_stack')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            stats_path = os.path.join(output_dir, 'diffusion_stats.npz')
            np.savez(stats_path,
                     cond_min=self.cond_min, cond_max=self.cond_max,
                     action_min=self.action_min, action_max=self.action_max)
            print(f"Stats saved to {stats_path}")
    
    def _load_condition_data(self, get_col, n_samples):
        self.cond_dim = 6 + 1 + self.horizon + 3
        self.cond_data = np.zeros((n_samples, self.cond_dim))
        
        self.cond_data[:, 0] = get_col('T_s_in')
        self.cond_data[:, 1] = get_col('T_s')
        self.cond_data[:, 2] = get_col('T_sep')
        self.cond_data[:, 3] = get_col('T_c_out')
        self.cond_data[:, 4] = get_col('n_liq')
        self.cond_data[:, 5] = get_col('n_gas')
        
        self.cond_data[:, 6] = get_col('T_ref')
        for k in range(self.horizon):
            col_name = f'P_ref_future_{k}'
            if col_name in self.col_map:
                self.cond_data[:, 7 + k] = get_col(col_name)
            else:
                self.cond_data[:, 7 + k] = self.data['P_ref'].shift(-k).ffill().values
        base_idx = 7 + self.horizon
        
        self.cond_data[:, base_idx] = get_col('I_prev')
        self.cond_data[:, base_idx + 1] = get_col('v_lye_prev')
        self.cond_data[:, base_idx + 2] = get_col('v_c_prev')
    
    def _load_action_data(self, get_col, n_samples):
        self.action_dim = 3
        self.action_seq_data = np.zeros((n_samples, self.action_dim, self.horizon))
        if f'plan_step_0_I' in self.col_map:
            for k in range(self.horizon):
                self.action_seq_data[:, 0, k] = get_col(f'plan_step_{k}_I')
                self.action_seq_data[:, 1, k] = get_col(f'plan_step_{k}_v_lye')
                self.action_seq_data[:, 2, k] = get_col(f'plan_step_{k}_v_c')
        else:
            raise ValueError("No action plan columns found in the dataset.")

    def _normalize_data(self):
        # Condition normalization
        self.cond_min = np.min(self.cond_data, axis=0)
        self.cond_max = np.max(self.cond_data, axis=0)
        
        # Avoid div by zero
        if np.any(self.cond_max == self.cond_min):
            self.cond_max[self.cond_max == self.cond_min] += 1.0
        
        self.cond_data = (self.cond_data - self.cond_min) / (self.cond_max - self.cond_min) * 2 - 1
        
        # Action normalization
        I_min, I_max = 0.0, 9360.0
        v_lye_min, v_lye_max = 0.0, 0.1
        v_c_min, v_c_max = 0.0, 1.0
        
        self.action_min = np.array([I_min, v_lye_min, v_c_min])
        self.action_max = np.array([I_max, v_lye_max, v_c_max])
        
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
        noise_scheduler = DDPMScheduler(device=device)
    elif args.model_type == 'diffusion_mlp':
        model = DiffusionMLP(
            action_dim=dataset.action_dim,
            obs_dim=dataset.cond_dim,
            horizon=args.horizon,
            hidden_dim=256,
            num_res_blocks=3
        )
        noise_scheduler = DDPMScheduler(device=device)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    model.to(device)
    
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
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    default_data_dir = os.path.join(project_root, 'output', 'single_stack', 'dataset')
    latest_csv = ""
    if os.path.exists(default_data_dir):
        files = [f for f in os.listdir(default_data_dir) if f.endswith('.csv')]
        nmpc_files = [f for f in files if f.lower().startswith('nmpc_dataset')]
        if nmpc_files:
            nmpc_files.sort(reverse=True)
            latest_csv = os.path.join(default_data_dir, nmpc_files[0])
        else:
            non_warmup = [f for f in files if not f.lower().startswith('warmup')]
            if non_warmup:
                non_warmup.sort(reverse=True)
                latest_csv = os.path.join(default_data_dir, non_warmup[0])
    
    default_output_dir = os.path.join(project_root, 'output', 'single_stack', 'policy')
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
