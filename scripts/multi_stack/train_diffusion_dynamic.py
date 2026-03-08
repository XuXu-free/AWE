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

from diffusion.models import DiffusionTCN
from diffusion.ddpm import DDPMScheduler

class DynamicsTrajectoryDataset(Dataset):
    def __init__(self, csv_file, horizon=5, normalize=True):
        self.horizon = horizon
        self.data_df = pd.read_csv(csv_file)
        
        # Define State and Action Columns
        self.state_cols = [
            'T_s_in', 
            'T_s_1', 'T_s_2', 'T_s_3', 'T_s_4',
            'T_sep', 
            'T_c_out',
            'n_H2_an_1', 'n_H2_an_2', 'n_H2_an_3', 'n_H2_an_4',
            'n_liq', 
            'n_gas'
        ]
        
        self.action_cols = [
            'I_1', 'I_2', 'I_3', 'I_4',
            'v_lye_1', 'v_lye_2', 'v_lye_3', 'v_lye_4',
            'v_c'
        ]
        
        # Verify columns exist
        for c in self.state_cols:
            if c not in self.data_df.columns:
                self.data_df[c] = 0.0
        for c in self.action_cols:
            if c not in self.data_df.columns:
                self.data_df[c] = 0.0
        
        # Convert to numpy
        self.states = self.data_df[self.state_cols].values.astype(np.float32)
        self.actions = self.data_df[self.action_cols].values.astype(np.float32)
        
        self.n_total = len(self.states)
        # We need sequences of length H+1 (1 for current state, H for future)
        # We also need action sequences of length H
        # Valid start indices: 0 to N - (H + 1)
        self.n_samples = self.n_total - self.horizon
        
        # Dimensions
        self.state_dim = len(self.state_cols)
        self.action_dim = len(self.action_cols)
        
        # Condition Dim: State (1) + Action Sequence (H) flattened
        self.cond_dim = self.state_dim + self.horizon * self.action_dim
        # Target Dim: State Sequence (H)
        self.target_dim = self.state_dim
        
        self.normalize = normalize
        if self.normalize:
            self._compute_stats()
            
        # Save stats
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
        output_dir = os.path.join(project_root, 'output', 'multi_stack')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        stats_path = os.path.join(output_dir, 'dynamics_trajectory_stats.npz')
        np.savez(stats_path, 
                 state_min=self.state_min, state_max=self.state_max,
                 action_min=self.action_min, action_max=self.action_max)
        print(f"Stats saved to {stats_path}")

    def _compute_stats(self):
        # Normalize States
        self.state_min = self.states.min(axis=0)
        self.state_max = self.states.max(axis=0)
        self.state_diff = self.state_max - self.state_min
        self.state_diff[self.state_diff < 1e-6] = 1.0
        
        # Normalize Actions
        self.action_min = self.actions.min(axis=0)
        self.action_max = self.actions.max(axis=0)
        self.action_diff = self.action_max - self.action_min
        self.action_diff[self.action_diff < 1e-6] = 1.0

    def _normalize_state(self, s):
        # To [-1, 1] for Diffusion Target
        return 2.0 * (s - self.state_min) / self.state_diff - 1.0
        
    def _normalize_state_cond(self, s):
        # To [0, 1] for Condition
        return (s - self.state_min) / self.state_diff

    def _normalize_action(self, a):
        # To [0, 1] for Condition (or [-1, 1]? Usually conditions are flexible, but consistent scaling helps)
        # Let's use [-1, 1] to be consistent with signal magnitude
        return 2.0 * (a - self.action_min) / self.action_diff - 1.0

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Current State: s_t
        s_t = self.states[idx]
        
        # Future States (Target): s_{t+1} ... s_{t+H}
        # Indices: idx+1 to idx+H+1
        s_future = self.states[idx+1 : idx+1+self.horizon]
        
        # Action Sequence (Condition): a_t ... a_{t+H-1}
        # Indices: idx to idx+H
        a_seq = self.actions[idx : idx+self.horizon]
        
        # Normalize
        if self.normalize:
            # Condition: s_t [0,1], a_seq [-1,1]
            s_t_norm = self._normalize_state_cond(s_t)
            a_seq_norm = self._normalize_action(a_seq)
            
            # Target: s_future [-1,1]
            s_future_norm = self._normalize_state(s_future)
        else:
            s_t_norm = s_t
            a_seq_norm = a_seq
            s_future_norm = s_future
            
        # Flatten Action Sequence for Condition
        # a_seq: (H, Action_Dim) -> (H * Action_Dim)
        a_seq_flat = a_seq_norm.flatten()
        
        # Construct Condition: [s_t, a_seq_flat]
        cond = np.concatenate([s_t_norm, a_seq_flat])
        
        # Construct Target: Transpose to (State_Dim, H) for TCN
        target = s_future_norm.T 
        
        return torch.FloatTensor(cond), torch.FloatTensor(target)

def train():
    parser = argparse.ArgumentParser(description='Train Diffusion Dynamics Trajectory Model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--horizon', type=int, default=5, help='Prediction horizon')
    args = parser.parse_args()

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    dataset_dir = os.path.join(project_root, 'output', 'multi_stack', 'dataset')
    model_output_dir = os.path.join(project_root, 'output', 'multi_stack', 'model')
    
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)

    # Find latest CSV
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory not found: {dataset_dir}")
        return

    csv_files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv') and 'nmpc_dataset' in f]
    if not csv_files:
        print("No dataset found!")
        return
        
    latest_csv = max([os.path.join(dataset_dir, f) for f in csv_files], key=os.path.getctime)
    print(f"Using dataset: {latest_csv}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset
    dataset = DynamicsTrajectoryDataset(latest_csv, horizon=args.horizon)
    
    # Split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Model
    # Target: s_{t+1:t+H} (dim: state_dim x H)
    # Condition: s_t, a_{t:t+H-1} flattened
    
    target_dim = dataset.target_dim # State Dim
    cond_dim = dataset.cond_dim     # State Dim + H * Action Dim
    
    print(f"Target Dim (State): {target_dim}, Horizon: {args.horizon}")
    print(f"Condition Dim (State Curr + Action Seq): {dataset.state_dim} + {args.horizon} * {dataset.action_dim} = {cond_dim}")
    
    # Use DiffusionTCN
    # action_dim arg -> target_dim (State)
    # obs_dim arg -> cond_dim
    model = DiffusionTCN(
        output_dim=target_dim, 
        cond_dim=cond_dim, 
        output_num=args.horizon,
        hidden_dim=256
    ).to(device)
    
    scheduler = DDPMScheduler(device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print("Starting training...")
    
    # Plotting setup
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    history_path = os.path.join(model_output_dir, 'dynamics_traj_loss_history.csv')
    best_test_loss = float('inf')
    loss_history = []
    
    try:
        with tqdm(range(args.epochs), desc="Training", unit="epoch") as pbar:
            for epoch in pbar:
                model.train()
                train_loss = 0
                
                for cond, target in train_loader:
                    cond = cond.to(device)
                    target = target.to(device) # (B, State_Dim, H)
                    
                    # Sample timesteps
                    t = torch.randint(0, scheduler.num_timesteps, (cond.shape[0],), device=device).long()
                    
                    # Compute loss
                    loss = scheduler.p_losses(model, target, t, cond)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    
                avg_train_loss = train_loss / len(train_loader)
                
                # Validation
                model.eval()
                test_loss = 0
                with torch.no_grad():
                    for cond, target in test_loader:
                        cond = cond.to(device)
                        target = target.to(device)
                        
                        t = torch.randint(0, scheduler.num_timesteps, (cond.shape[0],), device=device).long()
                        loss = scheduler.p_losses(model, target, t, cond)
                        test_loss += loss.item()
                        
                avg_test_loss = test_loss / len(test_loader)
                
                loss_history.append([epoch+1, avg_train_loss, avg_test_loss])
                
                pbar.set_postfix({
                    "Train": f"{avg_train_loss:.6f}",
                    "Test": f"{avg_test_loss:.6f}"
                })
                
                if avg_test_loss < best_test_loss:
                    best_test_loss = avg_test_loss
                    torch.save(model.state_dict(), os.path.join(model_output_dir, 'best_diffusion_dynamics_traj_model.pth'))
                    
                if (epoch + 1) % 10 == 0:
                    # Save CSV
                    df_history = pd.DataFrame(loss_history, columns=['epoch', 'train_loss', 'test_loss'])
                    df_history.to_csv(history_path, index=False)
                    
                    # Save Plot
                    plt.figure(figsize=(10, 6))
                    plt.plot(df_history['epoch'], df_history['train_loss'], label='Train Loss')
                    plt.plot(df_history['epoch'], df_history['test_loss'], label='Test Loss')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.title('Diffusion Dynamics Training Loss')
                    plt.legend()
                    plt.grid(True)
                    plot_path = os.path.join(model_output_dir, 'dynamics_traj_loss.png')
                    plt.savefig(plot_path)
                    plt.close()
                
    except KeyboardInterrupt:
        print("Training interrupted.")
        
    # Save Final
    torch.save(model.state_dict(), os.path.join(model_output_dir, 'diffusion_dynamics_traj_model.pth'))
    print(f"Model saved to {os.path.join(model_output_dir, 'diffusion_dynamics_traj_model.pth')}")
    
    # Final Save History and Plot
    df_history = pd.DataFrame(loss_history, columns=['epoch', 'train_loss', 'test_loss'])
    df_history.to_csv(history_path, index=False)
    print(f"History saved to {history_path}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(df_history['epoch'], df_history['train_loss'], label='Train Loss')
    plt.plot(df_history['epoch'], df_history['test_loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Diffusion Dynamics Training Loss')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(model_output_dir, 'dynamics_traj_loss.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Loss plot saved to {plot_path}")

if __name__ == "__main__":
    train()
