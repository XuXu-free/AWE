
import os
import sys
import torch
import numpy as np
from ..base_controller import BaseController
from diffusion.model import DiffusionMLP, DiffusionTCN, FlowMatchingTCN
from diffusion.ddpm import DDPMScheduler
from diffusion.flow_matching import FlowMatchingScheduler

class MultiStackDiffusionController(BaseController):
    def __init__(self, dt=1.0, horizon=10, model_type='tcn', model_path=None, stats_path=r'd:\Projects\AWE\output\multi_stack\diffusion_stats.npz'):
        super().__init__(dt)
        self.horizon = horizon
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Default model path based on type if not provided
        if model_path is None:
            model_path = r'd:\Projects\AWE\output\multi_stack\diffusion_policy_model_{}.pth'.format(model_type)
        
        # Load Stats
        if not os.path.exists(stats_path):
            # Try looking in parent/root if not found
            if os.path.exists(os.path.join('..', stats_path)):
                stats_path = os.path.join('..', stats_path)
            else:
                raise FileNotFoundError(f"Stats file not found: {stats_path}")
                
        stats = np.load(stats_path)
        self.cond_min = torch.FloatTensor(stats['cond_min']).to(self.device)
        self.cond_max = torch.FloatTensor(stats['cond_max']).to(self.device)
        self.action_min = torch.FloatTensor(stats['action_min']).to(self.device)
        self.action_max = torch.FloatTensor(stats['action_max']).to(self.device)
        
        # Handle constant columns to avoid div/0
        self.cond_diff = self.cond_max - self.cond_min
        self.cond_diff[self.cond_diff < 1e-6] = 1.0
        
        self.action_diff = self.action_max - self.action_min
        self.action_diff[self.action_diff < 1e-6] = 1.0
        
        # Load Model
        # Action: I(4), v_lye(4), v_c(1) => 9
        # Cond: 33 (13 + 1 + N + 9)
        self.action_dim = 9
        self.obs_dim = 1 + 4 + 1 + 1 + 4 + 1 + 1 + 1 + horizon + 9
        
        if model_type == 'mlp':
            self.model = DiffusionMLP(action_dim=self.action_dim, obs_dim=self.obs_dim).to(self.device)
        elif model_type == 'flow_matching':
            self.model = FlowMatchingTCN(action_dim=self.action_dim, obs_dim=self.obs_dim, horizon=horizon).to(self.device)
        else:
            self.model = DiffusionTCN(action_dim=self.action_dim, obs_dim=self.obs_dim, horizon=horizon).to(self.device)
            
        if not os.path.exists(model_path):
             if os.path.exists(os.path.join('..', model_path)):
                model_path = os.path.join('..', model_path)
             else:
                raise FileNotFoundError(f"Model file not found: {model_path}")
                
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        if model_type == 'flow_matching':
            self.scheduler = FlowMatchingScheduler(device=self.device)
        else:
            self.scheduler = DDPMScheduler(num_timesteps=100, device=self.device)
        
        # Internal state for previous action
        # I(4), v_lye(4), v_c(1)
        self.last_action = np.zeros(9)
        # Initialize with nominal values if needed (e.g. v_lye=0.03)
        self.last_action[4:8] = 0.03
        
        # Constraints (from NMPC)
        self.I_min = 0.0
        self.I_max = 7800.0
        self.v_lye_min = 0.0
        self.v_lye_max = 0.1
        self.v_c_min = 0.0
        self.v_c_max = 1.0

    def _prepare_condition(self, state, P_ref, T_ref):
        # 1. Construct Condition Vector
        cond = np.zeros(self.obs_dim)
        
        # Extract state components
        # state is numpy array (Kelvin)
        T_s_in = state[0]
        T_s_vec = state[1:5]
        T_sep = state[5]
        T_c_out = state[6]
        n_H2_an_vec = state[7:11]
        n_liq = state[11]
        n_gas = state[12]
        
        # Feature 0: T_s_in
        cond[0] = T_s_in
        
        # Feature 1-4: T_s_vec
        cond[1:5] = T_s_vec
        
        # Feature 5: T_sep
        cond[5] = T_sep
        
        # Feature 6: T_c_out
        cond[6] = T_c_out
        
        # Feature 7-10: n_H2_an_vec
        cond[7:11] = n_H2_an_vec
        
        # Feature 11: n_liq
        cond[11] = n_liq
        
        # Feature 12: n_gas
        cond[12] = n_gas
        
        # Feature 13: T_ref
        cond[13] = T_ref
        
        # Feature 14..14+N: P_ref(N)
        # Ensure P_ref has at least horizon elements
        P_ref_arr = np.array(P_ref)
        if len(P_ref_arr) >= self.horizon:
            cond[14 : 14 + self.horizon] = P_ref_arr[:self.horizon]
        else:
            # Pad with last element
            cond[14 : 14 + len(P_ref_arr)] = P_ref_arr
            cond[14 + len(P_ref_arr) : 14 + self.horizon] = P_ref_arr[-1]
            
        # Feature 14+N..: Previous Action
        cond[14 + self.horizon : 14 + self.horizon + 9] = self.last_action
        
        # 2. Normalize
        cond_tensor = torch.FloatTensor(cond).to(self.device).unsqueeze(0) # Batch size 1
        cond_norm = (cond_tensor - self.cond_min) / self.cond_diff
        
        return cond_norm

    def get_action(self, state, P_ref, T_ref=358.15):
        """
        state: [T_s_in, T_s1...4, T_sep, T_c_out, n_H2_an1...4, n_liq, n_gas] (13 elements)
        P_ref: List or array of future power references (length N or more)
        T_ref: Scalar target temperature
        """
        
        cond_norm = self._prepare_condition(state, P_ref, T_ref)
        
        # 3. Sample
        # This is the slow part (100 steps). For real-time control, DDIM or fewer steps is better, 
        # but we stick to DDPM 100 steps as per current implementation.
        # Now output is sequence (Batch, Action_Dim, Horizon)
        samples_norm = self.scheduler.sample(self.model, cond_norm, (1, self.action_dim, self.horizon))
        
        # 4. Denormalize
        samples_norm = torch.clamp(samples_norm, -1.0, 1.0)
        
        # Expand diff/min for broadcasting over horizon (1, 9, 1)
        action_diff_b = self.action_diff.view(1, -1, 1)
        action_min_b = self.action_min.view(1, -1, 1)
        
        action = ((samples_norm + 1) / 2) * action_diff_b + action_min_b
        
        # Extract first step (index 0 along horizon dim)
        # Shape (1, 9, Horizon) -> (1, 9)
        action_0 = action[:, :, 0]
        
        action_np = action_0.cpu().numpy().flatten()
        
        # Update last action
        self.last_action = action_np
        
        # Unpack
        I_cmd = action_np[0:4]
        v_lye_cmd = action_np[4:8]
        v_c_cmd = action_np[8]
        
        # Apply Constraints
        I_cmd = np.clip(I_cmd, self.I_min, self.I_max)
        v_lye_cmd = np.clip(v_lye_cmd, self.v_lye_min, self.v_lye_max)
        v_c_cmd = np.clip(v_c_cmd, self.v_c_min, self.v_c_max)
        
        return I_cmd, v_lye_cmd, v_c_cmd

    def get_all_actions(self, state, P_ref, T_ref=358.15):
        """
        Returns all generated actions in the horizon.
        Output shape: (N, n_controls)
        n_controls = 9 [I_1..4, v_lye_1..4, v_c]
        """
        cond_norm = self._prepare_condition(state, P_ref, T_ref)
        
        # Sample sequence
        samples_norm = self.scheduler.sample(self.model, cond_norm, (1, self.action_dim, self.horizon))
        
        # Denormalize
        samples_norm = torch.clamp(samples_norm, -1.0, 1.0)
        action_diff_b = self.action_diff.view(1, -1, 1)
        action_min_b = self.action_min.view(1, -1, 1)
        action = ((samples_norm + 1) / 2) * action_diff_b + action_min_b
        
        # Shape: (1, 9, Horizon) -> (Horizon, 9)
        action_seq = action.squeeze(0).permute(1, 0).cpu().numpy()
        
        # Update last action (with first step)
        self.last_action = action_seq[0]
        
        # Apply Constraints to all steps
        action_seq[:, 0:4] = np.clip(action_seq[:, 0:4], self.I_min, self.I_max)
        action_seq[:, 4:8] = np.clip(action_seq[:, 4:8], self.v_lye_min, self.v_lye_max)
        action_seq[:, 8] = np.clip(action_seq[:, 8], self.v_c_min, self.v_c_max)
        
        return action_seq
