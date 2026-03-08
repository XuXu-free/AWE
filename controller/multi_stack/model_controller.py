
import os
import sys
import torch
import numpy as np
from ..base_controller import BaseController
from diffusion.models import DiffusionMLP, DiffusionTCN, FlowMatchingTCN, FlowMatchingMLP
from diffusion.ddpm import DDPMScheduler
from diffusion.flow_matching import FlowMatchingScheduler
try:
    from plant.multi_stack_simulator import MultiStackSimulator
except ImportError:
    # Fallback if running from a different context, try to adjust path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from plant.multi_stack_simulator import MultiStackSimulator

class MultiStackModelController(BaseController):
    def __init__(self, dt=1.0, horizon=10, model_type='tcn', model_path=None, stats_path=r'd:\Projects\AWE\output\multi_stack\diffusion_stats.npz'):
        super().__init__(dt)
        self.horizon = horizon
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Weights for Cost Function (Matched to NMPC)
        self.lambda_prod = 1.0
        self.lambda_track = 1.2 
        self.lambda_temp = 0.15
        self.lambda_I = 0.0002
        self.lambda_lye = 25000.0
        self.lambda_c = 0.5
        
        # Simulator for Rollout
        self.sim_rollout = MultiStackSimulator(dt=self.dt)
        
        # Default model path based on type if not provided
        if model_path is None:
            model_path = r'd:\Projects\AWE\output\multi_stack\best_diffusion_policy_model_{}.pth'.format(model_type)
      
        print(f"Loading model from: {model_path}")
        
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
        # Action: I(4), v_lye(4), v_c(1) => 9. Or + States => 16.
        # Cond: 33 (13 + 1 + N + 9)
        self.action_dim = len(self.action_min)
        print(f"Detected Action Dim: {self.action_dim}")
        
        self.obs_dim = 1 + 4 + 1 + 1 + 4 + 1 + 1 + 1 + horizon + 9 # Fixed 9 for prev action
        
        if model_type == 'diffusion_mlp':
            self.model = DiffusionMLP(action_dim=self.action_dim, obs_dim=self.obs_dim, horizon=horizon).to(self.device)
            self.scheduler = DDPMScheduler(device=self.device)
        elif model_type == 'flow_mlp':
            self.model = FlowMatchingMLP(action_dim=self.action_dim, obs_dim=self.obs_dim, horizon=horizon).to(self.device)
            self.scheduler = FlowMatchingScheduler(device=self.device)
        elif model_type == 'flow_tcn':
            self.model = FlowMatchingTCN(action_dim=self.action_dim, obs_dim=self.obs_dim, horizon=horizon).to(self.device)
            self.scheduler = FlowMatchingScheduler(device=self.device)
        elif model_type == 'diffusion_tcn':
            self.model = DiffusionTCN(output_dim=self.action_dim, cond_dim=self.obs_dim, output_num=horizon, levels=4).to(self.device)
            self.scheduler = DDPMScheduler(device=self.device)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        if not os.path.exists(model_path):
             if os.path.exists(os.path.join('..', model_path)):
                model_path = os.path.join('..', model_path)
             else:
                raise FileNotFoundError(f"Model file not found: {model_path}")
                
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        print("-" * 50)
        print(f"Model Loaded Successfully")
        print(f"Path: {model_path}")
        print(f"Type: {model_type}")
        print(f"Device: {self.device}")
        print(f"Action Dim: {self.action_dim}")
        print(f"Observation Dim: {self.obs_dim}")
        print(f"Horizon: {self.horizon}")
        print(f"Hidden Dim: 256")
        if 'tcn' in model_type:
             print("Levels: 4")
        else:
             print("Res Blocks: 3")
             
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params}")
        print(f"Trainable Parameters: {trainable_params}")
        print("-" * 50)
        
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
        
        # 3. Sample 16 candidates
        num_candidates = 128
        # Expand condition for batch processing
        cond_norm_batch = cond_norm.repeat(num_candidates, 1)
        
        # Sample sequence: (Batch, Action_Dim, Horizon)
        # Using a small noise_scale to improve diversity/robustness
        noise_scale = 0.05 if isinstance(self.scheduler, FlowMatchingScheduler) else 0.0
        if isinstance(self.scheduler, FlowMatchingScheduler):
             samples_norm = self.scheduler.sample(self.model, cond_norm_batch, (num_candidates, self.action_dim, self.horizon), noise_scale=noise_scale)
        else:
             # DDPMScheduler
             samples_norm = self.scheduler.sample(self.model, cond_norm_batch, (num_candidates, self.action_dim, self.horizon))
        
        # 4. Denormalize
        samples_norm = torch.clamp(samples_norm, -1.0, 1.0)
        
        # Expand diff/min for broadcasting: (1, 9, 1) -> (Batch, 9, Horizon)
        action_diff_b = self.action_diff.view(1, -1, 1)
        action_min_b = self.action_min.view(1, -1, 1)
        
        # Denormalized actions: (Batch, 9, Horizon)
        actions = ((samples_norm + 1) / 2) * action_diff_b + action_min_b
        
        # Convert to numpy for rollout evaluation
        actions_np = actions.cpu().numpy() # Shape: (64, 9, Horizon)
        
        # 5. Rollout and Evaluate Cost
        best_cost = float('inf')
        best_idx = 0
        
        # Ensure P_ref covers the horizon
        P_ref_arr = np.array(P_ref)
        if len(P_ref_arr) < self.horizon:
            P_ref_eval = np.pad(P_ref_arr, (0, self.horizon - len(P_ref_arr)), 'edge')
        else:
            P_ref_eval = P_ref_arr[:self.horizon]
            
        for i in range(num_candidates):
            cost = 0.0
            
            # Reset simulator to current state
            self.sim_rollout.reset(initial_state=state)
            
            # Previous actions for smoothness cost
            # Initial previous action is self.last_action
            u_prev = self.last_action.copy()
            I_prev = u_prev[0:4]
            v_lye_prev = u_prev[4:8]
            v_c_prev = u_prev[8]
            
            # Iterate over horizon
            for k in range(self.horizon):
                # Get action for step k: Shape (9,) or (16,)
                u_k_full = actions_np[i, :, k]
                u_k = u_k_full[:9] # Take only controls
                
                # Apply Constraints (Clip)
                u_k[0:4] = np.clip(u_k[0:4], self.I_min, self.I_max)
                u_k[4:8] = np.clip(u_k[4:8], self.v_lye_min, self.v_lye_max)
                u_k[8] = np.clip(u_k[8], self.v_c_min, self.v_c_max)
                
                I_k = u_k[0:4]
                v_lye_k = u_k[4:8]
                v_c_k = u_k[8]
                
                # Step Simulator
                # Note: Simulator step returns NEW state
                # We need to calculate Power BEFORE or DURING step to match NMPC cost?
                # NMPC calculates Power based on u_k and current state x_k
                
                # Calculate Power and Properties for Cost
                # Use current state of sim (before step) or let sim step?
                # NMPC: Power_k = f(x_k, u_k). 
                # So we use sim.state (which is x_k) and u_k.
                
                # Calculate Real Power
                # Access protected method or reimplement power calc?
                # Using protected method for consistency
                curr_state = self.sim_rollout.state
                T_s_vec = curr_state[1:5]
                _, U_cell_vec, _ = self.sim_rollout._calculate_electrochemical_properties(I_k, T_s_vec)
                P_real = np.sum(U_cell_vec * I_k * self.sim_rollout.N_cell)
                
                # Step
                next_state = self.sim_rollout.step(u_k)
                T_s_vec_next = next_state[1:5]
                
                # --- Cost Calculation ---
                # 1. Power Tracking
                cost += self.lambda_track * ((P_real - P_ref_eval[k])/1e6)**2
                
                # 2. Temperature Regulation (using next state or current? NMPC uses x_k (current) or x_k+1?)
                # NMPC obj += (T_s_k - T_ref)**2. T_s_k is the decision variable for state at step k.
                # Usually MPC penalizes deviation over the trajectory.
                # Let's use next_state (result of action).
                cost += self.lambda_temp * np.sum((T_s_vec_next - T_ref)**2)
                
                # 3. Smoothness
                dI = I_k - I_prev
                dv_lye = v_lye_k - v_lye_prev
                dv_c = v_c_k - v_c_prev
                
                cost += self.lambda_I * np.sum(dI**2)
                cost += self.lambda_lye * np.sum(dv_lye**2)
                cost += self.lambda_c * (dv_c**2)
                
                # Update prev
                I_prev = I_k
                v_lye_prev = v_lye_k
                v_c_prev = v_c_k
                
            if cost < best_cost:
                best_cost = cost
                best_idx = i
                
        # Select Best Action Sequence
        best_action_seq = actions_np[best_idx] # (9, Horizon) or (16, Horizon)
        
        # Extract first step
        action_0_full = best_action_seq[:, 0]
        action_0 = action_0_full[:9] # Take only controls
        
        # Apply Constraints (again to be sure)
        action_0[0:4] = np.clip(action_0[0:4], self.I_min, self.I_max)
        action_0[4:8] = np.clip(action_0[4:8], self.v_lye_min, self.v_lye_max)
        action_0[8] = np.clip(action_0[8], self.v_c_min, self.v_c_max)
        
        # Update last action
        self.last_action = action_0
        
        # Unpack
        I_cmd = action_0[0:4]
        v_lye_cmd = action_0[4:8]
        v_c_cmd = action_0[8]
        
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
