
import os
import sys
import torch
import numpy as np
from ..base_controller import BaseController
from diffusion.models import DiffusionMLP, DiffusionTCN, FlowMatchingTCN, FlowMatchingMLP
from diffusion.ddpm import DDPMScheduler
from diffusion.flow_matching import FlowMatchingScheduler
from diffusion.hardflow_scheduler import HardFlowScheduler

try:
    from plant.single_stack_simulator import SingleStackSimulator
except ImportError:
    # Fallback if running from a different context
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from plant.single_stack_simulator import SingleStackSimulator

class SingleStackModelController(BaseController):
    def __init__(self, dt=60.0, horizon=5, model_type='tcn', model_path=None, stats_path=None):
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
        self.sim_rollout = SingleStackSimulator(sim_dt=dt)
        
        
        self._load_norm_params(stats_path)
        
        # Load Model
        self._load_model(model_type, model_path, horizon)
        
        # Constraints (from NMPC)
        self.I_min = 0.0
        self.I_max = 7800.0 * 1.2
        self.v_lye_min = 0.0
        self.v_lye_max = 0.1
        self.v_c_min = 0.0
        self.v_c_max = 1.0

    def _as_last_action_vec(self, last_action):
        if last_action is None:
            # Default to zeros if not provided
            return np.zeros(3)
            
        if isinstance(last_action, (list, tuple)) and len(last_action) == 3:
            return np.array(last_action, dtype=float)
        
        last_action_vec = np.asarray(last_action, dtype=float).flatten()
        if last_action_vec.shape[0] != 3:
             # Try to handle if it's passed differently, but for single stack it should be 3
             if last_action_vec.shape[0] > 3:
                 return last_action_vec[:3]
             raise ValueError(f"last_action must have 3 elements, got {last_action_vec.shape[0]}")
        return last_action_vec

    def _load_model(self, model_type, model_path, horizon):
        if model_path is None:
            # Try to find a default
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            model_path = os.path.join(base_dir, 'output', 'single_stack', 'policy', f'{model_type}_policy_best.pth')
            
        print(f"Loading model from: {model_path}")
        
        # Determine architecture class
        if model_type == 'diffusion_mlp':
            self.model = DiffusionMLP(
                action_dim=self.action_dim, 
                obs_dim=self.obs_dim, 
                horizon=horizon,
                hidden_dim=256,
                num_res_blocks=3
            ).to(self.device)
            self.scheduler = DDPMScheduler(device=self.device)
        elif model_type == 'flow_mlp':
            self.model = FlowMatchingMLP(
                action_dim=self.action_dim, 
                obs_dim=self.obs_dim, 
                horizon=horizon,
                hidden_dim=256,
                num_res_blocks=3
            ).to(self.device)
            self.scheduler = FlowMatchingScheduler(device=self.device)
        elif model_type == 'flow_tcn':
            self.model = FlowMatchingTCN(
                action_dim=self.action_dim, 
                obs_dim=self.obs_dim, 
                horizon=horizon,
                hidden_dim=256,
                levels=4
            ).to(self.device)
            self.scheduler = FlowMatchingScheduler(device=self.device)
        elif model_type == 'diffusion_tcn':
            self.model = DiffusionTCN(
                output_dim=self.action_dim, 
                cond_dim=self.obs_dim, 
                output_num=horizon, 
                hidden_dim=256,
                levels=4
            ).to(self.device)
            self.scheduler = DDPMScheduler(device=self.device)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
                
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print("Model loaded successfully.")

    def _load_norm_params(self, stats_path):
        # Normalization Parameters
        if stats_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            stats_path = os.path.join(base_dir, 'output', 'single_stack', 'diffusion_stats.npz')
        stats = np.load(stats_path)
        self.cond_min = torch.FloatTensor(stats['cond_min']).to(self.device)
        self.cond_max = torch.FloatTensor(stats['cond_max']).to(self.device)
        self.action_min = torch.FloatTensor(stats['action_min']).to(self.device)
        self.action_max = torch.FloatTensor(stats['action_max']).to(self.device)
        
        self.cond_diff = self.cond_max - self.cond_min
        self.cond_diff[self.cond_diff < 1e-6] = 1.0
        
        self.action_diff = self.action_max - self.action_min
        self.action_diff[self.action_diff < 1e-6] = 1.0
        
        self.action_dim = len(self.action_min)
        self.obs_dim = len(self.cond_min)
        print("cond_min:", self.cond_min)
        print("cond_max:", self.cond_max)
        print("action_min:", self.action_min)
        print("action_max:", self.action_max)
        print(f"Action Dim: {self.action_dim} (Expected 3)")
        print(f"Obs Dim: {self.obs_dim}")

    def _prepare_condition(self, state, P_ref, T_ref, last_action_vec):
        # 1. Construct Condition Vector
        cond = np.zeros(self.obs_dim)
        
        # Extract state components
        # state: [T_s_in, T_s, T_sep, T_c_out, n_H2_an, n_liq, n_gas]
        T_s_in = state[0]
        T_s = state[1]
        T_sep = state[2]
        T_c_out = state[3]
        n_H2_an = state[4]
        n_liq = state[5]
        n_gas = state[6]
        
        # Order must match SingleStackDataset in train_policy_model.py
        # 1. System State (7): T_s_in, T_s, T_sep, T_c_out, n_H2_an, n_liq, n_gas
        cond[0] = T_s_in
        cond[1] = T_s
        cond[2] = T_sep
        cond[3] = T_c_out
        cond[4] = n_H2_an
        cond[5] = n_liq
        cond[6] = n_gas
        
        # 2. Reference (1 + N)
        # T_ref
        cond[7] = T_ref
        
        # P_ref_future
        P_ref_arr = np.array(P_ref)
        base_idx = 8
        if len(P_ref_arr) >= self.horizon:
            cond[base_idx : base_idx + self.horizon] = P_ref_arr[:self.horizon]
        else:
            cond[base_idx : base_idx + len(P_ref_arr)] = P_ref_arr
            cond[base_idx + len(P_ref_arr) : base_idx + self.horizon] = P_ref_arr[-1]
            
        # 3. Prev Actions (3)
        base_idx = 8 + self.horizon
        cond[base_idx : base_idx + 3] = last_action_vec
        
        # Normalize
        cond_tensor = torch.FloatTensor(cond).to(self.device).unsqueeze(0) # Batch size 1
        cond_norm = (cond_tensor - self.cond_min) / self.cond_diff * 2 - 1
        
        # Clamp to [-1, 1] to handle out-of-distribution values (e.g. higher power in Dec vs Jan)
        cond_norm = torch.clamp(cond_norm, -1.0, 1.0)
        
        return cond_norm

    def _denormalize_clamp_action(self, samples_norm):
        # Shape: (Batch, Action_Dim, Horizon)
        action_diff_b = self.action_diff.view(1, -1, 1)
        action_min_b = self.action_min.view(1, -1, 1)
        
        actions = ((samples_norm + 1) / 2) * action_diff_b + action_min_b
        
        # Apply Constraints
        actions[:, 0, :] = torch.clamp(actions[:, 0, :], self.I_min, self.I_max)
        actions[:, 1, :] = torch.clamp(actions[:, 1, :], self.v_lye_min, self.v_lye_max)
        actions[:, 2, :] = torch.clamp(actions[:, 2, :], self.v_c_min, self.v_c_max)
        
        return actions

    def get_action(self, state, P_ref, T_ref, last_action):
        """
        state: [T_s_in, T_s, T_sep, T_c_out, n_liq, n_gas] (6 elements)
        P_ref: List or array of future power references (length N or more)
        T_ref: Scalar target temperature
        """
        last_action_vec = self._as_last_action_vec(last_action)
        cond_norm = self._prepare_condition(state, P_ref, T_ref, last_action_vec)
        
        # 3. Sample 128 candidates
        num_candidates = 1
        cond_norm_batch = cond_norm.repeat(num_candidates, 1)
        
        # Sample sequence: (Batch, Action_Dim, Horizon)
        noise_scale = 0.0
        if self.scheduler is None:
             # Pure MLP direct prediction
             samples_norm = self.model(cond_norm_batch)
        elif isinstance(self.scheduler, DDPMScheduler):
            samples_norm = self.scheduler.sample(self.model, cond_norm_batch, (num_candidates, self.action_dim, self.horizon))
        elif isinstance(self.scheduler, FlowMatchingScheduler):
            samples_norm = self.scheduler.sample(self.model, cond_norm_batch, (num_candidates, self.action_dim, self.horizon), noise_scale=noise_scale)
        else:
            raise ValueError("Unsupported scheduler type")
        
        # 4. Denormalize
        actions = self._denormalize_clamp_action(samples_norm)
        
        # Convert to numpy for rollout evaluation
        actions_np = actions.cpu().numpy() # Shape: (128, 3, Horizon)
        
        # 5. Rollout and Evaluate Cost
        best_cost = float('inf')
        best_idx = 0
        
        # Ensure P_ref covers the horizon
        P_ref_arr = np.array(P_ref)
        if len(P_ref_arr) < self.horizon:
            P_ref_eval = np.pad(P_ref_arr, (0, self.horizon - len(P_ref_arr)), 'edge')
        else:
            P_ref_eval = P_ref_arr[:self.horizon]
            
        steps_per_ctrl = int(self.dt / self.sim_rollout.dt)
        if steps_per_ctrl < 1:
            # If rollout sim dt > ctrl dt (unlikely), force it to be smaller or 1
            steps_per_ctrl = 1
            self.sim_rollout.dt = self.dt
        
        # If sim_rollout.dt is large (60s), force smaller steps for accuracy
        if self.sim_rollout.dt > 1.0:
            self.sim_rollout.dt = 0.2
            steps_per_ctrl = int(self.dt / self.sim_rollout.dt)

        for i in range(num_candidates):
            cost = 0.0
            
            # Reset simulator to current state
            self.sim_rollout.reset(initial_state=state)
            
            # Previous actions for smoothness cost
            u_prev = last_action_vec.copy()
            I_prev = u_prev[0]
            v_lye_prev = u_prev[1]
            v_c_prev = u_prev[2]
            I_0 = u_prev[0]
            v_lye_0 = u_prev[1]
            v_c_0 = u_prev[2]
            
            # Iterate over horizon
            for k in range(self.horizon):
                # Get action for step k: Shape (3,)
                u_k = actions_np[i, :, k]
                
                I_k = u_k[0]
                v_lye_k = u_k[1]
                v_c_k = u_k[2]
                
                # Calculate Real Power using CURRENT state (Start of Interval)
                # To match NMPC behavior which optimizes Power at step k
                T_s_curr = self.sim_rollout.state[1]
                _, U_cell, _ = self.sim_rollout._calculate_electrochemical_properties(I_k, T_s_curr)
                P_real_curr = U_cell * I_k * self.sim_rollout.N_cell

                # Step (multiple small steps)
                for _ in range(steps_per_ctrl):
                    next_state = self.sim_rollout.step(u_k)
                
                # State after control interval
                T_s_next = next_state[1] # T_s is at index 1
                
                # --- Cost Calculation ---
                # 1. Power Tracking
                cost += self.lambda_track * ((P_real_curr - P_ref_eval[k])/1e6)**2
                
                # 2. Temperature Regulation
                cost += self.lambda_temp * ((T_s_next - T_ref)**2)
                
                # 3. Smoothness
                dI = I_k - I_prev
                dv_lye = v_lye_k - v_lye_0
                dv_c = v_c_k - v_c_0
                
                cost += self.lambda_I * (dI**2)
                cost += self.lambda_lye * (dv_lye**2)
                cost += self.lambda_c * (dv_c**2)
                
                # Update prev
                I_prev = I_k
                v_lye_prev = v_lye_k
                v_c_prev = v_c_k
                
            if cost < best_cost:
                best_cost = cost
                best_idx = i
                
        # Select Best Action Sequence
        best_action_seq = actions_np[best_idx] # (3, Horizon)
        
        # Extract first step
        action_0 = best_action_seq[:, 0]
        
        return action_0[0], action_0[1], action_0[2]

    def get_all_actions_states(self, state, P_ref, T_ref, last_action):
        # Return full trajectory
        last_action_vec = self._as_last_action_vec(last_action)
        cond_norm = self._prepare_condition(state, P_ref, T_ref, last_action_vec)
        
        if self.scheduler is None:
             # Pure MLP direct prediction
             samples_norm = self.model(cond_norm)
        elif isinstance(self.scheduler, DDPMScheduler):
            samples_norm = self.scheduler.sample(self.model, cond_norm, (1, self.action_dim, self.horizon))
        elif isinstance(self.scheduler, FlowMatchingScheduler):
            samples_norm = self.scheduler.sample(self.model, cond_norm, (1, self.action_dim, self.horizon), noise_scale=0.0)
        else:
            raise ValueError("Unsupported scheduler type")
            
        actions = self._denormalize_clamp_action(samples_norm)
        actions_np = actions.cpu().numpy()[0] # (Action_Dim, Horizon)
        
        # Transpose to (Horizon, Action_Dim)
        u_opt_matrix = actions_np.T
        
        # Predict states (Open loop rollout)
        states_matrix = np.zeros((self.horizon, 7)) # Single stack state dim is 7
        
        # Reset internal simulator state
        self.sim_rollout.state = state.copy()
        
        # Steps per control interval
        steps_per_ctrl = int(self.dt / self.sim_rollout.dt)
        
        for k in range(self.horizon):
            u_k = u_opt_matrix[k]
            # Hold action constant for steps_per_ctrl
            for _ in range(steps_per_ctrl):
                self.sim_rollout.step(u_k)
            
            states_matrix[k] = self.sim_rollout.state.copy()
            
        return u_opt_matrix, states_matrix
