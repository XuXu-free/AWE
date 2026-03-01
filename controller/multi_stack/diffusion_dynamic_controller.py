
import os
import sys
import torch
import numpy as np
from .diffusion_controller import MultiStackDiffusionController
from diffusion.model import DiffusionTCN
from diffusion.ddpm import DDPMScheduler

class MultiStackDiffusionDynamicController(MultiStackDiffusionController):
    def __init__(self, dt=60.0, horizon=5, model_type='tcn', 
                 model_path=None, stats_path=None,
                 dyn_model_path=None, dyn_stats_path=None):
        
        # Initialize Base Controller (Policy Model)
        # We pass model_path and stats_path for the POLICY model
        super().__init__(dt, horizon, model_type, model_path, stats_path)
        
        # --- Load Dynamics Model ---
        # Default paths if not provided
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
        
        if dyn_model_path is None:
            dyn_model_path = os.path.join(project_root, 'output', 'multi_stack', 'model', 'best_diffusion_dynamics_traj_model.pth')
            
        if dyn_stats_path is None:
            dyn_stats_path = os.path.join(project_root, 'output', 'multi_stack', 'dynamics_trajectory_stats.npz')
            
        print(f"Loading Dynamics Model from: {dyn_model_path}")
        print(f"Loading Dynamics Stats from: {dyn_stats_path}")
        
        # Load Dynamics Stats
        if not os.path.exists(dyn_stats_path):
             raise FileNotFoundError(f"Dynamics stats not found: {dyn_stats_path}")
             
        dyn_stats = np.load(dyn_stats_path)
        
        # Stats for Dynamics Model
        # Note: Keys from train_diffusion_dynamic.py: state_min, state_max, action_min, action_max
        self.dyn_state_min = torch.FloatTensor(dyn_stats['state_min']).to(self.device)
        self.dyn_state_max = torch.FloatTensor(dyn_stats['state_max']).to(self.device)
        self.dyn_action_min = torch.FloatTensor(dyn_stats['action_min']).to(self.device)
        self.dyn_action_max = torch.FloatTensor(dyn_stats['action_max']).to(self.device)
        
        self.dyn_state_diff = self.dyn_state_max - self.dyn_state_min
        self.dyn_state_diff[self.dyn_state_diff < 1e-6] = 1.0
        
        self.dyn_action_diff = self.dyn_action_max - self.dyn_action_min
        self.dyn_action_diff[self.dyn_action_diff < 1e-6] = 1.0
        
        # Dimensions
        self.state_dim = len(self.dyn_state_min) # 13
        self.dyn_action_dim = len(self.dyn_action_min) # 9
        
        # Dynamics Model Dimensions
        # Target: State Sequence (H, State_Dim) -> Model output dim = State_Dim
        # Condition: State (1) + Action Sequence (H) flattened
        self.dyn_target_dim = self.state_dim
        self.dyn_cond_dim = self.state_dim + self.horizon * self.dyn_action_dim
        
        # Initialize Dynamics Model (DiffusionTCN)
        # Note: hidden_dim=256 used in training script
        self.dyn_model = DiffusionTCN(
            output_dim=self.dyn_target_dim, 
            cond_dim=self.dyn_cond_dim, 
            output_num=self.horizon,
            hidden_dim=256
        ).to(self.device)
        
        if not os.path.exists(dyn_model_path):
            raise FileNotFoundError(f"Dynamics model file not found: {dyn_model_path}")
            
        self.dyn_model.load_state_dict(torch.load(dyn_model_path, map_location=self.device))
        self.dyn_model.eval()
        
        # Scheduler for Dynamics (DDPM)
        self.dyn_scheduler = DDPMScheduler(device=self.device)

    def get_action(self, state, P_ref, T_ref=358.15):
        """
        Override get_action to use learned dynamics model for evaluation.
        """
        # 1. Prepare Condition for Policy Model
        # (Same as base class)
        cond_norm = self._prepare_condition(state, P_ref, T_ref)
        
        # 2. Sample Candidates from Policy Model
        num_candidates = 64
        cond_norm_batch = cond_norm.repeat(num_candidates, 1)
        
        # Sample sequence: (Batch, Action_Dim, Horizon)
        samples_norm = self.scheduler.sample(self.model, cond_norm_batch, (num_candidates, self.action_dim, self.horizon))
        
        # Denormalize Policy Actions
        samples_norm = torch.clamp(samples_norm, -1.0, 1.0)
        action_diff_b = self.action_diff.view(1, -1, 1)
        action_min_b = self.action_min.view(1, -1, 1)
        actions_candidates = ((samples_norm + 1) / 2) * action_diff_b + action_min_b
        
        # actions_candidates: (64, 16, H) or (64, 9, H)
        # We only need the control actions (first 9) for dynamics model
        # Controls: I(4), v_lye(4), v_c(1)
        ctrl_actions = actions_candidates[:, :9, :] # (Batch, 9, H)
        
        # Apply Constraints to candidates
        # (Broadcasting constraints)
        # I (0-4)
        ctrl_actions[:, 0:4, :] = torch.clamp(ctrl_actions[:, 0:4, :], self.I_min, self.I_max)
        # v_lye (4-8)
        ctrl_actions[:, 4:8, :] = torch.clamp(ctrl_actions[:, 4:8, :], self.v_lye_min, self.v_lye_max)
        # v_c (8)
        ctrl_actions[:, 8, :] = torch.clamp(ctrl_actions[:, 8, :], self.v_c_min, self.v_c_max)
        
        # --- 3. Evaluate Candidates using Dynamics Model ---
        
        # A. Prepare Condition for Dynamics Model
        # Condition: [s_t_norm, a_seq_flat]
        
        # Normalize Current State (s_t) for Dynamics Model
        # state is numpy array (13,)
        state_tensor = torch.FloatTensor(state).to(self.device)
        # Normalize to [0, 1] using DYNAMICS stats
        # (s - min) / diff
        state_norm_dyn = (state_tensor - self.dyn_state_min) / self.dyn_state_diff
        
        # Normalize Candidate Actions for Dynamics Model
        # ctrl_actions is (Batch, 9, H)
        # Normalize to [-1, 1] using DYNAMICS stats
        # 2 * (a - min) / diff - 1
        # Need to broadcast dyn_action_min/max: (1, 9, 1)
        dyn_act_min_b = self.dyn_action_min.view(1, -1, 1)
        dyn_act_diff_b = self.dyn_action_diff.view(1, -1, 1)
        
        ctrl_actions_norm_dyn = 2.0 * (ctrl_actions - dyn_act_min_b) / dyn_act_diff_b - 1.0
        
        # Flatten actions: (Batch, 9, H) -> (Batch, 9*H)
        # Note: Flattening order must match training (H * Action_Dim). 
        # In training: a_seq (H, 9) -> flatten -> (row-major?)
        # PyTorch flatten defaults to row-major if contiguous?
        # In training: `a_seq_flat = a_seq_norm.flatten()` where a_seq_norm is (H, 9) (numpy)
        # So it is [a_0, a_1, ... a_H-1].
        # Here ctrl_actions_norm_dyn is (Batch, 9, H). 
        # We need to permute to (Batch, H, 9) before flattening to match [a_0, a_1...]
        ctrl_actions_flat = ctrl_actions_norm_dyn.permute(0, 2, 1).reshape(num_candidates, -1)
        
        # Construct Batch Condition: (Batch, State_Dim + 9*H)
        # Expand state_norm_dyn: (13,) -> (Batch, 13)
        state_norm_batch = state_norm_dyn.unsqueeze(0).expand(num_candidates, -1)
        
        dyn_cond_batch = torch.cat([state_norm_batch, ctrl_actions_flat], dim=1)
        
        # B. Predict Trajectories (Sample from Dynamics Model)
        # Output shape: (Batch, State_Dim, Horizon) -> (Batch, 13, H)
        # Note: Dynamics model "action_dim" is actually State_Dim (13)
        
        pred_states_norm = self.dyn_scheduler.sample(
            self.dyn_model, 
            dyn_cond_batch, 
            (num_candidates, self.state_dim, self.horizon)
        )
        
        # C. Denormalize Predicted States
        # To [-1, 1] -> Physical
        # Training target was: 2 * (s - min) / diff - 1
        # Inverse: (y + 1) / 2 * diff + min
        
        pred_states_norm = torch.clamp(pred_states_norm, -1.0, 1.0)
        
        dyn_state_diff_b = self.dyn_state_diff.view(1, -1, 1)
        dyn_state_min_b = self.dyn_state_min.view(1, -1, 1)
        
        pred_states = ((pred_states_norm + 1.0) / 2.0) * dyn_state_diff_b + dyn_state_min_b
        
        # --- 4. Calculate Cost ---
        # We perform cost calculation in PyTorch for speed (vectorized over batch)
        
        # Prepare References
        # P_ref: (Horizon,)
        P_ref_arr = np.array(P_ref)
        if len(P_ref_arr) < self.horizon:
            P_ref_eval = np.pad(P_ref_arr, (0, self.horizon - len(P_ref_arr)), 'edge')
        else:
            P_ref_eval = P_ref_arr[:self.horizon]
        
        P_ref_tensor = torch.FloatTensor(P_ref_eval).to(self.device).unsqueeze(0) # (1, H)
        
        # T_ref
        T_ref_tensor = torch.tensor(T_ref, device=self.device)
        
        # Previous Action for Smoothness
        # self.last_action is numpy (9,)
        # Need to broadcast to (Batch, 9) for first step diff
        u_prev_tensor = torch.FloatTensor(self.last_action[:9]).to(self.device).unsqueeze(0) # (1, 9)
        
        # Extract Control Components from Candidates
        # ctrl_actions: (Batch, 9, H)
        I_seq = ctrl_actions[:, 0:4, :] # (Batch, 4, H)
        v_lye_seq = ctrl_actions[:, 4:8, :]
        v_c_seq = ctrl_actions[:, 8, :] # (Batch, H)
        
        # Extract State Components from Predictions
        # pred_states: (Batch, 13, H)
        # Indices: T_s_in(0), T_s(1-4), T_sep(5), T_c_out(6), ...
        T_s_vec_seq = pred_states[:, 1:5, :] # (Batch, 4, H)
        
        # Calculate Power (Approximation / Model)
        P_real_seq = self._calculate_power_torch(I_seq, T_s_vec_seq) # (Batch, H)
        
        # Costs
        # 1. Power Tracking
        # ((P_real - P_ref) / 1e6)^2
        cost_track = self.lambda_track * ((P_real_seq - P_ref_tensor) / 1e6)**2
        cost_track = torch.sum(cost_track, dim=1) # Sum over horizon -> (Batch,)
        
        # 2. Temperature Regulation
        # Sum((T_s - T_ref)^2)
        # T_s_vec_seq: (Batch, 4, H)
        # Sum over stacks and horizon
        cost_temp = self.lambda_temp * torch.sum((T_s_vec_seq - T_ref_tensor)**2, dim=(1, 2))
        
        # 3. Smoothness
        # Diff along horizon
        # First step: u_0 - u_prev
        # Subsequent: u_k - u_{k-1}
        
        # Concatenate u_prev to sequence
        # ctrl_actions: (Batch, 9, H)
        # u_prev_tensor: (1, 9) -> expand -> (Batch, 9, 1)
        u_prev_exp = u_prev_tensor.unsqueeze(2).expand(num_candidates, -1, -1)
        
        # Full seq with prev: (Batch, 9, H+1)
        ctrl_seq_full = torch.cat([u_prev_exp, ctrl_actions], dim=2)
        
        # Diff: (Batch, 9, H)
        diff_seq = ctrl_seq_full[:, :, 1:] - ctrl_seq_full[:, :, :-1]
        
        dI = diff_seq[:, 0:4, :]
        dv_lye = diff_seq[:, 4:8, :]
        dv_c = diff_seq[:, 8, :] 
        
        cost_I = self.lambda_I * torch.sum(dI**2, dim=(1, 2))
        cost_lye = self.lambda_lye * torch.sum(dv_lye**2, dim=(1, 2))
        cost_c = self.lambda_c * torch.sum(dv_c**2, dim=1)
        
        total_cost = cost_track + cost_temp + cost_I + cost_lye + cost_c
        
        # Find Best
        best_idx = torch.argmin(total_cost).item()
        
        # Return Best Action (First step)
        best_action_seq = ctrl_actions[best_idx] # (9, H)
        
        # First step
        action_0 = best_action_seq[:, 0].cpu().numpy() # (9,)
        
        # Update last action
        self.last_action = action_0
        
        I_cmd = action_0[0:4]
        v_lye_cmd = action_0[4:8]
        v_c_cmd = action_0[8]
        
        return I_cmd, v_lye_cmd, v_c_cmd

    def _calculate_power_torch(self, I_seq, T_s_seq):
        """
        Calculate Power in PyTorch.
        I_seq: (Batch, 4, H)
        T_s_seq: (Batch, 4, H)
        Returns: P_real_seq (Batch, H)
        """
        # Constants (from MultiStackSimulator)
        N_cell = 368
        U_rev = 1.229
        r1 = 3.202e-5
        r2 = 8.970e-8
        r3 = -4.193e-12
        t1 = -1.070e-1
        t2 = 14.43
        t3 = 38.8
        s = 7.572e-2
        p_sys = 1.6e6
        
        # T in Celsius
        T_C = T_s_seq - 273.15
        
        # Ohmic Overpotential
        # r = r1 + r2*T + r3*p_sys (Note: r3 multiplies p_sys in simulator, NOT T^2!)
        # Wait, checking simulator again:
        # R_ohm = self.r1 + self.r2 * T_s_i + self.r3 * self.p_sys
        # Correct.
        R_ohm = r1 + r2 * T_s_seq + r3 * p_sys
        V_ohm = R_ohm * I_seq
        
        # Activation Overpotential
        # term_act = t1 + t2/T_C + t3/T_C^2
        term_act = t1 + t2 / T_C + t3 / (T_C**2)
        
        # arg = term_act * I + 1
        arg = term_act * I_seq + 1.0
        
        # Safe log (Natural Log!)
        arg = torch.clamp(arg, min=1.0001)
        V_act = s * torch.log(arg)
        
        U_cell = U_rev + V_ohm + V_act
        
        # Power = Sum(U_cell * I * N_cell)
        P_stack = U_cell * I_seq * N_cell
        P_total = torch.sum(P_stack, dim=1) # Sum over stacks -> (Batch, H)
        
        return P_total
