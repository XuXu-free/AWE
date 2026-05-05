
import os
import sys
import torch
import numpy as np
from ..base_controller import BaseController
from diffusion.models import DiffusionMLP, DiffusionPureMLP, DiffusionTCN, FlowMatchingTCN, FlowMatchingMLP, PureMLP, PureTCN, LSTMPolicy
from diffusion.ddpm import DDPMScheduler
from diffusion.guided_ddpm import GuidedDDPMScheduler
from diffusion.deterministic_ddpm import DeterministicDDPMScheduler, DeterministicGuidedDDPMScheduler
from diffusion.early_stop_ddpm import EarlyStopNoiseDDPMScheduler, EarlyStopNoiseGuidedDDPMScheduler
from diffusion.flow_matching import FlowMatchingScheduler
from diffusion.hardflow_scheduler import HardFlowScheduler
try:
    from plant.multi_stack_simulator import MultiStackSimulator
except ImportError:
    # Fallback if running from a different context, try to adjust path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from plant.multi_stack_simulator import MultiStackSimulator

class MultiStackModelController(BaseController):
    def __init__(self, dt=60.0, horizon=5, model_type='tcn', model_path=None, stats_path=None):
        super().__init__(dt)
        self.horizon = horizon
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Weights for Cost Function (Matched to NMPC)
        self.lambda_prod = 1e-6
        self.lambda_track = 1.2
        self.lambda_temp = 0.15
        self.lambda_I = 0.0002
        self.lambda_lye = 25000.0
        self.lambda_c = 2500.0

        # Number of candidates for sampling-based controllers
        self.num_candidates = 128

        # Simulator for Rollout
        self.sim_rollout = MultiStackSimulator(dt=self.dt)

        # Batch GPU simulator for fast parallel rollout (optional)
        try:
            from plant.multi_stack_simulator_batch_jit import BatchMultiStackSimulatorJIT
            self.batch_sim = BatchMultiStackSimulatorJIT(dt=self.dt, dt_sub=2.0, device=self.device)
            self.use_batch_rollout = True
        except Exception as e:
            print(f"Batch simulator initialization failed: {e}")
            self.batch_sim = None
            self.use_batch_rollout = False

        # Warm-start for flow models: previous best action sequence (normalized)
        self.prev_best_action_norm = None

        # Normalization Parameters
        if stats_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            stats_path = os.path.join(base_dir, 'output', 'multi_stack', 'diffusion_stats.npz')
        self._load_norm_params(stats_path)
        
        
        # Load Model
        self._load_model(model_type, model_path, horizon)
        
        # Constraints (from NMPC)
        self.I_min = 0.0
        self.I_max = 7500.0
        self.v_lye_min = 0.01
        self.v_lye_max = 0.1
        self.v_c_min = 0.0
        self.v_c_max = 1.0

    def _as_last_action_vec(self, last_action):
        if last_action is None:
            raise ValueError("last_action must be provided as [I_prev, v_lye_prev, v_c_prev] or a 9D vector.")
        if isinstance(last_action, (list, tuple)) and len(last_action) == 3:
            I_prev = np.asarray(last_action[0], dtype=float).reshape(-1)
            v_lye_prev = np.asarray(last_action[1], dtype=float).reshape(-1)
            v_c_prev = float(last_action[2])
            last_action_vec = np.concatenate([I_prev, v_lye_prev, [v_c_prev]])
        else:
            last_action_vec = np.asarray(last_action, dtype=float).reshape(-1)
        if last_action_vec.shape[0] != 9:
            raise ValueError(f"last_action must have 9 elements, got {last_action_vec.shape[0]}")
        return last_action_vec

    def _load_model(self, model_type, model_path, horizon):
        # Default model path based on type if not provided
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            model_path = os.path.join(base_dir, 'output', 'multi_stack', f'best_diffusion_policy_model_{model_type}.pth')
      
        print(f"Loading model from: {model_path}")
        if model_type == 'diffusion_mlp':
            self.model = DiffusionMLP(
                action_dim=self.action_dim,
                obs_dim=self.obs_dim,
                horizon=horizon).to(self.device)
            self.scheduler = DDPMScheduler(device=self.device)
        elif model_type == 'diffusion_pure_mlp':
            self.model = DiffusionPureMLP(
                action_dim=self.action_dim,
                obs_dim=self.obs_dim,
                horizon=horizon,
                hidden_dim=512,
                num_layers=4).to(self.device)
            self.scheduler = DDPMScheduler(device=self.device)
        elif model_type == 'deterministic_diffusion_pure_mlp':
            self.model = DiffusionPureMLP(
                action_dim=self.action_dim,
                obs_dim=self.obs_dim,
                horizon=horizon,
                hidden_dim=256,
                num_layers=4).to(self.device)
            self.scheduler = DeterministicDDPMScheduler(device=self.device)
        elif model_type == 'flow_mlp':
            self.model = FlowMatchingMLP(
                action_dim=self.action_dim, 
                obs_dim=self.obs_dim, 
                horizon=horizon).to(self.device)
            self.scheduler = FlowMatchingScheduler(device=self.device)
        elif model_type == 'flow_mlp_hardflow':
            self.model = FlowMatchingMLP(
                action_dim=self.action_dim,
                obs_dim=self.obs_dim,
                horizon=horizon).to(self.device)
            self.scheduler = HardFlowScheduler(device=self.device)
        elif model_type == 'flow_tcn':
            self.model = FlowMatchingTCN(
                action_dim=self.action_dim, 
                obs_dim=self.obs_dim, 
                horizon=horizon).to(self.device)
            self.scheduler = FlowMatchingScheduler(device=self.device)
        elif model_type == 'flow_tcn_hardflow':
            self.model = FlowMatchingTCN(
                action_dim=self.action_dim,
                obs_dim=self.obs_dim,
                horizon=horizon).to(self.device)
            self.scheduler = HardFlowScheduler(device=self.device)
        elif model_type == 'diffusion_tcn':
            self.model = DiffusionTCN(
                output_dim=self.action_dim,
                cond_dim=self.obs_dim,
                output_num=horizon,
                levels=4).to(self.device)
            self.scheduler = DDPMScheduler(device=self.device)
        elif model_type == 'diffusion_tcn_l3':
            self.model = DiffusionTCN(
                output_dim=self.action_dim,
                cond_dim=self.obs_dim,
                output_num=horizon,
                levels=3).to(self.device)
            self.scheduler = DDPMScheduler(device=self.device)
        elif model_type == 'guided_diffusion_mlp':
            self.model = DiffusionMLP(
                action_dim=self.action_dim,
                obs_dim=self.obs_dim,
                horizon=horizon).to(self.device)
            self.scheduler = GuidedDDPMScheduler(device=self.device, guidance_weight=0.01)
        elif model_type == 'guided_diffusion_tcn':
            self.model = DiffusionTCN(
                output_dim=self.action_dim,
                cond_dim=self.obs_dim,
                output_num=horizon,
                levels=4).to(self.device)
            self.scheduler = GuidedDDPMScheduler(device=self.device, guidance_weight=0.01)
        elif model_type == 'deterministic_diffusion_mlp':
            self.model = DiffusionMLP(
                action_dim=self.action_dim,
                obs_dim=self.obs_dim,
                horizon=horizon).to(self.device)
            self.scheduler = DeterministicDDPMScheduler(device=self.device)
        elif model_type == 'deterministic_diffusion_tcn':
            self.model = DiffusionTCN(
                output_dim=self.action_dim,
                cond_dim=self.obs_dim,
                output_num=horizon,
                levels=4).to(self.device)
            self.scheduler = DeterministicDDPMScheduler(device=self.device)
        elif model_type == 'deterministic_guided_diffusion_mlp':
            self.model = DiffusionMLP(
                action_dim=self.action_dim,
                obs_dim=self.obs_dim,
                horizon=horizon).to(self.device)
            self.scheduler = DeterministicGuidedDDPMScheduler(device=self.device, guidance_weight=0.01)
        elif model_type == 'deterministic_guided_diffusion_tcn':
            self.model = DiffusionTCN(
                output_dim=self.action_dim,
                cond_dim=self.obs_dim,
                output_num=horizon,
                levels=4).to(self.device)
            self.scheduler = DeterministicGuidedDDPMScheduler(device=self.device, guidance_weight=0.01)
        elif model_type == 'early_stop_diffusion_mlp':
            self.model = DiffusionMLP(
                action_dim=self.action_dim,
                obs_dim=self.obs_dim,
                horizon=horizon).to(self.device)
            self.scheduler = EarlyStopNoiseDDPMScheduler(device=self.device, noise_stop_timestep=5)
        elif model_type == 'early_stop_diffusion_tcn':
            self.model = DiffusionTCN(
                output_dim=self.action_dim,
                cond_dim=self.obs_dim,
                output_num=horizon,
                levels=4).to(self.device)
            self.scheduler = EarlyStopNoiseDDPMScheduler(device=self.device, noise_stop_timestep=5)
        elif model_type == 'early_stop_guided_diffusion_mlp':
            self.model = DiffusionMLP(
                action_dim=self.action_dim,
                obs_dim=self.obs_dim,
                horizon=horizon).to(self.device)
            self.scheduler = EarlyStopNoiseGuidedDDPMScheduler(device=self.device, guidance_weight=0.01, noise_stop_timestep=5)
        elif model_type == 'early_stop_guided_diffusion_tcn':
            self.model = DiffusionTCN(
                output_dim=self.action_dim,
                cond_dim=self.obs_dim,
                output_num=horizon,
                levels=4).to(self.device)
            self.scheduler = EarlyStopNoiseGuidedDDPMScheduler(device=self.device, guidance_weight=0.01, noise_stop_timestep=5)
        elif model_type == 'pure_mlp':
            self.model = PureMLP(
                action_dim=self.action_dim,
                obs_dim=self.obs_dim,
                horizon=horizon,
                hidden_dim=256,
                num_res_blocks=3).to(self.device)
            self.scheduler = None
        elif model_type == 'pure_tcn':
            self.model = PureTCN(
                action_dim=self.action_dim,
                obs_dim=self.obs_dim,
                horizon=horizon,
                hidden_dim=256,
                levels=4).to(self.device)
            self.scheduler = None
        elif model_type == 'lstm':
            self.model = LSTMPolicy(
                action_dim=self.action_dim,
                obs_dim=self.obs_dim,
                horizon=horizon,
                hidden_dim=256,
                num_layers=2).to(self.device)
            self.scheduler = None
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        if not os.path.exists(model_path):
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

    def _load_norm_params(self, stats_path):
        # Load Stats
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Stats file not found: {stats_path}")
        stats = np.load(stats_path)
        self.cond_min = torch.FloatTensor(stats['cond_min']).to(self.device)
        self.cond_max = torch.FloatTensor(stats['cond_max']).to(self.device)
        self.action_min = torch.FloatTensor(stats['action_min']).to(self.device)
        self.action_max = torch.FloatTensor(stats['action_max']).to(self.device)
        
        self.cond_diff = self.cond_max - self.cond_min
        self.cond_diff[self.cond_diff < 1e-6] = 1.0
        
        self.action_diff = self.action_max - self.action_min
        self.action_diff[self.action_diff < 1e-6] = 1.0
        
        # Set Action and Observation Dimensions
        # Action: I(4), v_lye(4), v_c(1) => 9
        # Cond: 33 (13 + 1 + N + 9)
        self.action_dim = len(self.action_min)
        self.obs_dim = 1 + 4 + 1 + 1 + 4 + 1 + 1 + 1 + self.horizon + 9 # Fixed 9 for prev action
        print(f"Detected Action Dim: {self.action_dim}")
        print(f"Detected Obs Dim: {self.obs_dim}")

    def _prepare_condition(self, state, P_ref, T_ref, last_action_vec):
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
        cond[14 + self.horizon : 14 + self.horizon + 9] = last_action_vec
        
        # 2. Normalize
        cond_tensor = torch.FloatTensor(cond).to(self.device).unsqueeze(0) # Batch size 1
        cond_norm = (cond_tensor - self.cond_min) / self.cond_diff
        
        return cond_norm

    def _denormalize_action(self, samples_norm):
        # Shape: (Batch, Action_Dim, Horizon)
        # samples_norm = torch.clamp(samples_norm, -1.0, 1.0)
        
        # Expand diff/min for broadcasting: (1, 9, 1) -> (Batch, 9, Horizon)
        action_diff_b = self.action_diff.view(1, -1, 1)
        action_min_b = self.action_min.view(1, -1, 1)
        
        # Denormalized actions: (Batch, 9, Horizon)
        actions = ((samples_norm + 1) / 2) * action_diff_b + action_min_b
        return actions

    def _clamp_action(self, actions):
        """
        Final safety clamp to ensure actions are within physical limits.
        """
        # Apply Constraints
        actions[:, 0:4, :] = torch.clamp(actions[:, 0:4, :], self.I_min, self.I_max)
        actions[:, 4:8, :] = torch.clamp(actions[:, 4:8, :], self.v_lye_min, self.v_lye_max)
        actions[:, 8, :] = torch.clamp(actions[:, 8, :], self.v_c_min, self.v_c_max)
        
        return actions

    def _projection(self, actions):
        """
        Project the action onto the feasible set C.
        P_C(x) = argmin ||y - x||^2 s.t. y in C
        For box constraints, this is equivalent to clamping.
        """
        return self._clamp_action(actions)

    def _evaluate_costs_batch(self, actions_t, state, P_ref_eval, T_ref, last_action_vec):
        """
        GPU-batched rollout evaluation for all candidates.
        actions_t: (num_candidates, 9, horizon) torch tensor on self.device
        state: (13,) numpy array
        P_ref_eval: (horizon,) numpy array
        T_ref: float
        last_action_vec: (9,) numpy array
        returns costs: (num_candidates,) numpy array
        """
        num_candidates = actions_t.shape[0]
        x = self.batch_sim.reset(initial_state=state, batch_size=num_candidates)

        I_prev = torch.from_numpy(last_action_vec[0:4]).float().to(self.device).unsqueeze(0).expand(num_candidates, -1)
        v_lye_prev = torch.from_numpy(last_action_vec[4:8]).float().to(self.device).unsqueeze(0).expand(num_candidates, -1)
        v_c_prev = torch.tensor(last_action_vec[8], dtype=torch.float32, device=self.device).unsqueeze(0).expand(num_candidates)
        P_ref_t = torch.from_numpy(P_ref_eval).float().to(self.device)
        T_ref_t = torch.tensor(T_ref, dtype=torch.float32, device=self.device)

        costs = torch.zeros(num_candidates, dtype=torch.float32, device=self.device)

        for k in range(self.horizon):
            u_k = actions_t[:, :, k]
            I_k = u_k[:, 0:4]
            v_lye_k = u_k[:, 4:8]
            v_c_k = u_k[:, 8]

            x = self.batch_sim.step(x, u_k)
            T_s_next = x[:, 1:5]

            P_total, _, _ = self.batch_sim.calculate_power(I_k, T_s_next)
            costs += self.lambda_track * ((P_total - P_ref_t[k]) / 1e6) ** 2
            costs += self.lambda_temp * torch.sum((T_s_next - T_ref_t) ** 2, dim=1)

            # Current equalization: hardcode 6 pairs for 4 stacks
            cost_eq = ((I_k[:, 0] - I_k[:, 1])**2 +
                       (I_k[:, 0] - I_k[:, 2])**2 +
                       (I_k[:, 0] - I_k[:, 3])**2 +
                       (I_k[:, 1] - I_k[:, 2])**2 +
                       (I_k[:, 1] - I_k[:, 3])**2 +
                       (I_k[:, 2] - I_k[:, 3])**2)
            costs += self.lambda_prod * cost_eq

            dI = I_k - I_prev
            dv_lye = v_lye_k - v_lye_prev
            dv_c = v_c_k - v_c_prev
            costs += self.lambda_I * torch.sum(dI**2, dim=1)
            costs += self.lambda_lye * torch.sum(dv_lye**2, dim=1)
            costs += self.lambda_c * (dv_c**2)

            I_prev = I_k

        return costs.cpu().numpy()

    def get_action(self, state, P_ref, T_ref, last_action):
        """
        state: [T_s_in, T_s1...4, T_sep, T_c_out, n_H2_an1...4, n_liq, n_gas] (13 elements)
        P_ref: List or array of future power references (length N or more)
        T_ref: Scalar target temperature
        """
        last_action_vec = self._as_last_action_vec(last_action)
        cond_norm = self._prepare_condition(state, P_ref, T_ref, last_action_vec)

        # 3. Sample candidates or direct forward pass
        if self.scheduler is None:
            with torch.no_grad():
                samples_norm = self.model(cond_norm)
            num_candidates = 1
        else:
            num_candidates = self.num_candidates
            cond_norm_batch = cond_norm.repeat(num_candidates, 1)
            noise_scale = 0.0
            last_action_t = torch.FloatTensor(last_action_vec).to(self.device)
            last_action_norm = 2 * (last_action_t - self.action_min) / self.action_diff - 1
            last_action_norm_b = last_action_norm.view(1, -1).repeat(num_candidates, 1)

            if isinstance(self.scheduler, GuidedDDPMScheduler):
                samples_norm = self.scheduler.sample(
                    self.model, cond_norm_batch, (num_candidates, self.action_dim, self.horizon),
                    prev_action_norm=last_action_norm_b)
            elif isinstance(self.scheduler, DDPMScheduler):
                samples_norm = self.scheduler.sample(self.model, cond_norm_batch, (num_candidates, self.action_dim, self.horizon))
            elif isinstance(self.scheduler, FlowMatchingScheduler):
                samples_norm = self.scheduler.sample(
                    self.model, cond_norm_batch, (num_candidates, self.action_dim, self.horizon),
                    noise_scale=noise_scale,
                    warm_start=self.prev_best_action_norm,
                    alpha=0.7)
            else:
                lb = torch.full((num_candidates, self.action_dim, self.horizon), -1.0, device=self.device)
                ub = torch.full((num_candidates, self.action_dim, self.horizon), 1.0, device=self.device)
                samples_norm = self.scheduler.sample(
                    self.model,
                    cond_norm_batch,
                    (num_candidates, self.action_dim, self.horizon),
                    steps=20,
                    num_iters=8,
                    lambda_oc=1.0,
                    bounds=(lb, ub),
                    smooth_ref=last_action_norm_b,
                    smooth_w=10.0,
                    step_size=0.3,
                    noise_scale=0.0,
                    warm_start=self.prev_best_action_norm,
                    alpha=0.7
                )

        # 4. Denormalize
        actions_denorm = self._denormalize_action(samples_norm)
        actions_proj = self._projection(actions_denorm)
        actions = self._clamp_action(actions_proj)

        # Ensure P_ref covers the horizon
        P_ref_arr = np.array(P_ref)
        if len(P_ref_arr) < self.horizon:
            P_ref_eval = np.pad(P_ref_arr, (0, self.horizon - len(P_ref_arr)), 'edge')
        else:
            P_ref_eval = P_ref_arr[:self.horizon]

        # 5. Rollout and Evaluate Cost
        best_idx = 0
        best_cost = float('inf')

        if self.use_batch_rollout and num_candidates > 1:
            # Fast GPU-batched rollout
            costs = self._evaluate_costs_batch(actions, state, P_ref_eval, T_ref, last_action_vec)
            best_idx = int(np.argmin(costs))
            best_cost = float(costs[best_idx])
        else:
            # Fallback to serial CPU rollout
            actions_np = actions.cpu().numpy()
            for i in range(num_candidates):
                cost = 0.0
                self.sim_rollout.reset(initial_state=state)
                u_prev = last_action_vec.copy()
                I_prev = u_prev[0:4]
                v_lye_prev = u_prev[4:8]
                v_c_prev = u_prev[8]
                for k in range(self.horizon):
                    u_k_full = actions_np[i, :, k]
                    u_k = u_k_full[:9]
                    I_k = u_k[0:4]
                    v_lye_k = u_k[4:8]
                    v_c_k = u_k[8]
                    next_state = self.sim_rollout.step(u_k)
                    T_s_vec_next = next_state[1:5]
                    _, U_cell_vec_next, _ = self.sim_rollout._calculate_electrochemical_properties(I_k, T_s_vec_next)
                    P_real_next = np.sum(U_cell_vec_next * I_k * self.sim_rollout.N_cell)
                    cost += self.lambda_track * ((P_real_next - P_ref_eval[k])/1e6)**2
                    cost += self.lambda_temp * np.sum((T_s_vec_next - T_ref)**2)
                    current_diff_sum = 0.0
                    for si in range(4):
                        for sj in range(si + 1, 4):
                            current_diff_sum += (I_k[si] - I_k[sj])**2
                    cost += self.lambda_prod * current_diff_sum
                    dI = I_k - I_prev
                    dv_lye = v_lye_k - v_lye_prev
                    dv_c = v_c_k - v_c_prev
                    cost += self.lambda_I * np.sum(dI**2)
                    cost += self.lambda_lye * np.sum(dv_lye**2)
                    cost += self.lambda_c * (dv_c**2)
                    I_prev = I_k
                if cost < best_cost:
                    best_cost = cost
                    best_idx = i

        best_action_seq = actions[best_idx].cpu().numpy() if isinstance(actions, torch.Tensor) else actions_np[best_idx]

        # Save normalized best sequence for warm-start in next timestep (flow models)
        if self.scheduler is not None and hasattr(self.scheduler, 'sample') and (
            isinstance(self.scheduler, FlowMatchingScheduler) or type(self.scheduler).__name__ == 'HardFlowScheduler'
        ):
            self.prev_best_action_norm = samples_norm[best_idx:best_idx+1].detach().clone()

        action_0 = best_action_seq[:, 0]
        I_cmd = action_0[0:4]
        v_lye_cmd = action_0[4:8]
        v_c_cmd = action_0[8]
        return I_cmd, v_lye_cmd, v_c_cmd

    def get_all_actions(self, state, P_ref, T_ref, last_action):
        """
        Returns all generated actions in the horizon.
        Output shape: (N, n_controls)
        n_controls = 9 [I_1..4, v_lye_1..4, v_c]
        """
        last_action_vec = self._as_last_action_vec(last_action)
        cond_norm = self._prepare_condition(state, P_ref, T_ref, last_action_vec)
        
        # Sample sequence
        samples_norm = self.scheduler.sample(self.model, cond_norm, (1, self.action_dim, self.horizon))
        
        # Denormalize
        action_denorm = self._denormalize_action(samples_norm)
        action_proj = self._projection(action_denorm)
        action = self._clamp_action(action_proj)
        
        # Shape: (1, 9, Horizon) -> (Horizon, 9)
        action_seq = action.squeeze(0).permute(1, 0).cpu().numpy()
        return action_seq

