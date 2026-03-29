"""
Single-Stack CBF Model Controller

Combines diffusion/flow-matching policy with CBF-based projection operator
to ensure safety constraints are satisfied via Control Barrier Functions.
"""

import os
import sys
import torch
import numpy as np
from ..base_controller import BaseController
from diffusion.models import DiffusionMLP, DiffusionTCN, FlowMatchingTCN, FlowMatchingMLP
from diffusion.ddpm import DDPMScheduler
from diffusion.flow_matching import FlowMatchingScheduler
from diffusion.hardflow_scheduler import HardFlowScheduler

# Import CBF projection
try:
    from .cbf_projection import SingleStackCBFProjection, SingleStackCBFProjectionSimplified
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from single_stack.cbf_projection import SingleStackCBFProjection, SingleStackCBFProjectionSimplified

try:
    from plant.single_stack_simulator import SingleStackSimulator
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from plant.single_stack_simulator import SingleStackSimulator


class SingleStackCBFModelController(BaseController):
    """
    Single-Stack Model Controller with CBF-based Projection.

    Uses a learned policy (diffusion/flow-matching) to generate action candidates,
    then applies CBF projection to ensure safety constraints are satisfied.

    CBF (Control Barrier Function) guarantees forward invariance of the safe set
    through Lyapunov-like conditions on the barrier function.
    """

    def __init__(self, dt=60.0, horizon=5, model_type='tcn', model_path=None, stats_path=None,
                 use_cbf_projection=True, projection_backend='casadi', gamma=1.0,
                 gamma_vec=None, rho_vec=None, h_margin_vec=None, normalize=True,
                 lambda_u_scale=1000.0, soft_mask=None, active_mask=None):
        """
        Args:
            dt: Control time step
            horizon: Prediction horizon
            model_type: Type of model ('diffusion_mlp', 'flow_mlp', 'flow_tcn', 'diffusion_tcn')
            model_path: Path to model checkpoint
            stats_path: Path to normalization statistics
            use_cbf_projection: Whether to apply CBF projection
            projection_backend: 'casadi' or 'scipy' for projection solver
            gamma: CBF gain parameter (larger = more aggressive safety enforcement)
            gamma_vec: Per-constraint gamma values [h_T, h_HTO, h_V, h_P, h_Tmin]
            rho_vec: Per-constraint rho values for soft constraints
            h_margin_vec: Per-constraint safety margins
            normalize: Whether to use normalized CBF constraints
            lambda_u_scale: Control change penalty scale
            soft_mask: Per-constraint soft slack mask [h_T, h_HTO, h_V, h_P, h_Tmin]
            active_mask: Per-constraint active mask [h_T, h_HTO, h_V, h_P, h_Tmin].
                         Default all True. Can be overridden per-call in project().
        """
        super().__init__(dt)
        self.horizon = horizon
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.use_cbf_projection = use_cbf_projection
        self.projection_backend = projection_backend
        self.gamma = gamma

        # Weights for Cost Function (Matched to NMPC)
        self.lambda_prod = 1.0
        self.lambda_track = 1.2
        self.lambda_temp = 0.15
        self.lambda_I = 0.0002
        self.lambda_lye = 25000.0
        self.lambda_c = 0.5

        # Simulator for Rollout
        self.sim_rollout = SingleStackSimulator(sim_dt=dt)

        # Initialize CBF projection if enabled
        if self.use_cbf_projection:
            proj_kwargs = {
                'dt': dt,
                'gamma': gamma,
                'gamma_vec': gamma_vec,
                'rho_vec': rho_vec,
                'h_margin_vec': h_margin_vec,
                'normalize': normalize,
                'lambda_u_scale': lambda_u_scale,
                'soft_mask': soft_mask,
                'active_mask': active_mask
            }
            if projection_backend == 'casadi':
                try:
                    self.projector = SingleStackCBFProjection(**proj_kwargs)
                    print(f"Using CasADi-based CBF projection (gamma={gamma}, rho_vec={rho_vec})")
                except Exception as e:
                    print(f"CasADi CBF projection failed: {e}, falling back to SciPy")
                    self.projector = SingleStackCBFProjectionSimplified(**proj_kwargs)
                    self.projection_backend = 'scipy'
            else:
                self.projector = SingleStackCBFProjectionSimplified(**proj_kwargs)
                print(f"Using SciPy-based CBF projection (gamma={gamma}, rho_vec={rho_vec})")

        self._load_norm_params(stats_path)
        self._load_model(model_type, model_path, horizon)

        # Constraints (from NMPC)
        self.I_min = 0.0
        self.I_max = 7800.0 * 1.2
        self.v_lye_min = 0.01
        self.v_lye_max = 0.1
        self.v_c_min = 0.0
        self.v_c_max = 1.0

    def _as_last_action_vec(self, last_action):
        if last_action is None:
            return np.zeros(3)

        if isinstance(last_action, (list, tuple)) and len(last_action) == 3:
            return np.array(last_action, dtype=float)

        last_action_vec = np.asarray(last_action, dtype=float).flatten()
        if last_action_vec.shape[0] != 3:
             if last_action_vec.shape[0] > 3:
                 return last_action_vec[:3]
             raise ValueError(f"last_action must have 3 elements, got {last_action_vec.shape[0]}")
        return last_action_vec

    def _load_model(self, model_type, model_path, horizon):
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            model_path = os.path.join(base_dir, 'output', 'single_stack', 'policy', f'{model_type}_policy_best.pth')

        print(f"Loading model from: {model_path}")

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
        print(f"Action Dim: {self.action_dim} (Expected 3)")
        print(f"Obs Dim: {self.obs_dim}")

    def _prepare_condition(self, state, P_ref, T_ref, last_action_vec):
        cond = np.zeros(self.obs_dim)

        T_s_in = state[0]
        T_s = state[1]
        T_sep = state[2]
        T_c_out = state[3]
        n_H2_an = state[4]
        n_liq = state[5]
        n_gas = state[6]

        cond[0] = T_s_in
        cond[1] = T_s
        cond[2] = T_sep
        cond[3] = T_c_out
        cond[4] = n_H2_an
        cond[5] = n_liq
        cond[6] = n_gas

        cond[7] = T_ref

        P_ref_arr = np.array(P_ref)
        base_idx = 8
        if len(P_ref_arr) >= self.horizon:
            cond[base_idx : base_idx + self.horizon] = P_ref_arr[:self.horizon]
        else:
            cond[base_idx : base_idx + len(P_ref_arr)] = P_ref_arr
            cond[base_idx + len(P_ref_arr) : base_idx + self.horizon] = P_ref_arr[-1]

        base_idx = 8 + self.horizon
        cond[base_idx : base_idx + 3] = last_action_vec

        cond_tensor = torch.FloatTensor(cond).to(self.device).unsqueeze(0)
        cond_norm = (cond_tensor - self.cond_min) / self.cond_diff * 2 - 1
        cond_norm = torch.clamp(cond_norm, -1.0, 1.0)

        return cond_norm

    def _denormalize_clamp_action(self, samples_norm):
        action_diff_b = self.action_diff.view(1, -1, 1)
        action_min_b = self.action_min.view(1, -1, 1)

        actions = ((samples_norm + 1) / 2) * action_diff_b + action_min_b

        actions[:, 0, :] = torch.clamp(actions[:, 0, :], self.I_min, self.I_max)
        actions[:, 1, :] = torch.clamp(actions[:, 1, :], self.v_lye_min, self.v_lye_max)
        actions[:, 2, :] = torch.clamp(actions[:, 2, :], self.v_c_min, self.v_c_max)

        return actions

    def _apply_cbf_projection(self, action, state, last_action=None, verbose=False, active_mask=None):
        """
        Apply CBF projection to ensure action satisfies safety constraints.

        Args:
            action: [I, v_lye, v_c] model-generated action
            state: Full state vector [T_s_in, T_s, T_sep, T_c_out, n_H2_an, n_liq, n_gas]
            last_action: Previous control [I, v_lye, v_c] for smoothness penalty
            verbose: Print CBF diagnostic information
            active_mask: Optional bool array overriding projector's default active_mask.

        Returns:
            safe_action: Projected action satisfying CBF conditions
            success: Whether projection succeeded
            cbf_values: Dict of CBF function values
        """
        if not self.use_cbf_projection:
            return action, True, {}

        u_safe, success, cbf_values = self.projector.project(
            action, state, u_last=last_action, verbose=verbose, active_mask=active_mask
        )
        return u_safe, success, cbf_values

    def get_action(self, state, P_ref, T_ref, last_action, verbose=False):
        """
        Get action with CBF projection.

        state: [T_s_in, T_s, T_sep, T_c_out, n_H2_an, n_liq, n_gas] (7 elements)
        P_ref: List or array of future power references
        T_ref: Scalar target temperature
        verbose: Print CBF diagnostic information
        """
        last_action_vec = self._as_last_action_vec(last_action)
        cond_norm = self._prepare_condition(state, P_ref, T_ref, last_action_vec)

        # Sample action from model
        num_candidates = 1
        cond_norm_batch = cond_norm.repeat(num_candidates, 1)

        noise_scale = 0.0
        if self.scheduler is None:
             samples_norm = self.model(cond_norm_batch)
        elif isinstance(self.scheduler, DDPMScheduler):
            samples_norm = self.scheduler.sample(self.model, cond_norm_batch, (num_candidates, self.action_dim, self.horizon))
        elif isinstance(self.scheduler, FlowMatchingScheduler):
            samples_norm = self.scheduler.sample(self.model, cond_norm_batch, (num_candidates, self.action_dim, self.horizon), noise_scale=noise_scale)
        else:
            raise ValueError("Unsupported scheduler type")

        actions = self._denormalize_clamp_action(samples_norm)
        actions_np = actions.cpu().numpy()

        # Apply CBF projection to first action
        raw_action = actions_np[0, :, 0]  # [I, v_lye, v_c]

        if self.use_cbf_projection:
            safe_action, success, cbf_values = self._apply_cbf_projection(raw_action, state, last_action=last_action_vec, verbose=verbose)
            if not success:
                print("Warning: CBF projection failed, using clipped action")
            elif verbose:
                print(f"CBF values: h_T={cbf_values.get('h_T', 0):.3f}, "
                      f"h_HTO={cbf_values.get('h_HTO', 0):.4f}, "
                      f"h_V={cbf_values.get('h_V', 0):.3f}")
        else:
            safe_action = raw_action

        return safe_action[0], safe_action[1], safe_action[2]

    def get_action_with_rollout(self, state, P_ref, T_ref, last_action, num_candidates=128, verbose=False):
        """
        Get action with candidate sampling, cost evaluation, and CBF projection.

        Similar to MultiStackModelController approach with added CBF safety projection.
        """
        last_action_vec = self._as_last_action_vec(last_action)
        cond_norm = self._prepare_condition(state, P_ref, T_ref, last_action_vec)

        # Sample multiple candidates
        cond_norm_batch = cond_norm.repeat(num_candidates, 1)

        noise_scale = 0.0
        if isinstance(self.scheduler, DDPMScheduler):
            samples_norm = self.scheduler.sample(self.model, cond_norm_batch, (num_candidates, self.action_dim, self.horizon))
        elif isinstance(self.scheduler, FlowMatchingScheduler):
            samples_norm = self.scheduler.sample(self.model, cond_norm_batch, (num_candidates, self.action_dim, self.horizon), noise_scale=noise_scale)
        else:
            raise ValueError("Unsupported scheduler type")

        actions = self._denormalize_clamp_action(samples_norm)
        actions_np = actions.cpu().numpy()

        # Apply CBF projection to all first-step actions
        projected_actions = np.zeros((num_candidates, 3))
        projection_success = np.zeros(num_candidates, dtype=bool)
        cbf_values_list = []

        for i in range(num_candidates):
            raw_action = actions_np[i, :, 0]
            safe_action, success, cbf_vals = self._apply_cbf_projection(raw_action, state, last_action=last_action_vec, verbose=(verbose and i==0))
            projected_actions[i] = safe_action
            projection_success[i] = success
            cbf_values_list.append(cbf_vals)

        # Rollout and evaluate cost with projected actions
        best_cost = float('inf')
        best_idx = 0

        P_ref_arr = np.array(P_ref)
        if len(P_ref_arr) < self.horizon:
            P_ref_eval = np.pad(P_ref_arr, (0, self.horizon - len(P_ref_arr)), 'edge')
        else:
            P_ref_eval = P_ref_arr[:self.horizon]

        steps_per_ctrl = int(self.dt / self.sim_rollout.dt)
        if steps_per_ctrl < 1:
            steps_per_ctrl = 1
            self.sim_rollout.dt = self.dt

        if self.sim_rollout.dt > 1.0:
            self.sim_rollout.dt = 0.2
            steps_per_ctrl = int(self.dt / self.sim_rollout.dt)

        for i in range(num_candidates):
            cost = 0.0
            self.sim_rollout.reset(initial_state=state)

            u_prev = last_action_vec.copy()
            I_prev = u_prev[0]
            v_lye_prev = u_prev[1]
            v_c_prev = u_prev[2]
            I_0 = u_prev[0]
            v_lye_0 = u_prev[1]
            v_c_0 = u_prev[2]

            # Use projected action for first step
            u_0 = projected_actions[i]

            for k in range(self.horizon):
                if k == 0:
                    u_k = u_0
                else:
                    u_k = actions_np[i, :, k]

                I_k = u_k[0]
                v_lye_k = u_k[1]
                v_c_k = u_k[2]

                T_s_curr = self.sim_rollout.state[1]
                _, U_cell, _ = self.sim_rollout._calculate_electrochemical_properties(I_k, T_s_curr)
                P_real_curr = U_cell * I_k * self.sim_rollout.N_cell

                for _ in range(steps_per_ctrl):
                    next_state = self.sim_rollout.step(u_k)

                T_s_next = next_state[1]

                cost += self.lambda_track * ((P_real_curr - P_ref_eval[k])/1e6)**2
                cost += self.lambda_temp * ((T_s_next - T_ref)**2)

                dI = I_k - I_prev
                dv_lye = v_lye_k - v_lye_0
                dv_c = v_c_k - v_c_0

                cost += self.lambda_I * (dI**2)
                cost += self.lambda_lye * (dv_lye**2)
                cost += self.lambda_c * (dv_c**2)

                I_prev = I_k
                v_lye_prev = v_lye_k
                v_c_prev = v_c_k

            if cost < best_cost:
                best_cost = cost
                best_idx = i

        best_action = projected_actions[best_idx]

        if verbose:
            success_rate = np.mean(projection_success)
            print(f"CBF Projection: {success_rate*100:.1f}% success rate ({np.sum(projection_success)}/{num_candidates})")
            if cbf_values_list[best_idx]:
                cv = cbf_values_list[best_idx]
                print(f"Best action CBF: h_T={cv.get('h_T', 0):.3f}, h_HTO={cv.get('h_HTO', 0):.4f}")

        return best_action[0], best_action[1], best_action[2]

    def get_cbf_diagnostics(self, state, action):
        """
        Get CBF diagnostic information for a given state-action pair.

        Returns:
            dict: CBF function values and constraint satisfaction status
        """
        if not self.use_cbf_projection:
            return {"error": "CBF projection not enabled"}

        _, _, cbf_values = self._apply_cbf_projection(action, state, verbose=False)
        return cbf_values
