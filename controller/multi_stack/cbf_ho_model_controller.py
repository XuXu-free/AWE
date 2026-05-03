"""
Multi-Stack HOCBF Model Controller

Combines diffusion/flow-matching policy with Higher-Order CBF-based projection
operator to ensure safety constraints are satisfied via second-order HOCBF.
"""

import os
import sys
import torch
import numpy as np
from .model_controller import MultiStackModelController

# Import HOCBF projection
try:
    from .cbf_projection_ho import MultiStackCBFProjectionHO
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from multi_stack.cbf_projection_ho import MultiStackCBFProjectionHO


class MultiStackCBFHOModelController(MultiStackModelController):
    """
    Multi-Stack Model Controller with Higher-Order CBF (HOCBF) Projection.

    Uses a learned policy (diffusion/flow-matching) to generate action candidates,
    then applies HOCBF projection to ensure safety constraints are satisfied.

    HOCBF enforces second-order barrier conditions for smoother control response.
    """

    def __init__(self, dt=60.0, horizon=5, model_type='tcn', model_path=None, stats_path=None,
                 use_cbf_projection=True, gamma=1.0,
                 gamma_vec=None, rho_vec=None, h_margin_vec=None, normalize=True,
                 lambda_u_scale=1000.0, soft_mask=None,
                 alpha1_hto=0.8, alpha2_hto=2.0,
                 alpha1_vec=None, alpha2_vec=None,
                 u_weight_scale=None,
                 adaptive_alpha=False, T_adaptive_threshold=358.15,
                 alpha1_vec_low=None, alpha2_vec_low=None, h_margin_vec_low=None):
        """
        Args:
            dt: Control time step
            horizon: Prediction horizon
            model_type: Type of model ('diffusion_mlp', 'flow_mlp', 'flow_tcn', 'diffusion_tcn')
            model_path: Path to model checkpoint
            stats_path: Path to normalization statistics
            use_cbf_projection: Whether to apply HOCBF projection
            gamma: Default CBF gain
            gamma_vec: Per-constraint gamma values
            rho_vec: Per-constraint rho values for soft constraints
            h_margin_vec: Per-constraint safety margins
            normalize: Whether to use normalized CBF constraints
            lambda_u_scale: Control change penalty scale
            soft_mask: Per-constraint soft slack mask
            alpha1_hto: First-order decay rate for HTO HOCBF
            alpha2_hto: Second-order damping for HTO HOCBF
            alpha1_vec: Per-constraint alpha1 (9 elements)
            alpha2_vec: Per-constraint alpha2 (9 elements)
            u_weight_scale: Per-control weight scale for reference tracking cost (9 elements)
            adaptive_alpha: Whether to enable temperature-adaptive alpha parameters
            T_adaptive_threshold: Temperature threshold (K) below which low-temp params are used
            alpha1_vec_low: Low-temperature alpha1 vector (9 elements)
            alpha2_vec_low: Low-temperature alpha2 vector (9 elements)
            h_margin_vec_low: Low-temperature h_margin vector (9 elements)
        """
        super().__init__(dt=dt, horizon=horizon, model_type=model_type,
                         model_path=model_path, stats_path=stats_path)

        self.use_cbf_projection = use_cbf_projection
        self.gamma = gamma
        self.adaptive_alpha = adaptive_alpha
        self.T_adaptive_threshold = T_adaptive_threshold

        # Initialize HOCBF projection if enabled
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
                'alpha1_hto': alpha1_hto,
                'alpha2_hto': alpha2_hto,
                'alpha1_vec': alpha1_vec,
                'alpha2_vec': alpha2_vec,
                'u_weight_scale': u_weight_scale,
            }
            self.projector = MultiStackCBFProjectionHO(**proj_kwargs)
            print(f"Using MultiStack HOCBF projection (alpha1_hto={alpha1_hto}, alpha2_hto={alpha2_hto})")

            if self.adaptive_alpha:
                # Save default parameters
                self.alpha1_vec_default = self.projector.alpha1_vec.copy()
                self.alpha2_vec_default = self.projector.alpha2_vec.copy()
                self.h_margin_vec_default = self.projector.h_margin_vec.copy()

                # Low-temperature parameters
                self.alpha1_vec_low = np.array(alpha1_vec_low) if alpha1_vec_low is not None else self.alpha1_vec_default.copy()
                self.alpha2_vec_low = np.array(alpha2_vec_low) if alpha2_vec_low is not None else self.alpha2_vec_default.copy()
                self.h_margin_vec_low = np.array(h_margin_vec_low) if h_margin_vec_low is not None else self.h_margin_vec_default.copy()

                print(f"Adaptive alpha enabled: T_threshold={T_adaptive_threshold-273.15:.1f}C")
                print(f"  Default alpha1[HTO]={self.alpha1_vec_default[4]:.2f}, alpha2[HTO]={self.alpha2_vec_default[4]:.2f}")
                print(f"  Low-temp alpha1[HTO]={self.alpha1_vec_low[4]:.2f}, alpha2[HTO]={self.alpha2_vec_low[4]:.2f}")

    def _update_adaptive_params(self, state):
        """Update projector parameters based on current temperature."""
        if not self.adaptive_alpha or not self.use_cbf_projection:
            return
        T_avg = np.mean(state[1:5])
        if T_avg < self.T_adaptive_threshold:
            self.projector.alpha1_vec = self.alpha1_vec_low.copy()
            self.projector.alpha2_vec = self.alpha2_vec_low.copy()
            self.projector.h_margin_vec = self.h_margin_vec_low.copy()
        else:
            self.projector.alpha1_vec = self.alpha1_vec_default.copy()
            self.projector.alpha2_vec = self.alpha2_vec_default.copy()
            self.projector.h_margin_vec = self.h_margin_vec_default.copy()

    def _apply_cbf_projection(self, action, state, last_action=None, verbose=False):
        """
        Apply HOCBF projection to ensure action satisfies safety constraints.

        Args:
            action: [I1..4, v_lye1..4, v_c] model-generated action (9-dim)
            state: Full state vector (13-dim)
            last_action: Previous control [I1..4, v_lye1..4, v_c] for smoothness penalty (9-dim)
            verbose: Print HOCBF diagnostic information

        Returns:
            safe_action: Projected action satisfying HOCBF conditions
            success: Whether projection succeeded
            info: Dict of HOCBF function values and diagnostics
        """
        if not self.use_cbf_projection:
            return action, True, {}

        u_safe, success, info = self.projector.project(action, state, u_last=last_action, verbose=verbose)
        return u_safe, success, info

    def get_action(self, state, P_ref, T_ref, last_action, verbose=False):
        """
        Get action with HOCBF projection.

        Uses parent class sampling (128 candidates + cost-weighted selection)
        then applies HOCBF projection to the best action for safety.

        state: [T_s_in, T_s1..4, T_sep, T_c_out, n_H2_an1..4, n_liq, n_gas] (13 elements)
        P_ref: List or array of future power references
        T_ref: Scalar target temperature
        verbose: Print HOCBF diagnostic information
        """
        # 1. Use parent class to get best action (128 candidates + rollout + cost-weighted selection)
        I_cmd, v_lye_cmd, v_c_cmd = super().get_action(state, P_ref, T_ref, last_action)
        raw_action = np.concatenate([I_cmd, v_lye_cmd, [v_c_cmd]])

        # 2. Apply HOCBF projection for safety
        last_action_vec = self._as_last_action_vec(last_action)

        # Update adaptive CBF parameters based on current temperature
        self._update_adaptive_params(state)

        if self.use_cbf_projection:
            safe_action, success, info = self._apply_cbf_projection(raw_action, state, last_action=last_action_vec, verbose=verbose)
            if not success:
                print("Warning: HOCBF projection failed, using clipped action")
            elif verbose:
                print(f"HOCBF info: {info}")
        else:
            safe_action = raw_action
            info = {
                'projection_needed': False,
                'cbf': np.zeros(9),
                'slack': np.zeros(9),
                'adjustment': 0.0,
                'max_slack': 0.0,
            }

        I_cmd = safe_action[0:4]
        v_lye_cmd = safe_action[4:8]
        v_c_cmd = safe_action[8]

        return I_cmd, v_lye_cmd, v_c_cmd, info

    def get_action_with_rollout(self, state, P_ref, T_ref, last_action, num_candidates=128, verbose=False):
        """
        Get action with candidate sampling, cost evaluation, and HOCBF projection.

        Similar to MultiStackModelController approach with added HOCBF safety projection.
        """
        last_action_vec = self._as_last_action_vec(last_action)
        cond_norm = self._prepare_condition(state, P_ref, T_ref, last_action_vec)

        # Update adaptive CBF parameters based on current temperature
        self._update_adaptive_params(state)

        # Sample multiple candidates
        cond_norm_batch = cond_norm.repeat(num_candidates, 1)

        noise_scale = 0.0
        from diffusion.ddpm import DDPMScheduler
        from diffusion.flow_matching import FlowMatchingScheduler
        if isinstance(self.scheduler, DDPMScheduler):
            samples_norm = self.scheduler.sample(self.model, cond_norm_batch, (num_candidates, self.action_dim, self.horizon))
        elif isinstance(self.scheduler, FlowMatchingScheduler):
            samples_norm = self.scheduler.sample(self.model, cond_norm_batch, (num_candidates, self.action_dim, self.horizon), noise_scale=noise_scale)
        else:
            lb = torch.full((num_candidates, self.action_dim, self.horizon), -1.0, device=self.device)
            ub = torch.full((num_candidates, self.action_dim, self.horizon), 1.0, device=self.device)
            last_action_t = torch.FloatTensor(last_action_vec).to(self.device)
            last_action_norm = 2 * (last_action_t - self.action_min) / self.action_diff - 1
            last_action_norm_b = last_action_norm.view(1, -1).repeat(num_candidates, 1)
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
                noise_scale=0.0
            )

        actions_denorm = self._denormalize_action(samples_norm)
        actions_proj = self._projection(actions_denorm)
        actions = self._clamp_action(actions_proj)
        actions_np = actions.cpu().numpy()

        # Apply HOCBF projection to all first-step actions
        projected_actions = np.zeros((num_candidates, 9))
        projection_success = np.zeros(num_candidates, dtype=bool)
        info_list = []

        for i in range(num_candidates):
            raw_action = actions_np[i, :, 0]
            safe_action, success, info = self._apply_cbf_projection(raw_action, state, last_action=last_action_vec, verbose=(verbose and i==0))
            projected_actions[i] = safe_action
            projection_success[i] = success
            info_list.append(info)

        # Rollout and evaluate cost with projected actions
        best_cost = float('inf')
        best_idx = 0

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
            u_prev = last_action_vec.copy()
            I_prev = u_prev[0:4]
            v_lye_prev = u_prev[4:8]
            v_c_prev = u_prev[8]
            I_0 = u_prev[0:4]
            v_lye_0 = u_prev[4:8]
            v_c_0 = u_prev[8]

            # Use projected action for first step
            u_0 = projected_actions[i]

            # Iterate over horizon
            for k in range(self.horizon):
                if k == 0:
                    u_k = u_0
                else:
                    u_k = actions_np[i, :, k]

                I_k = u_k[0:4]
                v_lye_k = u_k[4:8]
                v_c_k = u_k[8]

                # Step
                next_state = self.sim_rollout.step(u_k)
                T_s_vec_next = next_state[1:5]

                # Calculate Real Power using NEXT state
                _, U_cell_vec_next, _ = self.sim_rollout._calculate_electrochemical_properties(I_k, T_s_vec_next)
                P_real_next = np.sum(U_cell_vec_next * I_k * self.sim_rollout.N_cell)

                # --- Cost Calculation ---
                # 1. Power Tracking
                cost += self.lambda_track * ((P_real_next - P_ref_eval[k])/1e6)**2

                # 2. Temperature Regulation
                cost += self.lambda_temp * np.sum((T_s_vec_next - T_ref)**2)

                # 3. Smoothness
                dI = I_k - I_prev
                dv_lye = v_lye_k - v_lye_0
                dv_c = v_c_k - v_c_0

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

        best_action = projected_actions[best_idx]

        if verbose:
            success_rate = np.mean(projection_success)
            print(f"HOCBF Projection: {success_rate*100:.1f}% success rate ({np.sum(projection_success)}/{num_candidates})")

        I_cmd = best_action[0:4]
        v_lye_cmd = best_action[4:8]
        v_c_cmd = best_action[8]

        return I_cmd, v_lye_cmd, v_c_cmd
