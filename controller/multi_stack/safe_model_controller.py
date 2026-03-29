"""
Multi-Stack Safe Model Controller with Projection

Combines diffusion/flow-matching policy with safe projection operator
to ensure state constraints are satisfied.
"""

import os
import sys
import torch
import numpy as np
from .model_controller import MultiStackModelController

# Import safe projection
try:
    from .safe_projection import MultiStackSafeProjection, MultiStackSafeProjectionScipy
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from multi_stack.safe_projection import MultiStackSafeProjection, MultiStackSafeProjectionScipy


class MultiStackSafeModelController(MultiStackModelController):
    """
    Multi-Stack Model Controller with Safe Projection.

    Uses a learned policy (diffusion/flow-matching) to generate action candidates,
    then applies safe projection to ensure state constraints are satisfied.
    """

    def __init__(self, dt=60.0, horizon=5, model_type='tcn', model_path=None, stats_path=None,
                 use_safe_projection=True, projection_backend='casadi'):
        """
        Args:
            dt: Control time step
            horizon: Prediction horizon
            model_type: Type of model ('diffusion_mlp', 'flow_mlp', 'flow_tcn', 'diffusion_tcn')
            model_path: Path to model checkpoint
            stats_path: Path to normalization statistics
            use_safe_projection: Whether to apply safe projection
            projection_backend: 'casadi' or 'scipy' for projection solver
        """
        super().__init__(dt=dt, horizon=horizon, model_type=model_type,
                         model_path=model_path, stats_path=stats_path)

        self.use_safe_projection = use_safe_projection
        self.projection_backend = projection_backend

        # Initialize safe projection if enabled
        if self.use_safe_projection:
            if projection_backend == 'casadi':
                try:
                    self.projector = MultiStackSafeProjection(dt=dt)
                    print("Using CasADi-based safe projection")
                except Exception as e:
                    print(f"CasADi projection failed: {e}, falling back to SciPy")
                    self.projector = MultiStackSafeProjectionScipy(dt=dt)
                    self.projection_backend = 'scipy'
            else:
                self.projector = MultiStackSafeProjectionScipy(dt=dt)
                print("Using SciPy-based safe projection")

    def _apply_safe_projection(self, action, state, verbose=False):
        """
        Apply safe projection to ensure action satisfies state constraints.

        Args:
            action: [I1..4, v_lye1..4, v_c] model-generated action (9-dim)
            state: Full state vector [T_s_in, T_s1..4, T_sep, T_c_out, n_H2_an1..4, n_liq, n_gas] (13-dim)
            verbose: Print diagnostic information

        Returns:
            safe_action: Projected action
            success: Whether projection succeeded
        """
        if not self.use_safe_projection:
            return action, True

        # Extract projection state: [T_s_in, T_s1..4, T_sep, T_c_out, n_gas] (8-dim)
        proj_state = np.array([state[0], *state[1:5], state[5], state[6], state[12]])

        u_safe, success = self.projector.project(action, proj_state)

        return u_safe, success

    def get_action(self, state, P_ref, T_ref, last_action, verbose=False):
        """
        Get action with safe projection.

        state: [T_s_in, T_s1..4, T_sep, T_c_out, n_H2_an1..4, n_liq, n_gas] (13 elements)
        P_ref: List or array of future power references
        T_ref: Scalar target temperature
        verbose: Print diagnostic information
        """
        last_action_vec = self._as_last_action_vec(last_action)
        cond_norm = self._prepare_condition(state, P_ref, T_ref, last_action_vec)

        # Sample action from model
        num_candidates = 1
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

        # Apply safe projection to first action
        raw_action = actions_np[0, :, 0]  # [I1..4, v_lye1..4, v_c]

        if self.use_safe_projection:
            safe_action, success = self._apply_safe_projection(raw_action, state, verbose=verbose)
            if not success:
                print("Warning: Safe projection failed, using clipped action")
        else:
            safe_action = raw_action

        I_cmd = safe_action[0:4]
        v_lye_cmd = safe_action[4:8]
        v_c_cmd = safe_action[8]

        return I_cmd, v_lye_cmd, v_c_cmd

    def get_action_with_rollout(self, state, P_ref, T_ref, last_action, num_candidates=128, verbose=False):
        """
        Get action with candidate sampling, cost evaluation, and safe projection.

        Similar to MultiStackModelController approach with added safety projection.
        """
        last_action_vec = self._as_last_action_vec(last_action)
        cond_norm = self._prepare_condition(state, P_ref, T_ref, last_action_vec)

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

        # Apply safe projection to all first-step actions
        projected_actions = np.zeros((num_candidates, 9))
        projection_success = np.zeros(num_candidates, dtype=bool)

        for i in range(num_candidates):
            raw_action = actions_np[i, :, 0]
            safe_action, success = self._apply_safe_projection(raw_action, state, verbose=(verbose and i==0))
            projected_actions[i] = safe_action
            projection_success[i] = success

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

                # Calculate Real Power using NEXT state (as requested)
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
            print(f"Safe Projection: {success_rate*100:.1f}% success rate ({np.sum(projection_success)}/{num_candidates})")

        I_cmd = best_action[0:4]
        v_lye_cmd = best_action[4:8]
        v_c_cmd = best_action[8]

        return I_cmd, v_lye_cmd, v_c_cmd
