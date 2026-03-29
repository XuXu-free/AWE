"""
Unified Safety Test Script for Single-Stack System

Integrates all safety-related tests:
1. Open-loop constraint test (Open-loop) - Verify unsafe behavior of raw control
2. Safe Projection test - Verify traditional safe projection effect
3. CBF Projection test - Verify CBF projection effect
4. HTO specific test - HTO constraints under low power conditions
5. Comprehensive comparison test - Direct comparison of three methods

Usage:
    python run_unified_safety_tests.py --test all       # Run all tests
    python run_unified_safety_tests.py --test openloop  # Open-loop test only
    python run_unified_safety_tests.py --test safe      # Only Safe Projection test
    python run_unified_safety_tests.py --test cbf       # Only CBF Projection test
    python run_unified_safety_tests.py --test hto       # Only HTO specific test
    python run_unified_safety_tests.py --test compare   # Three-way comparison only
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from plant.single_stack_simulator import SingleStackSimulator

# Import projectors
try:
    from controller.single_stack.safe_projection import SingleStackSafeProjection, SingleStackSafeProjectionScipy
    from controller.single_stack.cbf_projection import SingleStackCBFProjection, SingleStackCBFProjectionSimplified
except ImportError as e:
    print(f"Failed to import projectors: {e}")
    sys.exit(1)


@dataclass
class SafetyConfig:
    """Safety Configuration Parameters"""
    # Constraint boundaries
    I_min: float = 0.0
    I_max: float = 7800.0 * 1.2  # 9360 A
    v_lye_min: float = 0.0
    v_lye_max: float = 0.1
    v_c_min: float = 0.0
    v_c_max: float = 1.0
    T_min: float = 293.15  # 20degC
    T_max: float = 363.15  # 90degC
    U_cell_max: float = 2.2
    HTO_max: float = 2.0
    P_stack_max: float = 6.0e6

    # CBF parameters
    cbf_gamma: float = 1.0

    # Simulation parameters
    sim_dt: float = 0.2
    ctrl_dt: float = 60.0


@dataclass
class TestResult:
    """Test Result Data Structure"""
    test_name: str
    method: str  # 'raw', 'safe', 'cbf'
    history: Dict
    violations: Dict[str, int]
    max_values: Dict[str, float]
    success: bool


class UnifiedSafetyTester:
    """Unified Safety Tester"""

    def __init__(self, config: SafetyConfig = None, output_dir: str = None):
        self.config = config or SafetyConfig()
        self.output_dir = output_dir or os.path.join('output', 'single_stack', 'unified_safety_tests')
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize projectors
        self._init_projectors()

    def _init_projectors(self):
        """Initialize safety projectors"""
        print("Initializing safety projectors...")

        # Safe Projection
        try:
            self.safe_projector = SingleStackSafeProjection(dt=self.config.ctrl_dt)
            print("  [OK] Safe Projection (CasADi) initialized")
        except Exception as e:
            print(f"  [FAIL] Safe Projection (CasADi): {e}")
            self.safe_projector = SingleStackSafeProjectionScipy(dt=self.config.ctrl_dt)
            print("  [OK] Safe Projection (SciPy) fallback initialized")

        # CBF Projection
        try:
            self.cbf_projector = SingleStackCBFProjection(dt=self.config.ctrl_dt, gamma=self.config.cbf_gamma)
            print(f"  [OK] Strict CBF Projection (CasADi, gamma={self.config.cbf_gamma}) initialized")
        except Exception as e:
            print(f"  [FAIL] CBF Projection (CasADi): {e}")
            self.cbf_projector = SingleStackCBFProjectionSimplified(dt=self.config.ctrl_dt, gamma=self.config.cbf_gamma)
            print("  [OK] Strict CBF Projection (SciPy) fallback initialized")

    def run_open_loop_test(self, test_name: str, actions: List[Tuple],
                          initial_state: np.ndarray = None, duration: float = 3600) -> TestResult:
        """
        Run open-loop safety test (raw control, no projection)

        Args:
            test_name: Test name
            actions: Control action sequence [(t_start, t_end, [I, v_lye, v_c]), ...]
            initial_state: Initial state
            duration: Test duration (seconds)
        """
        print(f"\n{'='*70}")
        print(f"Open-loop test: {test_name}")
        print(f"{'='*70}")

        sim = SingleStackSimulator(sim_dt=self.config.sim_dt)
        sim.reset(initial_state=initial_state)

        dt = self.config.sim_dt
        steps = int(duration / dt)

        # Create action lookup table
        action_table = {}
        for t_start, t_end, action in actions:
            start_step = int(t_start / dt)
            end_step = int(t_end / dt)
            for s in range(start_step, min(end_step, steps)):
                action_table[s] = np.array(action)

        default_action = np.array([2000.0, 0.03, 0.0])

        # Record data
        history = self._create_history_dict()

        for i in range(steps):
            t = i * dt
            action = action_table.get(i, default_action)
            state = sim.state

            # Record data
            if i % 50 == 0:
                self._record_state(history, t, action, state, sim)

            # Execute simulation step
            sim.step(action)

        # Analyze results
        violations, max_values = self._analyze_history(history)

        print(f"\nTest Results:")
        print(f"  Max current: {max_values['I']:.1f} A")
        print(f"  Max temperature: {max_values['T']:.2f} K ({max_values['T']-273.15:.1f}degC)")
        print(f"  Max voltage: {max_values['U']:.3f} V")
        print(f"  Max HTO: {max_values['HTO']:.4f}%")
        print(f"  Constraint violations: {sum(violations.values())} times")

        return TestResult(test_name, 'raw', history, violations, max_values, True)

    def run_projection_test(self, test_name: str, actions: List[Tuple],
                           projection_type: str = 'safe',
                           initial_state: np.ndarray = None, duration: float = 3600) -> TestResult:
        """
        Run projection comparison test

        Args:
            test_name: Test name
            actions: Raw Control action sequence
            projection_type: 'safe' or 'cbf'
            initial_state: Initial state
            duration: Test duration
        """
        print(f"\n{'='*70}")
        print(f"{projection_type.upper()} Projection test: {test_name}")
        print(f"{'='*70}")

        sim = SingleStackSimulator(sim_dt=self.config.sim_dt)
        sim.reset(initial_state=initial_state)

        projector = self.safe_projector if projection_type == 'safe' else self.cbf_projector

        dt = self.config.sim_dt
        steps = int(duration / dt)
        ctrl_steps = int(self.config.ctrl_dt / dt)

        # Create action lookup table
        action_table = {}
        for t_start, t_end, action in actions:
            start_step = int(t_start / dt)
            end_step = int(t_end / dt)
            for s in range(start_step, min(end_step, steps)):
                action_table[s] = np.array(action)

        default_action = np.array([2000.0, 0.03, 0.0])

        # Record data
        history = self._create_history_dict()
        projection_diffs = {'t': [], 'dI': [], 'dv_lye': [], 'dv_c': [], 'norm': []}

        current_action = default_action.copy()

        for i in range(steps):
            t = i * dt
            raw_action = action_table.get(i, default_action)

            # Apply projection every control cycle
            if i % ctrl_steps == 0:
                state = sim.state

                if projection_type == 'safe':
                    # Safe projection uses reduced state
                    proj_state = np.array([state[0], state[1], state[2], state[3], state[6]])
                    current_action, success = projector.project(raw_action, proj_state)
                else:
                    # CBF projection uses full state
                    current_action, success, _ = projector.project(raw_action, state)

                # Record projection difference
                if i % (ctrl_steps * 5) == 0:
                    projection_diffs['t'].append(t)
                    projection_diffs['dI'].append(current_action[0] - raw_action[0])
                    projection_diffs['dv_lye'].append(current_action[1] - raw_action[1])
                    projection_diffs['dv_c'].append(current_action[2] - raw_action[2])
                    projection_diffs['norm'].append(np.linalg.norm(current_action - raw_action))

            # Record data
            if i % 50 == 0:
                self._record_state(history, t, current_action, sim.state, sim)

            # Execute simulation step
            sim.step(current_action)

        # Analyze results
        violations, max_values = self._analyze_history(history)

        # Add projection diff to history
        history['projection_diffs'] = projection_diffs

        print(f"\nTest Results:")
        print(f"  Max current: {max_values['I']:.1f} A")
        print(f"  Max temperature: {max_values['T']:.2f} K ({max_values['T']-273.15:.1f}degC)")
        print(f"  Max HTO: {max_values['HTO']:.4f}%")
        print(f"  Constraint violations: {sum(violations.values())} times")

        if projection_diffs['t']:
            avg_diff = np.mean(projection_diffs['norm'])
            max_diff = np.max(projection_diffs['norm'])
            print(f"  Avg projection adjustment: {avg_diff:.3f}")
            print(f"  Max projection adjustment: {max_diff:.3f}")

        return TestResult(test_name, projection_type, history, violations, max_values, True)

    def run_three_way_comparison(self, test_name: str, actions: List[Tuple],
                                  initial_state: np.ndarray = None, duration: float = 3600) -> Dict[str, TestResult]:
        """
        Run three-way comparison test：Raw vs Safe vs CBF
        """
        print(f"\n{'='*70}")
        print(f"Three-Way Method Comparison Test: {test_name}")
        print(f"{'='*70}")

        results = {}

        # 1. Raw control
        results['raw'] = self.run_open_loop_test(f"{test_name}_raw", actions, initial_state, duration)

        # 2. Safe Projection
        results['safe'] = self.run_projection_test(f"{test_name}_safe", actions, 'safe', initial_state, duration)

        # 3. CBF Projection
        results['cbf'] = self.run_projection_test(f"{test_name}_cbf", actions, 'cbf', initial_state, duration)

        # 打印Comparison summary
        print(f"\n{'='*70}")
        print(f"Comparison summary: {test_name}")
        print(f"{'='*70}")
        print(f"{'Method':<15} {'Max HTO':<12} {'Max Temp':<12} {'Violations':<10}")
        print("-" * 60)
        for method in ['raw', 'safe', 'cbf']:
            r = results[method]
            print(f"{method.upper():<15} {r.max_values['HTO']:>10.4f}% {r.max_values['T']-273.15:>10.2f}degC {sum(r.violations.values()):>8}times")

        return results

    def run_hto_specific_test(self, test_name: str, raw_actions: List[Tuple],
                              initial_state: np.ndarray = None, duration: float = 7200) -> Dict[str, TestResult]:
        """
        HTO specific test - 验证HTO constraints under low power conditions
        """
        print(f"\n{'='*70}")
        print(f"HTO specific test: {test_name}")
        print(f"{'='*70}")

        results = {}

        # 运行三种Comparison
        results = self.run_three_way_comparison(test_name, raw_actions, initial_state, duration)

        # HTO specific analysis
        print(f"\nHTO specific analysis:")
        print("-" * 50)
        for method in ['raw', 'safe', 'cbf']:
            hto_values = results[method].history['HTO']
            max_hto = max(hto_values)
            hto_violations = sum(results[method].history['HTO_violation'])

            print(f"{method.upper()}:")
            print(f"  Max HTO: {max_hto:.4f}%")
            print(f"  HTOviolation count: {hto_violations}")
            print(f"  Final HTO: {hto_values[-1]:.4f}%")

        return results

    def _create_history_dict(self) -> Dict:
        """Create history dict"""
        return {
            't': [], 'I': [], 'v_lye': [], 'v_c': [],
            'T_s': [], 'T_sep': [], 'T_c_out': [],
            'P_real': [], 'U_cell': [], 'HTO': [], 'eta': [],
            'I_violation': [], 'v_lye_violation': [], 'v_c_violation': [],
            'T_violation': [], 'U_violation': [], 'HTO_violation': [], 'P_violation': []
        }

    def _record_state(self, history: Dict, t: float, action: np.ndarray, state: np.ndarray, sim):
        """Record current state"""
        T_s_in, T_s, T_sep, T_c_out = state[0], state[1], state[2], state[3]
        n_H2_an, n_liq, n_gas = state[4], state[5], state[6]

        _, U_cell, eta = sim._calculate_electrochemical_properties(action[0], T_s)
        P_real = U_cell * action[0] * sim.N_cell
        hto_pct = (n_gas * sim.R * T_sep) / (sim.P_sys * sim.V_sep_gas) * 100

        cfg = self.config

        history['t'].append(t)
        history['I'].append(action[0])
        history['v_lye'].append(action[1])
        history['v_c'].append(action[2])
        history['T_s'].append(T_s)
        history['T_sep'].append(T_sep)
        history['T_c_out'].append(T_c_out)
        history['P_real'].append(P_real)
        history['U_cell'].append(U_cell)
        history['HTO'].append(hto_pct)
        history['eta'].append(eta)

        history['I_violation'].append(action[0] < cfg.I_min or action[0] > cfg.I_max)
        history['v_lye_violation'].append(action[1] < cfg.v_lye_min or action[1] > cfg.v_lye_max)
        history['v_c_violation'].append(action[2] < cfg.v_c_min or action[2] > cfg.v_c_max)
        history['T_violation'].append(T_s < cfg.T_min or T_s > cfg.T_max)
        history['U_violation'].append(U_cell > cfg.U_cell_max)
        history['HTO_violation'].append(hto_pct > cfg.HTO_max)
        history['P_violation'].append(P_real > cfg.P_stack_max)

    def _analyze_history(self, history: Dict) -> Tuple[Dict, Dict]:
        """分析历史数据中的Constraint violations情况"""
        violations = {
            'I': sum(history['I_violation']),
            'v_lye': sum(history['v_lye_violation']),
            'v_c': sum(history['v_c_violation']),
            'T': sum(history['T_violation']),
            'U': sum(history['U_violation']),
            'HTO': sum(history['HTO_violation']),
            'P': sum(history['P_violation'])
        }

        max_values = {
            'I': max(history['I']),
            'v_lye': max(history['v_lye']),
            'v_c': max(history['v_c']),
            'T': max(history['T_s']),
            'U': max(history['U_cell']),
            'HTO': max(history['HTO']),
            'P': max(history['P_real']) / 1e6
        }

        return violations, max_values

    # ==================== 绘图Method ====================

    def plot_single_result(self, result: TestResult, output_subdir: str = None):
        """绘制单个Test Results"""
        t_arr = np.array(result.history['t'])

        fig, axes = plt.subplots(4, 2, figsize=(14, 16))
        fig.suptitle(f'{result.method.upper()} - {result.test_name}', fontsize=14, fontweight='bold')

        cfg = self.config

        # 电流
        ax = axes[0, 0]
        ax.plot(t_arr/60, result.history['I'], 'b-', linewidth=2)
        ax.axhline(y=cfg.I_max, color='r', linestyle='--', label=f'I_max={cfg.I_max:.0f}')
        ax.set_title('Current')
        ax.set_ylabel('A')
        ax.legend()
        ax.grid(True)

        # HTO
        ax = axes[0, 1]
        ax.plot(t_arr/60, result.history['HTO'], 'orange', linewidth=2)
        ax.axhline(y=cfg.HTO_max, color='r', linestyle='--', label=f'HTO_max={cfg.HTO_max}%')
        ax.fill_between(t_arr/60, cfg.HTO_max, max(result.history['HTO']) * 1.1,
                        alpha=0.2, color='red')
        ax.set_title('HTO (Safety Critical)')
        ax.set_ylabel('%')
        ax.legend()
        ax.grid(True)

        # 温deg
        ax = axes[1, 0]
        ax.plot(t_arr/60, np.array(result.history['T_s'])-273.15, 'r-', linewidth=2)
        ax.axhline(y=cfg.T_max-273.15, color='r', linestyle='--', label=f'T_max={cfg.T_max-273.15:.0f}degC')
        ax.set_title('Stack Temperature')
        ax.set_ylabel('degC')
        ax.legend()
        ax.grid(True)

        # 电压
        ax = axes[1, 1]
        ax.plot(t_arr/60, result.history['U_cell'], 'm-', linewidth=2)
        ax.axhline(y=cfg.U_cell_max, color='r', linestyle='--', label=f'U_max={cfg.U_cell_max}V')
        ax.set_title('Cell Voltage')
        ax.set_ylabel('V')
        ax.legend()
        ax.grid(True)

        # 功率
        ax = axes[2, 0]
        ax.plot(t_arr/60, np.array(result.history['P_real'])/1e6, 'purple', linewidth=2)
        ax.axhline(y=cfg.P_stack_max/1e6, color='r', linestyle='--', label=f'P_max={cfg.P_stack_max/1e6:.0f}MW')
        ax.set_title('Stack Power')
        ax.set_ylabel('MW')
        ax.legend()
        ax.grid(True)

        # 碱液流速
        ax = axes[2, 1]
        ax.plot(t_arr/60, np.array(result.history['v_lye'])*1000, 'g-', linewidth=2)
        ax.axhline(y=cfg.v_lye_max*1000, color='r', linestyle='--', label=f'v_lye_max={cfg.v_lye_max*1000:.0f}L/s')
        ax.set_title('Lye Flow Rate')
        ax.set_ylabel('L/s')
        ax.legend()
        ax.grid(True)

        # Constraint violations汇总
        ax = axes[3, 0]
        violation_sum = (
            np.array(result.history['I_violation']) +
            np.array(result.history['T_violation']) +
            np.array(result.history['U_violation']) +
            np.array(result.history['HTO_violation'])
        )
        ax.fill_between(t_arr/60, 0, violation_sum, alpha=0.3, color='red')
        ax.set_title('Constraint Violations')
        ax.set_ylabel('Count')
        ax.set_xlabel('Time (min)')
        ax.grid(True)

        # 法拉第效率
        ax = axes[3, 1]
        ax.plot(t_arr/60, result.history['eta'], 'c-', linewidth=2)
        ax.set_title('Faraday Efficiency')
        ax.set_ylabel('η')
        ax.set_xlabel('Time (min)')
        ax.grid(True)

        plt.tight_layout()

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{result.method}_{result.test_name}_{timestamp}.png"

        if output_subdir:
            save_dir = os.path.join(self.output_dir, output_subdir)
            os.makedirs(save_dir, exist_ok=True)
            filepath = os.path.join(save_dir, filename)
        else:
            filepath = os.path.join(self.output_dir, filename)

        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Figure saved: {filepath}")
        return filepath

    def plot_three_way_comparison(self, results: Dict[str, TestResult], test_name: str):
        """绘制三种MethodComparison图"""
        fig = plt.figure(figsize=(16, 20))
        gs = fig.add_gridspec(5, 2, hspace=0.3, wspace=0.3)

        colors = {'raw': 'red', 'safe': 'green', 'cbf': 'blue'}
        labels = {'raw': 'Raw (No Projection)', 'safe': 'Safe Projection', 'cbf': 'CBF Projection'}

        cfg = self.config

        # Current comparison
        ax1 = fig.add_subplot(gs[0, 0])
        for method in ['raw', 'safe', 'cbf']:
            t = np.array(results[method].history['t']) / 60
            ax1.plot(t, results[method].history['I'], color=colors[method],
                    label=labels[method], linewidth=2, alpha=0.8)
        ax1.axhline(y=cfg.I_max, color='k', linestyle='--', alpha=0.5)
        ax1.set_title('Current Comparison', fontweight='bold')
        ax1.set_ylabel('A')
        ax1.legend()
        ax1.grid(True)

        # HTOComparison（Critical）
        ax2 = fig.add_subplot(gs[0, 1])
        for method in ['raw', 'safe', 'cbf']:
            t = np.array(results[method].history['t']) / 60
            ax2.plot(t, results[method].history['HTO'], color=colors[method],
                    label=labels[method], linewidth=2, alpha=0.8)
        ax2.axhline(y=cfg.HTO_max, color='k', linestyle='--', alpha=0.5)
        ax2.fill_between(t, cfg.HTO_max, max(max(results[m].history['HTO']) for m in ['raw', 'safe', 'cbf']) * 1.05,
                        alpha=0.1, color='red')
        ax2.set_title('HTO Comparison (Safety Critical)', fontweight='bold')
        ax2.set_ylabel('%')
        ax2.legend()
        ax2.grid(True)

        # Temperature comparison
        ax3 = fig.add_subplot(gs[1, 0])
        for method in ['raw', 'safe', 'cbf']:
            t = np.array(results[method].history['t']) / 60
            ax3.plot(t, np.array(results[method].history['T_s'])-273.15, color=colors[method],
                    label=labels[method], linewidth=2, alpha=0.8)
        ax3.axhline(y=cfg.T_max-273.15, color='k', linestyle='--', alpha=0.5)
        ax3.set_title('Temperature Comparison', fontweight='bold')
        ax3.set_ylabel('degC')
        ax3.legend()
        ax3.grid(True)

        # Voltage comparison
        ax4 = fig.add_subplot(gs[1, 1])
        for method in ['raw', 'safe', 'cbf']:
            t = np.array(results[method].history['t']) / 60
            ax4.plot(t, results[method].history['U_cell'], color=colors[method],
                    label=labels[method], linewidth=2, alpha=0.8)
        ax4.axhline(y=cfg.U_cell_max, color='k', linestyle='--', alpha=0.5)
        ax4.set_title('Voltage Comparison', fontweight='bold')
        ax4.set_ylabel('V')
        ax4.legend()
        ax4.grid(True)

        # Power comparison
        ax5 = fig.add_subplot(gs[2, 0])
        for method in ['raw', 'safe', 'cbf']:
            t = np.array(results[method].history['t']) / 60
            ax5.plot(t, np.array(results[method].history['P_real'])/1e6, color=colors[method],
                    label=labels[method], linewidth=2, alpha=0.8)
        ax5.axhline(y=cfg.P_stack_max/1e6, color='k', linestyle='--', alpha=0.5)
        ax5.set_title('Power Comparison', fontweight='bold')
        ax5.set_ylabel('MW')
        ax5.legend()
        ax5.grid(True)

        # Lye flow comparison
        ax6 = fig.add_subplot(gs[2, 1])
        for method in ['raw', 'safe', 'cbf']:
            t = np.array(results[method].history['t']) / 60
            ax6.plot(t, np.array(results[method].history['v_lye'])*1000, color=colors[method],
                    label=labels[method], linewidth=2, alpha=0.8)
        ax6.set_title('Lye Flow Comparison', fontweight='bold')
        ax6.set_ylabel('L/s')
        ax6.legend()
        ax6.grid(True)

        # HTO violationsComparison
        ax7 = fig.add_subplot(gs[3, :])
        for method in ['raw', 'safe', 'cbf']:
            t = np.array(results[method].history['t']) / 60
            viol = np.array(results[method].history['HTO_violation'], dtype=float)
            ax7.fill_between(t, 0, viol, alpha=0.3, color=colors[method], label=f'{labels[method]} Violations')
        ax7.set_title('HTO Constraint Violations', fontweight='bold')
        ax7.set_ylabel('Violation')
        ax7.set_xlabel('Time (min)')
        ax7.legend()
        ax7.grid(True)

        # Constraint violations统计柱状图
        ax8 = fig.add_subplot(gs[4, 0])
        methods = ['raw', 'safe', 'cbf']
        violation_types = ['HTO', 'T', 'U', 'P']
        x = np.arange(len(methods))
        width = 0.2

        for i, vtype in enumerate(violation_types):
            counts = [results[m].violations[vtype] for m in methods]
            ax8.bar(x + i*width, counts, width, label=vtype)

        ax8.set_title('Violation Count by Type', fontweight='bold')
        ax8.set_ylabel('Count')
        ax8.set_xticks(x + width * 1.5)
        ax8.set_xticklabels([m.upper() for m in methods])
        ax8.legend()
        ax8.grid(True, axis='y')

        # Projection difference (safe & cbf only)
        ax9 = fig.add_subplot(gs[4, 1])
        for method in ['safe', 'cbf']:
            if 'projection_diffs' in results[method].history:
                diffs = results[method].history['projection_diffs']
                if diffs['t']:
                    t = np.array(diffs['t']) / 60
                    ax9.plot(t, diffs['norm'], color=colors[method],
                            label=f'{labels[method]} Adjustment', linewidth=2)
        ax9.set_title('Projection Adjustment Magnitude', fontweight='bold')
        ax9.set_ylabel('||u_safe - u_raw||')
        ax9.set_xlabel('Time (min)')
        ax9.legend()
        ax9.grid(True)

        plt.suptitle(f'Three-Way Comparison: {test_name}', fontsize=16, fontweight='bold', y=0.995)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"comparison_{test_name}_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Comparison figure saved: {filepath}")
        return filepath

    def generate_summary_report(self, all_results: List[Dict]):
        """Generate summary report"""
        report_path = os.path.join(self.output_dir, 'summary_report.txt')

        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("Single-Stack Safety Test Summary Report\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")

            f.write("Safety Configuration:\n")
            f.write(f"  Current limits: [{self.config.I_min}, {self.config.I_max}] A\n")
            f.write(f"  Lye flow limits: [{self.config.v_lye_min}, {self.config.v_lye_max}] m³/s\n")
            f.write(f"  Temperature limits: [{self.config.T_min-273.15}, {self.config.T_max-273.15}] degC\n")
            f.write(f"  Voltage limit: < {self.config.U_cell_max} V\n")
            f.write(f"  HTO limit: < {self.config.HTO_max} %\n")
            f.write(f"  Power limit: < {self.config.P_stack_max/1e6} MW\n")
            f.write(f"  CBF gain: γ = {self.config.cbf_gamma}\n\n")

            for i, result_set in enumerate(all_results):
                f.write(f"\nTest scenario {i+1}:\n")
                f.write("-"*80 + "\n")

                if isinstance(result_set, dict):
                    # 三MethodComparison结果
                    f.write(f"{'Method':<15} {'HTO violations':<10} {'Temp violations':<10} {'Voltage violations':<10} {'Power violations':<10} {'Max HTO':<12}\n")
                    for method in ['raw', 'safe', 'cbf']:
                        if method in result_set:
                            r = result_set[method]
                            f.write(f"{method.upper():<15} {r.violations['HTO']:<10} {r.violations['T']:<10} "
                                   f"{r.violations['U']:<10} {r.violations['P']:<10} {r.max_values['HTO']:<12.4f}\n")
                else:
                    # 单结果
                    r = result_set
                    f.write(f"Method: {r.method.upper()}\n")
                    f.write(f"  Constraint violations统计:\n")
                    for vtype, count in r.violations.items():
                        if count > 0:
                            f.write(f"    {vtype}: {count} times\n")
                    f.write(f"  Maximum recorded values:\n")
                    for mtype, value in r.max_values.items():
                        f.write(f"    {mtype}: {value:.4f}\n")

        print(f"\nSummary report saved: {report_path}")


# ==================== 预定义Test scenario ====================

def get_test_scenario(scenario_name: str) -> Tuple[str, List[Tuple], np.ndarray, float]:
    """
    获取预定义Test scenario

    Returns:
        (test_name, actions, initial_state, duration)
    """
    # 默认Initial state
    default_state = np.array([345.0, 358.0, 358.0, 325.0, 0.52, 0.52, 0.52])

    scenarios = {
        'extreme_high_current': (
            "极端高电流",
            [(0, 300, [9000, 0.01, 0.1]), (300, 3600, [2000, 0.03, 0.1])],
            default_state, 3600
        ),
        'zero_lye_flow': (
            "零碱液流速",
            [(0, 600, [5000, 0.0, 0.5]), (600, 3600, [3000, 0.03, 0.5])],
            default_state, 3600
        ),
        'extreme_lye_flow': (
            "极高碱液流速",
            [(0, 600, [3000, 0.15, 0.3]), (600, 3600, [3000, 0.03, 0.3])],
            default_state, 3600
        ),
        'combined_extreme': (
            "Combined Extreme",
            [(0, 200, [9500, 0.0, 0.0]), (200, 600, [8000, 0.005, 0.05]), (600, 3600, [3000, 0.03, 0.3])],
            default_state, 3600
        ),
        'sustained_high_power': (
            "持续高功率",
            [(0, 3600, [6500, 0.02, 0.3])],
            default_state, 3600
        ),
        'hto_low_power': (
            "HTO Low Power Risk",
            [(0, 1800, [800, 0.01, 0.0]), (1800, 3600, [500, 0.005, 0.0]),
             (3600, 5400, [2000, 0.02, 0.1]), (5400, 7200, [3000, 0.03, 0.2])],
            default_state, 7200
        ),
        'hto_very_low': (
            "HTO Very Low Power",
            [(0, 3600, [600, 0.002, 0.0]), (3600, 7200, [3000, 0.03, 0.2])],
            default_state, 7200
        ),
        'power_fluctuation': (
            "Power Fluctuation",
            [(0, 300, [400, 0.005, 0.0]), (300, 600, [2000, 0.02, 0.1]),
             (600, 900, [400, 0.005, 0.0]), (900, 1200, [2000, 0.02, 0.1]),
             (1200, 3600, [3000, 0.03, 0.2])],
            default_state, 3600
        ),
    }

    return scenarios.get(scenario_name, scenarios['combined_extreme'])


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Unified Safety Test for Single-Stack System')
    parser.add_argument('--test', type=str, default='all',
                       choices=['all', 'openloop', 'safe', 'cbf', 'hto', 'compare'],
                       help='Select test type')
    parser.add_argument('--scenario', type=str, default='combined_extreme',
                       help='Test scenario名称')
    parser.add_argument('--gamma', type=float, default=1.0,
                       help='CBF gain参数')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory')

    args = parser.parse_args()

    # 初始化配置
    config = SafetyConfig(cbf_gamma=args.gamma)
    tester = UnifiedSafetyTester(config=config, output_dir=args.output)

    print("="*80)
    print("Unified Safety Test for Single-Stack System")
    print("="*80)
    print(f"Test type: {args.test}")
    print(f"CBF gain: γ = {args.gamma}")
    print(f"Output directory: {tester.output_dir}")
    print("="*80)

    all_results = []

    if args.test in ['all', 'openloop']:
        # 运行所有Open-loop test场景
        for scenario in ['extreme_high_current', 'zero_lye_flow', 'extreme_lye_flow', 'combined_extreme']:
            name, actions, init_state, duration = get_test_scenario(scenario)
            result = tester.run_open_loop_test(name, actions, init_state, duration)
            all_results.append(result)
            tester.plot_single_result(result, 'openloop')

    if args.test in ['all', 'safe']:
        # Safe Projection test
        for scenario in ['extreme_high_current', 'combined_extreme']:
            name, actions, init_state, duration = get_test_scenario(scenario)
            result = tester.run_projection_test(name, actions, 'safe', init_state, duration)
            all_results.append(result)
            tester.plot_single_result(result, 'safe')

    if args.test in ['all', 'cbf']:
        # CBF Projection test
        for scenario in ['extreme_high_current', 'combined_extreme']:
            name, actions, init_state, duration = get_test_scenario(scenario)
            result = tester.run_projection_test(name, actions, 'cbf', init_state, duration)
            all_results.append(result)
            tester.plot_single_result(result, 'cbf')

    if args.test in ['all', 'hto']:
        # HTO specific test
        for scenario in ['hto_low_power', 'hto_very_low', 'power_fluctuation']:
            name, actions, init_state, duration = get_test_scenario(scenario)
            results = tester.run_hto_specific_test(name, actions, init_state, duration)
            all_results.append(results)
            tester.plot_three_way_comparison(results, name)

    if args.test in ['all', 'compare']:
        # Three-Way Method Comparison Test
        for scenario in ['combined_extreme', 'hto_low_power']:
            name, actions, init_state, duration = get_test_scenario(scenario)
            results = tester.run_three_way_comparison(name, actions, init_state, duration)
            all_results.append(results)
            tester.plot_three_way_comparison(results, name)

    # Generate summary report
    tester.generate_summary_report(all_results)

    print("\n" + "="*80)
    print("All tests completed!")
    print(f"Results saved to: {tester.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
