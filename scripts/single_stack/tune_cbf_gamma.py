"""
CBF Gamma Parameter Tuning Script
Finds suitable gamma values that prevent constraint violations
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from controller.single_stack.cbf_projection import SingleStackCBFProjection
from plant.single_stack_simulator import SingleStackSimulator


def run_simulation(projector, duration=14400, u_ref=None):
    """Run simulation with given CBF projector"""
    sim = SingleStackSimulator(sim_dt=0.2)
    sim.reset()

    dt = 0.2
    steps = int(duration / dt)
    ctrl_steps = int(60.0 / dt)

    if u_ref is None:
        u_ref = np.array([600.0, 0.005, 0.0])
    current_action = u_ref.copy()

    max_hto = 0
    max_temp = 0
    max_I = 0
    n_failures = 0

    for i in range(steps):
        t = i * dt

        if i % ctrl_steps == 0:
            state = sim.state
            current_action, success, info = projector.project(u_ref, state)
            if not success:
                n_failures += 1

        state = sim.state
        n_gas = state[6]
        T_sep = state[2]
        hto = (n_gas * projector.R * T_sep) / (projector.P_sys * projector.V_sep_gas)

        max_hto = max(max_hto, hto * 100)
        max_temp = max(max_temp, state[1] - 273.15)
        max_I = max(max_I, current_action[0])

        sim.step(current_action)

    return {
        'max_hto': max_hto,
        'max_temp': max_temp,
        'max_I': max_I,
        'failure_rate': n_failures / (steps / ctrl_steps) * 100
    }


def test_gamma_combination(gamma_vec, duration=7200):
    """Test a specific gamma combination"""
    u_ref = np.array([600.0, 0.005, 0.0])

    projector = SingleStackCBFProjection(
        dt=60.0,
        gamma_vec=gamma_vec,
        rho_vec=[1e4, 1e10, 1e4, 1e4, 1e4],
        h_margin_vec=[0.0, 0.002, 0.0, 0.0, 0.0],
        normalize=True
    )

    results = run_simulation(projector, duration=duration, u_ref=u_ref)
    results['gamma_vec'] = gamma_vec
    return results


def main():
    print("="*70)
    print("CBF Gamma Parameter Tuning")
    print("="*70)

    # Test different gamma combinations
    # Format: [gamma_T, gamma_HTO, gamma_V, gamma_P, gamma_Tmin]

    gamma_candidates = [
        # Baseline
        [1.0, 5.0, 1.0, 1.0, 1.0],
        # Very high gamma values
        [50.0, 50.0, 5.0, 5.0, 5.0],
        [100.0, 50.0, 5.0, 5.0, 5.0],
        [50.0, 100.0, 5.0, 5.0, 5.0],
        [100.0, 100.0, 5.0, 5.0, 5.0],
        [200.0, 100.0, 10.0, 10.0, 10.0],
        [100.0, 200.0, 10.0, 10.0, 10.0],
        [200.0, 200.0, 10.0, 10.0, 10.0],
        # Extreme
        [500.0, 500.0, 50.0, 50.0, 50.0],
    ]

    duration = 7200  # 2 hours for faster testing
    results = []

    print(f"\nTesting {len(gamma_candidates)} gamma combinations (duration={duration}s)...")
    print("-"*70)
    print(f"{'gamma_T':>8} {'gamma_HTO':>10} {'gamma_V':>8} {'Max HTO':>10} {'Max Temp':>10} {'Max I':>10} {'Fail%':>8}")
    print("-"*70)

    for gamma_vec in gamma_candidates:
        result = test_gamma_combination(gamma_vec, duration)
        results.append(result)

        print(f"{gamma_vec[0]:>8.1f} {gamma_vec[1]:>10.1f} {gamma_vec[2]:>8.1f} "
              f"{result['max_hto']:>9.3f}% {result['max_temp']:>9.2f}C {result['max_I']:>9.0f}A {result['failure_rate']:>7.1f}%")

    # Find best result
    print("-"*70)
    print("\nConstraint Satisfaction Check:")

    valid_results = [r for r in results if r['max_hto'] <= 2.0 and r['max_temp'] <= 90.0]

    if valid_results:
        # Prefer lower max_I (less aggressive control)
        best = min(valid_results, key=lambda x: x['max_I'])
        print(f"\n[OK] Best gamma combination found:")
        print(f"  gamma_vec = {best['gamma_vec']}")
        print(f"  Max HTO: {best['max_hto']:.3f}%")
        print(f"  Max Temp: {best['max_temp']:.2f}°C")
        print(f"  Max Current: {best['max_I']:.0f}A")
        print(f"  Failure Rate: {best['failure_rate']:.1f}%")
    else:
        print("\n[FAIL] No gamma combination fully satisfied constraints.")
        print("  Best attempt (closest to constraints):")
        # Find result with minimum constraint violation
        best = min(results, key=lambda x: max(0, x['max_hto']-2.0) + max(0, x['max_temp']-90.0)/10)
        print(f"  gamma_vec = {best['gamma_vec']}")
        print(f"  Max HTO: {best['max_hto']:.3f}%")
        print(f"  Max Temp: {best['max_temp']:.2f}°C")

    # Run full 4h test with best parameters
    if valid_results:
        print(f"\nRunning 4-hour validation test with best parameters...")
        best_result = test_gamma_combination(best['gamma_vec'], duration=14400)
        print(f"4-Hour Results:")
        hto_ok = "OK" if best_result['max_hto'] <= 2.0 else "FAIL"
        temp_ok = "OK" if best_result['max_temp'] <= 90.0 else "FAIL"
        print(f"  Max HTO: {best_result['max_hto']:.3f}% [{hto_ok}]")
        print(f"  Max Temp: {best_result['max_temp']:.2f}°C [{temp_ok}]")
        print(f"  Max Current: {best_result['max_I']:.0f}A")
        print(f"  Failure Rate: {best_result['failure_rate']:.1f}%")

    print("="*70)


if __name__ == "__main__":
    main()
