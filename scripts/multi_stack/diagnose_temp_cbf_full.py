"""
全面诊断温度 CBF 触发阈值

测试不同温度/电流/碱液/冷凝水组合下，温度 CBF 是否触发。
目标：85C 以下、电流 7500A、约束范围内碱液和冷凝水，不触发温度 CBF。
"""

import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from controller.multi_stack.cbf_projection_ho import MultiStackCBFProjectionHO
from plant.multi_stack_simulator import MultiStackSimulator


def build_state(T_s_degC, T_s_in_degC=75.0, T_sep_degC=80.0, T_c_out_degC=20.0):
    sim = MultiStackSimulator(dt=0.2)
    sim.reset()
    u_low = np.array([500.0] * 4 + [0.05] * 4 + [0.01])
    for _ in range(5000):
        sim.step(u_low)

    state = sim.state.copy()
    state[0] = T_s_in_degC + 273.15
    state[1:5] = T_s_degC + 273.15
    state[5] = T_sep_degC + 273.15
    state[6] = T_c_out_degC + 273.15
    return state


def test(projector, state, I_val, v_lye_val, v_c_val):
    u_ref = np.array([I_val] * 4 + [v_lye_val] * 4 + [v_c_val])
    u_safe, success, info = projector.project(u_ref, state, u_last=u_ref)

    cbf_temp = info['cbf'][0:4]
    cbf_hto = info['cbf'][4]
    temp_triggered = np.any(cbf_temp < -1e-4)
    hto_triggered = cbf_hto < -1e-4

    return {
        'T_s': state[1] - 273.15,
        'I': I_val,
        'v_lye': v_lye_val,
        'v_c': v_c_val,
        'cbf_temp_min': np.min(cbf_temp),
        'cbf_hto': cbf_hto,
        'temp_triggered': temp_triggered,
        'hto_triggered': hto_triggered,
        'dI_max': np.max(np.abs(u_safe[0:4] - u_ref[0:4])),
    }


def main():
    projector = MultiStackCBFProjectionHO(
        dt=60.0,
        gamma_vec=[10.0]*4 + [1.0] + [10.0]*4,
        rho_vec=[5000]*9,
        h_margin_vec=[0.0]*4 + [0.0002] + [0.0]*4,
        normalize=True,
        lambda_u_scale=0.0,
        alpha1_hto=2.0,
        alpha2_hto=5.0,
    )

    print("=" * 100)
    print("温度 CBF 全面诊断")
    print("目标：85C 以下、7500A、碱液和冷凝水在约束范围内，不触发温度 CBF")
    print("=" * 100)

    temps = [80.0, 82.0, 85.0]
    currents = [7500.0]
    v_lye_vals = [0.01, 0.03, 0.05, 0.07, 0.10]
    v_c_vals = [0.0, 0.1, 0.3, 0.5, 1.0]

    results = []
    for T in temps:
        state = build_state(T)
        for I in currents:
            for v_lye in v_lye_vals:
                for v_c in v_c_vals:
                    r = test(projector, state, I, v_lye, v_c)
                    results.append(r)

    # 打印表格
    print(f"\n{'T(C)':<6} {'I(A)':<6} {'v_lye':<7} {'v_c':<6} {'CBF_temp':<10} {'CBF_hto':<10} {'TempTrig':<9} {'HtoTrig':<9} {'dI_max':<8}")
    print("-" * 90)
    for r in results:
        print(f"{r['T_s']:<6.0f} {r['I']:<6.0f} {r['v_lye']:<7.2f} {r['v_c']:<6.1f} {r['cbf_temp_min']:<10.4f} {r['cbf_hto']:<10.4f} "
              f"{'YES' if r['temp_triggered'] else 'NO':<9} {'YES' if r['hto_triggered'] else 'NO':<9} {r['dI_max']:<8.1f}")

    # 汇总
    print("\n" + "=" * 100)
    temp_trig_below_85 = [r for r in results if r['T_s'] < 85.0 and r['temp_triggered']]
    temp_trig_at_85 = [r for r in results if r['T_s'] == 85.0 and r['temp_triggered']]

    print(f"总测试点数: {len(results)}")
    print(f"85C 以下温度 CBF 触发: {len(temp_trig_below_85)}")
    print(f"85C 温度 CBF 触发: {len(temp_trig_at_85)}")

    if temp_trig_below_85:
        print("[WARNING] 85C 以下存在温度 CBF 触发点！")
        for r in temp_trig_below_85:
            print(f"    T={r['T_s']:.0f}C, I={r['I']:.0f}A, v_lye={r['v_lye']:.2f}, v_c={r['v_c']:.1f}, CBF_temp={r['cbf_temp_min']:.4f}")
    else:
        print("[OK] 85C 以下、7500A、全约束范围内碱液/冷凝水，温度 CBF 未触发")

    if temp_trig_at_85:
        print("[INFO] 85C 时温度 CBF 触发")
    else:
        print("[INFO] 85C 时温度 CBF 也未触发")

    cbf_temps = [r['cbf_temp_min'] for r in results]
    print(f"\n温度 CBF 值范围: [{min(cbf_temps):.4f}, {max(cbf_temps):.4f}]")
    print("=" * 100)


if __name__ == "__main__":
    main()
