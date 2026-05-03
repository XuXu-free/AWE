"""
诊断温度 CBF 触发阈值

测试不同温度/电流组合下，温度 CBF 是否触发。
目标：85C 以下、电流 3000~7500A 不触发温度 CBF。
"""

import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from controller.multi_stack.cbf_projection_ho import MultiStackCBFProjectionHO
from plant.multi_stack_simulator import MultiStackSimulator


def build_state(T_s_degC, T_s_in_degC=75.0, T_sep_degC=80.0, T_c_out_degC=20.0):
    """构建一个典型状态，仅改变温度。"""
    sim = MultiStackSimulator(dt=0.2)
    sim.reset()
    u_low = np.array([500.0] * 4 + [0.05] * 4 + [0.01])
    for _ in range(5000):
        sim.step(u_low)

    state = sim.state.copy()
    T_s_k = T_s_degC + 273.15
    T_s_in_k = T_s_in_degC + 273.15
    T_sep_k = T_sep_degC + 273.15
    T_c_out_k = T_c_out_degC + 273.15

    state[0] = T_s_in_k
    state[1:5] = T_s_k
    state[5] = T_sep_k
    state[6] = T_c_out_k
    return state


def test_trigger(projector, state, I_val, v_lye_val=0.05, v_c_val=0.01):
    """测试给定状态下，参考控制是否触发温度 CBF。"""
    u_ref = np.array([I_val] * 4 + [v_lye_val] * 4 + [v_c_val])
    u_safe, success, info = projector.project(u_ref, state, u_last=u_ref)

    # 温度约束索引 0-3 (Tmax)
    cbf_temp = info['cbf'][0:4]
    h_temp = info['h'][0:4]
    cbf_hto = info['cbf'][4]

    # 判断温度 CBF 是否触发：温度 CBF 值 < 0 才表示触发
    temp_triggered = np.any(cbf_temp < -1e-4)
    hto_triggered = cbf_hto < -1e-4

    # 控制量变化
    dI = u_safe[0:4] - u_ref[0:4]
    dv_c = u_safe[8] - u_ref[8]

    return {
        'I': I_val,
        'T_s': state[1] - 273.15,
        'cbf_temp_min': np.min(cbf_temp),
        'cbf_hto': cbf_hto,
        'h_temp_min': np.min(h_temp),
        'temp_triggered': temp_triggered,
        'hto_triggered': hto_triggered,
        'dI_max': np.max(np.abs(dI)),
        'dv_c': dv_c,
        'u_safe': u_safe,
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

    print("=" * 80)
    print("温度 CBF 触发诊断")
    print("目标：85C 以下、3000~7500A 不触发温度 CBF")
    print("=" * 80)

    temps = [80.0, 82.0, 85.0, 87.0, 89.0]
    currents = [3000.0, 5000.0, 7500.0]

    results = []
    for T in temps:
        state = build_state(T)
        for I in currents:
            r = test_trigger(projector, state, I)
            results.append(r)

    # 打印表格
    print(f"\n{'T (C)':<8} {'I (A)':<8} {'h (K)':<10} {'CBF_temp':<12} {'CBF_hto':<12} {'TempTrig?':<10} {'HtoTrig?':<10} {'dI_max':<10}")
    print("-" * 90)
    for r in results:
        print(f"{r['T_s']:<8.1f} {r['I']:<8.0f} {r['h_temp_min']:<10.3f} {r['cbf_temp_min']:<12.4f} {r['cbf_hto']:<12.4f} "
              f"{'YES' if r['temp_triggered'] else 'NO':<10} {'YES' if r['hto_triggered'] else 'NO':<10} {r['dI_max']:<10.2f}")

    # 汇总
    print("\n" + "=" * 80)
    temp_trig_below = [r for r in results if r['T_s'] < 85.0 and r['temp_triggered']]
    temp_trig_above = [r for r in results if r['T_s'] >= 85.0 and r['temp_triggered']]
    hto_trig_all = [r for r in results if r['hto_triggered']]

    print(f"HTO CBF 触发次数: {len(hto_trig_all)} / {len(results)}")
    print(f"温度 CBF 触发 (<85C): {len(temp_trig_below)}")
    print(f"温度 CBF 触发 (>=85C): {len(temp_trig_above)}")

    if temp_trig_below:
        print("[WARNING] 85C 以下存在温度 CBF 触发点！")
        for r in temp_trig_below:
            print(f"    T={r['T_s']:.1f}C, I={r['I']:.0f}A, CBF_temp={r['cbf_temp_min']:.4f}")
    else:
        print("[OK] 85C 以下、3000~7500A 范围内温度 CBF 未触发")

    if temp_trig_above:
        print("[INFO] 85C 以上温度 CBF 触发（预期行为）")
    else:
        print("[INFO] 85C 以上温度 CBF 也未触发")

    print("=" * 80)

    # 额外：打印温度 CBF 值范围
    cbf_temps = [r['cbf_temp_min'] for r in results]
    print(f"\n温度 CBF 值范围: [{min(cbf_temps):.4f}, {max(cbf_temps):.4f}]")
    print("(值 > 0 表示温度条件满足，值 < 0 表示触发)")


if __name__ == "__main__":
    main()
