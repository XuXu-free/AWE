"""
绘制电解槽电堆效率图 (修正版本 - 使用真实极化曲线模型)
基于公式: η_stack = (η_cell * LHV) / (2 * F * U_cell)
其中 U_cell 使用系统实际的极化曲线模型
"""
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 物理常数
F = 96485.0
LHV_MJ_per_kg = 120.1
M_H2 = 2.016
LHV_J_per_mol = LHV_MJ_per_kg * 1e6 * M_H2 / 1000

# 电堆参数
N_cell = 180
P_sys = 30e5  # 系统压力 30 bar
s = 0.185  # Tafel斜率

# 极化曲线参数 (来自 multi_stack_simulator.py)
r1, r2, r3 = 3.202e-5, 8.970e-8, -4.193e-12
t1, t2, t3 = -1.070e-1, 14.43, 38.8


def calculate_faraday_efficiency(I, T_celsius):
    """计算法拉第效率"""
    f1 = 50.0 + 2.5 * T_celsius
    f2 = 0.92 - 6.25e-6 * T_celsius
    I_sq = (0.1 * I) ** 2
    eta_cell = (I_sq / (f1 + I_sq)) * f2
    return eta_cell


def calculate_cell_voltage_real(I, T_celsius):
    """
    计算电解小室电压 U_cell (使用系统真实模型)
    U_cell = U_rev + V_ohm + V_act
    """
    U_rev = 1.229  # 可逆电压 (V)
    T_K = T_celsius + 273.15

    # 1. 欧姆过电位: V_ohm = (r1 + r2*T + r3*P) * I
    R_ohm = r1 + r2 * T_K + r3 * P_sys
    V_ohm = R_ohm * I

    # 2. 活化过电位: V_act = s * ln((t1 + t2/T_C + t3/T_C^2) * I + 1)
    term_act = t1 + t2 / T_celsius + t3 / (T_celsius ** 2)
    arg = term_act * I + 1
    arg_safe = max(arg, 1e-9)
    V_act = s * np.log(arg_safe)

    U_cell = U_rev + V_ohm + V_act
    return max(U_cell, 1.48)


def calculate_stack_efficiency(I, T_celsius, use_lhv=True):
    """计算电堆效率"""
    eta_cell = calculate_faraday_efficiency(I, T_celsius)
    U_cell = calculate_cell_voltage_real(I, T_celsius)

    if use_lhv:
        energy_content = LHV_J_per_mol
    else:
        energy_content = 141.9e6 * M_H2 / 1000

    eta_stack = (eta_cell * energy_content) / (2 * F * U_cell)
    return eta_stack, eta_cell, U_cell


def calculate_h2_production_rate(I, T_celsius):
    """计算产氢率"""
    eta_cell = calculate_faraday_efficiency(I, T_celsius)
    n_dot_H2 = eta_cell * N_cell * I / (2 * F)
    return n_dot_H2


def calculate_power_input(I, T_celsius):
    """计算电堆输入功率"""
    U_cell = calculate_cell_voltage_real(I, T_celsius)
    P_stack = N_cell * U_cell * I
    return P_stack


# 生成电流范围 (0-10000A)
I_range = np.linspace(0, 10000, 500)

# 不同温度
temperatures = [60, 70, 80, 90]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# 创建图形
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 图1: 电堆效率 vs 电流
ax1 = axes[0, 0]
for T_c, color in zip(temperatures, colors):
    eta_stack_values = []
    for I in I_range:
        eta_stack, _, _ = calculate_stack_efficiency(I, T_c)
        eta_stack_values.append(eta_stack * 100)
    ax1.plot(I_range, eta_stack_values, color=color, linewidth=2.5, label=f'T = {T_c}C')

ax1.set_xlabel('Current (A)', fontsize=11)
ax1.set_ylabel('Stack Efficiency (%)', fontsize=11)
ax1.set_title('Stack Efficiency vs Current (LHV)', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 10000)
ax1.set_ylim(0, 100)

# 图2: 法拉第效率
ax2 = axes[0, 1]
for T_c, color in zip(temperatures, colors):
    eta_cell_values = [calculate_faraday_efficiency(I, T_c) * 100 for I in I_range]
    ax2.plot(I_range, eta_cell_values, color=color, linewidth=2, label=f'{T_c}C')

ax2.set_xlabel('Current (A)', fontsize=11)
ax2.set_ylabel('Faraday Efficiency (%)', fontsize=11)
ax2.set_title('Faraday Efficiency', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 10000)
ax2.set_ylim(0, 100)

# 图3: 电解小室电压 (真实模型)
ax3 = axes[0, 2]
for T_c, color in zip(temperatures, colors):
    U_cell_values = [calculate_cell_voltage_real(I, T_c) for I in I_range]
    ax3.plot(I_range, U_cell_values, color=color, linewidth=2, label=f'{T_c}C')

ax3.set_xlabel('Current (A)', fontsize=11)
ax3.set_ylabel('Cell Voltage U_cell (V)', fontsize=11)
ax3.set_title('Cell Voltage (Real Polarization Model)', fontsize=12)
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 10000)
ax3.set_ylim(1.5, 3.5)
ax3.axhline(y=1.48, color='gray', linestyle='--', alpha=0.5, label='Thermal neutral')

# 图4: 产氢率
ax4 = axes[1, 0]
for T_c, color in zip(temperatures, colors):
    h2_rate_values = [calculate_h2_production_rate(I, T_c) for I in I_range]
    ax4.plot(I_range, h2_rate_values, color=color, linewidth=2, label=f'{T_c}C')

ax4.set_xlabel('Current (A)', fontsize=11)
ax4.set_ylabel('H2 Production Rate (mol/s)', fontsize=11)
ax4.set_title('H2 Production Rate', fontsize=12)
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, 10000)
ax4.set_ylim(0, 8)

# 图5: 电堆功率
ax5 = axes[1, 1]
for T_c, color in zip(temperatures, colors):
    P_values = [calculate_power_input(I, T_c) / 1e6 for I in I_range]
    ax5.plot(I_range, P_values, color=color, linewidth=2, label=f'{T_c}C')

ax5.set_xlabel('Current (A)', fontsize=11)
ax5.set_ylabel('Stack Power (MW)', fontsize=11)
ax5.set_title('Stack Power', fontsize=12)
ax5.legend()
ax5.grid(True, alpha=0.3)
ax5.set_xlim(0, 10000)
ax5.set_ylim(0, 5)

# 图6: 效率分解 (80C)
ax6 = axes[1, 2]
T_ref = 80
eta_stack_80 = []
eta_cell_80 = []
voltage_eff = []

for I in I_range:
    eta_s, eta_c, U_c = calculate_stack_efficiency(I, T_ref)
    eta_stack_80.append(eta_s * 100)
    eta_cell_80.append(eta_c * 100)
    voltage_eff.append(1.48 / U_c * 100)

ax6.plot(I_range, eta_stack_80, 'b-', linewidth=2.5, label='Stack Efficiency')
ax6.plot(I_range, eta_cell_80, 'g--', linewidth=2, label='Faraday Efficiency')
ax6.plot(I_range, voltage_eff, 'r:', linewidth=2, label='Voltage Efficiency (1.48/U_cell)')
ax6.axhline(y=100, color='gray', linestyle='--', alpha=0.3)

ax6.set_xlabel('Current (A)', fontsize=11)
ax6.set_ylabel('Efficiency (%)', fontsize=11)
ax6.set_title('Efficiency Breakdown (T=80C)', fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)
ax6.set_xlim(0, 10000)
ax6.set_ylim(0, 100)

plt.suptitle('Electrolysis Stack Efficiency (Real Polarization Model)', fontsize=14, fontweight='bold')
plt.tight_layout()

# 保存
output_file = os.path.join(os.path.dirname(__file__), 'stack_efficiency_real_model.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Plot saved: {output_file}")

# 打印数据表
print("\n" + "="*90)
print("Stack Efficiency (LHV basis, Real Polarization Model)")
print("="*90)
print(f"{'Current (A)':<12} {'Faraday(%)':<12} {'U_cell(V)':<12} {'V_ohm(V)':<12} {'V_act(V)':<12} {'H2(mol/s)':<12} {'Power(MW)':<12} {'Stack(%)':<12}")
print("-"*90)

test_currents = [1000, 2000, 3000, 4000, 5000, 6000, 8000]
for I in test_currents:
    T_c = 80
    eta_cell = calculate_faraday_efficiency(I, T_c) * 100
    U_cell = calculate_cell_voltage_real(I, T_c)

    # 计算各分量
    T_K = T_c + 273.15
    R_ohm = r1 + r2 * T_K + r3 * P_sys
    V_ohm = R_ohm * I
    term_act = t1 + t2 / T_c + t3 / (T_c ** 2)
    V_act = s * np.log(term_act * I + 1)

    h2_rate = calculate_h2_production_rate(I, T_c)
    P_stack = calculate_power_input(I, T_c) / 1e6
    eta_stack, _, _ = calculate_stack_efficiency(I, T_c)

    print(f"{I:<12} {eta_cell:<12.2f} {U_cell:<12.3f} {V_ohm:<12.4f} {V_act:<12.4f} {h2_rate:<12.3f} {P_stack:<12.3f} {eta_stack*100:<12.2f}")

print("\n" + "="*90)
print("Cell Voltage Model: U_cell = U_rev + V_ohm + V_act")
print("  U_rev = 1.229 V")
print("  V_ohm = (r1 + r2*T + r3*P) * I")
print("  V_act = s * ln((t1 + t2/T + t3/T^2) * I + 1)")
print("="*90)

plt.show()
