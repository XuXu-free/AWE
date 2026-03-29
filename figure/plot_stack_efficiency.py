"""
绘制电解槽电堆效率图
基于公式: η_stack = (η_cell * LHV) / (2 * F * U_cell)
其中:
- η_cell: 法拉第效率 (Faraday efficiency)
- LHV: 氢气低热值 (120.1 MJ/kg 或 242 kJ/mol)
- F: 法拉第常数 (96485 C/mol)
- U_cell: 电解小室电压 (V)
- N_cell: 电堆小室数
"""
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 物理常数
F = 96485.0  # 法拉第常数 (C/mol)
LHV_MJ_per_kg = 120.1  # 低热值 MJ/kg
M_H2 = 2.016  # 氢气摩尔质量 (g/mol)
LHV_J_per_mol = LHV_MJ_per_kg * 1e6 * M_H2 / 1000  # 转换为 J/mol = 242.0 kJ/mol

# 电堆参数
N_cell = 180  # 电解小室数


def calculate_faraday_efficiency(I, T_celsius):
    """计算法拉第效率 η_cell"""
    f1 = 50.0 + 2.5 * T_celsius
    f2 = 0.92 - 6.25e-6 * T_celsius
    I_sq = I ** 2
    eta_cell = (I_sq / (f1 + I_sq)) * f2
    return eta_cell


def calculate_cell_voltage(I, T_celsius, T_ref=80):
    """
    计算电解小室电压 U_cell
    基于简化极化曲线模型
    """
    # 基础可逆电压 (随温度降低)
    U_rev = 1.23 - 0.00085 * (T_celsius - 25)

    # 活化过电位 (Tafel方程简化)
    I_ref = 1000  # 参考电流
    a = 0.15  # 活化过电位系数
    eta_act = a * np.log10(I / I_ref + 1)

    # 欧姆过电位
    R_ohm = 0.0002  # 等效电阻
    eta_ohm = R_ohm * I

    # 浓差过电位 (高电流时显著)
    eta_conc = 0.05 * np.log10(1 + I / 8000)

    U_cell = U_rev + eta_act + eta_ohm + eta_conc
    return max(U_cell, 1.48)  # 理论最小电压约1.48V (热中性电压)


def calculate_stack_efficiency(I, T_celsius, use_lhv=True):
    """
    计算电堆效率 η_stack
    公式: η_stack = (η_cell * LHV) / (2 * F * U_cell)
    """
    eta_cell = calculate_faraday_efficiency(I, T_celsius)
    U_cell = calculate_cell_voltage(I, T_celsius)

    if use_lhv:
        energy_content = LHV_J_per_mol  # 使用低热值 LHV
    else:
        # 高热值 HHV = 141.9 MJ/kg
        energy_content = 141.9e6 * M_H2 / 1000  # J/mol

    # 电堆效率 = 氢气能量输出 / 电能输入
    # = (η_cell * LHV) / (2 * F * U_cell)
    eta_stack = (eta_cell * energy_content) / (2 * F * U_cell)

    return eta_stack, eta_cell, U_cell


def calculate_h2_production_rate(I, T_celsius):
    """计算产氢率 (mol/s)"""
    eta_cell = calculate_faraday_efficiency(I, T_celsius)
    n_dot_H2 = eta_cell * N_cell * I / (2 * F)  # mol/s
    return n_dot_H2


def calculate_power_input(I, T_celsius):
    """计算电堆输入功率 (W)"""
    U_cell = calculate_cell_voltage(I, T_celsius)
    P_stack = N_cell * U_cell * I  # W
    return P_stack


# 生成电流范围
I_range = np.linspace(200, 9000, 500)

# 不同温度下的效率曲线
temperatures = [60, 70, 80, 90]  # °C
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
linestyles = ['-', '--', '-.', ':']

# 创建图形
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 图1: 电堆效率 vs 电流 (主图)
ax1 = fig.add_subplot(gs[0, :2])
for T_c, color, ls in zip(temperatures, colors, linestyles):
    eta_stack_values = []
    for I in I_range:
        eta_stack, _, _ = calculate_stack_efficiency(I, T_c, use_lhv=True)
        eta_stack_values.append(eta_stack * 100)  # 转换为百分比
    ax1.plot(I_range, eta_stack_values, color=color, linestyle=ls, linewidth=2.5,
             label=f'T = {T_c}°C')

ax1.set_xlabel('Current (A)', fontsize=12)
ax1.set_ylabel('Stack Efficiency η_stack (%)', fontsize=12)
ax1.set_title('Stack Efficiency vs Current (LHV basis)', fontsize=13, fontweight='bold')
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 9000)
ax1.set_ylim(55, 85)

# 添加公式文本
formula_text = r'$\eta_{stack} = \frac{\dot{n}_{H_2} \cdot LHV}{P_{stack}} = \frac{\eta_{cell} \cdot LHV}{2F \cdot U_{cell}}$'
ax1.text(0.02, 0.98, formula_text, transform=ax1.transAxes, fontsize=11,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 图2: 法拉第效率 vs 电流
ax2 = fig.add_subplot(gs[0, 2])
for T_c, color, ls in zip(temperatures, colors, linestyles):
    eta_cell_values = [calculate_faraday_efficiency(I, T_c) * 100 for I in I_range]
    ax2.plot(I_range, eta_cell_values, color=color, linestyle=ls, linewidth=2,
             label=f'{T_c}°C')

ax2.set_xlabel('Current (A)', fontsize=11)
ax2.set_ylabel('Faraday Efficiency η_cell (%)', fontsize=11)
ax2.set_title('Faraday Efficiency', fontsize=12)
ax2.legend(loc='lower right', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 9000)
ax2.set_ylim(90, 95)

# 图3: 电解小室电压 vs 电流
ax3 = fig.add_subplot(gs[1, 0])
for T_c, color, ls in zip(temperatures, colors, linestyles):
    U_cell_values = [calculate_cell_voltage(I, T_c) for I in I_range]
    ax3.plot(I_range, U_cell_values, color=color, linestyle=ls, linewidth=2,
             label=f'{T_c}°C')

ax3.set_xlabel('Current (A)', fontsize=11)
ax3.set_ylabel('Cell Voltage U_cell (V)', fontsize=11)
ax3.set_title('Cell Voltage vs Current', fontsize=12)
ax3.legend(loc='best', fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.axhline(y=1.48, color='gray', linestyle='--', alpha=0.5, label='Thermal neutral')

# 图4: 产氢率 vs 电流
ax4 = fig.add_subplot(gs[1, 1])
for T_c, color, ls in zip(temperatures, colors, linestyles):
    h2_rate_values = [calculate_h2_production_rate(I, T_c) for I in I_range]
    ax4.plot(I_range, h2_rate_values, color=color, linestyle=ls, linewidth=2,
             label=f'{T_c}°C')

ax4.set_xlabel('Current (A)', fontsize=11)
ax4.set_ylabel('H₂ Production Rate (mol/s)', fontsize=11)
ax4.set_title('H₂ Production Rate vs Current', fontsize=12)
ax4.legend(loc='best', fontsize=9)
ax4.grid(True, alpha=0.3)

# 图5: 电堆功率 vs 电流
ax5 = fig.add_subplot(gs[1, 2])
for T_c, color, ls in zip(temperatures, colors, linestyles):
    P_values = [calculate_power_input(I, T_c) / 1e6 for I in I_range]  # MW
    ax5.plot(I_range, P_values, color=color, linestyle=ls, linewidth=2,
             label=f'{T_c}°C')

ax5.set_xlabel('Current (A)', fontsize=11)
ax5.set_ylabel('Stack Power P_stack (MW)', fontsize=11)
ax5.set_title('Stack Power vs Current', fontsize=12)
ax5.legend(loc='best', fontsize=9)
ax5.grid(True, alpha=0.3)

# 图6: 效率分解分析 (80°C)
ax6 = fig.add_subplot(gs[2, :])
T_ref = 80
eta_cell_80 = [calculate_faraday_efficiency(I, T_ref) * 100 for I in I_range]
U_cell_80 = [calculate_cell_voltage(I, T_ref) for I in I_range]
eta_stack_80 = []
voltage_efficiency = []  # 电压效率 = 1.48 / U_cell

for I in I_range:
    eta_s, _, U_c = calculate_stack_efficiency(I, T_ref, use_lhv=True)
    eta_stack_80.append(eta_s * 100)
    voltage_efficiency.append(1.48 / U_c * 100)  # 热中性电压效率

ax6.plot(I_range, eta_stack_80, 'b-', linewidth=2.5, label='Stack Efficiency η_stack')
ax6.plot(I_range, eta_cell_80, 'g--', linewidth=2, label='Faraday Efficiency η_cell')
ax6.plot(I_range, voltage_efficiency, 'r:', linewidth=2, label='Voltage Efficiency (1.48/U_cell)')
ax6.axhline(y=100, color='gray', linestyle='--', alpha=0.5)

ax6.set_xlabel('Current (A)', fontsize=12)
ax6.set_ylabel('Efficiency (%)', fontsize=12)
ax6.set_title('Efficiency Breakdown at T = 80°C (LHV basis)', fontsize=13, fontweight='bold')
ax6.legend(loc='best', fontsize=10)
ax6.grid(True, alpha=0.3)
ax6.set_xlim(0, 9000)
ax6.set_ylim(50, 105)

plt.suptitle('Alkaline Water Electrolysis Stack Efficiency Analysis', fontsize=15, fontweight='bold', y=1.02)

# 保存图片
output_dir = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(output_dir, 'stack_efficiency_formula.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Stack efficiency plot saved to: {output_file}")

# 打印关键数据表
print("\n" + "="*100)
print("Stack Efficiency Analysis (LHV basis, T = 80°C)")
print("="*100)
print(f"{'Current (A)':<12} {'Faraday η_cell (%)':<20} {'U_cell (V)':<15} {'H2 Rate (mol/s)':<18} {'Power (MW)':<15} {'Stack η (%)':<15}")
print("-"*100)

test_currents = [500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
for I in test_currents:
    eta_cell = calculate_faraday_efficiency(I, 80) * 100
    U_cell = calculate_cell_voltage(I, 80)
    h2_rate = calculate_h2_production_rate(I, 80)
    P_stack = calculate_power_input(I, 80) / 1e6
    eta_stack, _, _ = calculate_stack_efficiency(I, 80, use_lhv=True)
    print(f"{I:<12} {eta_cell:<20.3f} {U_cell:<15.3f} {h2_rate:<18.3f} {P_stack:<15.3f} {eta_stack*100:<15.2f}")

print("\n" + "="*100)
print("Formula:")
print("  η_stack = (η_cell × LHV) / (2F × U_cell)")
print("  where:")
print("    η_cell = Faraday efficiency = I² / (f1 + I²) × f2")
print("    LHV    = Lower Heating Value = 242 kJ/mol")
print("    F      = Faraday constant = 96,485 C/mol")
print("    U_cell = Cell voltage (V)")
print("="*100)

plt.show()
