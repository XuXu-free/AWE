"""
绘制系统总效率和产氢率与电流的关系图
包括法拉第效率、系统效率和产氢率
"""
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 常数定义
F = 96485.0  # 法拉第常数 (C/mol)
N_cell = 180  # 电堆小室数
HHV = 141.9e6  # 氢气高热值 (J/kg)
LHV = 120.1e6  # 氢气低热值 (J/kg)
M_H2 = 2.016e-3  # 氢气摩尔质量 (kg/mol)


def calculate_faraday_efficiency(I, T_celsius):
    """计算法拉第效率"""
    f1 = 50.0 + 2.5 * T_celsius
    f2 = 0.92 - 6.25e-6 * T_celsius
    I_sq = I ** 2
    eta_F = (I_sq / (f1 + I_sq)) * f2
    return eta_F, f1, f2


def calculate_cell_voltage(I, T_celsius):
    """
    简化计算电解小室电压
    基于典型AWE极化曲线
    """
    # 简化模型: U = U_rev + a + b*log(I) + c*I
    U_rev = 1.23 - 0.00085 * (T_celsius - 25)  # 可逆电压随温度变化
    a = 0.1  # 活化过电位
    b = 0.05  # Tafel斜率相关
    c = 1.5e-5  # 欧姆过电位系数

    if I < 100:
        I = 100

    U_cell = U_rev + a + b * np.log10(I/1000) + c * I
    return max(U_cell, 1.5)  # 最小电压1.5V


def calculate_h2_production_rate(I, T_celsius):
    """计算产氢率 (mol/s)"""
    eta_F, _, _ = calculate_faraday_efficiency(I, T_celsius)
    n_dot_H2 = eta_F * N_cell * I / (2 * F)
    return n_dot_H2


def calculate_system_efficiency(I, T_celsius, use_hhv=True):
    """计算系统总效率"""
    eta_F, _, _ = calculate_faraday_efficiency(I, T_celsius)
    U_cell = calculate_cell_voltage(I, T_celsius)

    # 能量输出 = 氢气热值 * 摩尔流量
    # 能量输入 = 电功率

    if use_hhv:
        energy_value = HHV * M_H2  # 每摩尔氢气的高热值 (J/mol)
    else:
        energy_value = LHV * M_H2  # 每摩尔氢气的低热值 (J/mol)

    # 理论产氢1摩尔需要2F电量
    # 系统效率 = (eta_F * 能量输出) / 能量输入
    # = (eta_F * HHV * M_H2) / (2 * F * U_cell)

    eta_system = (eta_F * energy_value) / (2 * F * U_cell)
    return eta_system


# 生成电流范围
I_range = np.linspace(100, 9000, 500)

# 目标温度 80°C (353.15K)
T_target = 80

# 计算各参数
eta_F_values = []
U_cell_values = []
h2_rate_values = []
eta_system_hhv_values = []
eta_system_lhv_values = []

for I in I_range:
    eta_F, f1, f2 = calculate_faraday_efficiency(I, T_target)
    U_cell = calculate_cell_voltage(I, T_target)
    h2_rate = calculate_h2_production_rate(I, T_target)
    eta_sys_hhv = calculate_system_efficiency(I, T_target, use_hhv=True)
    eta_sys_lhv = calculate_system_efficiency(I, T_target, use_hhv=False)

    eta_F_values.append(eta_F)
    U_cell_values.append(U_cell)
    h2_rate_values.append(h2_rate)
    eta_system_hhv_values.append(eta_sys_hhv)
    eta_system_lhv_values.append(eta_sys_lhv)

# 创建图形
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1: 法拉第效率 vs 电流
ax1 = axes[0, 0]
ax1.plot(I_range, eta_F_values, 'b-', linewidth=2.5, label=f'Faraday Efficiency (T={T_target}°C)')
ax1.set_xlabel('Current (A)', fontsize=11)
ax1.set_ylabel('Faraday Efficiency', fontsize=11)
ax1.set_title('Faraday Efficiency vs Current', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_ylim(0.91, 0.93)

# 图2: 电解小室电压 vs 电流
ax2 = axes[0, 1]
ax2.plot(I_range, U_cell_values, 'r-', linewidth=2.5, label='Cell Voltage')
ax2.set_xlabel('Current (A)', fontsize=11)
ax2.set_ylabel('Cell Voltage (V)', fontsize=11)
ax2.set_title('Cell Voltage vs Current', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend()

# 图3: 产氢率 vs 电流
ax3 = axes[1, 0]
ax3.plot(I_range, h2_rate_values, 'g-', linewidth=2.5, label='H2 Production Rate')
ax3.set_xlabel('Current (A)', fontsize=11)
ax3.set_ylabel('H2 Production Rate (mol/s)', fontsize=11)
ax3.set_title('H2 Production Rate vs Current', fontsize=12)
ax3.grid(True, alpha=0.3)
ax3.legend()

# 图4: 系统总效率 vs 电流
ax4 = axes[1, 1]
ax4.plot(I_range, eta_system_hhv_values, 'purple', linewidth=2.5, label='System Efficiency (HHV)')
ax4.plot(I_range, eta_system_lhv_values, 'orange', linewidth=2.5, linestyle='--', label='System Efficiency (LHV)')
ax4.set_xlabel('Current (A)', fontsize=11)
ax4.set_ylabel('System Efficiency', fontsize=11)
ax4.set_title('System Overall Efficiency vs Current', fontsize=12)
ax4.grid(True, alpha=0.3)
ax4.legend()

plt.tight_layout()

# 保存图片
output_dir = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(output_dir, 'system_efficiency_analysis.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"System efficiency plot saved to: {output_file}")

# 创建汇总表格
print("\n" + "="*80)
print("System Efficiency Analysis Summary (T = 80°C)")
print("="*80)
print(f"{'Current (A)':<12} {'Faraday η':<12} {'U_cell (V)':<12} {'H2 Rate':<12} {'System η_HHV':<12} {'System η_LHV':<12}")
print("-"*80)

test_currents = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
for I in test_currents:
    eta_F, _, _ = calculate_faraday_efficiency(I, T_target)
    U_cell = calculate_cell_voltage(I, T_target)
    h2_rate = calculate_h2_production_rate(I, T_target)
    eta_hhv = calculate_system_efficiency(I, T_target, use_hhv=True)
    eta_lhv = calculate_system_efficiency(I, T_target, use_hhv=False)
    print(f"{I:<12} {eta_F:<12.4f} {U_cell:<12.3f} {h2_rate:<12.3f} {eta_hhv:<12.3f} {eta_lhv:<12.3f}")

plt.show()
