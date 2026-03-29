"""
绘制法拉第效率与电流的关系图
基于多槽电解槽模型的效率公式
"""
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 法拉第效率计算函数
def calculate_faraday_efficiency(I, T_celsius):
    """
    计算法拉第效率

    Args:
        I: 电流 (A)
        T_celsius: 电堆温度 (°C)

    Returns:
        eta_F: 法拉第效率 (0-1)
    """
    # 经验参数
    f1 = 50.0 + 2.5 * T_celsius
    f2 = 0.92 - 6.25e-6 * T_celsius

    # 法拉第效率公式: eta = I^2 / (f1 + I^2) * f2
    I_sq = I ** 2
    eta_F = (I_sq / (f1 + I_sq)) * f2

    return eta_F

# 生成电流范围 (A)
I_range = np.linspace(100, 10000, 500)  # 100A 到 10000A

# 不同温度下的效率曲线
temperatures = [60, 70, 80, 90]  # °C
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
linestyles = ['-', '--', '-.', ':']

# 创建图形
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 图1: 效率-电流曲线
ax1 = axes[0]
for T_c, color, ls in zip(temperatures, colors, linestyles):
    eta_values = [calculate_faraday_efficiency(I, T_c) for I in I_range]
    ax1.plot(I_range, eta_values, color=color, linestyle=ls, linewidth=2,
             label=f'T = {T_c}°C (T = {T_c + 273.15:.2f}K)')

ax1.set_xlabel('Current (A)', fontsize=12)
ax1.set_ylabel('Faraday Efficiency η_F', fontsize=12)
ax1.set_title('Faraday Efficiency vs Current at Different Temperatures', fontsize=13)
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 10000)
ax1.set_ylim(0.85, 0.95)

# 图2: 归一化效率 (eta/f2) 与电流关系
ax2 = axes[1]
for T_c, color, ls in zip(temperatures, colors, linestyles):
    f1 = 50.0 + 2.5 * T_c
    f2 = 0.92 - 6.25e-6 * T_c
    # 归一化效率: I^2 / (f1 + I^2)
    eta_normalized = I_range**2 / (f1 + I_range**2)
    ax2.plot(I_range, eta_normalized, color=color, linestyle=ls, linewidth=2,
             label=f'T = {T_c}°C, f1={f1:.1f}, f2={f2:.4f}')

ax2.set_xlabel('Current (A)', fontsize=12)
ax2.set_ylabel('Normalized Efficiency η_F/f2 = I²/(f1+I²)', fontsize=12)
ax2.set_title('Normalized Faraday Efficiency vs Current', fontsize=13)
ax2.legend(loc='lower right', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 10000)
ax2.set_ylim(0.9, 1.0)

plt.tight_layout()

# 保存图片
output_dir = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(output_dir, 'faraday_efficiency_vs_current.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"图片已保存至: {output_file}")

# 打印关键数据点
print("\n=== 关键数据点 ===")
print(f"{'Current (A)':<15} {'60°C':<10} {'70°C':<10} {'80°C':<10} {'90°C':<10}")
print("-" * 55)
test_currents = [500, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000]
for I in test_currents:
    eta_60 = calculate_faraday_efficiency(I, 60)
    eta_70 = calculate_faraday_efficiency(I, 70)
    eta_80 = calculate_faraday_efficiency(I, 80)
    eta_90 = calculate_faraday_efficiency(I, 90)
    print(f"{I:<15} {eta_60:<10.4f} {eta_70:<10.4f} {eta_80:<10.4f} {eta_90:<10.4f}")

plt.show()
