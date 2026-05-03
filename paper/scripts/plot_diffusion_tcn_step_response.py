"""
绘制 TCN Diffusion 阶跃响应图
数据：step_model_diffusion_tcn_data_20260426_135007.csv（10 MW → 20 MW @ 120 min）
风格：参考 paper/figures/controller_step_comparison.png（Image #7 标准）
布局：2×3，第二行为控制量
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# ========== 全局配置 ==========
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

# ========== 数据加载 ==========
csv_path = '../../output/multi_stack/test_step/run_20260426_165250/step_model_diffusion_tcn_data_20260426_135007.csv'
df = pd.read_csv(csv_path)
t = df['t'].values / 60.0  # 转换为分钟

P_ref = df['P_ref'].values / 1e6
P_real = df['P_real'].values / 1e6
T_s = np.stack([df[f'T_s_all_{i}'].values for i in range(1, 5)], axis=1)
HTO = df['HTO'].values
I_all = np.stack([df[f'I_all_{i}'].values for i in range(1, 5)], axis=1)
v_lye_all = np.stack([df[f'v_lye_all_{i}'].values for i in range(1, 5)], axis=1)
v_c = df['v_c'].values

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

def setup_ax(ax):
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
    ax.set_xlim([t.min(), t.max()])
    ax.tick_params(axis='both', which='both', length=3)

# ========== 画布 ==========
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.subplots_adjust(hspace=0.32, wspace=0.28, left=0.07, right=0.97, top=0.90, bottom=0.08)
# 总标题由 LaTeX ption 控制，此处不设置 suptitle

# (a) 总功率跟踪
ax = axes[0, 0]
ax.plot(t, P_ref, 'k--', linewidth=1.2, alpha=0.7, label='参考功率')
ax.plot(t, P_real, color=colors[0], linewidth=1.5, label='实际功率')
ax.set_title('(a) 总功率跟踪', fontsize=14, fontweight='bold')
ax.set_ylabel('功率 (MW)')
ax.legend(loc='best', frameon=True)
setup_ax(ax)
ax.set_ylim([0, 22])
ax.set_yticks([0, 5, 10, 15, 20])

# (b) 电解槽温度（纵轴显示摄氏度，数据仍为K）
ax = axes[0, 1]
for i in range(4):
    ax.plot(t, T_s[:, i], color=colors[i], linewidth=1.5, label=f'槽{i+1}')
ax.axhline(y=353.15, color='k', linestyle='--', linewidth=1.2, alpha=0.7, label='设定温度')
ax.set_title('(b) 电解槽温度', fontsize=14, fontweight='bold')
ax.set_ylabel('温度 (°C)')
ax.legend(loc='best', frameon=True, ncol=2)
setup_ax(ax)
ax.set_ylim([273.15 + 10, 273.15 + 100])
ax.set_yticks([273.15 + c for c in range(10, 101, 10)])
ax.set_yticklabels([str(c) for c in range(10, 101, 10)])

# (c) HTO
ax = axes[0, 2]
ax.plot(t, HTO, color=colors[0], linewidth=1.5, label='HTO')
ax.axhline(y=2.0, color='r', linestyle='--', linewidth=1.5, label='安全限 (2%)')
ax.set_title('(c) 氢氧杂质含量', fontsize=14, fontweight='bold')
ax.set_ylabel('HTO (%)')
ax.legend(loc='upper right', frameon=True)
setup_ax(ax)
ax.set_ylim([0, 2.3])
ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])

# (d) 电解槽电流
ax = axes[1, 0]
for i in range(4):
    ax.plot(t, I_all[:, i] / 1000, color=colors[i], linewidth=1.5, label=f'槽{i+1}')
ax.set_title('(d) 电解槽电流', fontsize=14, fontweight='bold')
ax.set_ylabel('电流 (kA)')
ax.set_xlabel('时间 (min)')
ax.legend(loc='best', frameon=True, ncol=2)
setup_ax(ax)
ax.set_ylim([0, 8])
ax.set_yticks([0, 2, 4, 6, 8])

# (e) 碱液流量
ax = axes[1, 1]
for i in range(4):
    ax.plot(t, v_lye_all[:, i] * 1e3, color=colors[i], linewidth=1.5, label=f'槽{i+1}')
ax.set_title('(e) 碱液流量', fontsize=14, fontweight='bold')
ax.set_ylabel('流量 (L/s)')
ax.set_xlabel('时间 (min)')
ax.legend(loc='best', frameon=True, ncol=2)
setup_ax(ax)
ax.set_ylim([10, 100])
ax.set_yticks([10, 30, 50, 70, 90])

# (f) 冷却水流量
ax = axes[1, 2]
ax.plot(t, v_c * 1e3, color=colors[0], linewidth=1.5, label='实际流量')
ax.set_title('(f) 冷却水流量', fontsize=14, fontweight='bold')
ax.set_ylabel('流量 (L/s)')
ax.set_xlabel('时间 (min)')
ax.legend(loc='best', frameon=True)
setup_ax(ax)
ax.set_ylim([5, 40])
ax.set_yticks([10, 20, 30, 40])

# ========== 保存 ==========
plt.savefig('../figures/diffusion_tcn_step_response.png', dpi=600, bbox_inches='tight', facecolor='white')
print('Saved: ../figures/diffusion_tcn_step_response.png')
plt.close()
