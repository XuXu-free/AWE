"""
绘制 CBF 高温约束保护实验图
数据：high_power_step_cbf_on_no_warmup 电流阶跃测试 (3000A -> 7500A)
风格：参考 paper/scripts/plot_cbf_hto_protection.py
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

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
csv_path = '../../output/multi_stack/test_step/high_power_step_long_cbf_on/current_step_diffusion_tcn_cbf_on_data_20260430_131809.csv'
df = pd.read_csv(csv_path)
t = df['t'].values / 60.0  # 转换为分钟

T_s_all = np.stack([df[f'T_s_all_{i}'].values for i in range(1, 5)], axis=1)
T_sep = df['T_sep'].values
T_c_out = df['T_c_out'].values
HTO = df['HTO'].values
cbf_triggered = df['cbf_triggered'].values.astype(bool)
I_all = np.stack([df[f'I_all_{i}'].values for i in range(1, 5)], axis=1)
v_lye_all = np.stack([df[f'v_lye_all_{i}'].values for i in range(1, 5)], axis=1)
v_c = df['v_c'].values

# 预计算 CBF 触发连续块
trigger_times = t[cbf_triggered]
cbf_blocks = []
if len(trigger_times) > 0:
    merge_threshold = 2.0  # min
    cbf_blocks = [[trigger_times[0], trigger_times[0]]]
    for tt in trigger_times[1:]:
        if tt - cbf_blocks[-1][1] <= merge_threshold:
            cbf_blocks[-1][1] = tt
        else:
            cbf_blocks.append([tt, tt])
    for i in range(len(cbf_blocks)):
        cbf_blocks[i][1] = min(cbf_blocks[i][1] + 0.2, t[-1])


def add_cbf_background(ax):
    """在所有子图中添加浅红色 CBF 触发背景"""
    for start, end in cbf_blocks:
        ax.axvspan(start, end, facecolor='red', alpha=0.08, edgecolor='none')


def setup_ax(ax):
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
    ax.set_xlim([t.min(), t.max()])
    ax.tick_params(axis='both', which='both', length=3)


colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
cbf_patch = Patch(facecolor='red', alpha=0.08, edgecolor='none', label='CBF触发')

# ========== 画布 ==========
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.subplots_adjust(hspace=0.32, wspace=0.28, left=0.08, right=0.97, top=0.90, bottom=0.08)
# 总标题由 LaTeX ption 控制，此处不设置 suptitle

# 电流参考：阶跃 3000A -> 7500A @ t=60min
I_ref = np.full_like(t, 3.0)
I_ref[t >= 60] = 7.5

# (a) 电解槽电流
ax = axes[0, 0]
add_cbf_background(ax)
ax.plot(t, I_ref, 'k--', linewidth=1.2, alpha=0.7, label='参考电流')
for i in range(4):
    ax.plot(t, I_all[:, i] / 1000, color=colors[i], linewidth=1.5, label=f'槽{i+1}')
handles, labels = ax.get_legend_handles_labels()
handles.append(cbf_patch)
labels.append('CBF触发')
ax.legend(handles=handles, labels=labels, loc='best', frameon=True, ncol=2)
ax.set_title('(a) 电解槽电流', fontsize=14)
ax.set_ylabel('电流 (kA)')
setup_ax(ax)
ax.set_ylim([2.5, 8.5])
ax.set_yticks([3, 4, 5, 6, 7, 8])

# (b) 碱液流量
ax = axes[0, 1]
add_cbf_background(ax)
for i in range(4):
    ax.plot(t, v_lye_all[:, i] * 1e3, color=colors[i], linewidth=1.5, label=f'槽{i+1}')
handles, labels = ax.get_legend_handles_labels()
handles.append(cbf_patch)
labels.append('CBF触发')
ax.legend(handles=handles, labels=labels, loc='upper right', frameon=True, ncol=2)
ax.set_title('(b) 碱液流量', fontsize=14)
ax.set_ylabel('流量 (L/s)')
setup_ax(ax)
ax.set_ylim([15, 35])
ax.set_yticks([15, 20, 25, 30, 35])

# (c) 冷却水流量
ax = axes[1, 0]
add_cbf_background(ax)
ax.plot(t, v_c * 1e3, colors[0], linewidth=1.5, label='实际流量')
ax.axhline(y=20.0, color='k', linestyle='--', linewidth=1.2, alpha=0.7, label='参考流量')
handles, labels = ax.get_legend_handles_labels()
handles.append(cbf_patch)
labels.append('CBF触发')
ax.legend(handles=handles, labels=labels, loc='lower right', frameon=True)
ax.set_title('(c) 冷却水流量', fontsize=14)
ax.set_ylabel('流量 (L/s)')
ax.set_xlabel('时间 (min)')
setup_ax(ax)
ax.set_ylim([18, 35])
ax.set_yticks([20, 25, 30, 35])

# (d) 温度安全控制（替换原HTO图）
ax = axes[1, 1]
add_cbf_background(ax)
for i in range(4):
    ax.plot(t, T_s_all[:, i] - 273.15, color=colors[i], linewidth=1.5, label=f'电堆{i+1}')
ax.axhline(y=90.0, color='r', linestyle='--', linewidth=1.5, label='安全限 (90°C)')
handles, labels = ax.get_legend_handles_labels()
handles.append(cbf_patch)
labels.append('CBF触发')
ax.legend(handles=handles, labels=labels, loc='lower right', frameon=True, ncol=2)
ax.set_title('(d) 温度安全控制', fontsize=14)
ax.set_ylabel('温度 (°C)')
ax.set_xlabel('时间 (min)')
setup_ax(ax)
ax.set_ylim([45, 95])
ax.set_yticks([50, 60, 70, 80, 90])

# ========== 保存 ==========
plt.savefig('../figures/cbf_temp_protection.png', dpi=600, bbox_inches='tight', facecolor='white')
print('Saved: figures/cbf_temp_protection.png')
plt.close()
