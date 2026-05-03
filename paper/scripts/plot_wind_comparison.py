"""
绘制风电功率跟踪控制器对比图
数据：run_20260426_121806（扣除预热1440行）
风格：参考 paper/figures/controller_step_comparison.png（Image #7 标准）
"""

import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

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

base_dir = '../../output/multi_stack/test/run_20260426_121806'
warmup_rows = 1440

paper_models = {
    'TCN Diffusion': 'model_diffusion_tcn_data_20260426_121809.csv',
    'Diffusion MLP': 'model_diffusion_pure_mlp_data_20260426_121809.csv',
    'NMPC': 'nmpc_simplified_data_20260426_192526.csv',
    'MLP': 'model_pure_mlp_data_20260426_151739.csv',
}

METHOD_COLORS = {
    'TCN Diffusion': '#d62728',   # red - proposed method
    'NMPC': '#1f77b4',            # blue - baseline
    'Diffusion MLP': '#2ca02c',   # green
    'MLP': '#ff7f0e',             # orange
}

def load_and_unify(path):
    df = pd.read_csv(path)
    rename = {}
    if 'P_real' in df.columns:
        rename['P_real'] = 'P_actual'
    if 'P_total' in df.columns:
        rename['P_total'] = 'P_actual'
    for i in range(1, 5):
        if f'T_s_all_{i}' in df.columns:
            rename[f'T_s_all_{i}'] = f'T_s_{i}'
        if f'I_all_{i}' in df.columns:
            rename[f'I_all_{i}'] = f'I_{i}'
        if f'v_lye_all_{i}' in df.columns:
            rename[f'v_lye_all_{i}'] = f'v_lye_{i}'
    df = df.rename(columns=rename)
    return df

def smooth(y, window=101, poly=3):
    if len(y) < window:
        return y
    return savgol_filter(y, window, poly)

def setup_ax(ax):
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
    ax.tick_params(axis='both', which='both', length=3)

# 加载数据
data = {}
for label, fname in paper_models.items():
    path = os.path.join(base_dir, fname)
    df = load_and_unify(path)
    df = df.iloc[warmup_rows:]
    data[label] = df
    print(f"{label}: len={len(df)}")

# ========== 时间序列对比图 ==========
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
fig.subplots_adjust(hspace=0.32, wspace=0.28, left=0.07, right=0.97, top=0.90, bottom=0.12)
# 总标题由 LaTeX ption 控制，此处不设置 suptitle

labels = list(paper_models.keys())

# (a) 总功率跟踪
ax = axes[0, 0]
for label in labels:
    color = METHOD_COLORS[label]
    df = data[label]
    t = df['t'].values / 3600
    ax.plot(t, df['P_ref'].values/1e6, '--', color=color, alpha=0.5, linewidth=1)
    ax.plot(t, smooth(df['P_actual'].values/1e6), color=color, linewidth=1.5, label=label)
ax.set_ylabel('功率 (MW)')
ax.set_xlabel('时间 (h)')
ax.legend(loc='best', frameon=True)
setup_ax(ax)
ax.set_ylim([0, 40])
ax.set_yticks([0, 10, 20, 30, 40])

# (b) 电解槽温度（纵轴显示摄氏度，数据仍为K）
ax = axes[0, 1]
for label in labels:
    color = METHOD_COLORS[label]
    df = data[label]
    t = df['t'].values / 3600
    T_stack_mean = df[['T_s_1','T_s_2','T_s_3','T_s_4']].mean(axis=1).values
    ax.plot(t, smooth(T_stack_mean), color=color, linewidth=1.5, label=label)
ax.axhline(y=353.15, color='k', linestyle='--', linewidth=1.2, alpha=0.7, label='设定温度')
ax.set_ylabel('温度 (°C)')
ax.set_xlabel('时间 (h)')
ax.legend(loc='best', frameon=True)
setup_ax(ax)
ax.set_ylim([273.15 + 20, 273.15 + 100])
ax.set_yticks([273.15 + c for c in range(20, 101, 10)])
ax.set_yticklabels([str(c) for c in range(20, 101, 10)])

# (c) HTO
ax = axes[0, 2]
for label in labels:
    color = METHOD_COLORS[label]
    df = data[label]
    t = df['t'].values / 3600
    if 'HTO' in df.columns:
        ax.plot(t, smooth(df['HTO'].values), color=color, linewidth=1.5, label=label)
    else:
        ax.plot(t, np.zeros_like(t), color=color, linewidth=1.5, label=label)
ax.axhline(2.0, color='r', linestyle='--', linewidth=1.5, label='安全限 (2%)')
ax.set_ylabel('HTO (%)')
ax.set_xlabel('时间 (h)')
ax.legend(loc='best', frameon=True)
setup_ax(ax)
ax.set_ylim([0, 2.0])
ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])

# (d) 电解槽电流（单槽均值，与总电流/4等价）
ax = axes[1, 0]
for label in labels:
    color = METHOD_COLORS[label]
    df = data[label]
    t = df['t'].values / 3600
    I_mean = df[['I_1','I_2','I_3','I_4']].mean(axis=1).values / 1000
    ax.plot(t, smooth(I_mean), color=color, linewidth=1.5, label=label)
ax.set_ylabel('电流 (kA)')
ax.set_xlabel('时间 (h)')
ax.legend(loc='best', frameon=True)
setup_ax(ax)
ax.set_ylim([0, 8])
ax.set_yticks([0, 2, 4, 6, 8])

# (e) 碱液流量
ax = axes[1, 1]
for label in labels:
    color = METHOD_COLORS[label]
    df = data[label]
    t = df['t'].values / 3600
    v_lye_mean = df[['v_lye_1','v_lye_2','v_lye_3','v_lye_4']].mean(axis=1).values
    ax.plot(t, smooth(v_lye_mean * 1000), color=color, linewidth=1.5, label=label)
ax.set_ylabel('流量 (L/s)')
ax.set_xlabel('时间 (h)')
ax.legend(loc='best', frameon=True)
setup_ax(ax)
ax.set_ylim([10, 100])
ax.set_yticks([10, 30, 50, 70, 90])

# (f) 冷却水流量
ax = axes[1, 2]
for label in labels:
    color = METHOD_COLORS[label]
    df = data[label]
    t = df['t'].values / 3600
    ax.plot(t, smooth(df['v_c'].values * 1000), color=color, linewidth=1.5, label=label)
ax.set_ylabel('流量 (L/s)')
ax.set_xlabel('时间 (h)')
ax.legend(loc='best', frameon=True)
setup_ax(ax)
ax.set_ylim([0, 75])
ax.set_yticks([0, 20, 40, 60])

# 统一x轴
for ax in axes.flat:
    ax.set_xlim([t.min(), t.max()])

# ========== 保存 ==========
plt.savefig('../figures/controller_wind_comparison.png', dpi=600, bbox_inches='tight', facecolor='white')
print('Saved: ../figures/controller_wind_comparison.png')

# 保存各子图为独立PNG（供LaTeX subfigure环境使用）
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for idx, ax in enumerate(axes.flat):
    label = chr(ord('a') + idx)
    bbox = ax.get_tightbbox(renderer)
    bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
    bbox_inches = bbox_inches.expanded(1.02, 1.02)
    fig.savefig(f'../figures/controller_wind_comparison_{label}.png', dpi=600, bbox_inches=bbox_inches, facecolor='white')
    print(f'Saved: ../figures/controller_wind_comparison_{label}.png')
plt.close()
