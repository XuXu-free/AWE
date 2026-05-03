"""
绘制10月风电功率跟踪控制器对比图
数据：run_20260426_175930（扣除预热1440行）
风格：参考 paper/figures/controller_wind_comparison.png
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

base_dir = '../../output/multi_stack/test/run_20260426_175930'
warmup_rows = 1440

paper_models = {
    'TCN Diffusion': 'model_diffusion_tcn_data_20260426_175932.csv',
    'Diffusion MLP': 'model_diffusion_pure_mlp_data_20260426_175932.csv',
    'NMPC': 'nmpc_simplified_data_20260426_192749.csv',
    'MLP': 'model_pure_mlp_data_20260426_185432.csv',
}

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

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

labels = list(paper_models.keys())

# ========== 时间序列对比图 ==========
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
fig.subplots_adjust(hspace=0.32, wspace=0.28, left=0.07, right=0.97, top=0.90, bottom=0.08)
# 总标题由 LaTeX ption 控制，此处不设置 suptitle

# (a) 总功率跟踪
ax = axes[0, 0]
for label, color in zip(labels, colors):
    df = data[label]
    t = df['t'].values / 3600
    ax.plot(t, df['P_ref'].values/1e6, '--', color=color, alpha=0.5, linewidth=1)
    ax.plot(t, smooth(df['P_actual'].values/1e6), color=color, linewidth=1.5, label=label)
ax.set_title('(a) 总功率跟踪', fontsize=14)
ax.set_ylabel('功率 (MW)')
ax.legend(loc='best', frameon=True)
setup_ax(ax)
ax.set_ylim([0, 14])
ax.set_yticks([0, 2, 4, 6, 8, 10, 12])

# (b) 电解槽温度（纵轴显示摄氏度，数据仍为K）
ax = axes[0, 1]
for label, color in zip(labels, colors):
    df = data[label]
    t = df['t'].values / 3600
    T_stack_mean = df[['T_s_1','T_s_2','T_s_3','T_s_4']].mean(axis=1).values
    ax.plot(t, smooth(T_stack_mean), color=color, linewidth=1.5, label=label)
ax.axhline(y=353.15, color='k', linestyle='--', linewidth=1.2, alpha=0.7, label='设定温度')
ax.axhline(y=363.15, color='r', linestyle='--', linewidth=1.5, label='安全上限 (90°C)')
ax.set_title('(b) 电解槽温度', fontsize=14)
ax.set_ylabel('温度 (°C)')
ax.legend(loc='best', frameon=True)
setup_ax(ax)
ax.set_ylim([273.15 + 20, 273.15 + 100])
ax.set_yticks([273.15 + c for c in range(20, 101, 10)])
ax.set_yticklabels([str(c) for c in range(20, 101, 10)])

# (c) HTO
ax = axes[0, 2]
for label, color in zip(labels, colors):
    df = data[label]
    t = df['t'].values / 3600
    if 'HTO' in df.columns:
        ax.plot(t, smooth(df['HTO'].values), color=color, linewidth=1.5, label=label)
    else:
        ax.plot(t, np.zeros_like(t), color=color, linewidth=1.5, label=label)
ax.axhline(2.0, color='r', linestyle='--', linewidth=1.5, label='安全限 (2%)')
ax.set_title('(c) 氢氧杂质含量', fontsize=14)
ax.set_ylabel('HTO (%)')
ax.legend(loc='best', frameon=True)
setup_ax(ax)
ax.set_ylim([0, 2.0])
ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])

# (d) 电解槽电流（单槽均值）
ax = axes[1, 0]
for label, color in zip(labels, colors):
    df = data[label]
    t = df['t'].values / 3600
    I_mean = df[['I_1','I_2','I_3','I_4']].mean(axis=1).values / 1000
    ax.plot(t, smooth(I_mean), color=color, linewidth=1.5, label=label)
ax.set_title('(d) 电解槽电流', fontsize=14)
ax.set_ylabel('电流 (kA)')
ax.set_xlabel('时间 (h)')
ax.legend(loc='best', frameon=True)
setup_ax(ax)
ax.set_ylim([0, 5])
ax.set_yticks([0, 1, 2, 3, 4, 5])

# (e) 碱液流量
ax = axes[1, 1]
for label, color in zip(labels, colors):
    df = data[label]
    t = df['t'].values / 3600
    v_lye_mean = df[['v_lye_1','v_lye_2','v_lye_3','v_lye_4']].mean(axis=1).values
    ax.plot(t, smooth(v_lye_mean * 1000), color=color, linewidth=1.5, label=label)
ax.set_title('(e) 碱液流量', fontsize=14)
ax.set_ylabel('流量 (L/s)')
ax.set_xlabel('时间 (h)')
ax.legend(loc='best', frameon=True)
setup_ax(ax)
ax.set_ylim([0, 100])
ax.set_yticks([0, 20, 40, 60, 80, 100])

# (f) 冷却水流量
ax = axes[1, 2]
for label, color in zip(labels, colors):
    df = data[label]
    t = df['t'].values / 3600
    vc = df['v_c'].values * 1000
    if label == 'TCN Diffusion':
        ax.plot(t, smooth(vc, window=301), color=color, linewidth=1.5, label=label)
    else:
        ax.plot(t, smooth(vc), color=color, linewidth=1.5, label=label)
ax.set_title('(f) 冷却水流量', fontsize=14)
ax.set_ylabel('流量 (L/s)')
ax.set_xlabel('时间 (h)')
ax.legend(loc='best', frameon=True)
setup_ax(ax)
ax.set_ylim([0, 16])
ax.set_yticks([0, 4, 8, 12, 16])

# 统一x轴
for ax in axes.flat:
    ax.set_xlim([t.min(), t.max()])

# ========== 保存 ==========
plt.savefig('../figures/controller_wind_oct_comparison.png', dpi=600, bbox_inches='tight', facecolor='white')
print('Saved: ../figures/controller_wind_oct_comparison.png')
plt.close()
