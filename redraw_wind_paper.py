import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

lambda_track = 1.2
lambda_temp = 0.15
lambda_I = 0.0002
lambda_lye = 25000
lambda_c = 2500
lambda_prod = 1e-6

base_dir = 'output/multi_stack/test/run_20260426_121806'
warmup_rows = 1440

paper_models = {
    'TCN Diffusion': 'model_diffusion_tcn_data_20260426_121809.csv',
    'Diffusion Pure MLP': 'model_diffusion_pure_mlp_data_20260426_121809.csv',
    'Simplified NMPC': 'nmpc_simplified_data_20260426_192526.csv',
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

def compute_cost(df):
    P_ref = df['P_ref'].values
    P_actual = df['P_actual'].values
    T_ref = df['T_ref'].values[0] if 'T_ref' in df.columns else 353.15
    T_stacks = df[['T_s_1','T_s_2','T_s_3','T_s_4']].values
    I = df[['I_1','I_2','I_3','I_4']].values
    v_lye = df[['v_lye_1','v_lye_2','v_lye_3','v_lye_4']].values
    v_c = df['v_c'].values

    power_cost = lambda_track * np.sum(((P_actual - P_ref) / 1e6) ** 2)
    temp_cost = lambda_temp * np.sum((T_stacks - T_ref) ** 2)
    dI = np.diff(I, axis=0)
    current_smooth_cost = lambda_I * np.sum(dI ** 2)
    prod_equal_cost = 0.0
    for i in range(4):
        for j in range(i+1, 4):
            prod_equal_cost += lambda_prod * np.sum((I[:, i] - I[:, j]) ** 2)
    dv_lye = np.diff(v_lye, axis=0)
    lye_smooth_cost = lambda_lye * np.sum(dv_lye ** 2)
    dv_c = np.diff(v_c)
    coolant_smooth_cost = lambda_c * np.sum(dv_c ** 2)
    total = power_cost + temp_cost + current_smooth_cost + prod_equal_cost + lye_smooth_cost + coolant_smooth_cost
    return {
        'power': power_cost, 'temp': temp_cost, 'current': current_smooth_cost,
        'prod': prod_equal_cost, 'lye': lye_smooth_cost, 'coolant': coolant_smooth_cost, 'total': total
    }

# 加载数据
data = {}
costs = {}
for label, fname in paper_models.items():
    path = os.path.join(base_dir, fname)
    df = load_and_unify(path)
    df = df.iloc[warmup_rows:]  # 扣除预热
    data[label] = df
    costs[label] = compute_cost(df)
    print(f"{label}: len={len(df)}, total={costs[label]['total']:.1f}")

# ========== 时间序列对比图 ==========
fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
axes = axes.flatten()

colors = ['#4472C4', '#ED7D31', '#70AD47']
labels = list(paper_models.keys())

def smooth(y, window=101, poly=3):
    if len(y) < window:
        return y
    return savgol_filter(y, window, poly)

# 1. 总功率
ax = axes[0]
for label, color in zip(labels, colors):
    df = data[label]
    t = df['t'].values / 3600
    ax.plot(t, smooth(df['P_ref'].values/1e6), '--', color=color, alpha=0.5, linewidth=1)
    ax.plot(t, smooth(df['P_actual'].values/1e6), color=color, linewidth=1.5, label=label)
ax.set_ylabel('Power (MW)')
ax.set_title('总功率跟踪')
ax.legend(fontsize=9, loc='best')
ax.grid(True, alpha=0.3)

# 2. 堆栈温度
ax = axes[1]
for label, color in zip(labels, colors):
    df = data[label]
    t = df['t'].values / 3600
    T_stack_mean = df[['T_s_1','T_s_2','T_s_3','T_s_4']].mean(axis=1).values
    ax.plot(t, smooth(T_stack_mean), color=color, linewidth=1.5, label=label)
ax.axhline(353.15, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.set_ylabel('Temperature (K)')
ax.set_title('平均堆栈温度')
ax.legend(fontsize=9, loc='best')
ax.grid(True, alpha=0.3)

# 3. HTO
ax = axes[2]
for label, color in zip(labels, colors):
    df = data[label]
    t = df['t'].values / 3600
    if 'HTO' in df.columns:
        ax.plot(t, smooth(df['HTO'].values * 100), color=color, linewidth=1.5, label=label)
    else:
        ax.plot(t, np.zeros_like(t), color=color, linewidth=1.5, label=label)
ax.axhline(2.0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.set_ylabel('HTO (%)')
ax.set_title('氢氧杂质含量')
ax.legend(fontsize=9, loc='best')
ax.grid(True, alpha=0.3)

# 4. 电流
ax = axes[3]
for label, color in zip(labels, colors):
    df = data[label]
    t = df['t'].values / 3600
    I_total = df[['I_1','I_2','I_3','I_4']].sum(axis=1).values
    ax.plot(t, smooth(I_total/1000), color=color, linewidth=1.5, label=label)
ax.set_ylabel('Current (kA)')
ax.set_title('总电流')
ax.legend(fontsize=9, loc='best')
ax.grid(True, alpha=0.3)

# 5. 碱液流量
ax = axes[4]
for label, color in zip(labels, colors):
    df = data[label]
    t = df['t'].values / 3600
    v_lye_mean = df[['v_lye_1','v_lye_2','v_lye_3','v_lye_4']].mean(axis=1).values
    ax.plot(t, smooth(v_lye_mean * 1000), color=color, linewidth=1.5, label=label)
ax.set_ylabel('Lye Flow (L/s)')
ax.set_title('平均碱液流量')
ax.legend(fontsize=9, loc='best')
ax.grid(True, alpha=0.3)

# 6. 冷却剂流量
ax = axes[5]
for label, color in zip(labels, colors):
    df = data[label]
    t = df['t'].values / 3600
    ax.plot(t, smooth(df['v_c'].values * 1000), color=color, linewidth=1.5, label=label)
ax.set_ylabel('Coolant Flow (L/s)')
ax.set_title('冷却剂流量')
ax.legend(fontsize=9, loc='best')
ax.grid(True, alpha=0.3)

for ax in axes:
    ax.set_xlabel('Time (h)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('paper/figures/controller_wind_comparison_draft.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved paper/figures/controller_wind_comparison.png")

# ========== 成本对比图 ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
components = ['功率跟踪', '温度控制', '电流平滑', '生产均衡', '碱液平滑', '冷却剂平滑']
keys = ['power', 'temp', 'current', 'prod', 'lye', 'coolant']
x = np.arange(len(components))
width = 0.25

for idx, label in enumerate(labels):
    vals = [costs[label][k] for k in keys]
    ax.bar(x + idx * width, vals, width, label=label, color=colors[idx], edgecolor='white', linewidth=0.5)

ax.set_ylabel('Cost')
ax.set_title('(a) 各成本分量对比')
ax.set_xticks(x + width)
ax.set_xticklabels(components, fontsize=10)
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax = axes[1]
totals = [costs[label]['total'] for label in labels]
bars = ax.bar(labels, totals, color=colors, edgecolor='white', linewidth=0.5)
ax.set_ylabel('Total Cost')
ax.set_title('(b) 总成本对比')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('paper/figures/controller_wind_cost_comparison_draft.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved paper/figures/controller_wind_cost_comparison.png")
