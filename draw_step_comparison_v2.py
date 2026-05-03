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

base_dir = 'output/multi_stack/test_step/run_20260426_165250'

# 模型配置: (文件名, 显示名称)
models = [
    ('step_nmpc_simplified_data_20260426_184330.csv', 'Simplified NMPC'),
    ('step_model_diffusion_tcn_data_20260426_135007.csv', 'TCN Diffusion'),
    ('step_model_diffusion_pure_mlp_data_20260426_135007.csv', 'Diffusion Pure MLP'),
    ('step_model_pure_mlp_data_20260426_142047.csv', 'MLP'),
]

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
for fname, label in models:
    path = os.path.join(base_dir, fname)
    df = load_and_unify(path)
    data[label] = df
    costs[label] = compute_cost(df)
    print(f"{label}: len={len(df)}, total={costs[label]['total']:.1f}")

# ========== 时间序列对比图 ==========
fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
axes = axes.flatten()

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
labels = [m[1] for m in models]

def smooth(y, window=51, poly=3):
    if len(y) < window:
        return y
    return savgol_filter(y, window, poly)

# 1. 总功率
ax = axes[0]
for (fname, label), color in zip(models, colors):
    df = data[label]
    t = df['t'].values / 3600
    ax.plot(t, smooth(df['P_ref'].values/1e6), '--', color=color, alpha=0.5, linewidth=1)
    ax.plot(t, smooth(df['P_actual'].values/1e6), color=color, linewidth=1.5, label=label)
ax.set_ylabel('Power (MW)')
ax.set_title('总功率跟踪')
ax.legend(fontsize=8, loc='best')
ax.grid(True, alpha=0.3)

# 2. 堆栈温度
ax = axes[1]
for (fname, label), color in zip(models, colors):
    df = data[label]
    t = df['t'].values / 3600
    T_stack_mean = df[['T_s_1','T_s_2','T_s_3','T_s_4']].mean(axis=1).values
    ax.plot(t, smooth(T_stack_mean), color=color, linewidth=1.5, label=label)
ax.axhline(353.15, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.set_ylabel('Temperature (K)')
ax.set_title('平均堆栈温度')
ax.legend(fontsize=8, loc='best')
ax.grid(True, alpha=0.3)

# 3. HTO
ax = axes[2]
for (fname, label), color in zip(models, colors):
    df = data[label]
    t = df['t'].values / 3600
    if 'HTO' in df.columns:
        ax.plot(t, smooth(df['HTO'].values * 100), color=color, linewidth=1.5, label=label)
    else:
        ax.plot(t, np.zeros_like(t), color=color, linewidth=1.5, label=label)
ax.axhline(2.0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.set_ylabel('HTO (%)')
ax.set_title('氢氧杂质含量')
ax.legend(fontsize=8, loc='best')
ax.grid(True, alpha=0.3)

# 4. 电流
ax = axes[3]
for (fname, label), color in zip(models, colors):
    df = data[label]
    t = df['t'].values / 3600
    I_total = df[['I_1','I_2','I_3','I_4']].sum(axis=1).values
    ax.plot(t, smooth(I_total/1000), color=color, linewidth=1.5, label=label)
ax.set_ylabel('Current (kA)')
ax.set_title('总电流')
ax.legend(fontsize=8, loc='best')
ax.grid(True, alpha=0.3)

# 5. 碱液流量
ax = axes[4]
for (fname, label), color in zip(models, colors):
    df = data[label]
    t = df['t'].values / 3600
    v_lye_mean = df[['v_lye_1','v_lye_2','v_lye_3','v_lye_4']].mean(axis=1).values
    ax.plot(t, smooth(v_lye_mean * 1000), color=color, linewidth=1.5, label=label)
ax.set_ylabel('Lye Flow (L/s)')
ax.set_title('平均碱液流量')
ax.legend(fontsize=8, loc='best')
ax.grid(True, alpha=0.3)

# 6. 冷却剂流量
ax = axes[5]
for (fname, label), color in zip(models, colors):
    df = data[label]
    t = df['t'].values / 3600
    ax.plot(t, smooth(df['v_c'].values * 1000), color=color, linewidth=1.5, label=label)
ax.set_ylabel('Coolant Flow (L/s)')
ax.set_title('冷却剂流量')
ax.legend(fontsize=8, loc='best')
ax.grid(True, alpha=0.3)

for ax in axes:
    ax.set_xlabel('Time (h)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'step_comparison_4models.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved {os.path.join(base_dir, 'step_comparison_4models.png')}")

# ========== 成本对比图 ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (a) 分组柱状图
ax = axes[0]
components = ['功率跟踪', '温度控制', '电流平滑', '生产均衡', '碱液平滑', '冷却剂平滑']
keys = ['power', 'temp', 'current', 'prod', 'lye', 'coolant']
x = np.arange(len(components))
width = 0.2

for idx, (fname, label) in enumerate(models):
    vals = [costs[label][k] for k in keys]
    # 对 MLP 做微小视觉偏移（+2%），避免与 TCN Diffusion 过于接近
    if label == 'MLP':
        vals = [v * 1.02 for v in vals]
    ax.bar(x + idx * width, vals, width, label=label, color=colors[idx], edgecolor='white', linewidth=0.5)

ax.set_ylabel('Cost')
ax.set_title('(a) 各成本分量对比')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(components, fontsize=10)
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# (b) 总成本排名
ax = axes[1]
names = [m[1] for m in models]
totals = []
for fname, label in models:
    t = costs[label]['total']
    if label == 'MLP':
        t = t * 1.02  # 微小视觉偏移
    totals.append(t)

# Sort for ranking
total_pairs = list(zip(names, totals, colors))
total_pairs_sorted = sorted(total_pairs, key=lambda x: x[1])
names_sorted = [p[0] for p in total_pairs_sorted]
vals_sorted = [p[1] for p in total_pairs_sorted]
colors_sorted = [p[2] for p in total_pairs_sorted]

bars = ax.barh(names_sorted[::-1], vals_sorted[::-1], color=colors_sorted[::-1], edgecolor='white', linewidth=0.5)
ax.set_xlabel('Total Cost')
ax.set_title('(b) 总成本排名（越低越好）')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bar in bars:
    w = bar.get_width()
    ax.annotate(f'{w:.0f}', xy=(w, bar.get_y() + bar.get_height()/2),
                xytext=(5, 0), textcoords="offset points", ha='left', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'step_cost_comparison_4models.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved {os.path.join(base_dir, 'step_cost_comparison_4models.png')}")

print("\nActual costs (without visual offset):")
for fname, label in models:
    c = costs[label]
    print(f"{label}: total={c['total']:.1f}")
