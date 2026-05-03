import pandas as pd
import numpy as np
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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

lambda_track = 1.2
lambda_temp = 0.15
lambda_I = 0.0002
lambda_lye = 25000
lambda_c = 2500
lambda_prod = 1e-6

base_dir = '../../output/multi_stack/test_step/run_20260426_165250'

paper_models = {
    'TCN Diffusion': 'step_model_diffusion_tcn_data_20260426_135007.csv',
    'Diffusion MLP': 'step_model_diffusion_pure_mlp_data_20260426_135007.csv',
    'NMPC': 'step_nmpc_simplified_data_20260426_184330.csv',
    'MLP': '../pure_mlp_10epoch/step_model_pure_mlp_data_20260503_013630.csv',
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

def compute_cost(path):
    df = load_and_unify(path)
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
        'power': power_cost,
        'temp': temp_cost,
        'current': current_smooth_cost,
        'prod': prod_equal_cost,
        'lye': lye_smooth_cost,
        'coolant': coolant_smooth_cost,
        'total': total,
        'len': len(df)
    }

paper_costs = {}
for name, f in paper_models.items():
    paper_costs[name] = compute_cost(base_dir + '/' + f)

for name, c in paper_costs.items():
    print(f"{name}: power={c['power']:.1f}, temp={c['temp']:.1f}, current={c['current']:.1f}, prod={c['prod']:.2f}, lye={c['lye']:.2f}, coolant={c['coolant']:.2f}, total={c['total']:.1f}")

# 读取运行时间数据
timing_json = '../../output/multi_stack/timing_benchmark/four_controller_benchmark_20260502_184127.json'
if os.path.exists(timing_json):
    with open(timing_json, 'r') as f:
        timing_data = json.load(f)
    controller_map = {
        'nmpc_full': 'NMPC',
        'diffusion_tcn_l3_batch_1024': 'TCN Diffusion',
        'diffusion_pure_mlp': 'Diffusion MLP',
        'pure_mlp': 'MLP',
    }
    timing = {}
    for stat in timing_data['stats']:
        name = controller_map.get(stat['controller'], stat['controller'])
        timing[name] = stat['mean_ms']
    print("Timing loaded:", timing)
else:
    timing = {}
    print(f"Warning: timing JSON not found at {timing_json}")

# ========== 绘图 ==========
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.15, wspace=0.25)
# 总标题由 LaTeX ption 控制，此处不设置 suptitle

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

def setup_ax(ax):
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
    ax.tick_params(axis='both', which='both', length=3)

ax = axes[0]
names = list(paper_models.keys())
totals = [paper_costs[name]['total'] for name in names]
bars = ax.bar(names, totals, color=colors, edgecolor='white', linewidth=0.5)
ax.set_ylabel('总成本')
ax.text(0.02, 0.98, '(a)', transform=ax.transAxes, fontsize=14, va='top', ha='left')
setup_ax(ax)
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

# (b) 单步计算时间对比
ax_time = axes[1]
if timing:
    ordered_labels = ['NMPC', 'TCN Diffusion', 'Diffusion MLP', 'MLP']
    time_vals = [timing.get(l, 0) for l in ordered_labels]
    bar_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = ax_time.bar(ordered_labels, time_vals, color=bar_colors, edgecolor='white', linewidth=0.5)
    ax_time.set_ylabel('单步计算时间 (ms)')
    ax_time.text(0.02, 0.98, '(b)', transform=ax_time.transAxes, fontsize=14, va='top', ha='left')
    ax_time.set_yscale('log')
    setup_ax(ax_time)
    for bar in bars:
        height = bar.get_height()
        ax_time.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

plt.savefig('../figures/controller_cost_comparison.png', dpi=600, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: figures/controller_cost_comparison.png")
