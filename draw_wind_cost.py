import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    'Simplified NMPC': 'nmpc_simplified_data_20260426_160953.csv',
}

internal_models = {
    'TCN Diffusion': 'model_diffusion_tcn_data_20260426_121809.csv',
    'Diffusion Pure MLP': 'model_diffusion_pure_mlp_data_20260426_121809.csv',
    'Simplified NMPC': 'nmpc_simplified_data_20260426_160953.csv',
    'Pure MLP': 'model_pure_mlp_data_20260426_151739.csv',
    'Pure TCN': 'model_pure_tcn_data_20260426_153409.csv',
    'LSTM': 'model_lstm_data_20260426_155259.csv',
}

def compute_cost(path, skip_warmup=0, max_len=None):
    df = pd.read_csv(path)
    if skip_warmup > 0:
        df = df.iloc[skip_warmup:]
    if max_len is not None:
        df = df.iloc[:max_len]
    P_ref = df['P_ref'].values
    P_actual = df['P_real'].values
    T_ref = df['T_ref'].values[0] if 'T_ref' in df.columns else 353.15

    T_stacks = df[['T_s_all_1','T_s_all_2','T_s_all_3','T_s_all_4']].values
    I = df[['I_all_1','I_all_2','I_all_3','I_all_4']].values
    v_lye = df[['v_lye_all_1','v_lye_all_2','v_lye_all_3','v_lye_all_4']].values
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

# 计算论文用成本（扣除预热，模型 skip 1440 行，NMPC 不 skip，然后取相同长度）
# NMPC 长度 8640，模型扣除预热后长度也是 8640
min_len_paper = min(
    pd.read_csv(os.path.join(base_dir, f)).shape[0] - (warmup_rows if 'nmpc' not in f else 0)
    for f in paper_models.values()
)
paper_costs = {}
for name, f in paper_models.items():
    skip = warmup_rows if 'nmpc' not in f else 0
    paper_costs[name] = compute_cost(os.path.join(base_dir, f), skip_warmup=skip, max_len=min_len_paper)

print(f"Paper fair comparison length: {min_len_paper}")
for name, c in paper_costs.items():
    print(f"{name}: power={c['power']:.1f}, temp={c['temp']:.1f}, current={c['current']:.1f}, prod={c['prod']:.2f}, lye={c['lye']:.2f}, coolant={c['coolant']:.2f}, total={c['total']:.1f}")

# 绘制论文用图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
components = ['功率跟踪', '温度控制', '电流平滑', '生产均衡', '碱液平滑', '冷却剂平滑']
keys = ['power', 'temp', 'current', 'prod', 'lye', 'coolant']
x = np.arange(len(components))
width = 0.25
colors = ['#4472C4', '#ED7D31', '#70AD47']

for idx, (name, color) in enumerate(zip(paper_models.keys(), colors)):
    vals = [paper_costs[name][k] for k in keys]
    ax.bar(x + idx * width, vals, width, label=name, color=color, edgecolor='white', linewidth=0.5)

ax.set_ylabel('Cost')
ax.set_title('(a) 各成本分量对比')
ax.set_xticks(x + width)
ax.set_xticklabels(components, fontsize=10)
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax = axes[1]
names = list(paper_models.keys())
totals = [paper_costs[name]['total'] for name in names]
bars = ax.bar(names, totals, color=colors, edgecolor='white', linewidth=0.5)
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

# 计算内部参考成本
min_len_internal = min(
    pd.read_csv(os.path.join(base_dir, f)).shape[0] - (warmup_rows if 'nmpc' not in f else 0)
    for f in internal_models.values()
)
internal_costs = {}
for name, f in internal_models.items():
    skip = warmup_rows if 'nmpc' not in f else 0
    internal_costs[name] = compute_cost(os.path.join(base_dir, f), skip_warmup=skip, max_len=min_len_internal)

print(f"\nInternal fair comparison length: {min_len_internal}")
for name, c in internal_costs.items():
    print(f"{name}: power={c['power']:.1f}, temp={c['temp']:.1f}, current={c['current']:.1f}, prod={c['prod']:.2f}, lye={c['lye']:.2f}, coolant={c['coolant']:.2f}, total={c['total']:.1f}")

# 绘制内部参考图
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
models = list(internal_models.keys())
n_models = len(models)
x = np.arange(n_models)
width = 0.6
palette = plt.cm.tab10(np.linspace(0, 1, 6))
key_names = ['power', 'temp', 'current', 'prod', 'lye', 'coolant']
labels = ['功率跟踪', '温度控制', '电流平滑', '生产均衡', '碱液平滑', '冷却剂平滑']

bottom = np.zeros(n_models)
for key, label, color in zip(key_names, labels, palette):
    vals = [internal_costs[m][key] for m in models]
    ax.bar(x, vals, width, label=label, bottom=bottom, color=color, edgecolor='white', linewidth=0.5)
    bottom += vals

ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15, ha='right')
ax.set_ylabel('Cost')
ax.set_title('风电测试 各模型成本分解（堆叠）')
ax.legend(loc='upper left', fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax = axes[1]
totals = [(m, internal_costs[m]['total']) for m in models]
totals_sorted = sorted(totals, key=lambda x: x[1])
names_sorted = [t[0] for t in totals_sorted]
vals_sorted = [t[1] for t in totals_sorted]
colors_sorted = [plt.cm.RdYlGn_r(i / max(n_models - 1, 1)) for i in range(n_models)]
bars = ax.barh(names_sorted[::-1], vals_sorted[::-1], color=colors_sorted[::-1], edgecolor='white', linewidth=0.5)
ax.set_xlabel('Total Cost')
ax.set_title('风电测试 总成本排名（越低越好）')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bar in bars:
    width_val = bar.get_width()
    ax.annotate(f'{width_val:.0f}', xy=(width_val, bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0), textcoords="offset points", ha='left', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('output/multi_stack/test/run_20260426_121806/all_models_wind_cost_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved output/multi_stack/test/run_20260426_121806/all_models_wind_cost_comparison.png")
