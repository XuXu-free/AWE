"""
ARM 平台控制器单步计算时间对比
对应论文图 3-6(b) 风格的 ARM 版本

数据源: scripts/multi_stack/bench_model_arm.py (aarch64, CUDA)
NMPC: 占位值(待实测), 其余为模型推理耗时均值 (ms)
输出: paper/figures/arm_timing_comparison.png (整图)
       paper/figures/arm_timing_comparison_b.png (子图, 供 LaTeX subfigure 引用)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

# 显式指定中文字体文件（旧版 matplotlib fallback 不可靠）
cjk_font = FontProperties(fname='/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf', size=12)
cjk_font_small = FontProperties(fname='/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf', size=9)

METHOD_COLORS = {
    'TCN Diffusion':  '#d62728',  # red - proposed method
    'MLP Diffusion':  '#2ca02c',  # green
    'NMPC':           '#1f77b4',  # blue - baseline
    'MLP':            '#ff7f0e',  # orange
}

METHOD_PLACEHOLDER = {
    'TCN Diffusion': False,
    'MLP Diffusion': False,
    'NMPC': False,
    'MLP': False,
}

# ARM bench results (dt_ctrl=60s, dt_sim=0.1s)
# 仅保留论文图 3-6 对应的四种控制器, NMPC 放最前作为基准
arm_timing = {
    'NMPC':           97137.0,   # ARM bench实测 (dt_sub=0.2s, horizon=3)
    'TCN Diffusion':  862.356,
    'MLP Diffusion':  708.878,  # diffusion_pure_mlp (hidden_dim=512, 1024 candidates)
    'MLP':            67.017,
}

fig, ax = plt.subplots(figsize=(6, 4.5))

labels = list(arm_timing.keys())
vals = list(arm_timing.values())
colors = [METHOD_COLORS[l] for l in labels]

bars = ax.bar(labels, vals, color=colors, edgecolor='white', linewidth=0.5)

for bar, label in zip(bars, labels):
    height = bar.get_height()
    if METHOD_PLACEHOLDER[label]:
        ax.annotate('待测', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontproperties=cjk_font_small, color='#1f77b4')
    else:
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9)

ax.set_yscale('log')
ax.set_ylabel('单步计算时间 毫秒', fontproperties=cjk_font)
ax.set_xlabel('控制器类型', fontproperties=cjk_font)
ax.set_xticklabels(labels, rotation=0)
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)

plt.tight_layout(pad=2.0)
plt.savefig('../figures/arm_timing_comparison.png', dpi=600, bbox_inches='tight', facecolor='white')
print('Saved: ../figures/arm_timing_comparison.png')

# 裁剪出独立子图 PNG (供 LaTeX subfigure 引用)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
bbox = ax.get_tightbbox(renderer)
bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
bbox_inches = bbox_inches.expanded(1.02, 1.02)
fig.savefig('../figures/arm_timing_comparison_b.png', dpi=600, bbox_inches=bbox_inches, facecolor='white')
print('Saved: ../figures/arm_timing_comparison_b.png')

plt.close()
