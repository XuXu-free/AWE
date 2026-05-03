"""
详细CBF约束绘图脚本 v2

为每种CBF约束单独绘制，并使用adjustment判断真正触发。
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'FangSong']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt


def plot_cbf_detailed(csv_path, output_dir=None):
    """绘制每种CBF约束的详细图。"""
    df = pd.read_csv(csv_path)
    df_main = df[df['t'] >= 0].copy()
    t_hours = df_main['t'] / 3600.0

    # 计算平均温度
    T_avg = df_main[['T_s_all_1', 'T_s_all_2', 'T_s_all_3', 'T_s_all_4']].mean(axis=1)
    T_avg_C = T_avg - 273.15

    # 是否有 adjustment 字段
    has_adj = 'cbf_adjustment' in df_main.columns
    if has_adj:
        adj = df_main['cbf_adjustment']
        max_slack = df_main['cbf_max_slack']
        # 真正触发：adjustment >= 0.1 或 max_slack > 1e-6
        real_trigger = (adj >= 0.1) | (max_slack > 1e-6)
    else:
        real_trigger = df_main['cbf_triggered']

    # 约束名称
    constraint_names = {
        0: 'Temp Upper Stack 1 (T_s < 90C)',
        1: 'Temp Upper Stack 2 (T_s < 90C)',
        2: 'Temp Upper Stack 3 (T_s < 90C)',
        3: 'Temp Upper Stack 4 (T_s < 90C)',
        4: 'HTO (HTO < 2%)',
        5: 'Temp Lower Stack 1 (T_s > 20C)',
        6: 'Temp Lower Stack 2 (T_s > 20C)',
        7: 'Temp Lower Stack 3 (T_s > 20C)',
        8: 'Temp Lower Stack 4 (T_s > 20C)',
    }

    # 创建输出目录
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(csv_path), 'cbf_detailed')
    os.makedirs(output_dir, exist_ok=True)

    for j in range(9):
        fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
        fig.suptitle(f'CBF Constraint {j}: {constraint_names[j]}', fontsize=14, fontweight='bold')

        h_col = f'cbf_h_{j}'
        val_col = f'cbf_val_{j}'
        slack_col = f'cbf_slack_{j}'

        # 1. h(t) - barrier函数值
        ax = axes[0]
        ax.plot(t_hours, df_main[h_col], 'b-', linewidth=0.8, label='h(t)')
        ax.axhline(y=0, color='r', linestyle='--', linewidth=1, label='h=0')
        if j < 4:
            ax.axhline(y=10, color='g', linestyle=':', linewidth=1, alpha=0.5, label='h=10 (approx)')
        elif j == 4:
            ax.axhline(y=0.02, color='g', linestyle=':', linewidth=1, alpha=0.5, label='h=0.02 (2%)')
        ax.set_ylabel('h(t)')
        ax.set_title('Barrier Function h(t)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # 2. cbf_val(t) - CBF约束值
        ax = axes[1]
        colors = ['green' if v >= 0 else 'red' for v in df_main[val_col]]
        ax.scatter(t_hours, df_main[val_col], c=colors, s=3, alpha=0.7)
        ax.plot(t_hours, df_main[val_col], 'b-', linewidth=0.5, alpha=0.5)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
        ax.set_ylabel('CBF Value')
        ax.set_title('CBF Constraint Value (cbf_val)')
        ax.legend(['CBF >= 0 (safe)', 'CBF < 0 (violated)'], loc='upper right')
        ax.grid(True, alpha=0.3)

        # 3. slack(t) - 松弛变量
        ax = axes[2]
        ax.plot(t_hours, df_main[slack_col], 'm-', linewidth=0.8)
        ax.set_ylabel('Slack')
        ax.set_title('Slack Variable (s_opt)')
        ax.grid(True, alpha=0.3)

        # 4. 触发标记（使用adjustment判断）
        ax = axes[3]
        if has_adj:
            # 两层：底层是projection_needed，上层是真正触发
            ax.fill_between(t_hours, 0, df_main['cbf_triggered'].astype(float), alpha=0.15, color='orange', label='QP Called')
            ax.fill_between(t_hours, 0, real_trigger.astype(float), alpha=0.4, color='red', label='Real Trigger (adj>=0.1 | slack>1e-6)')
            ax.set_title('CBF Trigger Status (adjustment-based)')
        else:
            ax.fill_between(t_hours, 0, df_main['cbf_triggered'].astype(float), alpha=0.3, color='red', label='Projection Active')
            ax.set_title('CBF Projection Triggered')
        ax.set_ylim(-0.1, 1.2)
        ax.set_ylabel('Triggered')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # 5. 温度/HTO参考
        ax = axes[4]
        if j < 4:
            ax.plot(t_hours, df_main[f'T_s_all_{j+1}'] - 273.15, 'r-', linewidth=0.8, label=f'T_s_{j+1}')
            ax.axhline(y=90, color='r', linestyle='--', linewidth=1, label='90C limit')
            ax.axhline(y=85, color='orange', linestyle=':', linewidth=1, label='85C threshold')
            ax.set_ylabel('Temperature (C)')
            ax.set_title(f'Stack {j+1} Temperature')
        elif j == 4:
            ax.plot(t_hours, df_main['HTO'], 'b-', linewidth=0.8, label='HTO')
            ax.axhline(y=2.0, color='r', linestyle='--', linewidth=1, label='2% limit')
            ax.set_ylabel('HTO (%)')
            ax.set_title('HTO (Hydrogen in Oxygen)')
        else:
            ax.plot(t_hours, df_main[f'T_s_all_{j-4}'] - 273.15, 'g-', linewidth=0.8, label=f'T_s_{j-4}')
            ax.axhline(y=20, color='b', linestyle=':', linewidth=1, label='20C limit')
            ax.set_ylabel('Temperature (C)')
            ax.set_title(f'Stack {j-4} Temperature (Lower Bound)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time (hours)')

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        output_path = os.path.join(output_dir, f'cbf_constraint_{j}_detailed.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f'Saved: {output_path}')

    # 额外绘制一个汇总图
    fig, axes = plt.subplots(4 if has_adj else 3, 1, figsize=(14, 12 if has_adj else 10), sharex=True)
    fig.suptitle('CBF Constraints Overview', fontsize=14, fontweight='bold')

    # 所有温度CBF
    ax = axes[0]
    for j in range(4):
        ax.plot(t_hours, df_main[f'cbf_val_{j}'], label=f'Temp Upper {j+1}', linewidth=0.8)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax.set_ylabel('CBF Value')
    ax.set_title('Temperature Upper CBF Values (0-3)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # HTO CBF
    ax = axes[1]
    ax.plot(t_hours, df_main['cbf_val_4'], 'b-', linewidth=0.8, label='HTO CBF')
    ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax.set_ylabel('CBF Value')
    ax.set_title('HTO CBF Value (4)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    if has_adj:
        # adjustment 和 max_slack
        ax = axes[2]
        ax.semilogy(t_hours, np.maximum(adj, 1e-10), 'b-', linewidth=0.8, label='Adjustment')
        ax.semilogy(t_hours, np.maximum(max_slack, 1e-10), 'r-', linewidth=0.8, label='Max Slack')
        ax.axhline(y=0.1, color='b', linestyle='--', linewidth=1, label='adj=0.1 threshold')
        ax.axhline(y=1e-6, color='r', linestyle='--', linewidth=1, label='slack=1e-6 threshold')
        ax.set_ylabel('Log Scale')
        ax.set_title('Adjustment and Max Slack (log scale)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    # 触发汇总
    ax = axes[-1]
    if has_adj:
        ax.fill_between(t_hours, 0, df_main['cbf_triggered'].astype(float), alpha=0.15, color='orange', label='QP Called')
        ax.fill_between(t_hours, 0, real_trigger.astype(float), alpha=0.4, color='red', label='Real Trigger')
    else:
        ax.fill_between(t_hours, 0, df_main['cbf_triggered'].astype(float), alpha=0.3, color='red', label='Projection Active')
    ax.plot(t_hours, T_avg_C, 'b-', linewidth=0.8, label='T_avg')
    ax.axhline(y=85, color='orange', linestyle=':', linewidth=1, label='85C threshold')
    ax.axhline(y=90, color='r', linestyle='--', linewidth=1, label='90C limit')
    ax.set_ylim(-0.5, 95)
    ax.set_ylabel('Triggered / Temp')
    ax.set_title('Trigger Status vs Temperature')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Time (hours)')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = os.path.join(output_dir, 'cbf_overview.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')

    # 打印统计
    print('\n' + '='*70)
    print('CBF统计')
    print('='*70)
    for j in range(9):
        val_col = f'cbf_val_{j}'
        neg_count = (df_main[val_col] < 0).sum()
        min_val = df_main[val_col].min()
        print(f'Constraint {j} ({constraint_names[j]}): 负值={neg_count}, 最小值={min_val:.6f}')

    if has_adj:
        print('\n--- 触发统计（adjustment-based）---')
        print(f'QP Called (projection_needed=True): {df_main["cbf_triggered"].sum()} / {len(df_main)} ({df_main["cbf_triggered"].mean()*100:.1f}%)')
        print(f'Real Trigger (adj>=0.1 | slack>1e-6): {real_trigger.sum()} / {len(df_main)} ({real_trigger.mean()*100:.1f}%)')
    print('='*70)


if __name__ == '__main__':
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'output/multi_stack/test/wind_cbf_ho_adaptive_test_12h/cbf_ho_model_data_20260427_174828.csv'
    plot_cbf_detailed(csv_path)
