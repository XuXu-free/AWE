# -*- coding: utf-8 -*-
"""
NMPC vs Model Controller RMSE对比图 - 答辩PPT专用
展示功率和温度跟踪性能对比
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置中文字体（Windows系统）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义配色方案
COLORS = {
    'nmpc': '#e74c3c',       # 红色 - NMPC
    'model': '#1f77b4',      # 蓝色 - Model Controller
    'grid': '#e0e0e0',       # 网格灰
    'text': '#2c3e50',       # 正文深灰
    'bg': '#f8f9fa',         # 浅灰背景
}

def load_and_filter_data(filepath):
    """加载数据并过滤掉预热段 (t < 0)"""
    df = pd.read_csv(filepath)
    df = df[df['t'] >= 0].copy()
    df['time_min'] = df['t'] / 60.0
    return df

def calculate_metrics(df):
    """计算各项性能指标"""
    # 功率误差 (MW)
    power_error = (df['P_real'] - df['P_ref']) / 1e6
    power_rmse = np.sqrt(np.mean(power_error ** 2))
    power_mae = np.mean(np.abs(power_error))

    # 温度误差 (°C, 目标80°C)
    temp_error = (df['T_s_all'] - 273.15) - 80
    temp_rmse = np.sqrt(np.mean(temp_error ** 2))
    temp_mae = np.mean(np.abs(temp_error))

    # HTO和电压
    hto_max = df['HTO'].max()
    volt_max = df['U_cell_all'].max()

    return {
        'power_rmse': power_rmse,
        'power_mae': power_mae,
        'temp_rmse': temp_rmse,
        'temp_mae': temp_mae,
        'hto_max': hto_max,
        'volt_max': volt_max,
    }

def create_rmse_bar_chart(metrics_nmpc, metrics_model, output_path):
    """创建RMSE柱状对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('none')

    categories = ['NMPC', 'Model Controller']
    x_pos = np.arange(len(categories))
    bar_width = 0.35

    # 左图：功率RMSE
    ax = axes[0]
    ax.set_facecolor('none')
    values = [metrics_nmpc['power_rmse'], metrics_model['power_rmse']]
    colors = [COLORS['nmpc'], COLORS['model']]

    bars = ax.bar(x_pos, values, bar_width, color=colors, alpha=0.85, edgecolor='white', linewidth=2)

    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.003,
                f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold', color=COLORS['text'])

    # 添加改进百分比
    improvement = (metrics_nmpc['power_rmse'] - metrics_model['power_rmse']) / metrics_nmpc['power_rmse'] * 100
    ax.text(0.5, max(values) * 0.9, f'改进: {improvement:.1f}%',
            ha='center', fontsize=11, color='green', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    ax.set_ylabel('RMSE (MW)', fontsize=13, color=COLORS['text'])
    ax.set_title('Power Tracking RMSE', fontsize=14, fontweight='bold', color=COLORS['text'], pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, fontsize=12)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4, color=COLORS['grid'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim([0, max(values) * 1.2])

    # 右图：温度RMSE
    ax = axes[1]
    ax.set_facecolor('none')
    values = [metrics_nmpc['temp_rmse'], metrics_model['temp_rmse']]

    bars = ax.bar(x_pos, values, bar_width, color=colors, alpha=0.85, edgecolor='white', linewidth=2)

    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold', color=COLORS['text'])

    # 添加改进百分比
    improvement = (metrics_nmpc['temp_rmse'] - metrics_model['temp_rmse']) / metrics_nmpc['temp_rmse'] * 100
    ax.text(0.5, max(values) * 0.9, f'改进: {improvement:.1f}%',
            ha='center', fontsize=11, color='green', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    ax.set_ylabel('RMSE (°C)', fontsize=13, color=COLORS['text'])
    ax.set_title('Temperature Tracking RMSE\n(Target: 80°C)', fontsize=14, fontweight='bold', color=COLORS['text'], pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, fontsize=12)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4, color=COLORS['grid'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim([0, max(values) * 1.2])

    plt.suptitle('NMPC vs Model Controller Performance Comparison',
                 fontsize=16, fontweight='bold', y=1.02, color=COLORS['text'])
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='none', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")

def create_comprehensive_metrics_chart(metrics_nmpc, metrics_model, output_path):
    """创建综合指标对比图（多指标）"""
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')

    # 指标名称
    metrics_names = ['Power RMSE\n(MW)', 'Power MAE\n(MW)', 'Temp RMSE\n(°C)', 'Temp MAE\n(°C)']

    # 数值
    nmpc_values = [
        metrics_nmpc['power_rmse'],
        metrics_nmpc['power_mae'],
        metrics_nmpc['temp_rmse'],
        metrics_nmpc['temp_mae']
    ]
    model_values = [
        metrics_model['power_rmse'],
        metrics_model['power_mae'],
        metrics_model['temp_rmse'],
        metrics_model['temp_mae']
    ]

    x = np.arange(len(metrics_names))
    width = 0.35

    # 绘制柱状图
    bars1 = ax.bar(x - width/2, nmpc_values, width, label='NMPC',
                   color=COLORS['nmpc'], alpha=0.85, edgecolor='white', linewidth=2)
    bars2 = ax.bar(x + width/2, model_values, width, label='Model Controller',
                   color=COLORS['model'], alpha=0.85, edgecolor='white', linewidth=2)

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 添加改进百分比标注
    for i, (nmpc_val, model_val) in enumerate(zip(nmpc_values, model_values)):
        improvement = (nmpc_val - model_val) / nmpc_val * 100
        y_pos = max(nmpc_val, model_val) * 1.15
        color = 'green' if improvement > 0 else 'red'
        ax.text(i, y_pos, f'{improvement:+.1f}%', ha='center', fontsize=10,
                color=color, fontweight='bold')

    ax.set_ylabel('Error Value', fontsize=13, color=COLORS['text'])
    ax.set_title('Controller Performance Metrics Comparison', fontsize=15, fontweight='bold',
                 color=COLORS['text'], pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=11)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4, color=COLORS['grid'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 调整Y轴范围
    max_val = max(max(nmpc_values), max(model_values))
    ax.set_ylim([0, max_val * 1.3])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='none', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")

def create_rmse_table_figure(metrics_nmpc, metrics_model, output_path):
    """创建RMSE表格图（适合PPT插入）"""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    ax.axis('off')

    # 准备表格数据
    table_data = [
        ['Metric', 'NMPC', 'Model Controller', 'Improvement'],
        ['Power RMSE (MW)', f"{metrics_nmpc['power_rmse']:.4f}",
         f"{metrics_model['power_rmse']:.4f}",
         f"{(metrics_nmpc['power_rmse']-metrics_model['power_rmse'])/metrics_nmpc['power_rmse']*100:.1f}%"],
        ['Power MAE (MW)', f"{metrics_nmpc['power_mae']:.4f}",
         f"{metrics_model['power_mae']:.4f}",
         f"{(metrics_nmpc['power_mae']-metrics_model['power_mae'])/metrics_nmpc['power_mae']*100:.1f}%"],
        ['Temp RMSE (°C)', f"{metrics_nmpc['temp_rmse']:.4f}",
         f"{metrics_model['temp_rmse']:.4f}",
         f"{(metrics_nmpc['temp_rmse']-metrics_model['temp_rmse'])/metrics_nmpc['temp_rmse']*100:.1f}%"],
        ['Temp MAE (°C)', f"{metrics_nmpc['temp_mae']:.4f}",
         f"{metrics_model['temp_mae']:.4f}",
         f"{(metrics_nmpc['temp_mae']-metrics_model['temp_mae'])/metrics_nmpc['temp_mae']*100:.1f}%"],
        ['HTO Max (%)', f"{metrics_nmpc['hto_max']:.4f}",
         f"{metrics_model['hto_max']:.4f}",
         f"{(metrics_nmpc['hto_max']-metrics_model['hto_max'])/metrics_nmpc['hto_max']*100:.1f}%"],
        ['Voltage Max (V)', f"{metrics_nmpc['volt_max']:.4f}",
         f"{metrics_model['volt_max']:.4f}",
         f"{(metrics_nmpc['volt_max']-metrics_model['volt_max'])/metrics_nmpc['volt_max']*100:.1f}%"],
    ]

    # 创建表格
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                     loc='center', cellLoc='center',
                     colColours=[COLORS['bg']]*4,
                     colWidths=[0.3, 0.2, 0.25, 0.2])

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # 设置表头样式
    for i in range(4):
        table[(0, i)].set_text_props(fontweight='bold', color=COLORS['text'])
        table[(0, i)].set_facecolor('#e9ecef')

    # 设置单元格样式
    for i in range(1, len(table_data)):
        for j in range(4):
            if j == 1:  # NMPC列
                table[(i, j)].set_facecolor('#ffeaea')
            elif j == 2:  # Model列
                table[(i, j)].set_facecolor('#eaf4ff')
            else:
                table[(i, j)].set_facecolor('white')

    ax.set_title('Controller Performance Metrics', fontsize=16, fontweight='bold',
                 color=COLORS['text'], pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='none', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")

def main():
    # 文件路径
    nmpc_file = "output/single_stack/test/nmpc_data_20260331_114727.csv"
    model_file = "output/single_stack/test/model_data_20260319_161737.csv"
    output_dir = "dataset/figure"

    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 加载数据
    print("Loading data...")
    df_nmpc = load_and_filter_data(nmpc_file)
    df_model = load_and_filter_data(model_file)

    # 计算指标
    print("Calculating metrics...")
    metrics_nmpc = calculate_metrics(df_nmpc)
    metrics_model = calculate_metrics(df_model)

    # 打印结果
    print("\n" + "="*60)
    print("RMSE Comparison Results".center(60))
    print("="*60)
    print(f"{'Metric':<25} {'NMPC':>12} {'Model':>12} {'Improvement':>12}")
    print("-"*60)
    print(f"{'Power RMSE (MW)':<25} {metrics_nmpc['power_rmse']:>12.4f} {metrics_model['power_rmse']:>12.4f} {(metrics_nmpc['power_rmse']-metrics_model['power_rmse'])/metrics_nmpc['power_rmse']*100:>11.1f}%")
    print(f"{'Power MAE (MW)':<25} {metrics_nmpc['power_mae']:>12.4f} {metrics_model['power_mae']:>12.4f} {(metrics_nmpc['power_mae']-metrics_model['power_mae'])/metrics_nmpc['power_mae']*100:>11.1f}%")
    print(f"{'Temp RMSE (°C)':<25} {metrics_nmpc['temp_rmse']:>12.4f} {metrics_model['temp_rmse']:>12.4f} {(metrics_nmpc['temp_rmse']-metrics_model['temp_rmse'])/metrics_nmpc['temp_rmse']*100:>11.1f}%")
    print(f"{'Temp MAE (°C)':<25} {metrics_nmpc['temp_mae']:>12.4f} {metrics_model['temp_mae']:>12.4f} {(metrics_nmpc['temp_mae']-metrics_model['temp_mae'])/metrics_nmpc['temp_mae']*100:>11.1f}%")
    print("="*60)

    # 生成图表
    print("\nGenerating figures...")
    create_rmse_bar_chart(metrics_nmpc, metrics_model,
                          f"{output_dir}/rmse_comparison_bar.png")
    create_comprehensive_metrics_chart(metrics_nmpc, metrics_model,
                                       f"{output_dir}/rmse_comprehensive.png")
    create_rmse_table_figure(metrics_nmpc, metrics_model,
                             f"{output_dir}/rmse_table.png")

    print(f"\nAll figures saved to: {output_dir}/")
    print("Files generated:")
    print("  - rmse_comparison_bar.png   (RMSE柱状对比图)")
    print("  - rmse_comprehensive.png    (综合指标对比图)")
    print("  - rmse_table.png            (指标表格图)")

if __name__ == "__main__":
    main()
