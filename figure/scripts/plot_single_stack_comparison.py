# -*- coding: utf-8 -*-
"""
单槽 NMPC vs Model Controller 对比绘图 - 答辩PPT专用
对比两个控制器的效果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# 设置中文字体（Windows系统）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义配色方案 - NMPC vs Model
COLORS = {
    'nmpc': '#e74c3c',       # 红色 - NMPC
    'model': '#1f77b4',      # 蓝色 - Model Controller
    'ref': '#7f8c8d',        # 灰色 - Reference
    'limit': '#e74c3c',      # 警示红
    'grid': '#e0e0e0',       # 网格灰
    'text': '#2c3e50',       # 正文深灰
    'temp': '#ff7f0e',       # 橙色
    'hto': '#2ca02c',        # 绿色
    'h2': '#2ca02c',         # 绿色
}

def load_and_filter_data(filepath):
    """加载数据并过滤掉预热段 (t < 0)"""
    df = pd.read_csv(filepath)
    df = df[df['t'] >= 0].copy()
    df['time_min'] = df['t'] / 60.0
    return df

def setup_axis_style(ax, title, xlabel, ylabel, fontsize=14):
    """统一设置坐标轴样式"""
    ax.set_title(title, fontsize=fontsize+4, fontweight='bold', color=COLORS['text'], pad=15)
    ax.set_xlabel(xlabel, fontsize=fontsize, color=COLORS['text'])
    ax.set_ylabel(ylabel, fontsize=fontsize, color=COLORS['text'])
    ax.grid(True, linestyle='--', alpha=0.5, color=COLORS['grid'])
    ax.tick_params(labelsize=fontsize-2, colors=COLORS['text'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color('#cccccc')

def plot_power_comparison(df_nmpc, df_model, ax, fontsize=14):
    """绘制功率跟踪对比图"""
    ax.plot(df_nmpc['time_min'], df_nmpc['P_ref'] / 1e6, '--', label='Reference',
            color=COLORS['ref'], linewidth=2.5, alpha=0.9)
    ax.plot(df_nmpc['time_min'], df_nmpc['P_real'] / 1e6, '-', label='NMPC',
            color=COLORS['nmpc'], linewidth=2.5, alpha=0.9)
    ax.plot(df_model['time_min'], df_model['P_real'] / 1e6, '-', label='Model Controller',
            color=COLORS['model'], linewidth=2.5, alpha=0.9)

    setup_axis_style(ax, 'Power Tracking Comparison', 'Time (min)', 'Power (MW)', fontsize)
    ax.legend(loc='best', fontsize=fontsize-2, framealpha=0.95)

def plot_temperature_comparison(df_nmpc, df_model, ax, fontsize=14):
    """绘制温度对比图"""
    ax.plot(df_nmpc['time_min'], df_nmpc['T_s_all'] - 273.15, '-', label='NMPC',
            color=COLORS['nmpc'], linewidth=2.5, alpha=0.9)
    ax.plot(df_model['time_min'], df_model['T_s_all'] - 273.15, '-', label='Model Controller',
            color=COLORS['model'], linewidth=2.5, alpha=0.9)
    ax.axhline(y=90, color=COLORS['limit'], linestyle='--', linewidth=2,
               label='Limit (90°C)', alpha=0.8)
    ax.axhline(y=80, color=COLORS['ref'], linestyle=':', linewidth=1.5,
               label='Target (80°C)', alpha=0.6)

    setup_axis_style(ax, 'Stack Temperature Comparison', 'Time (min)', 'Temperature (°C)', fontsize)
    ax.legend(loc='best', fontsize=fontsize-3, framealpha=0.95)

def plot_current_comparison(df_nmpc, df_model, ax, fontsize=14):
    """绘制电流对比图"""
    ax.plot(df_nmpc['time_min'], df_nmpc['I'], '-', label='NMPC',
            color=COLORS['nmpc'], linewidth=2.5, alpha=0.9)
    ax.plot(df_model['time_min'], df_model['I'], '-', label='Model Controller',
            color=COLORS['model'], linewidth=2.5, alpha=0.9)

    setup_axis_style(ax, 'Current Comparison', 'Time (min)', 'Current (A)', fontsize)
    ax.legend(loc='best', fontsize=fontsize-2, framealpha=0.95)

def plot_voltage_comparison(df_nmpc, df_model, ax, fontsize=14):
    """绘制电压对比图"""
    ax.plot(df_nmpc['time_min'], df_nmpc['U_cell_all'], '-', label='NMPC',
            color=COLORS['nmpc'], linewidth=2.5, alpha=0.9)
    ax.plot(df_model['time_min'], df_model['U_cell_all'], '-', label='Model Controller',
            color=COLORS['model'], linewidth=2.5, alpha=0.9)
    ax.axhline(y=2.2, color=COLORS['limit'], linestyle='--', linewidth=2,
               label='Limit (2.2V)', alpha=0.8)

    setup_axis_style(ax, 'Cell Voltage Comparison', 'Time (min)', 'Voltage (V)', fontsize)
    ax.legend(loc='best', fontsize=fontsize-3, framealpha=0.95)

def plot_lye_flow_comparison(df_nmpc, df_model, ax, fontsize=14):
    """绘制碱液流量对比图"""
    ax.plot(df_nmpc['time_min'], df_nmpc['v_lye'] * 1e3, '-', label='NMPC',
            color=COLORS['nmpc'], linewidth=2.5, alpha=0.9)
    ax.plot(df_model['time_min'], df_model['v_lye'] * 1e3, '-', label='Model Controller',
            color=COLORS['model'], linewidth=2.5, alpha=0.9)

    setup_axis_style(ax, 'Lye Flow Rate Comparison', 'Time (min)', 'Flow Rate (L/s)', fontsize)
    ax.legend(loc='best', fontsize=fontsize-3, framealpha=0.95)

def plot_coolant_comparison(df_nmpc, df_model, ax, fontsize=14):
    """绘制冷却水流量对比图"""
    ax.plot(df_nmpc['time_min'], df_nmpc['v_c'] * 1e3, '-', label='NMPC',
            color=COLORS['nmpc'], linewidth=2.5, alpha=0.9)
    ax.plot(df_model['time_min'], df_model['v_c'] * 1e3, '-', label='Model Controller',
            color=COLORS['model'], linewidth=2.5, alpha=0.9)

    setup_axis_style(ax, 'Coolant Flow Comparison', 'Time (min)', 'Flow Rate (L/s)', fontsize)
    ax.legend(loc='best', fontsize=fontsize-3, framealpha=0.95)

def plot_hto_comparison(df_nmpc, df_model, ax, fontsize=14):
    """绘制HTO对比图"""
    ax.plot(df_nmpc['time_min'], df_nmpc['HTO'], '-', label='NMPC',
            color=COLORS['nmpc'], linewidth=2.5, alpha=0.9)
    ax.plot(df_model['time_min'], df_model['HTO'], '-', label='Model Controller',
            color=COLORS['model'], linewidth=2.5, alpha=0.9)
    ax.axhline(y=2, color=COLORS['limit'], linestyle='--', linewidth=2,
               label='Safety Limit (2%)', alpha=0.8)

    setup_axis_style(ax, 'HTO Comparison', 'Time (min)', 'HTO (%)', fontsize)
    ax.legend(loc='best', fontsize=fontsize-3, framealpha=0.95)
    ax.set_ylim([0, max(2.5, df_nmpc['HTO'].max(), df_model['HTO'].max()) * 1.1])

def plot_h2_comparison(df_nmpc, df_model, ax, fontsize=14):
    """绘制氢气产率对比图"""
    ax.plot(df_nmpc['time_min'], df_nmpc['H2_rate'], '-', label='NMPC',
            color=COLORS['nmpc'], linewidth=2.5, alpha=0.9)
    ax.plot(df_model['time_min'], df_model['H2_rate'], '-', label='Model Controller',
            color=COLORS['model'], linewidth=2.5, alpha=0.9)
    ax.fill_between(df_nmpc['time_min'], df_nmpc['H2_rate'], alpha=0.2, color=COLORS['nmpc'])
    ax.fill_between(df_model['time_min'], df_model['H2_rate'], alpha=0.2, color=COLORS['model'])

    setup_axis_style(ax, 'H₂ Production Rate Comparison', 'Time (min)', 'Rate (mol/s)', fontsize)
    ax.legend(loc='best', fontsize=fontsize-3, framealpha=0.95)

def plot_power_error_comparison(df_nmpc, df_model, ax, fontsize=14):
    """绘制功率跟踪误差对比图"""
    error_nmpc = (df_nmpc['P_real'] - df_nmpc['P_ref']) / df_nmpc['P_ref'] * 100
    error_model = (df_model['P_real'] - df_model['P_ref']) / df_model['P_ref'] * 100

    ax.plot(df_nmpc['time_min'], error_nmpc, '-', label='NMPC',
            color=COLORS['nmpc'], linewidth=2.5, alpha=0.9)
    ax.plot(df_model['time_min'], error_model, '-', label='Model Controller',
            color=COLORS['model'], linewidth=2.5, alpha=0.9)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.fill_between(df_nmpc['time_min'], error_nmpc, 0, alpha=0.2, color=COLORS['nmpc'])
    ax.fill_between(df_model['time_min'], error_model, 0, alpha=0.2, color=COLORS['model'])

    setup_axis_style(ax, 'Power Tracking Error', 'Time (min)', 'Error (%)', fontsize)
    ax.legend(loc='best', fontsize=fontsize-3, framealpha=0.95)

def create_side_by_side_comparison(df_nmpc, df_model, output_path):
    """创建并排对比图 - 2x3布局"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.patch.set_facecolor('white')

    plot_power_comparison(df_nmpc, df_model, axes[0, 0], fontsize=12)
    plot_temperature_comparison(df_nmpc, df_model, axes[0, 1], fontsize=12)
    plot_hto_comparison(df_nmpc, df_model, axes[0, 2], fontsize=12)
    plot_current_comparison(df_nmpc, df_model, axes[1, 0], fontsize=12)
    plot_voltage_comparison(df_nmpc, df_model, axes[1, 1], fontsize=12)
    plot_power_error_comparison(df_nmpc, df_model, axes[1, 2], fontsize=12)

    plt.suptitle('Single-Stack AWE: NMPC vs Model Controller Comparison\n(Warmup Removed)',
                 fontsize=18, fontweight='bold', y=1.02, color=COLORS['text'])
    plt.tight_layout(pad=3.0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

def create_power_temp_comparison(df_nmpc, df_model, output_path):
    """创建功率和温度专用对比图 - 用于PPT"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.patch.set_facecolor('white')

    # 功率对比
    ax = axes[0]
    ax.plot(df_nmpc['time_min'], df_nmpc['P_ref'] / 1e6, '--', label='Reference',
            color=COLORS['ref'], linewidth=2.5, alpha=0.9)
    ax.plot(df_nmpc['time_min'], df_nmpc['P_real'] / 1e6, '-', label='NMPC',
            color=COLORS['nmpc'], linewidth=2.5, alpha=0.9)
    ax.plot(df_model['time_min'], df_model['P_real'] / 1e6, '-', label='Model Controller',
            color=COLORS['model'], linewidth=2.5, alpha=0.9)
    setup_axis_style(ax, 'Power Tracking Performance', 'Time (min)', 'Power (MW)', 16)
    ax.legend(loc='best', fontsize=14, framealpha=0.95)

    # 温度对比
    ax = axes[1]
    ax.plot(df_nmpc['time_min'], df_nmpc['T_s_all'] - 273.15, '-', label='NMPC',
            color=COLORS['nmpc'], linewidth=2.5, alpha=0.9)
    ax.plot(df_model['time_min'], df_model['T_s_all'] - 273.15, '-', label='Model Controller',
            color=COLORS['model'], linewidth=2.5, alpha=0.9)
    ax.axhline(y=90, color=COLORS['limit'], linestyle='--', linewidth=2,
               label='Limit (90°C)', alpha=0.8)
    setup_axis_style(ax, 'Stack Temperature', 'Time (min)', 'Temperature (°C)', 16)
    ax.legend(loc='best', fontsize=13, framealpha=0.95)

    plt.tight_layout(pad=3.0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

def create_metrics_table(df_nmpc, df_model):
    """计算并打印性能指标对比"""
    def calc_metrics(df):
        power_error = np.abs((df['P_real'] - df['P_ref']) / df['P_ref'] * 100)
        return {
            'Power Error Mean (%)': power_error.mean(),
            'Power Error Std (%)': power_error.std(),
            'Power Error Max (%)': power_error.max(),
            'Temp Mean (°C)': (df['T_s_all'] - 273.15).mean(),
            'Temp Max (°C)': (df['T_s_all'] - 273.15).max(),
            'HTO Mean (%)': df['HTO'].mean(),
            'HTO Max (%)': df['HTO'].max(),
            'Voltage Mean (V)': df['U_cell_all'].mean(),
            'Voltage Max (V)': df['U_cell_all'].max(),
        }

    metrics_nmpc = calc_metrics(df_nmpc)
    metrics_model = calc_metrics(df_model)

    print("\n" + "="*70)
    print("Performance Metrics Comparison".center(70))
    print("="*70)
    print(f"{'Metric':<25} {'NMPC':>15} {'Model Ctrl':>15} {'Improvement':>12}")
    print("-"*70)

    for key in metrics_nmpc.keys():
        nmpc_val = metrics_nmpc[key]
        model_val = metrics_model[key]
        improvement = ((nmpc_val - model_val) / nmpc_val * 100) if nmpc_val != 0 else 0
        print(f"{key:<25} {nmpc_val:>15.4f} {model_val:>15.4f} {improvement:>11.2f}%")

    print("="*70)

    return metrics_nmpc, metrics_model

def main():
    # 文件路径
    nmpc_file = "output/single_stack/test/nmpc_data_20260331_114727.csv"
    model_file = "output/single_stack/test/model_data_20260319_161737.csv"
    output_dir = "dataset/figure"

    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 加载数据
    print("Loading NMPC data...")
    df_nmpc = load_and_filter_data(nmpc_file)
    print(f"  NMPC: {len(df_nmpc)} data points")

    print("Loading Model Controller data...")
    df_model = load_and_filter_data(model_file)
    print(f"  Model: {len(df_model)} data points")

    # 生成图表
    print("\nGenerating comparison figures...")
    create_side_by_side_comparison(df_nmpc, df_model,
                                   f"{output_dir}/single_stack_comparison_full.png")
    create_power_temp_comparison(df_nmpc, df_model,
                                 f"{output_dir}/single_stack_comparison_power_temp.png")

    # 计算指标
    create_metrics_table(df_nmpc, df_model)

    print(f"\nAll figures saved to: {output_dir}/")
    print("Files generated:")
    print("  - single_stack_comparison_full.png       (6-panel comparison)")
    print("  - single_stack_comparison_power_temp.png (Power & Temp focus)")

if __name__ == "__main__":
    main()
