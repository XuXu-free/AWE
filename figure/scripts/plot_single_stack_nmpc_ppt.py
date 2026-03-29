# -*- coding: utf-8 -*-
"""
美化绘制单槽NMPC test数据 - 答辩PPT专用版本
数据文件: nmpc_data_20260331_114727.csv
过滤预热段 (t < 0)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path

# 设置中文字体（Windows系统）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义专业配色方案
COLORS = {
    'stack1': '#e74c3c',      # 红色 - NMPC
    'ref': '#9467bd',         # 紫色
    'real': '#8c564b',        # 棕色
    'limit': '#e74c3c',       # 警示红
    'grid': '#e0e0e0',        # 网格灰
    'text': '#2c3e50',        # 正文深灰
    'temp': '#ff7f0e',        # 橙色
    'hto': '#2ca02c',         # 绿色
    'h2': '#2ca02c',          # 绿色
}

def load_and_filter_data(filepath):
    """加载数据并过滤掉预热段 (t < 0)"""
    df = pd.read_csv(filepath)
    # 过滤掉预热段
    df = df[df['t'] >= 0].copy()
    # 转换时间为分钟
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

def plot_power_tracking(df, ax, fontsize=14):
    """绘制功率跟踪图"""
    time = df['time_min']
    ax.plot(time, df['P_ref'] / 1e6, '--', label='Reference Power',
            color=COLORS['ref'], linewidth=2.5, alpha=0.9)
    ax.plot(time, df['P_real'] / 1e6, '-', label='Actual Power',
            color=COLORS['stack1'], linewidth=2.5, alpha=0.9)

    setup_axis_style(ax, 'Power Tracking Performance', 'Time (min)', 'Power (MW)', fontsize)
    ax.legend(loc='best', fontsize=fontsize-2, framealpha=0.95)

def plot_stack_temperature(df, ax, fontsize=14):
    """绘制单槽温度图"""
    time = df['time_min']
    ax.plot(time, df['T_s_all'] - 273.15, '-', color=COLORS['temp'],
            linewidth=2.5, alpha=0.9, label='Stack Temperature')

    # 添加温度限制线
    ax.axhline(y=90, color=COLORS['limit'], linestyle='--', linewidth=2,
               label='Upper Limit (90°C)', alpha=0.8)
    ax.axhline(y=80, color=COLORS['ref'], linestyle=':', linewidth=1.5,
               label='Target (80°C)', alpha=0.6)

    setup_axis_style(ax, 'Stack Temperature', 'Time (min)', 'Temperature (°C)', fontsize)
    ax.legend(loc='best', fontsize=fontsize-3, framealpha=0.95)

def plot_current(df, ax, fontsize=14):
    """绘制电流图"""
    time = df['time_min']
    ax.plot(time, df['I'], '-', color=COLORS['stack1'],
            linewidth=2.5, alpha=0.9, label='Current')

    setup_axis_style(ax, 'Stack Current', 'Time (min)', 'Current (A)', fontsize)
    ax.legend(loc='best', fontsize=fontsize-3, framealpha=0.95)

def plot_lye_flow(df, ax, fontsize=14):
    """绘制碱液流量图"""
    time = df['time_min']
    ax.plot(time, df['v_lye'] * 1e3, '-', color=COLORS['temp'],
            linewidth=2.5, alpha=0.9, label='Lye Flow')

    setup_axis_style(ax, 'Lye Flow Rate', 'Time (min)', 'Flow Rate (L/s)', fontsize)
    ax.legend(loc='best', fontsize=fontsize-3, framealpha=0.95)

def plot_hydrogen_safety(df, ax, fontsize=14):
    """绘制氢气安全指标（HTO）图"""
    time = df['time_min']

    ax.plot(time, df['HTO'], '-', color=COLORS['hto'],
            linewidth=2.5, alpha=0.9, label='HTO')
    ax.axhline(y=2, color=COLORS['limit'], linestyle='--', linewidth=2,
               label='Safety Limit (2%)', alpha=0.8)

    # 填充超过限制的区域
    ax.fill_between(time, df['HTO'], 2, where=(df['HTO'] > 2),
                    color=COLORS['limit'], alpha=0.3, interpolate=True)

    setup_axis_style(ax, 'Hydrogen Safety (HTO)', 'Time (min)', 'HTO (%)', fontsize)
    ax.legend(loc='best', fontsize=fontsize-3, framealpha=0.95)
    ax.set_ylim([0, max(2.5, df['HTO'].max() * 1.1)])

def plot_cell_voltage(df, ax, fontsize=14):
    """绘制单槽电压图"""
    time = df['time_min']
    ax.plot(time, df['U_cell_all'], '-', color=COLORS['stack1'],
            linewidth=2.5, alpha=0.9, label='Cell Voltage')

    ax.axhline(y=2.2, color=COLORS['limit'], linestyle='--', linewidth=2,
               label='Limit (2.2V)', alpha=0.8)

    setup_axis_style(ax, 'Cell Voltage', 'Time (min)', 'Voltage (V)', fontsize)
    ax.legend(loc='best', fontsize=fontsize-3, framealpha=0.95)

def plot_coolant_flow(df, ax, fontsize=14):
    """绘制冷却水流量图"""
    time = df['time_min']

    ax.plot(time, df['v_c'] * 1e3, '-', color=COLORS['stack1'],
            linewidth=2.5, alpha=0.9, label='Coolant Flow')

    setup_axis_style(ax, 'Coolant Flow Rate', 'Time (min)', 'Flow Rate (L/s)', fontsize)
    ax.legend(loc='best', fontsize=fontsize-3, framealpha=0.95)

def plot_h2_production(df, ax, fontsize=14):
    """绘制氢气产率图"""
    time = df['time_min']

    ax.plot(time, df['H2_rate'], '-', color=COLORS['h2'],
            linewidth=2.5, alpha=0.9, label='H2 Production Rate')
    ax.fill_between(time, df['H2_rate'], alpha=0.2, color=COLORS['h2'])

    setup_axis_style(ax, 'Hydrogen Production Rate', 'Time (min)', 'H2 Rate (mol/s)', fontsize)

def create_ppt_figure1_power_and_temp(df, output_path):
    """创建PPT图表1: 功率跟踪与温度"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.patch.set_facecolor('white')

    plot_power_tracking(df, axes[0], fontsize=16)
    plot_stack_temperature(df, axes[1], fontsize=16)

    plt.tight_layout(pad=3.0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

def create_ppt_figure2_currents_and_flows(df, output_path):
    """创建PPT图表2: 电流与流量"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('white')

    plot_current(df, axes[0, 0], fontsize=14)
    plot_lye_flow(df, axes[0, 1], fontsize=14)
    plot_cell_voltage(df, axes[1, 0], fontsize=14)
    plot_coolant_flow(df, axes[1, 1], fontsize=14)

    plt.tight_layout(pad=3.0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

def create_ppt_figure3_safety(df, output_path):
    """创建PPT图表3: 安全指标"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.patch.set_facecolor('white')

    plot_hydrogen_safety(df, axes[0], fontsize=16)
    plot_h2_production(df, axes[1], fontsize=16)

    plt.tight_layout(pad=3.0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

def create_widescreen_figure(df, output_path):
    """创建宽屏PPT专用图表 (16:9)"""
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor('white')

    # 创建自定义布局
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :2])  # 顶部横跨两列
    ax2 = fig.add_subplot(gs[0, 2])   # 顶部右侧
    ax3 = fig.add_subplot(gs[1, 0])   # 底部左侧
    ax4 = fig.add_subplot(gs[1, 1])   # 底部中间
    ax5 = fig.add_subplot(gs[1, 2])   # 底部右侧

    # 绘制各子图
    time = df['time_min']

    # 1. 功率与温度 (大图)
    ax1_twin = ax1.twinx()
    ax1.plot(time, df['P_real'] / 1e6, '-', color=COLORS['stack1'], linewidth=2.5, label='Power')
    ax1.plot(time, df['P_ref'] / 1e6, '--', color=COLORS['ref'], linewidth=2, label='Ref Power')
    ax1_twin.plot(time, df['T_s_all'] - 273.15, '-', color=COLORS['temp'],
                  linewidth=1.5, alpha=0.9, label='Temperature')
    ax1.set_xlabel('Time (min)', fontsize=12)
    ax1.set_ylabel('Power (MW)', fontsize=12, color=COLORS['stack1'])
    ax1_twin.set_ylabel('Temperature (°C)', fontsize=12, color=COLORS['temp'])
    ax1.set_title('Power Tracking & Stack Temperature', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(loc='upper left', fontsize=9)
    ax1_twin.legend(loc='upper right', fontsize=9)

    # 2. HTO
    ax2.plot(time, df['HTO'], '-', color=COLORS['hto'], linewidth=2.5)
    ax2.axhline(y=2, color=COLORS['limit'], linestyle='--', linewidth=2)
    ax2.fill_between(time, df['HTO'], 2, where=(df['HTO'] > 2),
                     color=COLORS['limit'], alpha=0.3)
    ax2.set_xlabel('Time (min)', fontsize=12)
    ax2.set_ylabel('HTO (%)', fontsize=12)
    ax2.set_title('HTO Safety', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.4)

    # 3. 电流
    ax3.plot(time, df['I'], '-', color=COLORS['stack1'], linewidth=2)
    ax3.set_xlabel('Time (min)', fontsize=12)
    ax3.set_ylabel('Current (A)', fontsize=12)
    ax3.set_title('Stack Current', fontsize=12, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.4)

    # 4. 电压
    ax4.plot(time, df['U_cell_all'], '-', color=COLORS['stack1'], linewidth=2)
    ax4.axhline(y=2.2, color=COLORS['limit'], linestyle='--', linewidth=2)
    ax4.set_xlabel('Time (min)', fontsize=12)
    ax4.set_ylabel('Voltage (V)', fontsize=12)
    ax4.set_title('Cell Voltage', fontsize=12, fontweight='bold')
    ax4.grid(True, linestyle='--', alpha=0.4)

    # 5. 氢气产率
    ax5.plot(time, df['H2_rate'], '-', color=COLORS['h2'], linewidth=2.5)
    ax5.fill_between(time, df['H2_rate'], alpha=0.3, color=COLORS['h2'])
    ax5.set_xlabel('Time (min)', fontsize=12)
    ax5.set_ylabel('Rate (mol/s)', fontsize=12)
    ax5.set_title('H2 Production', fontsize=12, fontweight='bold')
    ax5.grid(True, linestyle='--', alpha=0.4)

    plt.suptitle('Single-Stack AWE System Test - NMPC Controller',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

def create_comprehensive_figure(df, output_path):
    """创建综合图表 - 九宫格布局"""
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.patch.set_facecolor('white')

    plot_power_tracking(df, axes[0, 0], fontsize=12)
    plot_stack_temperature(df, axes[0, 1], fontsize=12)
    plot_hydrogen_safety(df, axes[0, 2], fontsize=12)
    plot_current(df, axes[1, 0], fontsize=12)
    plot_cell_voltage(df, axes[1, 1], fontsize=12)
    plot_lye_flow(df, axes[1, 2], fontsize=12)
    plot_coolant_flow(df, axes[2, 0], fontsize=12)
    plot_h2_production(df, axes[2, 1], fontsize=12)

    # 第九个图：功率误差
    ax = axes[2, 2]
    time = df['time_min']
    power_error = (df['P_real'] - df['P_ref']) / df['P_ref'] * 100
    ax.plot(time, power_error, '-', color=COLORS['stack1'], linewidth=2)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.fill_between(time, power_error, 0, alpha=0.3, color=COLORS['stack1'])
    ax.set_xlabel('Time (min)', fontsize=12)
    ax.set_ylabel('Error (%)', fontsize=12)
    ax.set_title('Power Tracking Error', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.suptitle('Single-Stack AWE System - NMPC Controller Performance\n(Warmup Removed)',
                 fontsize=18, fontweight='bold', y=1.02, color=COLORS['text'])
    plt.tight_layout(pad=2.5)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

def main():
    # 文件路径
    input_file = "output/single_stack/test/nmpc_data_20260331_114727.csv"
    output_dir = "dataset/figure"

    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 加载数据并过滤预热段
    print(f"Loading data from: {input_file}")
    df = load_and_filter_data(input_file)
    print(f"Loaded {len(df)} data points (warmup removed)")
    print(f"Time range: {df['time_min'].min():.1f} - {df['time_min'].max():.1f} min")

    # 生成各种图表
    print("\nGenerating figures...")

    # 1. PPT图1: 功率与温度
    create_ppt_figure1_power_and_temp(df, f"{output_dir}/nmpc_power_temp.png")

    # 2. PPT图2: 电流与流量
    create_ppt_figure2_currents_and_flows(df, f"{output_dir}/nmpc_currents_flows.png")

    # 3. PPT图3: 安全指标
    create_ppt_figure3_safety(df, f"{output_dir}/nmpc_safety.png")

    # 4. 综合图表
    create_comprehensive_figure(df, f"{output_dir}/nmpc_comprehensive.png")

    # 5. 宽屏PPT图表
    create_widescreen_figure(df, f"{output_dir}/nmpc_widescreen.png")

    print(f"\nAll figures saved to: {output_dir}/")
    print("Files generated:")
    print("  - nmpc_power_temp.png      (Power & Temperature)")
    print("  - nmpc_currents_flows.png  (Currents & Flows)")
    print("  - nmpc_safety.png          (Safety Metrics)")
    print("  - nmpc_comprehensive.png   (9-panel comprehensive)")
    print("  - nmpc_widescreen.png      (16:9 widescreen layout)")

if __name__ == "__main__":
    main()
