# -*- coding: utf-8 -*-
"""
美化绘制12月份风电数据单槽test结果 - 答辩PPT专用版本
数据文件: model_data_20260331_202442.csv (12-month wind data)
过滤预热段 (t < 0)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置中文字体（Windows系统）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义专业配色方案
COLORS = {
    'stack1': '#1f77b4',      # 蓝色
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
    # 转换时间为小时（12月份数据时间更长）
    df['time_hour'] = df['t'] / 3600.0
    df['time_day'] = df['t'] / 86400.0
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

def create_ppt_figure1_power_and_temp(df, output_path, use_day=True):
    """创建PPT图表1: 功率跟踪与温度"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.patch.set_facecolor('white')

    time_col = 'time_day' if use_day else 'time_hour'
    time_label = 'Time (days)' if use_day else 'Time (hours)'
    time = df[time_col]

    # 功率跟踪
    axes[0].plot(time, df['P_ref'] / 1e6, '--', label='Reference Power',
            color=COLORS['ref'], linewidth=2.5, alpha=0.9)
    axes[0].plot(time, df['P_real'] / 1e6, '-', label='Actual Power',
            color=COLORS['stack1'], linewidth=2.5, alpha=0.9)
    setup_axis_style(axes[0], 'Power Tracking Performance (12-Month Wind)', time_label, 'Power (MW)', 16)
    axes[0].legend(loc='best', fontsize=14, framealpha=0.95)

    # 温度
    axes[1].plot(time, df['T_s_all'] - 273.15, '-', color=COLORS['temp'],
            linewidth=2.5, alpha=0.9, label='Stack Temperature')
    axes[1].axhline(y=90, color=COLORS['limit'], linestyle='--', linewidth=2,
               label='Upper Limit (90°C)', alpha=0.8)
    axes[1].axhline(y=80, color=COLORS['ref'], linestyle=':', linewidth=1.5,
               label='Target (80°C)', alpha=0.6)
    setup_axis_style(axes[1], 'Stack Temperature', time_label, 'Temperature (°C)', 16)
    axes[1].legend(loc='best', fontsize=13, framealpha=0.95)

    plt.tight_layout(pad=3.0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

def create_ppt_figure2_currents_and_flows(df, output_path, use_day=True):
    """创建PPT图表2: 电流与流量"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('white')

    time_col = 'time_day' if use_day else 'time_hour'
    time_label = 'Time (days)' if use_day else 'Time (hours)'
    time = df[time_col]

    # 电流
    axes[0, 0].plot(time, df['I'], '-', color=COLORS['stack1'],
            linewidth=2.5, alpha=0.9, label='Current')
    setup_axis_style(axes[0, 0], 'Stack Current', time_label, 'Current (A)', 14)
    axes[0, 0].legend(loc='best', fontsize=11, framealpha=0.95)

    # 碱液流量
    axes[0, 1].plot(time, df['v_lye'] * 1e3, '-', color=COLORS['temp'],
            linewidth=2.5, alpha=0.9, label='Lye Flow')
    setup_axis_style(axes[0, 1], 'Lye Flow Rate', time_label, 'Flow Rate (L/s)', 14)
    axes[0, 1].legend(loc='best', fontsize=11, framealpha=0.95)

    # 电压
    axes[1, 0].plot(time, df['U_cell_all'], '-', color=COLORS['stack1'],
            linewidth=2.5, alpha=0.9, label='Cell Voltage')
    axes[1, 0].axhline(y=2.2, color=COLORS['limit'], linestyle='--', linewidth=2,
               label='Limit (2.2V)', alpha=0.8)
    setup_axis_style(axes[1, 0], 'Cell Voltage', time_label, 'Voltage (V)', 14)
    axes[1, 0].legend(loc='best', fontsize=11, framealpha=0.95)

    # 冷却水流量
    axes[1, 1].plot(time, df['v_c'] * 1e3, '-', color=COLORS['stack1'],
            linewidth=2.5, alpha=0.9, label='Coolant Flow')
    setup_axis_style(axes[1, 1], 'Coolant Flow Rate', time_label, 'Flow Rate (L/s)', 14)
    axes[1, 1].legend(loc='best', fontsize=11, framealpha=0.95)

    plt.tight_layout(pad=3.0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

def create_ppt_figure3_safety(df, output_path, use_day=True):
    """创建PPT图表3: 安全指标"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.patch.set_facecolor('white')

    time_col = 'time_day' if use_day else 'time_hour'
    time_label = 'Time (days)' if use_day else 'Time (hours)'
    time = df[time_col]

    # HTO
    axes[0].plot(time, df['HTO'], '-', color=COLORS['hto'],
            linewidth=2.5, alpha=0.9, label='HTO')
    axes[0].axhline(y=2, color=COLORS['limit'], linestyle='--', linewidth=2,
               label='Safety Limit (2%)', alpha=0.8)
    axes[0].fill_between(time, df['HTO'], 2, where=(df['HTO'] > 2),
                    color=COLORS['limit'], alpha=0.3, interpolate=True)
    setup_axis_style(axes[0], 'Hydrogen Safety (HTO)', time_label, 'HTO (%)', 16)
    axes[0].legend(loc='best', fontsize=13, framealpha=0.95)
    axes[0].set_ylim([0, max(2.5, df['HTO'].max() * 1.1)])

    # H2产率
    axes[1].plot(time, df['H2_rate'], '-', color=COLORS['h2'],
            linewidth=2.5, alpha=0.9, label='H2 Production Rate')
    axes[1].fill_between(time, df['H2_rate'], alpha=0.2, color=COLORS['h2'])
    setup_axis_style(axes[1], 'Hydrogen Production Rate', time_label, 'H2 Rate (mol/s)', 16)

    plt.tight_layout(pad=3.0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

def create_widescreen_figure(df, output_path, use_day=True):
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

    time_col = 'time_day' if use_day else 'time_hour'
    time = df[time_col]

    # 1. 功率与温度 (大图)
    ax1_twin = ax1.twinx()
    ax1.plot(time, df['P_real'] / 1e6, '-', color=COLORS['stack1'], linewidth=2.5, label='Power')
    ax1.plot(time, df['P_ref'] / 1e6, '--', color=COLORS['ref'], linewidth=2, label='Ref Power')
    ax1_twin.plot(time, df['T_s_all'] - 273.15, '-', color=COLORS['temp'],
                  linewidth=1.5, alpha=0.9, label='Temperature')
    time_label = 'Time (days)' if use_day else 'Time (hours)'
    ax1.set_xlabel(time_label, fontsize=12)
    ax1.set_ylabel('Power (MW)', fontsize=12, color=COLORS['stack1'])
    ax1_twin.set_ylabel('Temperature (°C)', fontsize=12, color=COLORS['temp'])
    ax1.set_title('Power Tracking & Stack Temperature (12-Month Wind)', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(loc='upper left', fontsize=9)
    ax1_twin.legend(loc='upper right', fontsize=9)

    # 2. HTO
    ax2.plot(time, df['HTO'], '-', color=COLORS['hto'], linewidth=2.5)
    ax2.axhline(y=2, color=COLORS['limit'], linestyle='--', linewidth=2)
    ax2.fill_between(time, df['HTO'], 2, where=(df['HTO'] > 2),
                     color=COLORS['limit'], alpha=0.3)
    ax2.set_xlabel(time_label, fontsize=12)
    ax2.set_ylabel('HTO (%)', fontsize=12)
    ax2.set_title('HTO Safety', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.4)

    # 3. 电流
    ax3.plot(time, df['I'], '-', color=COLORS['stack1'], linewidth=2)
    ax3.set_xlabel(time_label, fontsize=12)
    ax3.set_ylabel('Current (A)', fontsize=12)
    ax3.set_title('Stack Current', fontsize=12, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.4)

    # 4. 电压
    ax4.plot(time, df['U_cell_all'], '-', color=COLORS['stack1'], linewidth=2)
    ax4.axhline(y=2.2, color=COLORS['limit'], linestyle='--', linewidth=2)
    ax4.set_xlabel(time_label, fontsize=12)
    ax4.set_ylabel('Voltage (V)', fontsize=12)
    ax4.set_title('Cell Voltage', fontsize=12, fontweight='bold')
    ax4.grid(True, linestyle='--', alpha=0.4)

    # 5. 氢气产率
    ax5.plot(time, df['H2_rate'], '-', color=COLORS['h2'], linewidth=2.5)
    ax5.fill_between(time, df['H2_rate'], alpha=0.3, color=COLORS['h2'])
    ax5.set_xlabel(time_label, fontsize=12)
    ax5.set_ylabel('Rate (mol/s)', fontsize=12)
    ax5.set_title('H2 Production', fontsize=12, fontweight='bold')
    ax5.grid(True, linestyle='--', alpha=0.4)

    plt.suptitle('Single-Stack AWE System Test - 12 Month Wind Data - Model Controller',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

def create_first_week_detail(df, output_path):
    """创建第一周详细视图（用于展示细节）"""
    # 筛选前7天的数据
    df_week = df[df['time_day'] <= 7].copy()

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('white')

    # 创建布局
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :])   # 顶部横跨
    ax2 = fig.add_subplot(gs[1, 0])   # 中间左
    ax3 = fig.add_subplot(gs[1, 1])   # 中间右
    ax4 = fig.add_subplot(gs[2, 0])   # 底部左
    ax5 = fig.add_subplot(gs[2, 1])   # 底部右

    time = df_week['time_day']

    # 1. 功率跟踪（大图）
    ax1.plot(time, df_week['P_ref'] / 1e6, '--', color=COLORS['ref'], linewidth=2.5, label='Reference', alpha=0.9)
    ax1.plot(time, df_week['P_real'] / 1e6, '-', color=COLORS['stack1'], linewidth=2.5, label='Actual', alpha=0.9)
    ax1.fill_between(time, df_week['P_real'] / 1e6, df_week['P_ref'] / 1e6,
                     alpha=0.2, color=COLORS['stack1'])
    ax1.set_xlabel('Time (days)', fontsize=12)
    ax1.set_ylabel('Power (MW)', fontsize=12)
    ax1.set_title('Power Tracking - First Week Detail', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.4)

    # 2. 温度
    ax2.plot(time, df_week['T_s_all'] - 273.15, '-', color=COLORS['temp'], linewidth=2.5)
    ax2.axhline(y=90, color=COLORS['limit'], linestyle='--', linewidth=2, alpha=0.8)
    ax2.axhline(y=80, color=COLORS['ref'], linestyle=':', linewidth=1.5, alpha=0.6)
    ax2.set_xlabel('Time (days)', fontsize=12)
    ax2.set_ylabel('Temperature (°C)', fontsize=12)
    ax2.set_title('Stack Temperature', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.4)

    # 3. HTO
    ax3.plot(time, df_week['HTO'], '-', color=COLORS['hto'], linewidth=2.5)
    ax3.axhline(y=2, color=COLORS['limit'], linestyle='--', linewidth=2, alpha=0.8)
    ax3.fill_between(time, df_week['HTO'], 2, where=(df_week['HTO'] > 2),
                     color=COLORS['limit'], alpha=0.3)
    ax3.set_xlabel('Time (days)', fontsize=12)
    ax3.set_ylabel('HTO (%)', fontsize=12)
    ax3.set_title('HTO Safety', fontsize=12, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.4)
    ax3.set_ylim([0, max(2.5, df_week['HTO'].max() * 1.1)])

    # 4. 电流
    ax4.plot(time, df_week['I'], '-', color=COLORS['stack1'], linewidth=2.5)
    ax4.set_xlabel('Time (days)', fontsize=12)
    ax4.set_ylabel('Current (A)', fontsize=12)
    ax4.set_title('Stack Current', fontsize=12, fontweight='bold')
    ax4.grid(True, linestyle='--', alpha=0.4)

    # 5. 电压
    ax5.plot(time, df_week['U_cell_all'], '-', color=COLORS['stack1'], linewidth=2.5)
    ax5.axhline(y=2.2, color=COLORS['limit'], linestyle='--', linewidth=2, alpha=0.8)
    ax5.set_xlabel('Time (days)', fontsize=12)
    ax5.set_ylabel('Voltage (V)', fontsize=12)
    ax5.set_title('Cell Voltage', fontsize=12, fontweight='bold')
    ax5.grid(True, linestyle='--', alpha=0.4)

    plt.suptitle('Single-Stack Test - First Week Detail (12-Month Wind Data)',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

def main():
    # 文件路径 - 12月份数据
    input_file = "output/single_stack/test/model_data_20260331_202442.csv"
    output_dir = "dataset/figure/single_stack/december"

    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 加载数据并过滤预热段
    print(f"Loading data from: {input_file}")
    df = load_and_filter_data(input_file)
    print(f"Loaded {len(df)} data points (warmup removed)")
    print(f"Time range: {df['time_day'].min():.2f} - {df['time_day'].max():.2f} days")
    print(f"Duration: {df['time_day'].max():.1f} days ({df['time_hour'].max():.1f} hours)")

    # 计算并显示RMSE
    power_rmse = np.sqrt(np.mean((df['P_real'] - df['P_ref'])**2)) / 1e6
    temp_rmse = np.sqrt(np.mean((df['T_s_all'] - 353.15)**2))
    print(f"\nPerformance Metrics:")
    print(f"  Power RMSE: {power_rmse:.4f} MW")
    print(f"  Temperature RMSE: {temp_rmse:.4f} K")

    # 生成各种图表
    print("\nGenerating figures...")

    # 1. PPT图1: 功率与温度（按天显示）
    create_ppt_figure1_power_and_temp(df, f"{output_dir}/dec_power_temp.png", use_day=True)

    # 2. PPT图2: 电流与流量
    create_ppt_figure2_currents_and_flows(df, f"{output_dir}/dec_currents_flows.png", use_day=True)

    # 3. PPT图3: 安全指标
    create_ppt_figure3_safety(df, f"{output_dir}/dec_safety.png", use_day=True)

    # 4. 宽屏PPT图表
    create_widescreen_figure(df, f"{output_dir}/dec_widescreen.png", use_day=True)

    # 5. 第一周详细视图
    create_first_week_detail(df, f"{output_dir}/dec_first_week_detail.png")

    print(f"\nAll figures saved to: {output_dir}/")
    print("Files generated:")
    print("  - dec_power_temp.png          (Power & Temperature)")
    print("  - dec_currents_flows.png      (Currents & Flows)")
    print("  - dec_safety.png              (Safety Metrics)")
    print("  - dec_widescreen.png          (16:9 widescreen layout)")
    print("  - dec_first_week_detail.png   (First week zoomed view)")

if __name__ == "__main__":
    main()
