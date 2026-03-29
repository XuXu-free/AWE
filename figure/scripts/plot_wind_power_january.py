# -*- coding: utf-8 -*-
"""
风电数据示意图 - 架构图专用
数据文件: wind_power_2025-01_1min.csv
风格与其他PPT图表保持一致
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置中文字体（Windows系统）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义配色方案 - 与其他PPT图表一致
COLORS = {
    'primary': '#1f77b4',      # 蓝色
    'secondary': '#ff7f0e',    # 橙色
    'grid': '#e0e0e0',         # 网格灰
    'text': '#2c3e50',         # 正文深灰
    'fill': '#1f77b4',         # 填充色
}

def load_wind_data(filepath):
    """加载风电数据"""
    df = pd.read_csv(filepath)
    # 转换时间为小时
    df['time_hr'] = df['t'] / 3600.0
    # 功率转换为 MW
    df['power_mw'] = df['P_ref'] / 1e6
    return df

def plot_wind_power_full(df, output_path):
    """绘制完整的1月份风电功率曲线（适合宽屏展示）"""
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor('none')  # 透明背景
    ax.set_facecolor('none')  # 透明背景

    time = df['time_hr']
    power = df['power_mw']

    # 绘制功率曲线
    ax.fill_between(time, power, alpha=0.3, color=COLORS['primary'])
    ax.plot(time, power, '-', color=COLORS['primary'], linewidth=1.2, alpha=0.9)

    # 添加平均线
    mean_power = power.mean()
    ax.axhline(y=mean_power, color=COLORS['secondary'], linestyle='--', linewidth=2,
               label=f'Mean: {mean_power:.2f} MW', alpha=0.8)

    # 设置样式
    ax.set_title('Wind Power Profile - January 2025', fontsize=16, fontweight='bold',
                 color=COLORS['text'], pad=15)
    ax.set_xlabel('Time (hours)', fontsize=13, color=COLORS['text'])
    ax.set_ylabel('Power (MW)', fontsize=13, color=COLORS['text'])

    ax.grid(True, linestyle='--', alpha=0.5, color=COLORS['grid'])
    ax.tick_params(labelsize=11, colors=COLORS['text'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color('#cccccc')

    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax.set_xlim([0, time.max()])
    ax.set_ylim([0, power.max() * 1.1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='none', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")

def plot_wind_power_compact(df, output_path):
    """绘制紧凑型风电功率示意图（适合架构图嵌入）"""
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('none')  # 透明背景
    ax.set_facecolor('none')  # 透明背景

    time = df['time_hr']
    power = df['power_mw']

    # 绘制功率曲线（简化样式）
    ax.fill_between(time, power, alpha=0.25, color=COLORS['primary'])
    ax.plot(time, power, '-', color=COLORS['primary'], linewidth=1.0, alpha=0.85)

    # 统计信息
    mean_power = power.mean()
    max_power = power.max()
    min_power = power.min()

    # 添加统计标注
    ax.axhline(y=mean_power, color=COLORS['secondary'], linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(time.max() * 0.98, mean_power * 1.05, f'Mean: {mean_power:.1f} MW',
            fontsize=9, color=COLORS['secondary'], ha='right')

    # 紧凑样式
    ax.set_title('Wind Power Input (Jan 2025)', fontsize=14, fontweight='bold',
                 color=COLORS['text'], pad=10)
    ax.set_xlabel('Time (hr)', fontsize=11, color=COLORS['text'])
    ax.set_ylabel('Power (MW)', fontsize=11, color=COLORS['text'])

    ax.grid(True, linestyle='--', alpha=0.4, color=COLORS['grid'])
    ax.tick_params(labelsize=10, colors=COLORS['text'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color('#cccccc')

    # 添加统计信息框
    stats_text = f'Max: {max_power:.1f} MW\nMin: {min_power:.1f} MW\nMean: {mean_power:.1f} MW'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white',
            edgecolor=COLORS['grid'], alpha=0.9))

    ax.set_xlim([0, time.max()])
    ax.set_ylim([0, max_power * 1.15])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='none', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")

def plot_wind_power_24h_sample(df, output_path):
    """绘制24小时示例片段（适合详细展示）"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor('none')  # 透明背景
    for ax in axes.flat:
        ax.set_facecolor('none')  # 透明背景

    # 选择4个不同的24小时片段
    samples = [
        (0, 24, 'Day 1-2'),
        (168, 192, 'Day 8-9'),
        (336, 360, 'Day 15-16'),
        (504, 528, 'Day 22-23')
    ]

    for idx, (start, end, label) in enumerate(samples):
        ax = axes[idx // 2, idx % 2]

        # 筛选数据
        mask = (df['time_hr'] >= start) & (df['time_hr'] <= end)
        sample_df = df[mask]

        time = sample_df['time_hr'] - start  # 相对时间
        power = sample_df['power_mw']

        ax.fill_between(time, power, alpha=0.3, color=COLORS['primary'])
        ax.plot(time, power, '-', color=COLORS['primary'], linewidth=1.5)

        ax.set_title(f'{label} (Sample {idx+1})', fontsize=12, fontweight='bold',
                     color=COLORS['text'])
        ax.set_xlabel('Time (hr)', fontsize=10, color=COLORS['text'])
        ax.set_ylabel('Power (MW)', fontsize=10, color=COLORS['text'])
        ax.grid(True, linestyle='--', alpha=0.4, color=COLORS['grid'])
        ax.tick_params(labelsize=9, colors=COLORS['text'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 添加统计
        ax.text(0.98, 0.95, f'Mean: {power.mean():.1f} MW\nMax: {power.max():.1f} MW',
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['grid'], alpha=0.8))

        ax.set_xlim([0, 24])
        ax.set_ylim([0, df['power_mw'].max() * 1.1])

    plt.suptitle('Wind Power Profile - Representative 24h Samples (Jan 2025)',
                 fontsize=14, fontweight='bold', y=1.02, color=COLORS['text'])
    plt.tight_layout(pad=2.0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='none', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")

def plot_wind_power_architecture_style(df, output_path):
    """
    架构图专用极简风格
    突出波动性特征，适合放在系统架构图中
    """
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor('none')  # 透明背景
    ax.set_facecolor('none')  # 透明背景

    time = df['time_hr']
    power = df['power_mw']

    # 极简线条
    ax.plot(time, power, '-', color=COLORS['primary'], linewidth=0.8, alpha=0.9)
    ax.fill_between(time, power, alpha=0.15, color=COLORS['primary'])

    # 只保留必要的标注
    ax.set_title('Wind Power', fontsize=12, fontweight='bold', color=COLORS['text'], pad=8)
    ax.set_xlabel('t (hr)', fontsize=10, color=COLORS['text'])
    ax.set_ylabel('P (MW)', fontsize=10, color=COLORS['text'])

    # 极简网格
    ax.grid(True, linestyle=':', alpha=0.3, color=COLORS['grid'])
    ax.tick_params(labelsize=9, colors=COLORS['text'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')

    # 紧凑边距
    ax.set_xlim([0, time.max()])
    ax.set_ylim([0, power.max() * 1.05])

    plt.tight_layout(pad=0.5)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='none', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")

def print_statistics(df):
    """打印数据统计信息"""
    power = df['power_mw']
    print("\n" + "="*50)
    print("Wind Power Statistics (January 2025)".center(50))
    print("="*50)
    print(f"Duration:     {df['time_hr'].max():.1f} hours ({df['time_hr'].max()/24:.1f} days)")
    print(f"Data points:  {len(df)}")
    print(f"Mean Power:   {power.mean():.2f} MW")
    print(f"Max Power:    {power.max():.2f} MW")
    print(f"Min Power:    {power.min():.2f} MW")
    print(f"Std Dev:      {power.std():.2f} MW")
    print(f"Capacity Factor: {power.mean()/power.max()*100:.1f}%")
    print("="*50)

def main():
    # 文件路径
    input_file = "output/power/wind/wind_power_2025-01_1min.csv"
    output_dir = "dataset/figure"

    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 加载数据
    print(f"Loading wind data from: {input_file}")
    df = load_wind_data(input_file)
    print(f"Loaded {len(df)} data points")

    # 打印统计信息
    print_statistics(df)

    # 生成图表
    print("\nGenerating figures...")

    # 1. 完整月度曲线（宽屏）
    plot_wind_power_full(df, f"{output_dir}/wind_power_january_full.png")

    # 2. 紧凑示意图（适合嵌入）
    plot_wind_power_compact(df, f"{output_dir}/wind_power_january_compact.png")

    # 3. 24小时样本展示
    plot_wind_power_24h_sample(df, f"{output_dir}/wind_power_january_samples.png")

    # 4. 架构图专用极简风格
    plot_wind_power_architecture_style(df, f"{output_dir}/wind_power_architecture.png")

    print(f"\nAll figures saved to: {output_dir}/")
    print("Files generated:")
    print("  - wind_power_january_full.png      (Full month, wide format)")
    print("  - wind_power_january_compact.png   (Compact with stats)")
    print("  - wind_power_january_samples.png   (4x 24h samples)")
    print("  - wind_power_architecture.png      (Minimal, for architecture diagram)")

if __name__ == "__main__":
    main()
