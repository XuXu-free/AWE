# -*- coding: utf-8 -*-
"""
计算单槽 NMPC 和 Model Controller 的 RMSE 指标
- 功率跟踪 RMSE
- 温度跟踪 RMSE
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_and_filter_data(filepath):
    """加载数据并过滤掉预热段 (t < 0)"""
    df = pd.read_csv(filepath)
    df = df[df['t'] >= 0].copy()
    return df

def calculate_power_rmse(df):
    """计算功率跟踪 RMSE (单位: MW)"""
    # 功率误差 (MW)
    power_error = (df['P_real'] - df['P_ref']) / 1e6
    rmse = np.sqrt(np.mean(power_error ** 2))
    mae = np.mean(np.abs(power_error))
    max_error = np.max(np.abs(power_error))
    return {
        'RMSE (MW)': rmse,
        'MAE (MW)': mae,
        'Max Error (MW)': max_error
    }

def calculate_temperature_rmse(df, T_ref_celsius=80):
    """计算温度跟踪 RMSE (单位: °C)"""
    # 温度误差 (°C)
    temp_error = (df['T_s_all'] - 273.15) - T_ref_celsius
    rmse = np.sqrt(np.mean(temp_error ** 2))
    mae = np.mean(np.abs(temp_error))
    max_error = np.max(np.abs(temp_error))
    mean_temp = (df['T_s_all'] - 273.15).mean()
    max_temp = (df['T_s_all'] - 273.15).max()
    return {
        'RMSE (°C)': rmse,
        'MAE (°C)': mae,
        'Max Error (°C)': max_error,
        'Mean Temp (°C)': mean_temp,
        'Max Temp (°C)': max_temp
    }

def main():
    # 文件路径
    nmpc_file = "output/single_stack/test/nmpc_data_20260331_114727.csv"
    model_file = "output/single_stack/test/model_data_20260319_161737.csv"

    # 加载数据
    print("Loading NMPC data...")
    df_nmpc = load_and_filter_data(nmpc_file)
    print(f"  NMPC: {len(df_nmpc)} data points")

    print("Loading Model Controller data...")
    df_model = load_and_filter_data(model_file)
    print(f"  Model: {len(df_model)} data points")

    # 计算功率 RMSE
    print("\n" + "="*70)
    print("Power Tracking Performance".center(70))
    print("="*70)

    power_nmpc = calculate_power_rmse(df_nmpc)
    power_model = calculate_power_rmse(df_model)

    print(f"{'Metric':<25} {'NMPC':>18} {'Model Ctrl':>18} {'Improvement':>12}")
    print("-"*70)
    for key in power_nmpc.keys():
        nmpc_val = power_nmpc[key]
        model_val = power_model[key]
        if nmpc_val != 0:
            improvement = (nmpc_val - model_val) / nmpc_val * 100
            print(f"{key:<25} {nmpc_val:>18.4f} {model_val:>18.4f} {improvement:>11.2f}%")
        else:
            print(f"{key:<25} {nmpc_val:>18.4f} {model_val:>18.4f} {'N/A':>12}")

    # 计算温度 RMSE
    print("\n" + "="*70)
    print("Temperature Tracking Performance (Target: 80°C)".center(70))
    print("="*70)

    temp_nmpc = calculate_temperature_rmse(df_nmpc)
    temp_model = calculate_temperature_rmse(df_model)

    print(f"{'Metric':<25} {'NMPC':>18} {'Model Ctrl':>18} {'Improvement':>12}")
    print("-"*70)
    for key in temp_nmpc.keys():
        nmpc_val = temp_nmpc[key]
        model_val = temp_model[key]
        if key in ['Mean Temp (°C)', 'Max Temp (°C)']:
            # 这些是绝对值，不是误差，所以计算差值
            diff = model_val - nmpc_val
            print(f"{key:<25} {nmpc_val:>18.4f} {model_val:>18.4f} {diff:>+11.4f}")
        elif nmpc_val != 0:
            improvement = (nmpc_val - model_val) / nmpc_val * 100
            print(f"{key:<25} {nmpc_val:>18.4f} {model_val:>18.4f} {improvement:>11.2f}%")
        else:
            print(f"{key:<25} {nmpc_val:>18.4f} {model_val:>18.4f} {'N/A':>12}")

    # 额外的安全指标
    print("\n" + "="*70)
    print("Safety Metrics".center(70))
    print("="*70)
    print(f"{'Metric':<25} {'NMPC':>18} {'Model Ctrl':>18} {'Improvement':>12}")
    print("-"*70)

    # HTO Max
    hto_max_nmpc = df_nmpc['HTO'].max()
    hto_max_model = df_model['HTO'].max()
    print(f"{'HTO Max (%)':<25} {hto_max_nmpc:>18.4f} {hto_max_model:>18.4f} {(hto_max_nmpc - hto_max_model)/hto_max_nmpc*100:>11.2f}%")

    # Voltage Max
    volt_max_nmpc = df_nmpc['U_cell_all'].max()
    volt_max_model = df_model['U_cell_all'].max()
    print(f"{'Voltage Max (V)':<25} {volt_max_nmpc:>18.4f} {volt_max_model:>18.4f} {(volt_max_nmpc - volt_max_model)/volt_max_nmpc*100:>11.2f}%")

    # Temp constraint violation
    temp_viol_nmpc = np.sum((df_nmpc['T_s_all'] - 273.15) > 90)
    temp_viol_model = np.sum((df_model['T_s_all'] - 273.15) > 90)
    print(f"{'Temp Violations (>90°C)':<25} {temp_viol_nmpc:>18} {temp_viol_model:>18}")

    print("="*70)

    # 保存结果到文件
    output_dir = "dataset/figure"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(f"{output_dir}/rmse_comparison.txt", "w") as f:
        f.write("Single-Stack AWE Controller Performance Comparison\n")
        f.write("="*70 + "\n\n")

        f.write("Power Tracking RMSE (MW)\n")
        f.write(f"  NMPC:           {power_nmpc['RMSE (MW)']:.4f}\n")
        f.write(f"  Model Ctrl:     {power_model['RMSE (MW)']:.4f}\n")
        f.write(f"  Improvement:    {(power_nmpc['RMSE (MW)'] - power_model['RMSE (MW)'])/power_nmpc['RMSE (MW)']*100:.2f}%\n\n")

        f.write("Temperature Tracking RMSE (°C, Target: 80°C)\n")
        f.write(f"  NMPC:           {temp_nmpc['RMSE (°C)']:.4f}\n")
        f.write(f"  Model Ctrl:     {temp_model['RMSE (°C)']:.4f}\n")
        f.write(f"  Improvement:    {(temp_nmpc['RMSE (°C)'] - temp_model['RMSE (°C)'])/temp_nmpc['RMSE (°C)']*100:.2f}%\n\n")

        f.write("Safety Metrics\n")
        f.write(f"  HTO Max (%)     NMPC: {hto_max_nmpc:.4f}, Model: {hto_max_model:.4f}\n")
        f.write(f"  Voltage Max (V) NMPC: {volt_max_nmpc:.4f}, Model: {volt_max_model:.4f}\n")

    print(f"\nResults saved to: {output_dir}/rmse_comparison.txt")

if __name__ == "__main__":
    main()
