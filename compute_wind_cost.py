import pandas as pd
import numpy as np
import os

lambda_track = 1.2
lambda_temp = 0.15
lambda_I = 0.0002
lambda_lye = 25000
lambda_c = 2500
lambda_prod = 1e-6

base_dir = 'output/multi_stack/test/run_20260426_121806'

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

def compute_cost(path):
    df = pd.read_csv(path)
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

print("=== Paper models ===")
for name, f in paper_models.items():
    c = compute_cost(os.path.join(base_dir, f))
    print(f"{name}: len={c['len']}, power={c['power']:.1f}, temp={c['temp']:.1f}, current={c['current']:.1f}, prod={c['prod']:.2f}, lye={c['lye']:.2f}, coolant={c['coolant']:.2f}, total={c['total']:.1f}")

print("\n=== Internal models ===")
for name, f in internal_models.items():
    c = compute_cost(os.path.join(base_dir, f))
    print(f"{name}: len={c['len']}, power={c['power']:.1f}, temp={c['temp']:.1f}, current={c['current']:.1f}, prod={c['prod']:.2f}, lye={c['lye']:.2f}, coolant={c['coolant']:.2f}, total={c['total']:.1f}")
