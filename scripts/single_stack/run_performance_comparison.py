# -*- coding: utf-8 -*-
"""
NMPC vs DiffusionTCN 控制性能对比 - 答辩PPT专用
对比两者的功率跟踪、温度控制、安全指标等
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from plant.single_stack_simulator import SingleStackSimulator
from controller.single_stack.nmpc_controller import SingleStackNMPCController
from controller.single_stack.model_controller import SingleStackModelController

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义专业配色方案
COLORS = {
    'nmpc': '#e74c3c',        # 红色 - NMPC
    'diffusion': '#1f77b4',   # 蓝色 - Diffusion
    'ref': '#9467bd',         # 紫色 - 参考
    'limit': '#e74c3c',       # 警示红
    'grid': '#e0e0e0',        # 网格灰
    'text': '#2c3e50',        # 正文深灰
    'temp': '#ff7f0e',        # 橙色
    'hto': '#2ca02c',         # 绿色
}


class SingleStackNMPCControllerPure(SingleStackNMPCController):
    """纯CasADi求解器版本（不使用预编译C库）"""

    def _setup_solver(self):
        """重写求解器设置，使用纯CasADi"""
        import casadi as ca

        self.n_controls = 3
        self.n_states = 7

        self.U = ca.MX.sym('U', self.n_controls * self.horizon)
        self.P = ca.MX.sym('P', self.n_states + 1 + self.horizon + 3)

        obj = 0
        g = []
        lbg = []
        ubg = []

        p_idx = 0
        T_s_in_K = self.P[p_idx]; p_idx += 1
        T_s_K = self.P[p_idx]; p_idx += 1
        T_sep_K = self.P[p_idx]; p_idx += 1
        T_c_out_K = self.P[p_idx]; p_idx += 1
        n_H2_an = self.P[p_idx]; p_idx += 1
        n_liq = self.P[p_idx]; p_idx += 1
        n_gas = self.P[p_idx]; p_idx += 1

        T_ref_val = self.P[p_idx]; p_idx += 1
        P_ref = self.P[p_idx:p_idx+self.horizon]; p_idx += self.horizon
        I_0 = self.P[p_idx]; p_idx += 1
        v_lye_0 = self.P[p_idx]; p_idx += 1
        v_c_0 = self.P[p_idx]; p_idx += 1

        T_s_in_k = T_s_in_K
        T_s_k = T_s_K
        T_sep_k = T_sep_K
        T_c_out_k = T_c_out_K

        for k in range(self.horizon):
            uk = self.U[k*self.n_controls : (k+1)*self.n_controls]
            I_k = uk[0]
            v_lye_k = uk[1]
            v_c_k = uk[2]

            T_s_in_k, T_s_k, T_sep_k, T_c_out_k, Power_k, V_cell, hto_pct = self._dynamics_step(
                T_s_in_k, T_s_k, T_sep_k, T_c_out_k, I_k, v_lye_k, v_c_k, n_gas
            )

            obj += self.lambda_track * ((Power_k - P_ref[k])/1e6)**2
            obj += self.lambda_temp * (T_s_k - T_ref_val)**2

            if k == 0:
                dI = I_k - I_0
            else:
                uk_prev = self.U[(k-1)*self.n_controls : k*self.n_controls]
                dI = I_k - uk_prev[0]

            dv_lye = v_lye_k - v_lye_0
            dv_c = v_c_k - v_c_0

            obj += self.lambda_I * dI**2
            obj += self.lambda_lye * dv_lye**2
            obj += self.lambda_c * dv_c**2

            g.append(T_s_k)
            lbg.append(self.T_min)
            ubg.append(self.T_max)

            g.append(Power_k)
            lbg.append(self.P_stack_min)
            ubg.append(self.P_stack_max)

            g.append(V_cell)
            lbg.append(self.U_cell_min)
            ubg.append(self.U_cell_max)

            g.append(hto_pct)
            lbg.append(self.HTO_pct_min)
            ubg.append(self.HTO_pct_max)

        lbx = []
        ubx = []
        for k in range(self.horizon):
            lbx.extend([self.I_min, self.v_lye_min, self.v_c_min])
            ubx.extend([self.I_max, self.v_lye_max, self.v_c_max])

        self.lbx = lbx
        self.ubx = ubx
        self.lbg = lbg
        self.ubg = ubg

        nlp = {'x': self.U, 'f': obj, 'g': ca.vertcat(*g), 'p': self.P}

        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.tol': 1e-4
        }

        self.solver = ca.nlpsol('nmpc_solver', 'ipopt', nlp, opts)

        # Setup state prediction function
        pred_states = []
        p_idx = 0
        T_s_in_K = self.P[p_idx]; p_idx += 1
        T_s_K = self.P[p_idx]; p_idx += 1
        T_sep_K = self.P[p_idx]; p_idx += 1
        T_c_out_K = self.P[p_idx]; p_idx += 1
        n_H2_an_dummy = self.P[p_idx]; p_idx += 1
        n_liq_dummy = self.P[p_idx]; p_idx += 1
        n_gas_val = self.P[p_idx]; p_idx += 1

        T_s_in_k = T_s_in_K
        T_s_k = T_s_K
        T_sep_k = T_sep_K
        T_c_out_k = T_c_out_K

        for k in range(self.horizon):
            uk = self.U[k*self.n_controls : (k+1)*self.n_controls]
            I_k = uk[0]
            v_lye_k = uk[1]
            v_c_k = uk[2]

            T_s_in_k, T_s_k, T_sep_k, T_c_out_k, _, _, _ = self._dynamics_step(
                T_s_in_k, T_s_k, T_sep_k, T_c_out_k, I_k, v_lye_k, v_c_k, n_gas_val
            )
            pred_states.append(ca.vertcat(T_s_in_k, T_s_k, T_sep_k, T_c_out_k))

        self.state_func = ca.Function('state_func', [self.U, self.P], [ca.vertcat(*pred_states)])

        self.prev_sol_x = None


def run_controller_test(controller_type='nmpc', duration_minutes=30, month='02'):
    """运行控制器测试并记录完整数据"""
    print(f"\nRunning {controller_type.upper()} test...")

    sim_dt = 0.2
    ctrl_dt = 60.0
    T_ref = 353.15

    sim = SingleStackSimulator(sim_dt=sim_dt)

    if controller_type == 'nmpc':
        ctrl = SingleStackNMPCControllerPure(dt=ctrl_dt, horizon=5, sim_dt=sim_dt)
    else:
        model_path = "output/single_stack/policy/diffusion_tcn_policy_best.pth"
        stats_path = "output/single_stack/diffusion_stats.npz"
        ctrl = SingleStackModelController(
            dt=ctrl_dt,
            horizon=5,
            model_type='diffusion_tcn',
            model_path=model_path,
            stats_path=stats_path
        )

    # 加载风电数据
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    profile_path = os.path.join(project_root, 'output', 'power', 'wind', f'wind_power_2025-{month}_1min.csv')
    df_power = pd.read_csv(profile_path)
    full_profile = df_power['P_ref'].values * 0.25

    # 初始化
    state = sim.reset()
    last_action = [1000.0, 0.05, 0.5]

    # 预热
    print("  Warmup...")
    warmup_steps = int(10 * 60 / ctrl_dt)
    P_warmup = 2.0e6

    for _ in range(warmup_steps):
        P_ref_vec = np.full(ctrl.horizon if hasattr(ctrl, 'horizon') else 5, P_warmup)
        action = ctrl.get_action(state, P_ref_vec, T_ref, last_action)
        if action is not None:
            last_action = [action[0], action[1], action[2]]
        state = sim.step(last_action)

    # 正式测试
    print(f"  Testing for {duration_minutes} minutes...")
    n_steps = int(duration_minutes * 60 / ctrl_dt)

    # 数据记录
    data = {
        'time': [],
        'P_ref': [],
        'P_real': [],
        'T_stack': [],
        'I': [],
        'v_lye': [],
        'v_c': [],
        'U_cell': [],
        'HTO': [],
        'H2_rate': []
    }

    for step in range(n_steps):
        t = step * ctrl_dt
        idx = int(t / 60) % len(full_profile)

        horizon = ctrl.horizon if hasattr(ctrl, 'horizon') else 5
        P_ref_vec = np.array([full_profile[(idx + i) % len(full_profile)] for i in range(horizon)])

        action = ctrl.get_action(state, P_ref_vec, T_ref, last_action)
        if action is not None:
            last_action = [action[0], action[1], action[2]]

        state = sim.step(last_action)

        # 记录数据
        data['time'].append(t / 60.0)  # 转换为分钟
        data['P_ref'].append(P_ref_vec[0])
        data['P_real'].append(state[0])
        data['T_stack'].append(state[1])
        data['I'].append(last_action[0])
        data['v_lye'].append(last_action[1])
        data['v_c'].append(last_action[2])
        data['U_cell'].append(state[3])
        data['HTO'].append(state[2])
        data['H2_rate'].append(state[4])

    df = pd.DataFrame(data)

    # 计算RMSE
    power_rmse = np.sqrt(np.mean((df['P_real'] - df['P_ref'])**2))
    temp_rmse = np.sqrt(np.mean((df['T_stack'] - T_ref)**2))

    print(f"  Power RMSE: {power_rmse/1e6:.4f} MW")
    print(f"  Temp RMSE: {temp_rmse:.4f} K")

    return df


def setup_axis_style(ax, title, xlabel, ylabel, fontsize=14):
    """统一设置坐标轴样式"""
    ax.set_title(title, fontsize=fontsize+2, fontweight='bold', color=COLORS['text'], pad=10)
    ax.set_xlabel(xlabel, fontsize=fontsize, color=COLORS['text'])
    ax.set_ylabel(ylabel, fontsize=fontsize, color=COLORS['text'])
    ax.grid(True, linestyle='--', alpha=0.5, color=COLORS['grid'])
    ax.tick_params(labelsize=fontsize-2, colors=COLORS['text'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color('#cccccc')


def create_comparison_figures(df_nmpc, df_diffusion, output_dir):
    """创建对比图表"""

    # 1. 功率跟踪对比
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.patch.set_facecolor('white')

    time = df_nmpc['time']

    axes[0].plot(time, df_nmpc['P_ref'] / 1e6, '--', color=COLORS['ref'], linewidth=2.5, label='Reference', alpha=0.9)
    axes[0].plot(time, df_nmpc['P_real'] / 1e6, '-', color=COLORS['nmpc'], linewidth=2.5, label='NMPC', alpha=0.9)
    axes[0].plot(time, df_diffusion['P_real'] / 1e6, '-', color=COLORS['diffusion'], linewidth=2.5, label='DiffusionTCN', alpha=0.9)
    setup_axis_style(axes[0], 'Power Tracking Comparison', 'Time (min)', 'Power (MW)', 14)
    axes[0].legend(loc='best', fontsize=11, framealpha=0.95)

    axes[1].plot(time, (df_nmpc['P_real'] - df_nmpc['P_ref']) / 1e6, '-', color=COLORS['nmpc'], linewidth=2, label='NMPC Error', alpha=0.8)
    axes[1].plot(time, (df_diffusion['P_real'] - df_diffusion['P_ref']) / 1e6, '-', color=COLORS['diffusion'], linewidth=2, label='DiffusionTCN Error', alpha=0.8)
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    setup_axis_style(axes[1], 'Power Tracking Error', 'Time (min)', 'Error (MW)', 14)
    axes[1].legend(loc='best', fontsize=11, framealpha=0.95)

    plt.tight_layout(pad=3.0)
    plt.savefig(f"{output_dir}/comparison_power.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_dir}/comparison_power.png")

    # 2. 温度控制对比
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.patch.set_facecolor('white')

    axes[0].plot(time, df_nmpc['T_stack'] - 273.15, '-', color=COLORS['nmpc'], linewidth=2.5, label='NMPC', alpha=0.9)
    axes[0].plot(time, df_diffusion['T_stack'] - 273.15, '-', color=COLORS['diffusion'], linewidth=2.5, label='DiffusionTCN', alpha=0.9)
    axes[0].axhline(y=80, color=COLORS['ref'], linestyle=':', linewidth=2, label='Target (80°C)', alpha=0.8)
    axes[0].axhline(y=90, color=COLORS['limit'], linestyle='--', linewidth=2, label='Limit (90°C)', alpha=0.8)
    setup_axis_style(axes[0], 'Stack Temperature Comparison', 'Time (min)', 'Temperature (°C)', 14)
    axes[0].legend(loc='best', fontsize=10, framealpha=0.95)

    temp_error_nmpc = (df_nmpc['T_stack'] - 353.15)
    temp_error_diff = (df_diffusion['T_stack'] - 353.15)
    axes[1].plot(time, temp_error_nmpc, '-', color=COLORS['nmpc'], linewidth=2, label='NMPC Error', alpha=0.8)
    axes[1].plot(time, temp_error_diff, '-', color=COLORS['diffusion'], linewidth=2, label='DiffusionTCN Error', alpha=0.8)
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    setup_axis_style(axes[1], 'Temperature Tracking Error', 'Time (min)', 'Error (K)', 14)
    axes[1].legend(loc='best', fontsize=11, framealpha=0.95)

    plt.tight_layout(pad=3.0)
    plt.savefig(f"{output_dir}/comparison_temperature.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_dir}/comparison_temperature.png")

    # 3. HTO安全对比
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('white')

    ax.plot(time, df_nmpc['HTO'], '-', color=COLORS['nmpc'], linewidth=2.5, label='NMPC', alpha=0.9)
    ax.plot(time, df_diffusion['HTO'], '-', color=COLORS['diffusion'], linewidth=2.5, label='DiffusionTCN', alpha=0.9)
    ax.axhline(y=2, color=COLORS['limit'], linestyle='--', linewidth=2.5, label='Safety Limit (2%)', alpha=0.9)
    setup_axis_style(ax, 'HTO Safety Comparison', 'Time (min)', 'HTO (%)', 14)
    ax.legend(loc='best', fontsize=11, framealpha=0.95)
    ax.set_ylim([0, max(2.5, df_nmpc['HTO'].max(), df_diffusion['HTO'].max()) * 1.1])

    plt.tight_layout(pad=3.0)
    plt.savefig(f"{output_dir}/comparison_hto.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_dir}/comparison_hto.png")

    # 4. 电流对比
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('white')

    ax.plot(time, df_nmpc['I'], '-', color=COLORS['nmpc'], linewidth=2.5, label='NMPC', alpha=0.9)
    ax.plot(time, df_diffusion['I'], '-', color=COLORS['diffusion'], linewidth=2.5, label='DiffusionTCN', alpha=0.9)
    setup_axis_style(ax, 'Stack Current Comparison', 'Time (min)', 'Current (A)', 14)
    ax.legend(loc='best', fontsize=11, framealpha=0.95)

    plt.tight_layout(pad=3.0)
    plt.savefig(f"{output_dir}/comparison_current.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_dir}/comparison_current.png")

    # 5. 综合宽屏对比图
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('white')

    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[1, 2])

    # 功率
    ax1.plot(time, df_nmpc['P_ref'] / 1e6, '--', color=COLORS['ref'], linewidth=2, label='Reference', alpha=0.9)
    ax1.plot(time, df_nmpc['P_real'] / 1e6, '-', color=COLORS['nmpc'], linewidth=2.5, label='NMPC', alpha=0.9)
    ax1.plot(time, df_diffusion['P_real'] / 1e6, '-', color=COLORS['diffusion'], linewidth=2.5, label='DiffusionTCN', alpha=0.9)
    ax1.set_xlabel('Time (min)', fontsize=11)
    ax1.set_ylabel('Power (MW)', fontsize=11)
    ax1.set_title('Power Tracking', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.4)

    # HTO
    ax2.plot(time, df_nmpc['HTO'], '-', color=COLORS['nmpc'], linewidth=2.5, alpha=0.9)
    ax2.plot(time, df_diffusion['HTO'], '-', color=COLORS['diffusion'], linewidth=2.5, alpha=0.9)
    ax2.axhline(y=2, color=COLORS['limit'], linestyle='--', linewidth=2)
    ax2.set_xlabel('Time (min)', fontsize=11)
    ax2.set_ylabel('HTO (%)', fontsize=11)
    ax2.set_title('HTO Safety', fontsize=13, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.4)

    # 温度
    ax3.plot(time, df_nmpc['T_stack'] - 273.15, '-', color=COLORS['nmpc'], linewidth=2.5, alpha=0.9)
    ax3.plot(time, df_diffusion['T_stack'] - 273.15, '-', color=COLORS['diffusion'], linewidth=2.5, alpha=0.9)
    ax3.axhline(y=80, color=COLORS['ref'], linestyle=':', linewidth=1.5, alpha=0.6)
    ax3.set_xlabel('Time (min)', fontsize=11)
    ax3.set_ylabel('Temperature (°C)', fontsize=11)
    ax3.set_title('Stack Temperature', fontsize=13, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.4)

    # 电流
    ax4.plot(time, df_nmpc['I'], '-', color=COLORS['nmpc'], linewidth=2.5, alpha=0.9)
    ax4.plot(time, df_diffusion['I'], '-', color=COLORS['diffusion'], linewidth=2.5, alpha=0.9)
    ax4.set_xlabel('Time (min)', fontsize=11)
    ax4.set_ylabel('Current (A)', fontsize=11)
    ax4.set_title('Stack Current', fontsize=13, fontweight='bold')
    ax4.grid(True, linestyle='--', alpha=0.4)

    # H2产率
    ax5.plot(time, df_nmpc['H2_rate'], '-', color=COLORS['nmpc'], linewidth=2.5, alpha=0.9)
    ax5.plot(time, df_diffusion['H2_rate'], '-', color=COLORS['diffusion'], linewidth=2.5, alpha=0.9)
    ax5.set_xlabel('Time (min)', fontsize=11)
    ax5.set_ylabel('Rate (mol/s)', fontsize=11)
    ax5.set_title('H2 Production', fontsize=13, fontweight='bold')
    ax5.grid(True, linestyle='--', alpha=0.4)

    plt.suptitle('NMPC vs DiffusionTCN Control Performance Comparison',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(f"{output_dir}/comparison_widescreen.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_dir}/comparison_widescreen.png")


def main():
    """主函数"""
    output_dir = 'output/single_stack/timing_comparison'
    os.makedirs(output_dir, exist_ok=True)

    duration_minutes = 30
    month = '02'

    print("="*60)
    print("NMPC vs DiffusionTCN Control Performance Comparison")
    print("="*60)

    # 运行测试
    df_nmpc = run_controller_test('nmpc', duration_minutes, month)
    df_diffusion = run_controller_test('diffusion_tcn', duration_minutes, month)

    # 生成对比图
    print("\nGenerating comparison figures...")
    create_comparison_figures(df_nmpc, df_diffusion, output_dir)

    # 计算统计指标
    T_ref = 353.15
    stats = {
        'power_rmse_nmpc': np.sqrt(np.mean((df_nmpc['P_real'] - df_nmpc['P_ref'])**2)) / 1e6,
        'power_rmse_diffusion': np.sqrt(np.mean((df_diffusion['P_real'] - df_diffusion['P_ref'])**2)) / 1e6,
        'temp_rmse_nmpc': np.sqrt(np.mean((df_nmpc['T_stack'] - T_ref)**2)),
        'temp_rmse_diffusion': np.sqrt(np.mean((df_diffusion['T_stack'] - T_ref)**2)),
    }

    print(f"\nPerformance Comparison:")
    print(f"  Power RMSE - NMPC: {stats['power_rmse_nmpc']:.4f} MW, DiffusionTCN: {stats['power_rmse_diffusion']:.4f} MW")
    print(f"  Temp RMSE  - NMPC: {stats['temp_rmse_nmpc']:.4f} K, DiffusionTCN: {stats['temp_rmse_diffusion']:.4f} K")

    print(f"\nAll figures saved to: {output_dir}/")


if __name__ == "__main__":
    main()
