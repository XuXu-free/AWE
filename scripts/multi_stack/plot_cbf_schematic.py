"""
CBF / HOCBF 理论示意图

展示：
1. CBF 函数 h(x) 与安全集/不安全集
2. 一阶 CBF 时域响应：h_dot + gamma*h >= 0
3. 二阶 HOCBF 时域响应：h_ddot + alpha2*h_dot + alpha2*alpha1*h >= 0
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Subplot 1: CBF function schematic ---
    ax = axes[0]
    x = np.linspace(-2, 3, 300)
    h = x
    ax.plot(x, h, 'k-', linewidth=2, label=r'$h(x)$')
    ax.axhline(y=0, color='r', linestyle='--', linewidth=1.5, label='Boundary ($h=0$)')
    ax.fill_between(x, 0, 5, where=(h >= 0), alpha=0.15, color='green', label=r'Safe set ($h \geq 0$)')
    ax.fill_between(x, -3, 0, where=(h < 0), alpha=0.15, color='red', label=r'Unsafe set ($h < 0$)')
    ax.annotate('Safe', xy=(1.5, 0.5), fontsize=14, color='darkgreen', fontweight='bold')
    ax.annotate('Unsafe', xy=(-1.2, -0.8), fontsize=14, color='darkred', fontweight='bold')
    ax.annotate(r'$\partial C$ (boundary)', xy=(0, 0), xytext=(1.2, -1.2),
                arrowprops=dict(arrowstyle='->', color='red'), fontsize=12, color='red')
    ax.set_xlim(-2, 3)
    ax.set_ylim(-2, 3)
    ax.set_xlabel('State x', fontsize=12)
    ax.set_ylabel('h(x)', fontsize=12)
    ax.set_title('CBF Function & Safe Set', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Subplot 2: First-order CBF time response ---
    ax = axes[1]
    t = np.linspace(0, 10, 300)
    h0 = 2.0
    h_unconstrained = h0 * np.exp(-0.1 * t) - 0.5 * t
    gamma = 1.0
    h_1st = h0 * np.exp(-gamma * t)
    ax.plot(t, h_unconstrained, 'r--', linewidth=2, label='Unconstrained (unsafe)')
    ax.plot(t, h_1st, 'C0-', linewidth=2.5, label=r'1st-order: $\dot{h} + \gamma h \geq 0$')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.8)
    ax.fill_between(t, 0, 3, alpha=0.1, color='green')
    ax.fill_between(t, -2, 0, alpha=0.1, color='red')
    ax.annotate('Allowed to decay rapidly\n(no rate limit on $\dot{h}$)', xy=(2.5, 0.15), fontsize=10, color='C0')
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.5, 2.5)
    ax.set_xlabel('Time t', fontsize=12)
    ax.set_ylabel('h(x(t))', fontsize=12)
    ax.set_title('First-Order CBF Response', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- Subplot 3: Second-order HOCBF time response ---
    ax = axes[2]
    alpha1_t = 1.0
    alpha2_t = 6.0
    wn_t = np.sqrt(alpha1_t * alpha2_t)
    zeta_t = alpha2_t / (2 * wn_t)
    omega_d_t = wn_t * np.sqrt(max(1e-9, 1 - zeta_t**2))
    h_tuned = h0 * np.exp(-zeta_t * wn_t * t) * (
        np.cos(omega_d_t * t) + (zeta_t / omega_d_t) * np.sin(omega_d_t * t)
    )

    ax.plot(t, h_1st, 'C0--', linewidth=2, label='1st-order CBF', alpha=0.6)
    ax.plot(t, h_tuned, 'C2-', linewidth=2.5, label=r'HOCBF: $\ddot{h} + \alpha_2 \dot{h} + \alpha_2\alpha_1 h \geq 0$')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.8)
    ax.fill_between(t, 0, 3, alpha=0.1, color='green')
    ax.fill_between(t, -2, 0, alpha=0.1, color='red')
    ax.annotate('Smooth approach\n(rate of $\dot{h}$ is limited)', xy=(2.5, 0.35), fontsize=10, color='C2')
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.5, 2.5)
    ax.set_xlabel('Time t', fontsize=12)
    ax.set_ylabel('h(x(t))', fontsize=12)
    ax.set_title('Second-Order HOCBF Response', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Control Barrier Functions: From First-Order to High-Order', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    out_path = 'output/multi_stack/cbf_tests/cbf_function_schematic.png'
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    print(f'Saved CBF schematic: {out_path}')


if __name__ == "__main__":
    main()
