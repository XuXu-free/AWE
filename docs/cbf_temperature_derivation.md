# AWE系统温度约束的CBF推导分析

## 1. 温度约束的CBF候选函数

单槽AWE系统的温度安全约束为：
- 温度上限：$T_s \leq T_{\max} = 363.15\,\text{K}$（$90^\circ\text{C}$）
- 温度下限：$T_s \geq T_{\min} = 293.15\,\text{K}$（$20^\circ\text{C}$）

选取CBF候选函数（仅依赖于状态$x_2 = T_s$）：
$$
\begin{aligned}
h_1(\mathbf{x}) &= T_{\max} - x_2 = 363.15 - T_s \\
h_2(\mathbf{x}) &= x_2 - T_{\min} = T_s - 293.15
\end{aligned}
$$

**有效性说明**：$h_1$和$h_2$均为仅关于状态$x_2$的纯状态函数，满足标准CBF要求$h: \mathbb{R}^n \to \mathbb{R}$的基本假设。

---

## 2. 温度上限 $h_1$ 的李导数推导

### 2.1 梯度向量

$$
\nabla h_1 = \left[ 0,\ -1,\ 0,\ 0,\ 0,\ 0,\ 0 \right]
$$

仅对$x_2 = T_s$的偏导数为$-1$，其余分量均为0。

### 2.2 时间导数（链式法则）

$$
\dot{h}_1 = \nabla h_1 \cdot f(\mathbf{x}, \mathbf{u}) = -\dot{x}_2 = -\dot{T}_s
$$

### 2.3 代入电解槽热平衡动态

电解槽温度动态为：
$$
\dot{x}_2 = \frac{1}{C_s}\left[ Q_{ele}(u_1, x_2) - Q_{diss}(x_2) - c_{lye}\rho_{lye} u_2 (x_2 - x_1) \right]
$$

其中：
- $Q_{ele}(u_1, x_2) = N_{cell} \cdot u_1 \cdot \left[ U_{cell}(u_1, x_2) - \eta_F(u_1, x_2) \cdot U_{th} \right]$ 为电化学产热功率
- $Q_{diss}(x_2) = \sigma_s(x_2 - T_{am}) + \epsilon_s \sigma_b A_{stack}(x_2^4 - T_{am}^4)$ 为散热功率
- $c_{lye}\rho_{lye} u_2 (x_2 - x_1)$ 为碱液对流带走的热量

### 2.4 完整的CBF时间导数

将$\dot{x}_2$代入$\dot{h}_1 = -\dot{x}_2$：

$$
\boxed{
\dot{h}_1(\mathbf{x}, \mathbf{u}) = -\frac{1}{C_s}\left[ Q_{ele}(u_1, x_2) - Q_{diss}(x_2) - c_{lye}\rho_{lye} u_2 (x_2 - x_1) \right]
}
$$

### 2.5 物理意义分析

| 控制动作 | 对$\dot{h}_1$的影响 | 物理机制 |
|:---|:---|:---|
| **增大电流 $u_1$** | 减小（负向） | $Q_{ele}$增大，电堆升温，$h_1$减小更快 |
| **增大碱液流速 $u_2$** | 增大（正向，当$x_2 > x_1$） | 增强对流换热，带走更多热量，抑制升温 |
| **增大冷却水 $u_3$** | **间接**正向 | 降低$x_1 = T_{s,in}$，增强换热驱动力 |

**关键结论**：当温度接近上限（$h_1 \to 0$）时，CBF条件要求$\dot{h}_1 \geq -\gamma_1 h_1$。此时必须**增大碱液流量**或**降低电流**以确保$\dot{h}_1$足够大（正），从而维持$h_1 \geq 0$。

### 2.6 CBF不等式

$$
-\frac{1}{C_s}\left[ Q_{ele}(u_1, x_2) - Q_{diss}(x_2) - c_{lye}\rho_{lye} u_2 (x_2 - x_1) \right] + \gamma_1 (363.15 - x_2) \geq 0
$$

---

## 3. 温度下限 $h_2$ 的李导数推导

### 3.1 梯度向量

$$
\nabla h_2 = \left[ 0,\ 1,\ 0,\ 0,\ 0,\ 0,\ 0 \right]
$$

### 3.2 时间导数

$$
\dot{h}_2 = \nabla h_2 \cdot f(\mathbf{x}, \mathbf{u}) = \dot{x}_2 = \dot{T}_s
$$

### 3.3 完整的CBF时间导数

$$
\boxed{
\dot{h}_2(\mathbf{x}, \mathbf{u}) = \frac{1}{C_s}\left[ Q_{ele}(u_1, x_2) - Q_{diss}(x_2) - c_{lye}\rho_{lye} u_2 (x_2 - x_1) \right]
}
$$

**注意**：$\dot{h}_2 = -\dot{h}_1$，两者互为相反数。这一对称性源于$h_1 + h_2 = T_{\max} - T_{\min} = \text{const}$。

### 3.4 物理意义分析

| 控制动作 | 对$\dot{h}_2$的影响 | 物理机制 |
|:---|:---|:---|
| **增大电流 $u_1$** | 增大（正向） | 产热增加，电堆升温，$h_2$增大 |
| **增大碱液流速 $u_2$** | 减小（负向，当$x_2 > x_1$） | 对流散热增强，电堆降温，$h_2$减小 |

**关键结论**：当温度接近下限（$h_2 \to 0$）时，必须**增大电流**或**减小碱液流量**以维持$\dot{h}_2 \geq -\gamma_2 h_2$，防止温度跌破安全下限。

### 3.5 CBF不等式

$$
\frac{1}{C_s}\left[ Q_{ele}(u_1, x_2) - Q_{diss}(x_2) - c_{lye}\rho_{lye} u_2 (x_2 - x_1) \right] + \gamma_2 (x_2 - 293.15) \geq 0
$$

---

## 4. 与HTO约束的对比

| 特性 | $h_1$（温度上限） | $h_2$（温度下限） | $h_{HTO}$（HTO上限） |
|:---|:---|:---|:---|
| **状态依赖性** | 仅$x_2$ | 仅$x_2$ | $x_3, x_7$ |
| **梯度稀疏性** | 仅$\partial h/\partial x_2 \neq 0$ | 仅$\partial h/\partial x_2 \neq 0$ | $x_3, x_7$两项非零 |
| **非仿射项** | $Q_{ele}(u_1, x_2)$中的$u_1 \cdot U_{cell}$ | 同上 | $u_1 \cdot \eta_F(u_1, x_2)$ |
| **控制耦合** | 电流产热+流量换热 | 电流产热+流量换热 | 电流产氧+流量降温 |
| **李导数复杂度** | 中等（需展开$Q_{ele}$） | 中等 | 较高（三阶状态耦合） |

---

## 5. 关于电压、功率约束的说明

电压约束$h_3(\mathbf{x}) = 2.2 - U_{cell}(x_2, u_1)$和功率约束$h_4(\mathbf{x}) = 6\times10^6 - P(x_2, u_1)$**显式依赖于控制输入$u_1$**，即：

$$
h_3(\mathbf{x}, u_1) = 2.2 - \underbrace{[U_{rev} + V_{ohm}(u_1, x_2) + V_{act}(u_1, x_2)]}_{U_{cell}}
$$

这违反了标准CBF定义中$h: \mathbb{R}^n \to \mathbb{R}$（仅状态函数）的基本要求。在模型预测控制的ZOH（零阶保持）假设下，$U_{cell}$和$P$在单个控制周期内被视为常数，其变化率概念上由$\dot{T}_s$主导，但更合理的处理方式是将它们作为**显式的输入约束**（$u_1 \in [U_{cell}^{-1}(2.2), U_{cell}^{-1}(U_{cell,min})]$）而非状态CBF。

因此，在论文的CBF理论框架中，重点给出了$h_1, h_2$和$h_{HTO}$的完整李导数推导，而电压、功率约束在实际实现中通过投影算子或直接输入饱和处理。

---

## 6. 联合CBF优化中的温度约束

在多约束CBF联合优化中，温度上下限作为两个独立的不等式约束：

$$
\begin{aligned}
&-\frac{1}{C_s}\left[ Q_{ele}(u_1, x_2) - Q_{diss}(x_2) - c_{lye}\rho_{lye} u_2 (x_2 - x_1) \right] + \gamma_1 (363.15 - x_2) \geq 0 \\
&\frac{1}{C_s}\left[ Q_{ele}(u_1, x_2) - Q_{diss}(x_2) - c_{lye}\rho_{lye} u_2 (x_2 - x_1) \right] + \gamma_2 (x_2 - 293.15) \geq 0
\end{aligned}
$$

这两个约束与HTO的CBF条件（式~\eqref{eq:cbf_condition}）及输入饱和约束$\mathbf{u}_{min} \leq \mathbf{u} \leq \mathbf{u}_{max}$共同构成完整的非线性规划问题，由CasADi/IPOPT统一求解。
