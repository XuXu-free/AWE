# 二阶 HTO HOCBF 详细数学推导

## 1. 问题动机与理论基础

### 1.1 一阶 CBF 的局限性

一阶控制障碍函数（CBF）要求安全集 $\mathcal{C} = \{x : h(x) \geq 0\}$ 满足：

$$\dot{h}(x,u) + \alpha(h(x)) \geq 0$$

其中 $\dot{h} = L_f h = \nabla h \cdot f(x,u)$。在 AWE 多槽系统中，$h_{HTO}$ 依赖于 $n_{gas}$（分离器气相氢气物质的量）和 $T_{sep}$（分离器温度）。控制量 $u = [I_{1..4}, v_{lye,1..4}, v_c]$ 通过系统动态 $f(x,u)$ 进入 $\dot{h}$，但在某些工况区域，$\partial \dot{h} / \partial u$（控制对 $h$ 变化率的灵敏度）很小。这意味着一阶 CBF 允许控制量大幅变化而不显著影响约束，导致电流曲线出现锯齿状（sawtooth）突变。

**二阶高阶控制障碍函数（HOCBF）** 通过约束 $\ddot{h}$ 来解决这个问题，即限制 $\dot{h}$ 的变化率，为安全边界引入"惯性"，从而平滑控制响应。

### 1.2 HOCBF 理论框架

考虑仿射控制系统：

$$\dot{x} = f(x) + g(x)u$$

虽然 AWE 系统包含非仿射项（$I \cdot \eta_F(I, T_s)$），但在 HOCBF 推导中仍采用符号李导数框架。对于相对阶为 $r$ 的约束函数 $h(x)$，定义 HOCBF 层级函数：

$$\begin{aligned}
\psi_0(x) &= h(x) \\
\psi_1(x,u) &= \dot{h}(x,u) + \alpha_1(\psi_0(x)) \\
\psi_2(x,u) &= \dot{\psi}_1(x,u) + \alpha_2(\psi_1(x,u)) \\
&\vdots \\
\psi_r(x,u) &= \dot{\psi}_{r-1}(x,u) + \alpha_r(\psi_{r-1}(x,u)) \geq 0
\end{aligned}$$

其中 $\alpha_i \in \mathcal{K}$（扩展类 $\mathcal{K}$ 函数，通常取线性形式 $\alpha_i(s) = \alpha_i \cdot s$）。

对于二阶 HOCBF（$r=2$）：

$$\begin{aligned}
\psi_0 &= h \\
\psi_1 &= \dot{h} + \alpha_1 h \\
\psi_2 &= \ddot{h} + \alpha_2 \psi_1 \geq 0
\end{aligned}$$

展开得：

$$\ddot{h} + \alpha_2 \dot{h} + \alpha_2 \alpha_1 h \geq 0$$

这等价于一个二阶低通滤波器作用于 $h$，要求 $h$ 以平滑动态趋近于零边界。

---

## 2. 多槽系统 HTO CBF 函数定义

### 2.1 状态与控制向量

多槽 AWE 系统状态向量 $x \in \mathbb{R}^{13}$：

$$x = \begin{bmatrix} T_{s,in} & T_{s,1..4} & T_{sep} & T_{c,out} & n_{H_2,1..4}^{an} & n_{liq} & n_{gas} \end{bmatrix}^T$$

控制向量 $u \in \mathbb{R}^9$：

$$u = \begin{bmatrix} I_{1..4} & v_{lye,1..4} & v_c \end{bmatrix}^T$$

### 2.2 HTO 约束函数

气相氢气摩尔分数（体积百分比）：

$$\text{HTO}_{pct} = \frac{n_{gas} \cdot R \cdot T_{sep}}{P_{sys} \cdot V_{sep,gas}} \times 100\%$$

安全约束要求 $\text{HTO}_{pct} \leq 2\%$。定义 CBF 候选函数：

$$h_{HTO}(x) = \text{HTO}_{max} - \frac{n_{gas} \cdot R \cdot T_{sep}}{P_{sys} \cdot V_{sep,gas}} - h_{margin}$$

其中 $h_{margin}$ 为安全裕度（通常取 $0.007$ 以提供额外保守性）。记常数：

$$K_{HTO} = \frac{R}{P_{sys} \cdot V_{sep,gas}}$$

则：

$$h_{HTO}(x) = \text{HTO}_{max} - K_{HTO} \cdot n_{gas} \cdot T_{sep} - h_{margin}$$

---

## 3. 一阶 Lie 导数推导

### 3.1 梯度向量

$h_{HTO}$ 仅依赖于 $x_5 = T_{sep}$（第 6 个状态分量，索引 5）和 $x_{12} = n_{gas}$（第 13 个状态分量，索引 12）：

$$\nabla h_{HTO} = \frac{\partial h_{HTO}}{\partial x} = \begin{bmatrix} 0 & \cdots & \underbrace{-K_{HTO} \cdot n_{gas}}_{\partial h / \partial T_{sep}} & \cdots & \underbrace{-K_{HTO} \cdot T_{sep}}_{\partial h / \partial n_{gas}} \end{bmatrix}$$

具体分量：

$$\frac{\partial h_{HTO}}{\partial T_{sep}} = -K_{HTO} \cdot n_{gas}, \quad \frac{\partial h_{HTO}}{\partial n_{gas}} = -K_{HTO} \cdot T_{sep}$$

### 3.2 一阶时间导数

$$\dot{h}_{HTO} = \nabla h_{HTO} \cdot f(x,u) = -K_{HTO} \cdot \left( n_{gas} \cdot \dot{T}_{sep} + T_{sep} \cdot \dot{n}_{gas} \right)$$

#### 3.2.1 分离器温度动态 $\dot{T}_{sep}$

$$\dot{T}_{sep} = \frac{1}{C_{sep}}\left[ \frac{1}{2} c_{lye} \rho_{lye} v_{tot} (T_{sep,in} - T_{sep}) - Q_{sep,diss}(T_{sep}) \right]$$

其中：
- $v_{tot} = \sum_{i=1}^4 v_{lye,i}$ 为总碱液流量
- $T_{sep,in} = \frac{\sum_{i=1}^4 v_{lye,i} T_{s,i}}{v_{tot}}$ 为进入分离器的混合碱液温度
- $Q_{sep,diss}(T_{sep}) = \sigma_{sep}(T_{sep} - T_{amb}) + \epsilon_{sep} \sigma_b A_{sep}(T_{sep}^4 - T_{amb}^4)$ 为分离器散热功率

#### 3.2.2 气相氢气动态 $\dot{n}_{gas}$

$$\dot{n}_{gas} = \frac{n_{liq}}{\tau_{sep}} - \frac{R \cdot T_{sep} \cdot n_{gas} \cdot \sum_{i=1}^4 \dot{n}_{O_2,i}}{P_{sys} \cdot V_{sep,gas}}$$

其中氧气产生率：

$$\dot{n}_{O_2,i} = \frac{N_{cell} \cdot I_i \cdot \eta_F(I_i, T_{s,i})}{4F}$$

法拉第效率：

$$\eta_F(I, T_s) = \frac{I^2}{f_1(T_s) + I^2} \cdot f_2(T_s)$$

温度依赖参数：

$$\begin{aligned}
f_1(T_s) &= 50 + 2.5(T_s - 273.15) \\
f_2(T_s) &= 0.92 - 6.25 \times 10^{-6}(T_s - 273.15)
\end{aligned}$$

### 3.3 一阶李导数的完整展开

将 $\dot{T}_{sep}$ 和 $\dot{n}_{gas}$ 代入 $\dot{h}_{HTO}$：

$$\dot{h}_{HTO}(x,u) = \underbrace{\gamma(x) \cdot \sum_{i=1}^4 I_i \eta_F(I_i, T_{s,i})}_{\text{电流控制（非仿射）}} - \underbrace{\beta(x) \cdot v_{tot}}_{\text{流量控制（仿射）}} + \phi(x) + \xi(x) \cdot v_c$$

各项系数定义如下：

**电流控制增益** $\gamma(x)$（始终为正）：

$$\gamma(x) = \frac{K_{HTO} \cdot R \cdot T_{sep}^2 \cdot n_{gas} \cdot N_{cell}}{4F \cdot P_{sys} \cdot V_{sep,gas}} > 0 \quad (\text{当 } T_{sep}, n_{gas} > 0)$$

**流量控制增益** $\beta(x)$：

$$\beta(x) = \frac{K_{HTO} \cdot n_{gas} \cdot c_{lye} \rho_{lye} (T_{sep,in} - T_{sep})}{2 C_{sep}}$$

其符号取决于 $(T_{sep,in} - T_{sep})$：
- 当 $T_{sep,in} > T_{sep}$（热碱液进入较冷分离器），$\beta > 0$，增大流量降低 $h$（不利）
- 当 $T_{sep,in} < T_{sep}$，$\beta < 0$，增大流量提高 $h$（有利）

**纯状态漂移项** $\phi(x)$：

$$\phi(x) = \frac{K_{HTO} \cdot n_{gas} \cdot Q_{sep,diss}(T_{sep})}{C_{sep}} - \frac{K_{HTO} \cdot T_{sep} \cdot n_{liq}}{\tau_{sep}}$$

**冷却水耦合项** $\xi(x)$：

$$\xi(x) \cdot v_c = -K_{HTO} \cdot n_{gas} \cdot \frac{\partial \dot{T}_{sep}}{\partial v_c} \cdot v_c$$

实际上 $v_c$ 对 $T_{sep}$ 的影响是间接的（通过换热器影响 $T_{s,in}$，再影响 $T_{s,i}$，最后影响 $T_{sep,in}$），因此 $\xi(x)$ 为间接耦合项，在高阶导数中才会显现。

### 3.4 一阶 CBF 条件

$$\psi_1(x,u) = \dot{h}_{HTO} + \alpha_1 h_{HTO} \geq 0$$

即：

$$\gamma(x) \sum_{i=1}^4 I_i \eta_F(I_i, T_{s,i}) - \beta(x) v_{tot} + \phi(x) + \alpha_1 h_{HTO} \geq 0$$

---

## 4. 二阶 Lie 导数推导

### 4.1 二阶导数定义

$$\ddot{h}_{HTO} = \frac{d}{dt}\dot{h}_{HTO} = \nabla_x \dot{h}_{HTO} \cdot f(x,u)$$

注意：我们将 $u$ 视为每个控制步长的决策变量（分段恒定），因此：

$$\ddot{h}_{HTO} = \frac{\partial \dot{h}_{HTO}}{\partial x} \cdot \dot{x} = \frac{\partial \dot{h}_{HTO}}{\partial x} \cdot f(x,u)$$

这是一个**近似**，假设控制步长 $\Delta t$ 内 $u$ 保持不变。在 CasADi 中计算为：

```python
lie1 = jacobian(h, x) @ f(x,u)
lie2 = jacobian(lie1, x) @ f(x,u)
```

### 4.2 $\ddot{h}_{HTO}$ 的显式展开

由 $\dot{h}_{HTO} = -K_{HTO}(n_{gas} \dot{T}_{sep} + T_{sep} \dot{n}_{gas})$，求时间导数：

$$\ddot{h}_{HTO} = -K_{HTO}\left[ \dot{n}_{gas} \dot{T}_{sep} + n_{gas} \ddot{T}_{sep} + \dot{T}_{sep} \dot{n}_{gas} + T_{sep} \ddot{n}_{gas} \right]$$

整理得：

$$\ddot{h}_{HTO} = -K_{HTO}\left[ 2\dot{n}_{gas}\dot{T}_{sep} + n_{gas}\ddot{T}_{sep} + T_{sep}\ddot{n}_{gas} \right]$$

#### 4.2.1 $\ddot{T}_{sep}$ 的推导

$$\ddot{T}_{sep} = \frac{d}{dt}\left( \frac{1}{C_{sep}}\left[ \frac{1}{2} c_{lye} \rho_{lye} v_{tot} (T_{sep,in} - T_{sep}) - Q_{sep,diss}(T_{sep}) \right] \right)$$

展开得：

$$\ddot{T}_{sep} = \frac{1}{C_{sep}}\left[ \frac{1}{2} c_{lye} \rho_{lye} v_{tot} (\dot{T}_{sep,in} - \dot{T}_{sep}) - \frac{\partial Q_{sep,diss}}{\partial T_{sep}} \dot{T}_{sep} \right]$$

其中：

$$\frac{\partial Q_{sep,diss}}{\partial T_{sep}} = \sigma_{sep} + 4 \epsilon_{sep} \sigma_b A_{sep} T_{sep}^3$$

而 $\dot{T}_{sep,in}$ 由各槽温度变化率和流量变化率共同决定：

$$\dot{T}_{sep,in} = \frac{\sum_{i=1}^4 (\dot{v}_{lye,i} T_{s,i} + v_{lye,i} \dot{T}_{s,i}) v_{tot} - \sum_{i=1}^4 v_{lye,i} T_{s,i} \cdot \dot{v}_{tot}}{v_{tot}^2}$$

在 $u$ 分段恒定假设下，$\dot{v}_{lye,i} = 0$，因此：

$$\dot{T}_{sep,in} = \frac{\sum_{i=1}^4 v_{lye,i} \dot{T}_{s,i}}{v_{tot}}$$

#### 4.2.2 $\ddot{n}_{gas}$ 的推导

$$\ddot{n}_{gas} = \frac{d}{dt}\left( \frac{n_{liq}}{\tau_{sep}} - \frac{R \cdot T_{sep} \cdot n_{gas} \cdot \dot{n}_{O_2,tot}}{P_{sys} \cdot V_{sep,gas}} \right)$$

展开得：

$$\ddot{n}_{gas} = \frac{\dot{n}_{liq}}{\tau_{sep}} - \frac{R}{P_{sys} V_{sep,gas}}\left[ \dot{T}_{sep} n_{gas} \dot{n}_{O_2,tot} + T_{sep} \dot{n}_{gas} \dot{n}_{O_2,tot} + T_{sep} n_{gas} \ddot{n}_{O_2,tot} \right]$$

其中：

$$\dot{n}_{O_2,tot} = \sum_{i=1}^4 \dot{n}_{O_2,i} = \sum_{i=1}^4 \frac{N_{cell} I_i \eta_F(I_i, T_{s,i})}{4F}$$

$$\ddot{n}_{O_2,tot} = \sum_{i=1}^4 \frac{N_{cell} I_i}{4F} \cdot \frac{\partial \eta_F}{\partial T_s}\bigg|_{T_{s,i}} \cdot \dot{T}_{s,i}$$

（$u$ 恒定假设下 $I_i$ 不变，仅 $T_{s,i}$ 变化影响 $\eta_F$）

法拉第效率对温度的偏导数：

$$\frac{\partial \eta_F}{\partial T_s} = \frac{I^2}{(f_1 + I^2)^2}\left[ (f_1 + I^2) \frac{\partial f_2}{\partial T_s} - f_2 \frac{\partial f_1}{\partial T_s} \right]$$

其中：

$$\frac{\partial f_1}{\partial T_s} = 2.5, \quad \frac{\partial f_2}{\partial T_s} = -6.25 \times 10^{-6}$$

#### 4.2.3 $\ddot{h}_{HTO}$ 的完整表达式

将 $\ddot{T}_{sep}$ 和 $\ddot{n}_{gas}$ 代入：

$$\ddot{h}_{HTO} = -K_{HTO}\Bigg\{ 2\dot{n}_{gas}\dot{T}_{sep} + n_{gas} \cdot \frac{1}{C_{sep}}\left[ \frac{1}{2} c_{lye} \rho_{lye} v_{tot} (\dot{T}_{sep,in} - \dot{T}_{sep}) - \frac{\partial Q_{sep,diss}}{\partial T_{sep}} \dot{T}_{sep} \right] + T_{sep}\left[ \frac{\dot{n}_{liq}}{\tau_{sep}} - \frac{R}{P_{sys} V_{sep,gas}}\left( \dot{T}_{sep} n_{gas} \dot{n}_{O_2,tot} + T_{sep} \dot{n}_{gas} \dot{n}_{O_2,tot} + T_{sep} n_{gas} \ddot{n}_{O_2,tot} \right) \right] \Bigg\}$$

该表达式涉及以下状态变量的时间导数：
- $\dot{T}_{s,i}$（各槽温度变化率）
- $\dot{T}_{sep}$（分离器温度变化率）
- $\dot{n}_{gas}$（气相氢气变化率）
- $\dot{n}_{liq}$（液相氢气变化率）
- $\dot{n}_{O_2,tot}$（总氧气产生率）
- $\ddot{n}_{O_2,tot}$（总氧气产生率的变化率）

---

## 5. HOCBF 层级约束条件

### 5.1 HOCBF 函数层级

对于 HTO 约束（二阶），定义：

$$\begin{aligned}
\psi_0(x) &= h_{HTO}(x) \\
\psi_1(x,u) &= \dot{h}_{HTO}(x,u) + \alpha_1 \cdot \psi_0(x) \\
\psi_2(x,u) &= \ddot{h}_{HTO}(x,u) + \alpha_2 \cdot \psi_1(x,u) \geq 0
\end{aligned}$$

### 5.2 二阶约束的显式形式

$$\ddot{h}_{HTO} + \alpha_2 \dot{h}_{HTO} + \alpha_2 \alpha_1 h_{HTO} \geq 0$$

代入 $\dot{h}_{HTO}$ 和 $\ddot{h}_{HTO}$ 的表达式：

$$-K_{HTO}\left[ 2\dot{n}_{gas}\dot{T}_{sep} + n_{gas}\ddot{T}_{sep} + T_{sep}\ddot{n}_{gas} \right] + \alpha_2 \left[ -K_{HTO}(n_{gas}\dot{T}_{sep} + T_{sep}\dot{n}_{gas}) \right] + \alpha_2 \alpha_1 h_{HTO} \geq 0$$

整理：

$$-K_{HTO}\left[ 2\dot{n}_{gas}\dot{T}_{sep} + n_{gas}\ddot{T}_{sep} + T_{sep}\ddot{n}_{gas} + \alpha_2(n_{gas}\dot{T}_{sep} + T_{sep}\dot{n}_{gas}) \right] + \alpha_2 \alpha_1 h_{HTO} \geq 0$$

### 5.3 物理意义解释

二阶 HOCBF 要求 $h_{HTO}$ 的变化满足二阶阻尼振荡特性：
- **$\alpha_1$**：一阶衰减率，决定 $h$ 趋近于边界的指数衰减速度
- **$\alpha_2$**：二阶阻尼系数，限制 $\dot{h}$ 的变化率，防止控制突变

类比二阶系统：

$$\ddot{h} + 2\zeta\omega_n \dot{h} + \omega_n^2 h \geq 0$$

参数对应关系：

$$\alpha_2 = 2\zeta\omega_n, \quad \alpha_1 = \frac{\omega_n}{2\zeta}$$

自然频率和阻尼比：

$$\omega_n = \sqrt{\alpha_1 \alpha_2}, \quad \zeta = \frac{1}{2}\sqrt{\frac{\alpha_2}{\alpha_1}}$$

---

## 6. 混合阶 CBF 策略

### 6.1 多约束系统

在多槽 AWE 系统中，共有 9 个安全约束：

| 索引 | 约束 | CBF 函数 | 阶数 | 原因 |
|:---:|:---|:---|:---:|:---|
| 0-3 | $T_{s,i} \leq T_{max}$ | $h_{Tmax,i} = T_{max} - T_{s,i}$ | 一阶 | 控制直接耦合（$I_i, v_{lye,i}$ 直接出现在 $\dot{T}_{s,i}$） |
| 4 | HTO $\leq$ 2% | $h_{HTO} = \text{HTO}_{max} - K_{HTO} n_{gas} T_{sep}$ | **二阶** | 间接控制耦合，一阶导致锯齿 |
| 5-8 | $T_{s,i} \geq T_{min}$ | $h_{Tmin,i} = T_{s,i} - T_{min}$ | 一阶 | 控制直接耦合 |

### 6.2 混合阶策略的理论依据

**为什么 HTO 需要二阶而温度不需要？**

1. **相对阶差异**：
   - 温度约束：$\dot{T}_{s,i}$ 直接依赖于 $I_i$ 和 $v_{lye,i}$，控制输入直接出现在一阶导数中，相对阶为 1
   - HTO 约束：$\dot{h}_{HTO}$ 依赖于 $\dot{T}_{sep}$ 和 $\dot{n}_{gas}$，而 $\dot{n}_{gas}$ 又间接依赖于各槽的 $I_i$ 通过 $T_{s,i} \to \eta_F \to \dot{n}_{O_2}$ 的耦合链，控制对 $h$ 的灵敏度经过系统动态"过滤"

2. **控制灵敏度分析**：
   - 温度：$\partial \dot{h}_{Tmax} / \partial I_i = -\frac{1}{C_s} \frac{\partial Q_{ele}}{\partial I_i} \sim O(10^3)$，灵敏度大
   - HTO：$\partial \dot{h}_{HTO} / \partial I_i$ 间接通过 $\dot{n}_{gas}$，灵敏度小且受状态耦合影响

3. **高频噪声抑制**：
   - 一阶 CBF 仅限制 $h$ 的下降速度，允许 $\dot{h}$ 突变
   - 二阶 HOCBF 限制 $\ddot{h}$，要求 $\dot{h}$ 平滑变化，从而抑制控制高频振荡

### 6.3 联合约束条件

优化问题中的 CBF 约束向量：

**一阶约束**（$j = 0,1,2,3,5,6,7,8$）：

$$L_f h_j + \gamma_j h_j + s_j \geq 0$$

**二阶约束**（$j = 4$，HTO）：

$$L_f^2 h_4 + \alpha_2 L_f h_4 + \alpha_2 \alpha_1 h_4 + s_4 \geq 0$$

其中 $s_j \geq 0$ 为松弛变量，用于处理不可行情形。

---

## 7. NLP 优化形式

### 7.1 决策变量

$$z = \begin{bmatrix} u & s \end{bmatrix}^T \in \mathbb{R}^{9 + n_{soft}}$$

其中：
- $u = [I_{1..4}, v_{lye,1..4}, v_c]$ 为控制量
- $s \in \mathbb{R}^{n_{soft}}$ 为松弛变量（仅对 soft_mask=True 的约束）

### 7.2 目标函数

$$\min_{z} \quad J(z) = \underbrace{\|u - u_{ref}\|^2_{W_u}}_{\text{参考跟踪}} + \underbrace{\lambda \|u - u_{last}\|^2}_{\text{控制平滑}} + \underbrace{\rho^T (s \odot s)}_{\text{松弛惩罚}}$$

权重矩阵：

$$W_u = \text{diag}\left( \frac{w_1}{(I_{max}-I_{min})^2}, \cdots, \frac{w_9}{(v_{c,max}-v_{c,min})^2} \right)$$

其中 $w_i$ 为 per-control 权重（`u_weight_scale`），用于优先调整某些控制量。

### 7.3 约束条件

**控制量边界**：

$$u_{min} \leq u \leq u_{max}$$

**CBF 约束**（通过 CasADi 符号函数构建）：

```python
# Symbolic CBF computation
lie1_sym, lie2_sym, h_sym = ca_lie(state, u_sym)

for j in range(9):
    if alpha2_vec[j] > 0:
        # Second-order HOCBF
        psi1 = lie1_sym[j] + alpha1_vec[j] * h_sym[j]
        cbf_j = lie2_sym[j] + alpha2_vec[j] * psi1
    else:
        # First-order CBF
        cbf_j = lie1_sym[j] + gamma_vec[j] * h_sym[j]

    if soft_mask[j]:
        g.append(cbf_j + s_sym[s_idx])  # soft: cbf + s >= 0
        s_idx += 1
    else:
        g.append(cbf_j)  # hard: cbf >= 0
```

**松弛变量边界**：

$$s \geq 0$$

### 7.4 归一化处理

为改善数值稳定性，对 CBF 函数和李导数进行归一化：

$$\tilde{h}_j = \frac{h_j}{h_{scale,j}}, \quad \tilde{L}_f h_j = \frac{L_f h_j}{L_{scale,j}}, \quad \tilde{L}_f^2 h_j = \frac{L_f^2 h_j}{L_{scale,j}^2}$$

典型取值：
- 温度约束：$h_{scale} = 70\,\text{K}$，$L_{scale} = 10^{-2}\,\text{K/s}$
- HTO 约束：$h_{scale} = 0.02$，$L_{scale} = 10^{-3}\,\text{s}^{-1}$

---

## 8. 参数设计准则

### 8.1 $\alpha_1$ 与 $\alpha_2$ 的选取

对于控制周期 $dt = 60\,\text{s}$ 的离散系统：

**一阶等效**：取 $\alpha_1 = 0.8$，对应离散衰减因子 $e^{-\alpha_1 dt} \approx e^{-48} \approx 0$，边界恢复极快。

**二阶阻尼**：取 $\alpha_2 = 2.0$，提供适度阻尼。

对应二阶系统参数：

$$\omega_n = \sqrt{\alpha_1 \alpha_2} = \sqrt{1.6} \approx 1.26\,\text{rad/s}$$

$$\zeta = \frac{1}{2}\sqrt{\frac{\alpha_2}{\alpha_1}} = \frac{1}{2}\sqrt{2.5} \approx 0.79$$

这是**轻微欠阻尼**（$\zeta < 1$），响应快速但允许少量超调。若需更强平滑效果，可增大 $\alpha_2$ 或减小 $\alpha_1$。

### 8.2 参数调节指南

| 现象 | 调节方案 |
|:---|:---|
| 电流仍有小幅锯齿 | 增大 $\alpha_2$（如 2.0 $\to$ 3.0） |
| HTO 边界响应过慢 | 增大 $\alpha_1$（如 0.8 $\to$ 1.2） |
| HTO 超调严重 | 减小 $\alpha_1$ 或增大 $\alpha_2$ |
| 优化求解困难 | 减小 $\alpha_2$，降低约束激进程度 |

### 8.3 与一阶 CBF 的等价条件

当 $\alpha_2 \to \infty$ 时，$\psi_2 \geq 0$ 要求 $\psi_1 \to 0$，即 $\dot{h} = -\alpha_1 h$，退化为一阶 CBF。

当 $\alpha_2 \to 0$ 时，二阶约束退化为 $\ddot{h} \geq 0$，仅要求 $h$ 的曲率非负，不提供边界保护。

---

## 9. 计算开销与实现

### 9.1 符号导数计算

使用 CasADi 自动微分计算：

1. $h(x)$ 的 Jacobian：$\partial h / \partial x$（$9 \times 13$ 稀疏矩阵，实际每行仅 1-2 个非零元）
2. $\dot{h} = L_f h$ 的 Jacobian：$\partial \dot{h} / \partial x$（$9 \times 13$ 矩阵）
3. $\ddot{h} = L_f^2 h$ 的计算：Jacobian 与 $f(x,u)$ 的矩阵-向量乘积

### 9.2 计算复杂度

| 操作 | 一阶 CBF | 二阶 HOCBF | 倍数 |
|:---|:---|:---|:---:|
| Jacobian $h$ | 1 次 | 1 次 | 1x |
| Jacobian $\dot{h}$ | 0 次 | 1 次 | — |
| 矩阵-向量积 | 1 次 | 2 次 | 2x |
| NLP 求解 | 9 约束 | 9 约束（1个二阶） | ~1.5x |

**实际测量**：二阶 HOCBF 的平均求解时间为一阶 CBF 的 1.5-2 倍，仍在实时范围内（典型值 < 30ms/步）。

### 9.3 稀疏性利用

CasADi 自动利用以下稀疏结构：
- $\partial h_{Tmax,i} / \partial x$：仅 $\partial h / \partial T_{s,i} = -1$ 非零
- $\partial h_{HTO} / \partial x$：仅 $\partial h / \partial T_{sep}$ 和 $\partial h / \partial n_{gas}$ 非零
- $\partial h_{Tmin,i} / \partial x$：仅 $\partial h / \partial T_{s,i} = 1$ 非零

---

## 10. 实现要点

### 10.1 代码中的关键计算

```python
# 一阶 Lie 导数
dh_dx = jacobian(h, x)          # 9x13 矩阵
lie1 = dh_dx @ f(x,u)           # 9-dim 向量

# 二阶 Lie 导数
dlie1_dx = jacobian(lie1, x)    # 9x13 矩阵
lie2 = dlie1_dx @ f(x,u)        # 9-dim 向量

# HOCBF 层级（仅 HTO 索引 4 使用二阶）
psi0 = h[4]
psi1 = lie1[4] + alpha1_hto * psi0
psi2 = lie2[4] + alpha2_hto * psi1  # >= 0
```

### 10.2 归一化实现

```python
# 归一化尺度
h_scales = [70.0]*4 + [0.02] + [70.0]*4
lie_deriv_scales = [1e-2]*4 + [1e-3] + [1e-2]*4

# 归一化
h_norm = h / h_scales
lie1_norm = lie1 / lie_deriv_scales
lie2_norm = lie2 / (lie_deriv_scales**2)
```

### 10.3 混合约束构建

```python
cbf = np.zeros(9)
for j in range(9):
    if alpha2_vec[j] > 0:
        # 二阶 HOCBF
        psi1 = lie1[j] + alpha1_vec[j] * h[j]
        cbf[j] = lie2[j] + alpha2_vec[j] * psi1
    else:
        # 一阶 CBF
        cbf[j] = lie1[j] + gamma_vec[j] * h[j]
```

---

## 11. 物理机制总结

### 11.1 为什么增大电流降低 HTO？

**反直觉特性**：在 HTO 危险时（$h_{HTO}$ 较小），必须通过**增大电流 $I_i$** 来提高氧气产率，从而"稀释"气相中的氢气浓度。

**物理链条**：

$$I_i \uparrow \Rightarrow \dot{n}_{O_2,i} \uparrow \Rightarrow \dot{n}_{H_2,out} \uparrow \Rightarrow n_{gas} \downarrow \Rightarrow \text{HTO} \downarrow$$

这是因为气相氢气的排出速率与氧气总产率成正比：

$$\dot{n}_{H_2,out} = \frac{R T_{sep} n_{gas} \sum_i \dot{n}_{O_2,i}}{P_{sys} V_{sep,gas}}$$

增大电流提高氧气产率，加速了氢气从气相中排出的速率，从而降低了 HTO。

### 11.2 二阶约束的平滑机制

一阶 CBF 仅要求：

$$\dot{h} \geq -\alpha_1 h$$

这允许 $\dot{h}$ 在每一步大幅跳变（只要满足不等式）。二阶 HOCBF 额外要求：

$$\ddot{h} \geq -\alpha_2 \psi_1$$

即 $\dot{h}$ 的变化率受限，从而：

1. 防止控制器为了快速恢复 $h$ 而突然大幅调整电流
2. 电流变化呈现惯性，锯齿被抑制
3. 在 HTO 边界附近形成"软着陆"行为

---

## 12. 结论

本推导建立了多槽 AWE 系统 HTO 约束的二阶 HOCBF 完整框架：

1. **CBF 函数**：$h_{HTO}(x) = \text{HTO}_{max} - K_{HTO} n_{gas} T_{sep} - h_{margin}$
2. **一阶李导数**：$\dot{h}_{HTO} = -K_{HTO}(n_{gas}\dot{T}_{sep} + T_{sep}\dot{n}_{gas})$
3. **二阶李导数**：$\ddot{h}_{HTO} = -K_{HTO}(2\dot{n}_{gas}\dot{T}_{sep} + n_{gas}\ddot{T}_{sep} + T_{sep}\ddot{n}_{gas})$
4. **HOCBF 条件**：$\psi_2 = \ddot{h}_{HTO} + \alpha_2\dot{h}_{HTO} + \alpha_2\alpha_1 h_{HTO} \geq 0$
5. **混合阶策略**：HTO 使用二阶，温度使用一阶
6. **控制策略**：增大电流以增加氧气产率，稀释气相氢气，但二阶约束平滑了调节过程

该框架通过显式处理高阶动态，在保持安全保证的同时显著改善了控制平滑性。

---

## 13. 参考文献

- Xiao, W., & Belta, C. (2019). Control barrier functions for systems with high relative degree. *IEEE Conference on Decision and Control*.
- Nguyen, Q., & Sreenath, K. (2016). Exponential control barrier functions for enforcing high relative-degree safety-critical constraints. *American Control Conference*.
- Ames, A. D., Coogan, S., Egerstedt, M., Notomista, G., Sreenath, K., & Tabuada, P. (2019). Control barrier functions: Theory and applications. *European Control Conference*.
