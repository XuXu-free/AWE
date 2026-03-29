# 单槽碱性水电解槽(AWE)系统：非线性非仿射CBF推导（修正版）

## 1. 系统模型（非线性非仿射形式）

### 1.1 状态与控制变量

$$x = \begin{bmatrix} T_{s,in} \\ T_s \\ T_{sep} \\ T_{c,out} \\ n_{H_2}^{an} \\ n_{H_2}^{sep,liq} \\ n_{H_2}^{sep,gas} \end{bmatrix} \in \mathbb{R}^7, \quad u = \begin{bmatrix} I \\ v_{lye} \\ v_c \end{bmatrix} \in \mathbb{R}^3$$

### 1.2 非仿射状态方程

系统动态具有一般非线性形式：
$$\dot{x} = f(x, u)$$

其中 $f$ 在控制输入 $u$ 上**非仿射**，主要来源：

| 非仿射项 | 数学形式 | 物理来源 |
|:---|:---|:---|
| 法拉第效率耦合 | $u_1 \cdot \eta_F(u_1, x_2)$ | 电流效率非线性 |
| 状态-输入乘积 | $x_5 u_2$, $x_6 u_2$ | 对流输运 |
| 三阶耦合 | $x_3 x_7 u_1 \eta_F$ | 气相反应消耗 |

## 2. HTO约束的精确定义

### 2.1 气相氢气摩尔分数（体积比）

根据理想气体状态方程 $PV = nRT$：

$$\text{HTO}_{pct} = \frac{n_{H_2}^{sep,gas} \cdot R \cdot T_{sep}}{P_{sys} \cdot V_{sep,gas}} \times 100\%$$

**量纲验证**：
$$\frac{[mol] \cdot [J/(mol \cdot K)] \cdot [K]}{[Pa] \cdot [m^3]} = \frac{[J]}{[J/m^3] \cdot [m^3]} = \text{无量纲} \checkmark$$

### 2.2 安全约束

安全运行要求：
$$\text{HTO}_{pct} \leq 2\% \quad \Leftrightarrow \quad \frac{R \cdot x_3 \cdot x_7}{P_{sys} \cdot V_{sep,gas}} \leq 0.02$$

## 3. 非仿射控制障碍函数（CBF）设计

### 3.1 CBF候选函数

定义安全集：
$$\mathcal{C} = \{x \in \mathbb{R}^7 : h(x) \geq 0\}$$

**CBF函数**（高相对度处理）：
$$h(x) = 0.02 - \underbrace{\frac{R}{P_{sys} V_{sep,gas}}}_{K_{HTO}} x_3 x_7$$

或无量纲形式：
$$h(x) = 1 - \frac{x_3 x_7}{0.02 / K_{HTO}} = 1 - \frac{x_3 x_7}{x_{3,ref} x_{7,ref}}$$

### 3.2 非仿射系统的CBF条件

对于一般非线性系统 $\dot{x} = f(x, u)$，标准CBF条件为：

$$\sup_{u \in \mathcal{U}} \left[ \frac{\partial h}{\partial x} f(x, u) \right] \geq -\alpha(h(x))$$

其中：
- $\frac{\partial h}{\partial x} = \nabla h^T$ 是CBF的梯度
- $\alpha(\cdot)$ 是类$\mathcal{K}$函数（通常取 $\alpha(h) = \gamma h$，$\gamma > 0$）

### 3.3 完整李导数计算

**CBF梯度**：
$$\nabla h = \left[0,\, 0,\, -K_{HTO} x_7,\, 0,\, 0,\, 0,\, -K_{HTO} x_3\right]$$

**时间导数**（链式法则）：
$$\dot{h} = \nabla h \cdot f(x, u) = -K_{HTO}(x_7 \dot{x}_3 + x_3 \dot{x}_7)$$

### 3.4 代入具体动态

#### 分离器温度动态
$$\dot{x}_3 = \frac{1}{C_{sep}}\left[0.5 c_{lye}\rho_{lye}(x_2 - x_3)u_2 - Q_{sep,diss}(x_3)\right]$$

#### 气相氢气动态（关键非仿射项）
$$\dot{x}_7 = \frac{x_6}{\tau_{sep}} - \dot{n}_{H_2}^{out}$$

其中 $\dot{n}_{H_2}^{out}$ 是气相氢气被排出的速率，与产氧速率相关：

$$\dot{n}_{H_2}^{out} = \frac{x_7}{x_7 + n_{O_2}^{sep,gas}} \cdot \dot{n}_{O_2}^{prod} \cdot \frac{P_{sys}V_{sep,gas}}{R x_3}$$

由于 $n_{O_2}^{sep,gas} \approx n_{O_2}^{prod}$（氧气产率）：
$$\dot{n}_{O_2}^{prod} = \frac{N_{cell} u_1 \eta_F(u_1, x_2)}{4F}$$

最终：
$$\dot{x}_7 = \frac{x_6}{\tau_{sep}} - \frac{R x_3 x_7 N_{cell} u_1 \eta_F(u_1, x_2)}{4F P_{sys}V_{sep,gas}(x_7 + n_{O_2})}$$

### 3.5 展开的非仿射CBF条件

$$\dot{h}(x, u) = L_f^{0} h(x) + L_f^{1} h(x, u_1) + L_f^{2} h(x, u_2)$$

其中各项：

**纯状态漂移项**：
$$L_f^{0} h(x) = -K_{HTO} x_7 \frac{-Q_{sep,diss}(x_3)}{C_{sep}} - K_{HTO} x_3 \frac{x_6}{\tau_{sep}}$$

**电流控制项（非仿射）**：
$$L_f^{1} h(x, u_1) = + \underbrace{\frac{K_{HTO} R x_3^2 x_7 N_{cell} u_1 \eta_F(u_1, x_2)}{4F P_{sys}V_{sep,gas}(x_7 + n_{O_2})}}_{\text{三阶非线性耦合}}$$

**流量控制项（仿射）**：
$$L_f^{2} h(x, u_2) = - \underbrace{\frac{0.5 K_{HTO} c_{lye}\rho_{lye} x_7 (x_2 - x_3)}{C_{sep}}}_{\beta(x)} u_2$$

## 4. 非仿射CBF约束优化问题

### 4.1 点wise优化（每个时刻）

在状态 $x$ 处，寻找控制 $u$ 满足：

$$\begin{aligned}
\min_{u \in \mathcal{U}} \quad & \|u - u_{ref}\|^2 \\
\text{s.t.} \quad & \dot{h}(x, u) \geq -\gamma h(x) \\
& u_{min} \leq u \leq u_{max}
\end{aligned}$$

其中约束展开为：

$$\frac{K_{HTO} R x_3^2 x_7 N_{cell}}{4F P_{sys}V_{sep,gas}} \cdot \frac{u_1 \eta_F(u_1, x_2)}{x_7 + n_{O_2}} - \beta(x) u_2 \geq -\gamma h(x) - L_f^{0} h(x)$$

### 4.2 非仿射项的处理方法

#### 方法1：直接非线性优化
使用CasADi/IPOPT直接求解非凸优化问题：

$$\min_{u_1, u_2} (u_1 - u_{1,ref})^2 + (u_2 - u_{2,ref})^2 + (u_3 - u_{3,ref})^2$$

$$\text{s.t.} \quad \gamma(x) \cdot u_1 \eta_F(u_1, x_2) - \beta(x) u_2 + \phi(x) + \gamma h(x) \geq 0$$

#### 方法2：序贯凸规划（SCP）
在第 $k$ 次迭代，在当前点 $(u_1^{(k)}, x_2)$ 处线性化：

$$u_1 \eta_F(u_1, x_2) \approx u_1^{(k)} \eta_F(u_1^{(k)}, x_2) + \frac{\partial (u_1 \eta_F)}{\partial u_1}\bigg|_{(k)} (u_1 - u_1^{(k)})$$

其中：
$$\frac{\partial (u_1 \eta_F)}{\partial u_1} = \eta_F + u_1 \frac{\partial \eta_F}{\partial u_1}$$

$$\frac{\partial \eta_F}{\partial u_1} = \frac{2 u_1 f_1 f_2}{(f_1 + u_1^2)^2}$$

#### 方法3：输入变换（反馈线性化思路）
定义新控制变量：

$$v = u_1 \eta_F(u_1, x_2)$$

反解 $u_1$（数值或近似）：

$$u_1 = \eta_F^{-1}(v / u_1, x_2) \quad \text{(隐式)}$$

在实际应用中可采用查表或牛顿迭代。

## 5. 实用CBF-QP控制器（近似仿射形式）

### 5.1 局部线性化CBF

在工作点 $(\bar{x}, \bar{u})$ 附近，将非仿射项近似为仿射：

$$u_1 \eta_F(u_1, x_2) \approx \bar{g} u_1 + \bar{c}$$

其中：
$$\bar{g} = \frac{\partial (u_1 \eta_F)}{\partial u_1}\bigg|_{(\bar{u}_1, \bar{x}_2)}, \quad \bar{c} = \bar{u}_1 \eta_F(\bar{u}_1, \bar{x}_2) - \bar{g} \bar{u}_1$$

### 5.2 标准CBF-QP形式

$$\begin{aligned}
\min_{u, \delta} \quad & \|u - u_{ref}\|^2 + \lambda \delta^2 \\
\text{s.t.} \quad & \bar{\gamma}(x) \cdot (\bar{g} u_1 + \bar{c}) - \beta(x) u_2 + \phi(x) \geq -\gamma h(x) - \delta \\
& u_{min} \leq u \leq u_{max} \\
& \delta \geq 0
\end{aligned}$$

其中 $\delta$ 是松弛变量（处理可能的不可行）。

## 6. 完整安全约束集

### 6.1 多重CBF

定义多个安全约束：

| 约束 | CBF函数 | 物理意义 |
|:---|:---|:---|
| $h_1(x)$ | $363.15 - x_2$ | 温度上限（90°C） |
| $h_2(x)$ | $x_2 - 293.15$ | 温度下限（20°C） |
| $h_3(x)$ | $2.2 - U_{cell}(x_2, u_1)$ | 电压上限 |
| $h_4(x)$ | $6\text{e}6 - P(x_2, u_1)$ | 功率上限 |
| $h_5(x)$ | $0.02 - K_{HTO} x_3 x_7$ | **HTO上限** |

### 6.2 联合CBF条件

对于多个CBF，需要同时满足：

$$\dot{h}_i(x, u) \geq -\gamma_i h_i(x), \quad \forall i = 1, \dots, 5$$

在QP中作为多个线性约束处理。

## 7. 实现要点

### 7.1 数值稳定性

- **温度边界处理**：使用 $\ln$ 变换处理大温度范围
- **HTO小量处理**：当 $h_5 < 0.001$ 时切换到更高增益 $\gamma$
- **输入饱和**：使用投影梯度或内点法

### 7.2 与NMPC的结合

CBF可作为NMPC的**终端约束**或**状态约束**：

$$\begin{aligned}
\min_{u_{0:N-1}} \quad & J(x, u) \\
\text{s.t.} \quad & x_{k+1} = f(x_k, u_k) \\
& h(x_k) \geq 0, \quad \forall k \\
& \dot{h}(x_k, u_k) \geq -\gamma h(x_k), \quad \forall k
\end{aligned}$$

## 8. 总结

本推导针对**非线性非仿射**AWE系统：

1. **明确了HTO的物理定义**（气相摩尔分数）
2. **处理了三阶非仿射耦合** $x_3 x_7 u_1 \eta_F$
3. **提供了三种实用方法**：
   - 直接非线性优化（精确但计算重）
   - 序贯凸规划（折中）
   - 局部线性化CBF-QP（实时可行）
4. **建立了完整的多约束CBF框架**

关键洞见：**在HTO危险时，必须通过增大电流 $I$ 来提高氧气产率，从而"稀释"气相中的氢气浓度。**
