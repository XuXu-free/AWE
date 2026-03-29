# 单槽碱性水电解槽(AWE)系统：非线性非仿射模型与CBF约束推导

## 1. 系统概述

### 1.1 物理系统
- **单槽碱性水电解槽（Single-Stack Alkaline Water Electrolyzer）**
- 额定电流：$I_{rated} = 7800\,\text{A}$
- 电解槽数：$N_{cell} = 368$
- 系统压力：$P_{sys} = 1.8\,\text{MPa}$

### 1.2 状态变量
$$
x = \begin{bmatrix} T_{s,in} \\ T_s \\ T_{sep} \\ T_{c,out} \\ n_{H_2}^{an} \\ n_{H_2}^{sep,liq} \\ n_{H_2}^{sep,gas} \end{bmatrix} \in \mathbb{R}^7
$$

| 状态 | 物理意义 | 单位 | 典型初始值 |
|:---|:---|:---|:---|
| $x_1 = T_{s,in}$ | 电解槽入口碱液温度 | K | 345.0 (72°C) |
| $x_2 = T_s$ | 电解槽（电堆）温度 | K | 358.0 (85°C) |
| $x_3 = T_{sep}$ | 气液分离器温度 | K | 358.0 (85°C) |
| $x_4 = T_{c,out}$ | 换热器冷却水出口温度 | K | 325.0 (52°C) |
| $x_5 = n_{H_2}^{an}$ | 阳极氢气物质的量 | mol | ~0.52 |
| $x_6 = n_{H_2}^{sep,liq}$ | 分离器液相氢气物质的量 | mol | ~0.52 |
| $x_7 = n_{H_2}^{sep,gas}$ | 分离器气相氢气物质的量 | mol | ~0.52 |

### 1.3 控制输入
$$
u = \begin{bmatrix} I \\ v_{lye} \\ v_c \end{bmatrix} \in \mathbb{R}^3
$$

| 输入 | 物理意义 | 单位 | 约束范围 |
|:---|:---|:---|:---|
| $u_1 = I$ | 电解电流 | A | $[0, 9360]$ |
| $u_2 = v_{lye}$ | 碱液体积流量 | $\text{m}^3/\text{s}$ | $[0, 0.1]$ |
| $u_3 = v_c$ | 冷却水体积流量 | $\text{m}^3/\text{s}$ | $[0, 1.0]$ |

---

## 2. 非线性非仿射状态空间方程

### 2.1 电化学子系统（核心非线性源）

#### 法拉第效率
$$
\eta_F(I, T_s) = \frac{I^2}{f_1(T_s) + I^2} \cdot f_2(T_s)
$$

其中温度依赖参数：
$$
\begin{aligned}
f_1(T_s) &= 50 + 2.5(T_s - 273.15) \\
f_2(T_s) &= 0.92 - 6.25 \times 10^{-6}(T_s - 273.15)
\end{aligned}
$$

#### 单槽电压
$$
U_{cell}(I, T_s) = U_{rev} + V_{ohm}(I, T_s) + V_{act}(I, T_s)
$$

**欧姆过电位**（温度-压力耦合）：
$$
V_{ohm} = \left(r_1 + r_2 T_s + r_3 P_{sys}\right) \cdot I
$$

**活化过电位**（强非线性Arrhenius型）：
$$
V_{act} = s \cdot \ln\left(t_1 + \frac{t_2}{T_s - 273.15} + \frac{t_3}{(T_s - 273.15)^2} \cdot I + 1\right)
$$

#### 电化学产热
$$
Q_{ele} = N_{cell} \cdot I \cdot \left[U_{cell}(I, T_s) - \eta_F(I, T_s) \cdot U_{th}\right]
$$

### 2.2 热力学子系统

#### 电解槽热平衡（状态 $x_2$）
$$
\dot{x}_2 = \frac{1}{C_{s,i}}\left[Q_{ele}(I, x_2) - Q_{diss}(x_2) - Q_{flow}(u_2, x_1, x_2)\right]
$$

**散热项**（辐射-对流非线性）：
$$
Q_{diss} = \underbrace{\sigma_s(x_2 - T_{am})}_{\text{对流}} + \underbrace{\epsilon_s \sigma_b A_{stack}(x_2^4 - T_{am}^4)}_{\text{辐射}}
$$

**对流换热项**（输入-状态乘积）：
$$
Q_{flow} = c_{lye}\rho_{lye} \cdot u_2 \cdot (x_2 - x_1)
$$

#### 分离器热平衡（状态 $x_3$）
$$
\dot{x}_3 = \frac{1}{C_{sep}}\left[\underbrace{0.5 c_{lye}\rho_{lye} u_2 (x_2 - x_3)}_{\text{仿射}} - \underbrace{Q_{sep,diss}(x_3)}_{\text{非线性 } T^4}\right]
$$

#### 换热器热平衡（状态 $x_1, x_4$）

**关键非线性：对数平均温差(LMTD)**
$$
\text{LMTD}(x_1, x_3, x_4) = \begin{cases}
\frac{\Delta T_1 - \Delta T_2}{\ln(\Delta T_1/\Delta T_2)} & \Delta T_1 \neq \Delta T_2, \Delta T_1 \cdot \Delta T_2 &gt; 0 \\
\Delta T_1 & |\Delta T_1 - \Delta T_2| &lt; 10^{-5} \\
0 & \Delta T_1 \cdot \Delta T_2 \leq 0
\end{cases}
$$

其中 $\Delta T_1 = x_1 - x_4$, $\Delta T_2 = x_3 - T_{c,in}$

$$
\dot{x}_1 = \frac{1}{C_{he}}\left[c_{lye}\rho_{lye}u_2(x_3 - x_1) - k_{he}A_{he} \cdot \text{LMTD}\right]
$$

$$
\dot{x}_4 = \frac{1}{C_c}\left[c_{cw}\rho_{cw}u_3(T_{c,in} - x_4) + k_{he}A_{he} \cdot \text{LMTD}\right]
$$

### 2.3 HTO（氢氧比）子系统

#### 氢气输入分离器（状态 $x_5 \to x_6$）
$$
\dot{x}_5 = \underbrace{\dot{n}_{H_2}^{im}}_{\text{总输入}} - \underbrace{\frac{x_5 u_2}{2V_{an}}}_{\text{非仿射：}x_5 \cdot u_2}
$$

总氢气输入（电化学+溶解+扩散+对流）：
$$
\dot{n}_{H_2}^{im} = \underbrace{\frac{S_{H_2}\rho_{lye}u_2}{4}}_{\text{仿射}} + \underbrace{\frac{A_{cell}N_{cell}D_{eff}S_{H_2}P_{sys}}{\delta}}_{\text{常数}} + \underbrace{\frac{A_{cell}N_{cell}K_{eff}S_{H_2}\rho_{lye}\Delta P}{\mu_{lye}\delta}}_{\text{常数}}
$$

#### 分离器液相氢气（状态 $x_6$）
$$
\dot{x}_6 = \frac{x_5 u_2}{2V_{an}} - \frac{x_6}{\tau_{sep}}
$$

#### 分离器气相氢气（状态 $x_7$，**强非仿射**）
$$
\dot{x}_7 = \frac{x_6}{\tau_{sep}} - \underbrace{\frac{R x_3 x_7 \cdot N_{cell} u_1 \eta_F(u_1, x_2)}{4F P_{sys}V_{sep,gas}}}_{\text{三阶耦合：}x_3 \cdot x_7 \cdot u_1 \cdot \eta_F(u_1, x_2)}
$$

### 2.4 非仿射特征总结

| 非线性类型 | 数学形式 | 物理来源 | 控制难度 |
|:---|:---|:---|:---|
| **状态-输入乘积** | $x_5 u_2$, $x_7 u_1$ | 对流输运、反应消耗 | 中等 |
| **输入-输入耦合** | $u_1 \cdot \eta_F(u_1, x_2)$ | 法拉第效率 | 高 |
| **状态-状态-输入三阶** | $x_3 x_7 u_1 \eta_F$ | 气相动态 | 极高 |
| **指数/对数非线性** | $\ln(\cdot)$, $T^4$ | 活化能、辐射 | 中等 |
| **隐式代数约束** | LMTD分段定义 | 换热器设计 | 中等 |

---

## 3. HTO约束的CBF推导

### 3.1 HTO定义（摩尔比形式）

$$
\text{HTO} = \frac{n_{H_2}^{sep,gas}}{n_{O_2}^{sep,gas}} = \frac{x_7 \cdot R x_3}{P_{sys}V_{sep,gas}}
$$

**物理意义**：气相中氢气与氧气的摩尔比。安全运行要求 $\text{HTO} \leq 0.02$（即2%）。

### 3.2 控制障碍函数（CBF）

**安全集定义**：
$$
\mathcal{C} = \left\{x \in \mathbb{R}^7 : \text{HTO} \leq 0.02\right\} = \{x : h(x) \geq 0\}
$$

**CBF候选函数**：
$$
\boxed{h(x) = 0.02 - \frac{R}{P_{sys}V_{sep,gas}} x_3 x_7}
$$

或等价地：
$$
h(x) = 0.02 - K_{HTO} \cdot x_3 x_7, \quad K_{HTO} = \frac{R}{P_{sys}V_{sep,gas}} \approx 4.04 \times 10^{-4}\, \text{K}^{-1}\text{mol}^{-1}
$$

### 3.3 李导数计算

**梯度向量**：
$$
\nabla h = \left[0,\, 0,\, -K_{HTO} x_7,\, 0,\, 0,\, 0,\, -K_{HTO} x_3\right]
$$

**时间导数**：
$$
\dot{h} = \nabla h \cdot f(x,u) = -K_{HTO}(x_7 \dot{x}_3 + x_3 \dot{x}_7)
$$

### 3.4 展开的控制依赖形式

代入热力学和HTO动态：

$$
\begin{aligned}
\dot{h}(x,u) &= -K_{HTO} x_7 \cdot \frac{0.5c_{lye}\rho_{lye}(x_2-x_3)u_2 - Q_{sep,diss}(x_3)}{C_{sep}} \\
&\quad - K_{HTO} x_3 \left[\frac{x_6}{\tau_{sep}} - \frac{R x_3 x_7 N_{cell} u_1 \eta_F(u_1, x_2)}{4F P_{sys}V_{sep,gas}}\right]
\end{aligned}
$$

### 3.5 非仿射CBF条件（核心结果）

整理为**控制仿射+非仿射混合形式**：

$$
\dot{h}(x,u) = \underbrace{\gamma(x) \cdot u_1 \eta_F(u_1, x_2)}_{\text{电流控制（非仿射）}} - \underbrace{\beta(x) \cdot u_2}_{\text{流量控制（仿射）}} + \phi(x)
$$

**控制增益系数**：
$$
\gamma(x) = \frac{R^2 N_{cell} x_3^2 x_7}{4F P_{sys}^2 V_{sep,gas}^2} \gt 0 \quad (\text{当 } x_3, x_7 \gt 0)
$$

$$
\beta(x) = \frac{0.5 K_{HTO} c_{lye}\rho_{lye} x_7 (x_2 - x_3)}{C_{sep}}
$$

**纯状态漂移项**：
$$
\phi(x) = \frac{K_{HTO} x_7 Q_{sep,diss}(x_3)}{C_{sep}} - \frac{K_{HTO} x_3 x_6}{\tau_{sep}}
$$

### 3.6 标准CBF不等式

对于类$\mathcal{K}$函数 $\alpha(h) = \gamma_\alpha h$（通常 $\gamma_\alpha \gt 0$）：

$$
\sup_{u \in \mathcal{U}} \left[\gamma(x) \cdot u_1 \eta_F(u_1, x_2) - \beta(x) \cdot u_2\right] \geq -\alpha(h(x)) - \phi(x)
$$

**物理意义**：
- **增大电流 $I$**：产生更多O₂，稀释H₂，降低HTO（$\dot{h}$增大，有利安全）
- **增大碱液流量 $v_{lye}$**：若$x_2 \gt x_3$，增强对流冷却，降低分离器温度（效果取决于工况）
