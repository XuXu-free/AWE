# 多槽碱性水电解 (AWE) 系统 Control Barrier Function (CBF) 实现文档

本文档详细介绍了多槽 AWE 系统中 Control Barrier Function (CBF) 的实现细节。该实现旨在确保系统在运行过程中满足各项安全约束，包括温度、电压、功率和氢氧混合浓度 (HTO) 等限制。

## 1. 系统状态与控制输入

### 1.1 状态变量 (State Variables)
系统状态 $x \in \mathbb{R}^{13}$ 包含以下分量：
- $T_{s,in}$: 碱液入口温度 (1)
- $T_{s,1}, \dots, T_{s,4}$: 4个电解槽的堆栈温度 (4)
- $T_{sep}$: 气液分离器温度 (1)
- $T_{c,out}$: 冷却水出口温度 (1)
- $n_{H2,an,1}, \dots, n_{H2,an,4}$: 4个阳极室中的溶解氢气量 (4)
- $n_{liq}$: 分离器液相摩尔数 (1)
- $n_{gas}$: 分离器气相摩尔数 (1)

### 1.2 控制输入 (Control Inputs)
控制输入 $u \in \mathbb{R}^{9}$ 包含：
- $I_1, \dots, I_4$: 4个电解槽的电流 (4)
- $v_{lye,1}, \dots, v_{lye,4}$: 4个电解槽的碱液流量 (4)
- $v_c$: 冷却水流量 (1)

## 2. CBF 一般形式

### 2.1 相对阶为 0 的约束 (Relative Degree 0)
对于直接依赖于控制输入 $u$ 的约束 $h(x, u) \ge 0$，我们采用泰勒展开进行线性化处理，转化为 QP (Quadratic Programming) 约束形式：
$$
h(x, u) \approx h(x, u_0) + \nabla_u h(x, u_0) \cdot (u - u_0) \ge 0
$$
整理为 $A_{cbf} u \le b_{cbf}$ 形式：
$$
-\nabla_u h(x, u_0) \cdot u \le h(x, u_0) - \nabla_u h(x, u_0) \cdot u_0
$$

### 2.2 相对阶为 1 的约束 (Relative Degree 1)
对于依赖于状态变化率 $\dot{x}$ (即 $\dot{h}$ 显式依赖于 $u$) 的约束 $h(x) \ge 0$，根据 CBF 理论，需满足：
$$
\dot{h}(x, u) \ge -\gamma h(x)
$$
其中 $\gamma > 0$ 为衰减系数。

使用李导数 (Lie Derivative) 形式表示：
$$
\dot{h}(x, u) = L_f h(x) + L_g h(x) u
$$
其中：
- $L_f h(x) = \nabla_x h(x) \cdot f(x)$ 是沿系统漂移项 $f(x)$ 的李导数。
- $L_g h(x) = \nabla_x h(x) \cdot g(x)$ 是沿控制输入矩阵 $g(x)$ 的李导数。

CBF 约束条件变为：
$$
L_f h(x) + L_g h(x) u \ge -\gamma h(x)
$$
整理为 $A_{cbf} u \le b_{cbf}$ 形式：
$$
-L_g h(x) u \le \gamma h(x) + L_f h(x)
$$

在实际实现中，由于系统关于控制输入 $u$ 是非线性的（非仿射系统），我们通过一阶泰勒展开来近似 $L_g h(x)$ 和 $L_f h(x)$。
$$
\dot{h}(x, u) \approx \dot{h}(x, u_0) + \nabla_u \dot{h}(x, u_0) \cdot (u - u_0)
$$
整理为 $A + B u$ 的形式，并对应到李导数符号：
$$
L_g h(x) \approx \nabla_u \dot{h}(x, u_0)
$$
$$
L_f h(x) \approx \dot{h}(x, u_0) - \nabla_u \dot{h}(x, u_0) u_0
$$

## 3. 具体约束实现

### 3.1 堆栈温度约束 (Stack Temperature)

**约束条件**：$T_{min} \le T_{s,i} \le T_{max}$

电解槽温度的动力学方程为：
$$
\dot{T}_{s,i} = \frac{1}{C_{s,i}} \left( Q_{gen,i}(I_i) - Q_{diss,i} - c_{lye} \rho_{lye} v_{lye,i} (T_{s,i} - T_{s,in}) \right)
$$
控制输入 $I_i$（影响产热 $Q_{gen,i}$）和 $v_{lye,i}$（影响散热）在 $\dot{T}_{s,i}$ 中显式出现，这是一个**相对阶为 1** 的约束。
为了在 QP 中使用，我们对 $\dot{T}_{s,i}$ 关于控制输入 $u$ 提取偏导数：
- $\frac{\partial \dot{T}_{s,i}}{\partial I_i} = \frac{1}{C_{s,i}} \frac{\partial Q_{gen,i}}{\partial I_i}$
  其中产热功率 $Q_{gen,i}$ 对电流的偏导数为：
  $$
  \frac{\partial Q_{gen,i}}{\partial I_i} = N_{cell} \left( (U_{cell,i} - \eta_F U_{th}) + I_i \left( \frac{\partial U_{cell,i}}{\partial I_i} - \frac{\partial \eta_F}{\partial I_i} U_{th} \right) \right)
  $$
  具体的偏导数项如下：
  1. 电压偏导数 $\frac{\partial U_{cell,i}}{\partial I_i}$：
     $$
     \frac{\partial U_{cell,i}}{\partial I_i} = R_{ohm} + s \frac{K_{act}}{K_{act} I_i + 1}
     $$
     其中 $K_{act} = t_1 + t_2/T_C + t_3/T_C^2$，$R_{ohm} = r_1 + r_2 T_{s,i} + r_3 P_{sys}$。
  2. 法拉第效率偏导数 $\frac{\partial \eta_F}{\partial I_i}$：
     $$
     \frac{\partial \eta_F}{\partial I_i} = f_2 \frac{2 I_i f_1}{(f_1 + I_i^2)^2}
     $$
     其中 $f_1 = 50 + 2.5 T_C$，$f_2 = 0.92 - 6.25 \times 10^{-6} T_C$。

- $\frac{\partial \dot{T}_{s,i}}{\partial v_{lye,i}} = - \frac{1}{C_{s,i}} c_{lye} \rho_{lye} (T_{s,i} - T_{s,in})$

#### 3.1.1 温度上限 ($T_{s,i} \le T_{max}$)
- Barrier Function: $h_1(x) = T_{max} - T_{s,i} \ge 0$
- 导数: $\dot{h}_1 = -\dot{T}_{s,i}$
- 李导数项:
  - $L_g h_1(x) = -\nabla_u \dot{T}_{s,i}$
  - $L_f h_1(x) = -\dot{T}_{s,i}(x, u_0) - L_g h_1(x) u_0$
- QP 形式: $-L_g h_1(x) u \le \gamma h_1(x) + L_f h_1(x)$

#### 3.1.2 温度下限 ($T_{s,i} \ge T_{min}$)
- Barrier Function: $h_{1b}(x) = T_{s,i} - T_{min} \ge 0$
- 导数: $\dot{h}_{1b} = \dot{T}_{s,i}$
- 李导数项:
  - $L_g h_{1b}(x) = \nabla_u \dot{T}_{s,i}$
  - $L_f h_{1b}(x) = \dot{T}_{s,i}(x, u_0) - L_g h_{1b}(x) u_0$
- QP 形式: $-L_g h_{1b}(x) u \le \gamma h_{1b}(x) + L_f h_{1b}(x)$

### 3.2 电压约束 (Voltage)

**约束条件**：$U_{min} \le U_{cell,i}(I_i, T_{s,i}) \le U_{max}$
这是相对阶为 0 的约束，因为 $U_{cell}$ 直接是非线性函数 $g(I, T)$。

#### 3.2.1 电压上限 ($U_{cell,i} \le U_{max}$)
- Barrier Function: $h_3(u, x) = U_{max} - U_{cell,i}(I_i, T_{s,i}) \ge 0$
- 梯度 $\nabla_u h_3$：
  由于 $U_{cell,i}$ 只依赖于 $I_i$，因此梯度向量中只有第 $i$ 个分量非零。
  $$
  \frac{\partial h_3}{\partial I_i} = -\frac{\partial U_{cell,i}}{\partial I_i} = -\left( R_{ohm} + s \frac{K_{act}}{K_{act} I_i + 1} \right)
  $$
- 线性化形式: $-\nabla_u h_3 \cdot u \le h_3(u_0) - \nabla_u h_3 \cdot u_0$

#### 3.2.2 电压下限 ($U_{cell,i} \ge U_{min}$)
- Barrier Function: $h_4(u, x) = U_{cell,i}(I_i, T_{s,i}) - U_{min} \ge 0$
- 梯度 $\nabla_u h_4$：
  $$
  \frac{\partial h_4}{\partial I_i} = \frac{\partial U_{cell,i}}{\partial I_i} = R_{ohm} + s \frac{K_{act}}{K_{act} I_i + 1}
  $$
- 线性化形式: $-\nabla_u h_4 \cdot u \le h_4(u_0) - \nabla_u h_4 \cdot u_0$

### 3.3 功率约束 (Power)

**约束条件**：$P_{min} \le P_{stack,i} \le P_{max}$
其中 $P_{stack,i} = N_{cell} \cdot U_{cell,i} \cdot I_i$。这也是相对阶为 0 的约束。

#### 3.3.1 功率上限 ($P_{stack,i} \le P_{max}$)
- Barrier Function: $h_5(u, x) = P_{max} - P_{stack,i} \ge 0$
- 梯度 $\nabla_u h_5$：
  $$
  \frac{\partial h_5}{\partial I_i} = -\frac{\partial P_{stack,i}}{\partial I_i} = -N_{cell} \left( U_{cell,i} + I_i \frac{\partial U_{cell,i}}{\partial I_i} \right)
  $$
- 线性化形式: $-\nabla_u h_5 \cdot u \le h_5(u_0) - \nabla_u h_5 \cdot u_0$

#### 3.3.2 功率下限 ($P_{stack,i} \ge P_{min}$)
- Barrier Function: $h_{5b}(u, x) = P_{stack,i} - P_{min} \ge 0$
- 梯度 $\nabla_u h_{5b}$：
  $$
  \frac{\partial h_{5b}}{\partial I_i} = \frac{\partial P_{stack,i}}{\partial I_i} = N_{cell} \left( U_{cell,i} + I_i \frac{\partial U_{cell,i}}{\partial I_i} \right)
  $$
- 线性化形式: $-\nabla_u h_{5b} \cdot u \le h_{5b}(u_0) - \nabla_u h_{5b} \cdot u_0$

### 3.4 氢氧混合浓度 (HTO) 约束

**约束条件**：$HTO_{min} \le HTO \le HTO_{max}$

精确的 HTO 浓度公式为：
$$
HTO(x) = \frac{n_{gas} R T_{sep}}{P_{sys} V_{sep\_gas}} \times 100
$$
令常数 $c_{HTO} = \frac{100 R}{P_{sys} V_{sep\_gas}}$，则 $HTO(x) = c_{HTO} n_{gas} T_{sep}$。

对其求导，根据链式法则：
$$
\dot{HTO} = c_{HTO} (\dot{n}_{gas} T_{sep} + n_{gas} \dot{T}_{sep})
$$
由于 $\dot{n}_{gas}$ 受电流 $I_i$ 控制（影响产气率），而 $\dot{T}_{sep}$ 受碱液流量 $v_{lye,i}$ 控制，因此 $\dot{HTO}$ 显式包含控制输入 $u$，这是一个**相对阶为 1** 的约束。
具体动力学方程如下：
1. 分离器气相摩尔数变化率 $\dot{n}_{gas}$：
   $$
   \dot{n}_{gas} = \frac{1}{\tau_{sep}} n_{liq} - \kappa_x \sum_{i=1}^4 \dot{n}_{O_2,i}(I_i)
   $$
   其中 $\kappa_x$ 是与压力、体积、温度有关的系数。
   氧气生成率 $\dot{n}_{O_2,i}$ 为：
   $$
   \dot{n}_{O_2,i} = \frac{N_{cell}}{4F} I_i \eta_F(I_i, T_{s,i})
   $$
   其对电流 $I_i$ 的导数为：
   $$
   \frac{\partial \dot{n}_{O_2,i}}{\partial I_i} = \frac{N_{cell}}{4F} \left( \eta_F + I_i \frac{\partial \eta_F}{\partial I_i} \right)
   $$

2. 分离器温度变化率 $\dot{T}_{sep}$：
   $$
   \dot{T}_{sep} = \frac{1}{C_{sep}} \left( 0.5 c_{lye} \rho_{lye} \sum_{i=1}^4 v_{lye,i} (T_{s,i} - T_{sep}) - Q_{sep\_diss} \right)
   $$
   其中 $v_{lye,i}$ 是各个电解槽的碱液流量控制输入。

为了在 QP 中使用，我们对 $\dot{HTO}$ 关于控制输入 $u$ 提取偏导数：

- 对电流 $I_i$ 的偏导：
  $$
  \frac{\partial \dot{HTO}}{\partial I_i} = c_{HTO} T_{sep} \frac{\partial \dot{n}_{gas}}{\partial I_i} = c_{HTO} T_{sep} \left( -\kappa_x \frac{\partial \dot{n}_{O_2,i}}{\partial I_i} \right)
  $$
  代入 $\frac{\partial \dot{n}_{O_2,i}}{\partial I_i}$：
  $$
  \frac{\partial \dot{HTO}}{\partial I_i} = -c_{HTO} T_{sep} \kappa_x \frac{N_{cell}}{4F} \left( \eta_F + I_i \frac{\partial \eta_F}{\partial I_i} \right)
  $$

- 对碱液流量 $v_{lye,i}$ 的偏导：
  $$
  \frac{\partial \dot{HTO}}{\partial v_{lye,i}} = c_{HTO} n_{gas} \frac{\partial \dot{T}_{sep}}{\partial v_{lye,i}}
  $$
  代入 $\dot{T}_{sep}$ 对 $v_{lye,i}$ 的导数：
  $$
  \frac{\partial \dot{HTO}}{\partial v_{lye,i}} = c_{HTO} n_{gas} \left( \frac{0.5 c_{lye} \rho_{lye} (T_{s,i} - T_{sep})}{C_{sep}} \right)
  $$

#### 3.4.1 HTO 上限 ($HTO \le HTO_{max}$)
- Barrier Function: $h_2(x) = HTO_{max} - HTO(x) \ge 0$
- 导数: $\dot{h}_2 = -\dot{HTO}$
- 李导数项:
  - $L_g h_2(x) = -\nabla_u \dot{HTO}$
  - $L_f h_2(x) = -\dot{HTO}(x, u_0) - L_g h_2(x) u_0$
- QP 形式: $-L_g h_2(x) u \le \gamma h_2(x) + L_f h_2(x)$

#### 3.4.2 HTO 下限 ($HTO \ge HTO_{min}$)
- Barrier Function: $h_6(x) = HTO(x) - HTO_{min} \ge 0$
- 导数: $\dot{h}_6 = \dot{HTO}$
- 李导数项:
  - $L_g h_6(x) = \nabla_u \dot{HTO}$
  - $L_f h_6(x) = \dot{HTO}(x, u_0) - L_g h_6(x) u_0$
- QP 形式: $-L_g h_6(x) u \le \gamma h_6(x) + L_f h_6(x)$

## 4. 求解与执行

所有上述约束最终被整合为一个统一的线性不等式组 $G u \le H$。
在每个控制周期，系统会求解以下 QP 问题：
$$
\min_{u} || u - u_{ref} ||^2
$$
$$
s.t. \quad G u \le H
$$
$$
u_{min} \le u \le u_{max}
$$
从而得到满足安全约束的最优控制输入 $u^*$。
