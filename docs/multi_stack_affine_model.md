# Multi-Stack AWE 模型的非线性仿射控制形式

本文档基于 `c:\Users\admin\Desktop\sjtu\AWE\plant\multi_stack_simulator.py` 中的 `_get_derivatives` 方法，将 4 槽碱性电解水（AWE）系统的动力学模型整理为标准的非线性控制仿射形式：

$$
\dot{x} = f(x) + G(x) u_{aff}
$$

## 1. 变量定义

### 1.1 状态向量 $x \in \mathbb{R}^{13}$
与代码中的状态定义完全一致：

$$
x = \begin{bmatrix}
T_{s,in} \\
T_{s,1} \\ T_{s,2} \\ T_{s,3} \\ T_{s,4} \\
T_{sep} \\
T_{c,out} \\
n_{H_2,an,1} \\ n_{H_2,an,2} \\ n_{H_2,an,3} \\ n_{H_2,an,4} \\
n_{H_2,sep,liq} \\
n_{H_2,sep,gas}
\end{bmatrix}
$$

### 1.2 物理控制输入 $u \in \mathbb{R}^{9}$
这是实际的物理控制量（4个电流，4个碱液流速，1个冷却水流速）：

$$
u = \begin{bmatrix}
I_1 \\ I_2 \\ I_3 \\ I_4 \\
v_1 \\ v_2 \\ v_3 \\ v_4 \\
v_c
\end{bmatrix}
$$

### 1.3 仿射输入向量 $u_{aff} \in \mathbb{R}^{13}$
由于电流 $I_i$ 对产热 $Q_{ele,i}$ 和产气 $\dot{n}_{O_2,i}$ 的影响是非线性的，为了写成标准的 $f(x)+G(x)u$ 仿射形式，我们需要定义一个中间仿射输入向量。该向量由物理输入 $u$ 和状态 $x$ 计算得到：

$$
u_{aff} = \begin{bmatrix}
Q_{ele,1}(I_1, T_{s,1}) \\
Q_{ele,2}(I_2, T_{s,2}) \\
Q_{ele,3}(I_3, T_{s,3}) \\
Q_{ele,4}(I_4, T_{s,4}) \\
\dot{n}_{O_2,1}(I_1, T_{s,1}) \\
\dot{n}_{O_2,2}(I_2, T_{s,2}) \\
\dot{n}_{O_2,3}(I_3, T_{s,3}) \\
\dot{n}_{O_2,4}(I_4, T_{s,4}) \\
v_1 \\
v_2 \\
v_3 \\
v_4 \\
v_c
\end{bmatrix} \in \mathbb{R}^{13}
$$

其中 $Q_{ele,i}$ 和 $\dot{n}_{O_2,i}$ 的计算公式在第 3 节给出。

## 2. 整体仿射模型矩阵

系统动力学方程为：

$$
\dot{x} = f(x) + G(x) u_{aff}
$$

### 2.1 漂移向量 $f(x) \in \mathbb{R}^{13}$

$$
f(x) = \begin{bmatrix}
-\frac{1}{C_{he}} Q_{hx}(x) \\
-\frac{1}{C_{s,1}} Q_{diss,1}(x) \\
-\frac{1}{C_{s,2}} Q_{diss,2}(x) \\
-\frac{1}{C_{s,3}} Q_{diss,3}(x) \\
-\frac{1}{C_{s,4}} Q_{diss,4}(x) \\
-\frac{1}{C_{sep}} Q_{sep,diss}(x) \\
\frac{1}{C_c} Q_{hx}(x) \\
d_{H_2} \\
d_{H_2} \\
d_{H_2} \\
d_{H_2} \\
-\frac{1}{\tau_{sep}} n_{H_2,sep,liq} \\
\frac{1}{\tau_{sep}} n_{H_2,sep,liq}
\end{bmatrix}
$$

### 2.2 控制输入矩阵 $G(x) \in \mathbb{R}^{13 \times 13}$

$$
G(x) = \left[ \begin{array}{cccc|cccc|cccc|c}
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \frac{k_{lye}}{C_{he}}\Delta T_{sep,s\_in} & \frac{k_{lye}}{C_{he}}\Delta T_{sep,s\_in} & \frac{k_{lye}}{C_{he}}\Delta T_{sep,s\_in} & \frac{k_{lye}}{C_{he}}\Delta T_{sep,s\_in} & 0 \\
\frac{1}{C_{s,1}} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & -\frac{k_{lye}}{C_{s,1}}\Delta T_{s1,s\_in} & 0 & 0 & 0 & 0 \\
0 & \frac{1}{C_{s,2}} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & -\frac{k_{lye}}{C_{s,2}}\Delta T_{s2,s\_in} & 0 & 0 & 0 \\
0 & 0 & \frac{1}{C_{s,3}} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & -\frac{k_{lye}}{C_{s,3}}\Delta T_{s3,s\_in} & 0 & 0 \\
0 & 0 & 0 & \frac{1}{C_{s,4}} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & -\frac{k_{lye}}{C_{s,4}}\Delta T_{s4,s\_in} & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \frac{k_{lye}}{2C_{sep}}\Delta T_{s1,sep} & \frac{k_{lye}}{2C_{sep}}\Delta T_{s2,sep} & \frac{k_{lye}}{2C_{sep}}\Delta T_{s3,sep} & \frac{k_{lye}}{2C_{sep}}\Delta T_{s4,sep} & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \frac{k_{cw}}{C_c}\Delta T_{c\_in,c\_out} \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \phi_1(x) & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \phi_2(x) & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \phi_3(x) & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \phi_4(x) & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \frac{n_{H_2,an,1}}{2V_{an}} & \frac{n_{H_2,an,2}}{2V_{an}} & \frac{n_{H_2,an,3}}{2V_{an}} & \frac{n_{H_2,an,4}}{2V_{an}} & 0 \\
0 & 0 & 0 & 0 & -\kappa(x) & -\kappa(x) & -\kappa(x) & -\kappa(x) & 0 & 0 & 0 & 0 & 0
\end{array} \right]
$$

为了排版紧凑，矩阵中使用了以下简写符号：

$$
\begin{aligned}
\Delta T_{sep,s\_in} &= T_{sep} - T_{s,in} \\
\Delta T_{si,s\_in} &= T_{s,i} - T_{s,in} \\
\Delta T_{si,sep} &= T_{s,i} - T_{sep} \\
\Delta T_{c\_in,c\_out} &= T_{c,in} - T_{c,out} \\
\phi_i(x) &= \frac{S_{H_2,lye}\rho_{lye}}{4} - \frac{n_{H_2,an,i}}{2V_{an,lye}} \\
\kappa(x) &= \frac{R T_{sep} n_{H_2,sep,gas}}{P_{sys} V_{sep,gas}}
\end{aligned}
$$

## 3. 辅助方程与参数

### 3.1 物理输入到仿射输入的映射
由 `_calculate_electrochemical_properties` 和 `_calculate_hto_derivatives` 方法定义：

$$
Q_{ele,i} = N_{cell} I_i (U_{cell,i}(I_i, T_{s,i}) - \eta_{F,i}(I_i, T_{s,i}) U_{th})
$$

$$
\dot{n}_{O_2,i} = \frac{N_{cell}}{4F} I_i \eta_{F,i}(I_i, T_{s,i})
$$

其中电压 $U_{cell}$ 和法拉第效率 $\eta_F$ 是电流 $I_i$ 和温度 $T_{s,i}$ 的非线性函数。

### 3.2 散热与热交换函数
$$
\begin{aligned}
Q_{diss,i}(x) &= \sigma_s(T_{s,i}-T_{am}) + \epsilon_{stack}\sigma_b A_{stack}(T_{s,i}^4-T_{am}^4) \\
Q_{sep,diss}(x) &= \sigma_{sep}(T_{sep}-T_{am}) + \epsilon_{sep}\sigma_b A_{sep}(T_{sep}^4-T_{am}^4) \\
Q_{hx}(x) &= k_{he} A_{he} \cdot \text{LMTD}(T_{s,in}, T_{sep}, T_{c,out})
\end{aligned}
$$

### 3.3 常数定义
$$
\begin{aligned}
k_{lye} &= c_{lye} \rho_{lye} \\
k_{cw} &= c_{cw} \rho_{cw} \\
d_{H_2} &= A_{cell} N_{cell} \left( \frac{D_{eff} S_{H_2,lye} P_{sys}}{\delta} + \frac{K_{eff} S_{H_2,lye} \rho_{lye} \Delta P}{\mu_{lye} \delta} \right)
\end{aligned}
$$

## 4. HOCBF 控制器线性化实现

为了在 BarrierNet 或 QP 控制器中使用该模型，我们需要处理输入非线性。通过在每一步对物理控制量 $u$ 进行一阶泰勒展开，可以将 HOCBF 约束转化为关于 $u$ 的标准线性不等式。

### 4.1 雅可比线性化

在当前工作点 $u_0$ 处对 $u_{aff}$ 线性化：

$$
u_{aff}(u) \approx u_{aff}(u_0) + J_u (u - u_0)
$$

其中 $J_u = \frac{\partial u_{aff}}{\partial u} \in \mathbb{R}^{13 \times 9}$ 为稀疏矩阵：

$$
J_u = \text{diag}\left( \frac{\partial Q_{ele,1}}{\partial I_1}, \dots, \frac{\partial Q_{ele,4}}{\partial I_4}, \frac{\partial \dot{n}_{O_2,1}}{\partial I_1}, \dots, \frac{\partial \dot{n}_{O_2,4}}{\partial I_4}, 1, 1, 1, 1, 1 \right)
$$

注意 $J_u$ 实际上是一个分块对角阵，因为 $v_i$ 和 $v_c$ 已经是线性的（偏导为 1）。

### 4.2 关键偏导数计算公式

根据电化学模型，我们需要计算以下两项偏导数：

**1. 产热对电流的偏导** $\frac{\partial Q_{ele,i}}{\partial I_i}$：

$$
\frac{\partial Q_{ele,i}}{\partial I_i} = N_{cell} \left[ (U_{cell,i} - \eta_{F,i} U_{th}) + I_i \left( \frac{\partial U_{cell,i}}{\partial I_i} - \frac{\partial \eta_{F,i}}{\partial I_i} U_{th} \right) \right]
$$

**2. 产气对电流的偏导** $\frac{\partial \dot{n}_{O_2,i}}{\partial I_i}$：

$$
\frac{\partial \dot{n}_{O_2,i}}{\partial I_i} = \frac{N_{cell}}{4F} \left( \eta_{F,i} + I_i \frac{\partial \eta_{F,i}}{\partial I_i} \right)
$$

这些偏导数可通过数值差分或解析求导获得。

### 4.3 局部线性化后的标准形式

将线性化后的输入代入原方程，得到关于物理输入 $u$ 的局部仿射形式：

$$
\dot{x} \approx \tilde{f}(x, u_0) + \tilde{G}(x) u
$$

其中：

$$
\begin{aligned}
\tilde{f}(x, u_0) &= f(x) + G(x) (u_{aff}(u_0) - J_u u_0) \\
\tilde{G}(x) &= G(x) J_u
\end{aligned}
$$

这就是标准的控制仿射形式 $\dot{x} = \tilde{f} + \tilde{g} u$，可以直接用于计算 Lie 导数和构建 CBF 约束。
