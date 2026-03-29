# HTO约束的控制障碍函数(CBF)理论推导

## 1. 系统模型与状态定义

### 1.1 状态变量

单槽碱性水电解槽(AWE)系统的状态向量：

$$
x = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \\ x_4 \\ x_5 \\ x_6 \\ x_7 \end{bmatrix} = \begin{bmatrix} T_{s,in} \\ T_s \\ T_{sep} \\ T_{c,out} \\ n_{H_2}^{an} \\ n_{H_2}^{sep,liq} \\ n_{H_2}^{sep,gas} \end{bmatrix} \in \mathbb{R}^7
$$

| 状态 | 物理意义 | 单位 |
|:---|:---|:---|
| $x_1 = T_{s,in}$ | 电解槽入口碱液温度 | K |
| $x_2 = T_s$ | 电解槽（电堆）温度 | K |
| $x_3 = T_{sep}$ | 气液分离器温度 | K |
| $x_4 = T_{c,out}$ | 换热器冷却水出口温度 | K |
| $x_5 = n_{H_2}^{an}$ | 阳极氢气物质的量 | mol |
| $x_6 = n_{H_2}^{sep,liq}$ | 分离器液相氢气物质的量 | mol |
| $x_7 = n_{H_2}^{sep,gas}$ | 分离器气相氢气物质的量 | mol |

### 1.2 控制输入

$$
u = \begin{bmatrix} u_1 \\ u_2 \\ u_3 \end{bmatrix} = \begin{bmatrix} I \\ v_{lye} \\ v_c \end{bmatrix} \in \mathbb{R}^3
$$

| 输入 | 物理意义 | 单位 | 约束范围 |
|:---|:---|:---|:---|
| $u_1 = I$ | 电解电流 | A | $[0, 9360]$ |
| $u_2 = v_{lye}$ | 碱液体积流量 | $\text{m}^3/\text{s}$ | $[0, 0.1]$ |
| $u_3 = v_c$ | 冷却水体积流量 | $\text{m}^3/\text{s}$ | $[0, 1.0]$ |

## 2. HTO约束的数学定义

### 2.1 气相氢气摩尔分数

根据理想气体状态方程 $PV = nRT$，气相中氢气的摩尔分数（体积百分比）为：

$$
\text{HTO}_{pct} = \frac{n_{H_2}^{sep,gas} \cdot R \cdot T_{sep}}{P_{sys} \cdot V_{sep,gas}} \times 100\%
$$

**量纲验证**：

$$
\frac{[\text{mol}] \cdot [\text{J}/(\text{mol} \cdot \text{K})] \cdot [\text{K}]}{[\text{Pa}] \cdot [\text{m}^3]} = \frac{[\text{J}]}{[\text{J}/\text{m}^3] \cdot [\text{m}^3]} = \text{无量纲} \checkmark
$$

### 2.2 状态空间表达式

定义系统常数：

$$
K_{HTO} = \frac{R}{P_{sys} \cdot V_{sep,gas}} \approx 4.04 \times 10^{-4} \, \text{K}^{-1}\text{mol}^{-1}
$$

则HTO可表示为状态变量的函数：

$$
\text{HTO}(x) = K_{HTO} \cdot x_3 \cdot x_7 \times 100
$$

### 2.3 安全约束

安全运行要求：

$$
\text{HTO}_{pct} \leq 2\% \quad \Leftrightarrow \quad \frac{R \cdot x_3 \cdot x_7}{P_{sys} \cdot V_{sep,gas}} \leq 0.02
$$

## 3. 控制障碍函数(CBF)设计

### 3.1 安全集定义

$$
\mathcal{C} = \{x \in \mathbb{R}^7 : h_{HTO}(x) \geq 0\}
$$

### 3.2 CBF候选函数

**有量纲形式**：

$$
h_{HTO}(x) = 0.02 - K_{HTO} \cdot x_3 \cdot x_7 \times 100
$$

**无量纲形式**：

$$
h_{HTO}(x) = 1 - \frac{x_3 \cdot x_7}{0.02 / K_{HTO}} = 1 - \frac{x_3 \cdot x_7}{x_{3,ref} \cdot x_{7,ref}}
$$

其中参考值 $x_{3,ref} \cdot x_{7,ref} = 0.02 / K_{HTO} \approx 49.5$。

### 3.3 CBF梯度向量

$$
\nabla h_{HTO} = \frac{\partial h}{\partial x} = \left[0,\, 0,\, -K_{HTO} \cdot x_7 \times 100,\, 0,\, 0,\, 0,\, -K_{HTO} \cdot x_3 \times 100\right]
$$

具体分量：
- $\frac{\partial h}{\partial x_3} = -K_{HTO} \cdot x_7 \times 100$（对分离器温度的敏感度）
- $\frac{\partial h}{\partial x_7} = -K_{HTO} \cdot x_3 \times 100$（对气相氢气的敏感度）
- 其他分量为0

## 4. 李导数计算

### 4.1 CBF时间导数

根据链式法则：

$$
\dot{h}_{HTO} = \nabla h_{HTO} \cdot \dot{x} = \nabla h_{HTO} \cdot f(x,u)
$$

展开为：

$$
\dot{h}_{HTO} = -K_{HTO} \times 100 \cdot \left(x_7 \cdot \dot{x}_3 + x_3 \cdot \dot{x}_7\right)
$$

### 4.2 分离器温度动态 ($\dot{x}_3$)

$$
\dot{x}_3 = \frac{1}{C_{sep}}\left[c_{lye}\rho_{lye} u_2 (x_2 - x_3) - Q_{sep,diss}(x_3)\right]
$$

其中散热项：

$$
Q_{sep,diss}(x_3) = h_{sep} \cdot (x_3 - T_{amb})
$$

### 4.3 气相氢气动态 ($\dot{x}_7$) - 核心非仿射项

$$
\dot{x}_7 = \frac{x_6}{\tau_{sep}} - \frac{R \cdot x_3 \cdot x_7 \cdot N_{cell} \cdot u_1 \cdot \eta_F(u_1, x_2)}{4F \cdot P_{sys} \cdot V_{sep,gas}}
$$

**关键非仿射耦合**：$u_1 \cdot \eta_F(u_1, x_2)$

其中法拉第效率：

$$
\eta_F(I, T_s) = \frac{I^2}{f_1(T_s) + I^2} \cdot f_2(T_s)
$$

温度依赖参数：

$$
\begin{aligned}
f_1(T_s) &= 50 + 2.5(T_s - 273.15) \\
f_2(T_s) &= 0.92 - 6.25 \times 10^{-6}(T_s - 273.15)
\end{aligned}
$$

## 5. 完整李导数展开

### 5.1 控制仿射+非仿射混合形式

将 $\dot{x}_3$ 和 $\dot{x}_7$ 代入 $\dot{h}_{HTO}$：

$$
\dot{h}_{HTO}(x,u) = \underbrace{\gamma(x) \cdot u_1 \cdot \eta_F(u_1, x_2)}_{\text{电流控制（非仿射）}} - \underbrace{\beta(x) \cdot u_2}_{\text{流量控制（仿射）}} + \phi(x)
$$

### 5.2 控制增益系数

**电流控制增益**（始终为正）：

$$
\gamma(x) = \frac{R^2 \cdot N_{cell} \cdot x_3^2 \cdot x_7}{4F \cdot P_{sys}^2 \cdot V_{sep,gas}^2} \times 100 > 0 \quad (\text{当 } x_3, x_7 > 0)
$$

**流量控制增益**：

$$
\beta(x) = \frac{K_{HTO} \times 100 \cdot c_{lye} \cdot \rho_{lye} \cdot x_7 \cdot (x_2 - x_3)}{C_{sep}}
$$

**纯状态漂移项**：

$$
\phi(x) = \frac{K_{HTO} \times 100 \cdot x_7 \cdot Q_{sep,diss}(x_3)}{C_{sep}} - \frac{K_{HTO} \times 100 \cdot x_3 \cdot x_6}{\tau_{sep}}
$$

## 6. 标准CBF条件

### 6.1 类$\mathcal{K}$函数

取类$\mathcal{K}$函数为线性形式：

$$
\alpha(h) = \gamma_{CBF} \cdot h, \quad \gamma_{CBF} > 0
$$

其中 $\gamma_{CBF}$ 是CBF增益参数（通常取0.1-10）。

### 6.2 CBF不等式

标准CBF条件要求：

$$
\sup_{u \in \mathcal{U}} \left[\dot{h}_{HTO}(x,u)\right] \geq -\alpha(h_{HTO}(x))
$$

展开为：

$$
\gamma(x) \cdot u_1 \cdot \eta_F(u_1, x_2) - \beta(x) \cdot u_2 + \phi(x) \geq -\gamma_{CBF} \cdot h_{HTO}(x)
$$

### 6.3 非仿射优化问题

在每个时刻求解：

$$
\begin{aligned}
\min_{u \in \mathcal{U}} \quad & \|u - u_{ref}\|^2 \\
\text{s.t.} \quad & \gamma(x) \cdot u_1 \cdot \eta_F(u_1, x_2) - \beta(x) \cdot u_2 + \phi(x) + \gamma_{CBF} \cdot h_{HTO}(x) \geq 0 \\
& u_{min} \leq u \leq u_{max}
\end{aligned}
$$

## 7. 物理意义与控制策略

### 7.1 控制作用分析

| 控制动作 | 对 $\dot{h}_{HTO}$ 的影响 | 物理机制 |
|:---|:---|:---|
| **增大电流 $I$** | $\uparrow$ (正向) | 增加氧气产率，稀释气相中的氢气，降低HTO |
| **增大碱液流速 $v_{lye}$** | 视符号而定 | 若 $x_2 > x_3$，增强换热，降低分离器温度 |
| **系统漂移 $\phi(x)$** | 被动项 | 散热和液相传质的影响 |

### 7.2 关键洞见

> **非直观特性**：在HTO危险时（$h_{HTO}$ 较小），必须通过**增大电流 $I$** 来提高氧气产率，从而"稀释"气相中的氢气浓度。
>
> 这与常规直觉相反——通常人们倾向于降低电流以减少氢气产生，但这会导致氧气产生同样减少，反而使HTO升高。

### 7.3 非仿射项处理

由于 $u_1 \cdot \eta_F(u_1, x_2)$ 的非线性耦合，标准QP方法不适用。实用处理方法：

1. **直接非线性优化**：使用CasADi/IPOPT求解非凸问题
2. **序贯凸规划(SCP)**：在当前点线性化 $u_1 \cdot \eta_F$
3. **输入变换**：定义 $v = u_1 \cdot \eta_F$，反解 $u_1$

## 8. 离散时间实现

### 8.1 数值近似

在离散实现中，李导数通过中心差分近似：

$$
\nabla h \approx \frac{h(x + \epsilon) - h(x - \epsilon)}{2\epsilon}
$$

$$
\dot{h} = \nabla h \cdot f(x,u) \approx \frac{\partial h}{\partial x} \cdot \frac{dx}{dt}
$$

### 8.2 离散CBF条件

$$
\frac{h(x_{k+1}) - h(x_k)}{\Delta t} + \gamma_{CBF} \cdot h(x_k) \geq 0
$$

等价于：

$$
h(x_{k+1}) \geq (1 - \gamma_{CBF} \Delta t) \cdot h(x_k)
$$

## 9. 数值稳定性考虑

### 9.1 小量处理

当 $h_{HTO} < 0.001$（接近边界）时：
- 切换到更高增益 $\gamma_{CBF}$
- 或添加松弛变量处理不可行情况

### 9.2 温度边界

使用对数变换处理大温度范围：

$$
\tilde{h}_T = \ln(T_{max} - T_s) - \ln(T_s - T_{min})
$$

## 10. 与其他约束的联合CBF

### 10.1 多重CBF

| 约束 | CBF函数 | 物理意义 |
|:---|:---|:---|
| $h_1(x)$ | $363.15 - x_2$ | 温度上限（90°C） |
| $h_2(x)$ | $x_2 - 293.15$ | 温度下限（20°C） |
| $h_3(x)$ | $2.2 - U_{cell}(x_2, u_1)$ | 电压上限 |
| $h_4(x)$ | $6\times10^6 - P(x_2, u_1)$ | 功率上限 |
| $h_5(x)$ | $0.02 - K_{HTO} \cdot x_3 \cdot x_7$ | **HTO上限** |

### 10.2 联合CBF条件

对于多个CBF，需要同时满足：

$$
\dot{h}_i(x, u) \geq -\gamma_i \cdot h_i(x), \quad \forall i = 1, \dots, 5
$$

在优化中作为多个不等式约束处理。

## 11. 实现代码示例

```python
def _compute_h_hto(self, state):
    """
    计算HTO的CBF函数值
    
    h_HTO = HTO_max - K_HTO * T_sep * n_gas * 100
    """
    T_sep = state[2]    # x_3
    n_gas = state[6]    # x_7
    
    K_HTO = self.R / (self.P_sys * self.V_sep_gas)
    hto_pct = K_HTO * n_gas * T_sep * 100
    h_HTO = self.HTO_max - hto_pct
    
    return h_HTO

def _compute_lie_derivative_hto(self, state, control, epsilon=1e-4):
    """
    数值计算HTO CBF的李导数
    
    L_f h = ∇h · f(x,u)
    """
    n_states = len(state)
    h_current = self._compute_h_hto(state)
    f_current = self._dynamics(state, control)
    
    # 数值计算梯度 ∇h
    grad_h = np.zeros(n_states)
    for i in range(n_states):
        # 前向扰动
        x_plus = state.copy()
        x_plus[i] += epsilon
        h_plus = self._compute_h_hto(x_plus)
        
        # 后向扰动
        x_minus = state.copy()
        x_minus[i] -= epsilon
        h_minus = self._compute_h_hto(x_minus)
        
        # 中心差分
        grad_h[i] = (h_plus - h_minus) / (2 * epsilon)
    
    # 李导数
    lie_derivative = grad_h @ f_current
    
    return lie_derivative, h_current, grad_h
```

## 12. 结论

本推导建立了单槽AWE系统HTO约束的严格CBF框架：

1. **CBF函数**：$h_{HTO}(x) = 0.02 - K_{HTO} \cdot x_3 \cdot x_7 \times 100$
2. **李导数**：$\dot{h}_{HTO} = \gamma(x) \cdot I \cdot \eta_F - \beta(x) \cdot v_{lye} + \phi(x)$
3. **CBF条件**：$\dot{h}_{HTO} + \gamma_{CBF} \cdot h_{HTO} \geq 0$
4. **控制策略**：增大电流以增加氧气产率，稀释气相氢气

该框架通过显式处理非仿射耦合 $I \cdot \eta_F(I, T_s)$，确保了HTO约束的前向不变性保证。
