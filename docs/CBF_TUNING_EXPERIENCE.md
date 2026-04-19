# 单槽 CBF 安全投影调试经验总结

> 记录于 2026-04-17，基于 `controller/single_stack/cbf_projection.py` 与 `scripts/single_stack/test_cbf_unified.py` 的联合调试。

---

## 1. 核心问题清单

### 1.1 Power 硬约束失效（最严重）
**现象**：`high_power` 场景下 Max Power 达到 7.979 MW，远超 6 MW 限制。  
**根因**：Power（以及 Voltage）的 CBF 函数 `h` **直接依赖于控制量 `I`**。标准 CBF 条件 `L_f h + gamma * h >= 0` 在这种情况下**不成立**，因为 Lie 导数 `L_f h = dh/dx * f(x,u)` 没有包含 `∂h/∂u` 项。 solver 即使满足 CBF 条件，也可能给出 `Power > 6 MW` 的解。  
**修复**：对控制相关硬约束（P:j=3, V:j=2），**仅直接施加 `h >= 0`**，不再附加无效的 CBF 动态条件。同时对 early-return 和可行性判断也加入 `h >= 0` 校验。

### 1.2 `actual_success` 判定 bug
**现象**：solver 报告失败后，系统仍返回 `success=True`，导致硬约束被大规模违反。  
**根因**：代码写的是 `solver_success or (...feasible...)`。  
**修复**：改为 `solver_success and control_feasible and cbf_feasible and slack_feasible and hard_feasible`。

### 1.3 Python 字节码缓存导致修复不生效
**现象**：修改 `cbf_projection.py` 后，测试结果仍显示旧行为。  
**根因**：`__pycache__` 中缓存了旧版本。  
**修复**：运行时使用 `python -B` 绕过缓存，或手动清除 `__pycache__`。

### 1.4 高温场景电流剧烈振荡
**现象**：`high_temp` 场景下电流在 9360A 与 1900A 之间大幅跳变，没有稳定工作点。  
**根因**：`v_c = 0.0` 时散热极弱，维持 90°C 仅需约 1500A。`gamma_T = 10.0` 过于激进，把电流压得过低，导致温度快速下降；温度一旦离开边界，参考电流 9360A 又变得可行，形成 bang-bang。  
**修复**：为 `high_temp` 场景单独调低 `gamma_T = 2.0`，使 CBF 条件更宽松，电流维持在约 1800–2200A 的真实热稳态附近，温度稳定在 88–89°C。

---

## 2. 关键物理发现

### 2.1 6 MW 功率上限在真实模型下的实际电流限制
使用 `sim._calculate_electrochemical_properties()` 计算（含对数活化过电位）：

| 温度 | 最大允许电流 (P ≤ 6 MW) |
|------|------------------------|
| 40°C | ~7421 A |
| 50°C | ~7475 A |
| 60°C | ~7523 A |
| 70°C | ~7568 A |
| 80°C | ~7613 A |
| 85°C | ~7636 A |

**结论**：在 85°C 以上时，`I = 9360 A` 的参考电流对应的实际功率约为 **7.74 MW**，因此功率硬约束必须有效才能防止越限。

### 2.2 高温场景（v_c = 0.0）的热稳态
在无冷却水、v_lye = 0.03 m³/s 的条件下，维持 `T_s = 90°C` 所需的稳态电流仅约 **1300–1500 A**。这意味着：
- 若参考电流为 9360 A，CBF 必须把电流压到约 1/6；
- 单一温度硬约束无法提供唯一的稳定控制点（高功率场景因 **P hard + T hard** 双重约束才有了 7480A/51°C 的稳定交点）。

---

## 3. 最终推荐参数（所有场景统一）

所有场景使用**同一套**CBF 调参，仅在测试时改变参考控制量 `u_ref`：

```python
gamma_vec = [3.0, 2.0, 100.0, 100.0, 5.0]
rho_vec   = [50000, 50000, 50000, 50000, 10000]
h_margin  = [1.0, 0.005, 0.0, 0.0, 0.0]       # HTO 余量 0.5% -> 有效边界 1.5%
soft_mask = [False, True, True, False, True]   # T hard, P hard, 其余 soft
lambda_u  = [500.0, 200.0, 1000.0]             # [I, v_lye, v_c] 分别调节
```

**HTO 调参说明**：
- `gamma_HTO = 2.0`（原为 1.0）：使 CBF 条件在 HTO 远离边界时更容易满足，避免系统过早限制电流。
- `h_margin_HTO = 0.005`（0.5% 余量）：有效边界为 1.5%，既保留了安全缓冲，又比原来的 1% 余量更宽松。
- `gamma_T = 3.0`（原为 2.0）：温度约束更快反应，提前压低电流，从而在 HTO 接近边界前间接保护它。

**说明**：`gamma_T = 2.0` 被统一用于全部场景。对 `high_power`/`low_power_hto` 而言，功率/电流本身远离温度边界，较小的 `gamma_T` 不影响安全性；对 `high_temp` 而言，`gamma_T = 2.0` 避免了 `gamma_T = 10.0` 造成的电流 overshoot 与 bang-bang 振荡，使系统稳定在热稳态附近（约 1800–2200A），温度保持在 88–89°C，HTO 低于 2%。

### 3.1 各场景 active_mask 与参考控制
| 场景 | `u_ref` (I, v_lye, v_c) | active_mask | 说明 |
|------|------------------------|-------------|------|
| `high_power` | `[9360, 0.03, 1.0]` | `[True, False, False, True, True]` | T hard, P hard, Tmin soft |
| `high_temp` | `[9360, 0.03, 0.0]` | `[True, True, False, True, False]` | T hard, P hard, HTO soft |
| `low_power_hto` | `[600, 0.04, 0.01]` | `[True, True, False, False, True]` | T hard, HTO soft, Tmin soft |

---

## 4. 测试结果摘要

| 场景 | Max Power | Max Temp | Max HTO | 电流行为 |
|------|-----------|----------|---------|----------|
| `high_power` | **6.000 MW** | 84.85 °C | 0.989 % | 稳定在 ~7430A |
| `high_temp` | **5.036 MW** | **88.91 °C** | **1.961 %** | 初始 transient 后稳定在 ~1800–2200A |
| `low_power_hto` | 2.886 MW | 84.85 °C | 1.703 % | 稳定在 ~3500–4000A |

---

## 5. CBF 保守性评估

### 5.1 评估方法

为验证 CBF 在非边界条件下是否存在过度反应（保守性），创建了系统化的网格扫描分析：
- **扫描范围**：I ∈ [0, 2500, 5000, 7500, 9360] A，v_lye ∈ [0.01, 0.03, 0.05, 0.10] m³/s，v_c ∈ [0.0, 0.3, 0.6, 1.0]
- **评估标准**：当 `u_ref` 处的原始 CBF 条件 `cbf_ref >= 0` 时，若优化器仍偏离 `u_ref`，则判定为保守
- **关键控制**：设置 `u_last = u_ref` 以消除平滑性惩罚的干扰

### 5.2 评估结果

| 评估配置 | 平均保守率 | 保守点数/总数 |
|----------|-----------|---------------|
| 初版（使用优化后 cbf + u_last 连续） | 35.26% | 33/80 |
| 修正版（使用原始 cbf_ref + u_last=u_ref） | **0.00%** | **0/80** |

**结论**：当前 CBF 调参**不存在保守性问题**。当参考控制量满足所有 CBF 条件时，优化器不会进行不必要的控制调整。

### 5.3 初版保守性假象的根因

初版分析中观察到 35% 的"保守率"，实际上是由以下两个因素共同造成的**评估方法缺陷**，而非 CBF 本身保守：

1. **使用了优化后的 `cbf` 而非原始 `cbf_ref`**  
   优化后的 `cbf >= 0` 是因为优化器通过调整控制量**修复了**原本不满足的约束。此时 `cbf_ref`（`u_ref` 处的值）实际上 `< 0`，所以优化是必要的。

2. **`u_last ≠ u_ref` 引入的平滑性惩罚**  
   在连续控制框架中，`delta_cost = lambda_u * (u_var - u_last)^2` 会推动优化器选择接近上一时刻控制量的解，即使 `u_ref` 本身是安全的。这是 MPC 的正常行为，不是保守性。

---

## 6. 调试建议（后续维护）

1. **控制相关约束必须直接施加 `h >= 0`**  
   对于 Power、Voltage 这类 `h(x,u)` 显式含 `u` 的约束，标准 CBF 动态条件 `L_f h + gamma*h >= 0` **不足够**，必须补充直接约束。

2. **避免盲目增大 `lambda_u_scale`**  
   大 `lambda_u` 虽能抑制控制跳变，但会让系统对所有软约束（包括 HTO）响应迟钝。若某场景需要平滑，优先调整该场景对应的 `gamma` 或 `active_mask`，而不是全局加大 `lambda_u`。

3. **不同场景的物理稳态不同，参数可适度差异化**  
   `gamma_T` 对高功率场景影响显著（过小会导致功率利用率下降），但对低功率场景几乎无影响。因此把 `gamma_T` 作为场景级可调参数是合理的。

4. **始终使用 `python -B` 或在修改核心模块后清除缓存**  
   CasADi 函数对象的编译结果会被 Python 缓存，Stale cache 是导致"修复无效"的常见陷阱。

---

## 7. 相关文件

- `controller/single_stack/cbf_projection.py` — CBF 投影核心逻辑
- `scripts/single_stack/test_cbf_unified.py` — 统一测试与可视化脚本
- `scripts/single_stack/test_cbf_grid_scan.py` — 网格扫描评估脚本
- `scripts/single_stack/test_cbf_conservatism.py` — 保守性专项分析脚本
- `plant/single_stack_simulator.py` — 真实电化学/热/HTO 模型
