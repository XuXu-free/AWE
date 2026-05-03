# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AWE (Alkaline Water Electrolysis) Multi-Stack Simulation & Control framework. It simulates a 4-stack electrolysis system and implements both classical NMPC control and modern generative AI-based control policies (Diffusion Models, Flow Matching).

## Architecture

### Core Components

**Plant** (`plant/multi_stack_simulator.py`):
- Physics-based simulator modeling 4-stack AWE system
- State (13-dim): `[T_s_in, T_s1..4, T_sep, T_c_out, n_H2_an1..4, n_liq, n_gas]`
- Action (9-dim): `[I1..4, v_lye1..4, v_c]` (currents, lye flows, coolant flow)
- Dynamics: Electrochemical, thermal (heat exchanger), and HTO (gas purity) models
- Uses `scipy.integrate.solve_ivp` with RK45 for ODE integration

**Controllers** (`controller/multi_stack/`):
- `nmpc_controller.py`: CasADi/IPOPT-based nonlinear MPC. Uses warm-starting for speed.
- `model_controller.py`: Inference wrapper for trained diffusion/flow-matching policies. Samples action sequences and optimizes via cost-weighted selection.
- `model_dynamic_controller.py`: Dynamics-based variant that uses learned models for trajectory rollouts.

**Diffusion Models** (`diffusion/`):
- `ddpm.py`: DDPM scheduler for diffusion training/sampling
- `flow_matching.py`: Flow Matching scheduler (continuous normalizing flows)
- `models/`: Neural network backbones
  - `tcn.py`: Temporal Convolutional Network (recommended)
  - `mlp.py`: MLP with residual blocks
  - TCN uses Mish activations, dilated convolutions, 4 levels

### Data Pipeline

1. **Dataset Generation**: NMPC generates expert trajectories → CSV with state/action sequences
2. **Training**: Normalize data (min-max) → Train diffusion/flow model → Save to `output/multi_stack/policy/`
3. **Inference**: Load model + stats → Sample action sequences → Cost-weighted selection → Execute first action

## Common Commands

### Running Simulations

```bash
# NMPC baseline (optimization-based)
python scripts/multi_stack/run_multi_stack_test.py --controller nmpc

# Flow Matching TCN (recommended AI controller)
python scripts/multi_stack/run_multi_stack_test.py --controller model --model_type flow_tcn

# Other model variants
python scripts/multi_stack/run_multi_stack_test.py --controller model --model_type diffusion_tcn
python scripts/multi_stack/run_multi_stack_test.py --controller model --model_type flow_mlp
python scripts/multi_stack/run_multi_stack_test.py --controller model --model_type diffusion_mlp
```

### Training Policies (Multi-Stack)

```bash
# Train Flow Matching TCN (recommended)
python scripts/multi_stack/train_policy_model.py --model_type flow_tcn --epochs 100 --batch_size 64

# Train other variants
python scripts/multi_stack/train_policy_model.py --model_type diffusion_tcn --epochs 100
python scripts/multi_stack/train_policy_model.py --model_type flow_mlp --epochs 100
```

### Training Policies (Single-Stack)

```bash
# Train TCN Diffusion on single-stack dataset (GPU recommended)
uv run python scripts/single_stack/train_policy_model.py \
    --model_type diffusion_tcn \
    --data_path output/single_stack/dataset/merged_20260319_150124/nmpc_dataset_merged_20260319_150124.csv \
    --epochs 200 --batch_size 128

# Resume from existing checkpoint
uv run python scripts/single_stack/train_policy_model.py \
    --model_type diffusion_tcn \
    --data_path output/single_stack/dataset/merged_20260319_150124/nmpc_dataset_merged_20260319_150124.csv \
    --epochs 200 --batch_size 128 \
    --resume output/single_stack/policy/diffusion_tcn_policy_best.pth
```

### Dataset Generation

```bash
# Generate training data using NMPC as expert
python scripts/multi_stack/generate_dataset.py
```

## Key File Locations

- **Models**: `output/multi_stack/policy/{model_type}_policy_best.pth`
- **Normalization stats**: `output/multi_stack/diffusion_stats.npz`
- **Datasets**: `output/multi_stack/dataset/nmpc_dataset_*.csv`
- **Test results**: `output/multi_stack/test/`
- **Power profiles**: `output/power/wind/wind_power_2025-12_1min.csv`

## Critical Implementation Details

### State Vector Indices (13-dim)
```
0:      T_s_in    (HX lye outlet temp)
1-4:    T_s1..4   (Stack temperatures)
5:      T_sep     (Separator temperature)
6:      T_c_out   (Cooling water outlet)
7-10:   n_H2_an   (H2 in anode per stack)
11:     n_liq     (H2 in separator liquid)
12:     n_gas     (H2 in separator gas)
```

### Action Vector (9-dim)
```
0-3:  I1..4      (Stack currents, A)
4-7:  v_lye1..4  (Lye flow rates, m³/s)
8:    v_c        (Coolant flow rate, m³/s)
```

### Controller Interface
All controllers implement `get_action(state_vec, P_ref_vec, T_ref, last_action)`:
- `state_vec`: Current 13-dim state
- `P_ref_vec`: Horizon-length power reference array
- `T_ref`: Target temperature (K), typically 353.15 (80°C)
- `last_action`: Previous action as `[I_vec, v_lye_vec, v_c]`
- Returns: `(I_cmd, v_lye_cmd, v_c_cmd)`

### NMPC Specifics
- Uses CasADi with IPOPT solver
- Warm-starting: Previous solution shifted by one step
- Sub-stepping: 60s control dt with 0.2s simulation dt (300 sub-steps)
- Constraints: Stack temps (20-90°C), cell voltage (<2.2V), HTO (<2%), power per stack (<6MW)

### Model Controller Specifics
- Samples multiple action sequences (default: 64 samples)
- Cost function weights: `lambda_track=1.2`, `lambda_temp=0.15`, `lambda_I=0.0002`, `lambda_lye=25000`, `lambda_c=0.5`
- Cost-weighted selection: Lower cost = higher selection probability
- Only first action from horizon is executed (MPC-style receding horizon)

### Dataset Format
Expected columns in training CSV:
- State: `T_s_in`, `T_s_1..4`, `T_sep`, `T_c_out`, `n_H2_an_1..4`, `n_liq`, `n_gas`, `T_ref`
- Actions: `I_1..4`, `v_lye_1..4`, `v_c`
- Future plans: `plan_step_{k}_I_{i}`, `plan_step_{k}_v_lye_{i}`, `plan_step_{k}_v_c` for k in horizon
- Previous actions: `I_prev_1..4`, `v_lye_prev_1..4`, `v_c_prev`
- Power: `P_ref`, `P_ref_future_0..{horizon-1}`

## Dependencies

Core requirements:
- `torch` - PyTorch for neural networks (GPU version: `uv pip install torch --index-url https://download.pytorch.org/whl/cu126`)
- `numpy`, `scipy` - Numerical computing
- `casadi` - Nonlinear optimization (NMPC)
- `pandas`, `matplotlib` - Data handling and visualization
- `tqdm` - Progress bars

Package management:
- Use `uv` for fast Python package management and script execution (`uv run python ...`)
- GPU training requires CUDA-capable PyTorch; install via `uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126`

## 论文数据图表统一绘图风格（Matplotlib）

所有用于论文的 matplotlib 图表必须遵循以下统一风格，以 `paper/figures/controller_step_comparison.png` 为标杆。

### 全局配置

```python
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
```

### 画布与布局

- **单图尺寸**：单张子图建议 `figsize=(6, 4.5)`；2×3 整图建议 `figsize=(18, 10)`
- **边距**：使用 `plt.tight_layout(pad=2.0)` 或手动调整 `subplots_adjust`，确保子图间不拥挤
- **多子图编号**：子图标题使用 `(a) 功率跟随` 格式，**居中**置于子图正上方，不加粗

### 标题规范

- **总标题**（如有）：整图顶部居中，黑体/加粗，例如 `"TCN Diffusion 控制器 — 多槽AWE系统响应"`
- **子图标题**：每个 subplot 上方居中，例如 `"(a) 功率跟随"`、`"(b) 电解槽温度控制"`、`"(c) HTO安全控制"`

### 坐标轴规范

- **轴标签**：中文 + 单位，用圆括号包裹单位，例如 `功率 (MW)`、`温度 (°C)`、`时间 (min)`、`电流 (A)`、`流量 (L/s)`
- **刻度**：根据数据范围合理设置，避免过多或过少；科学计数法仅在必要时使用
- **范围**：y 轴范围应略宽于数据实际范围（上下留白约 5%~10%），但不可过大导致曲线扁平

### 网格与边框

- **网格线**：`ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)`，浅灰色虚线
- **边框**：保留四周边框（不隐藏 top/right），线宽默认或 `0.8`

### 曲线与颜色

- **主曲线线宽**：`linewidth=1.5` ~ `2.0`
- **参考线线宽**：`linewidth=1.0` ~ `1.5`，黑色虚线 `'k--'`
- **安全限线宽**：红色虚线 `'r--'`，`linewidth=1.5`
- **颜色序列**：优先使用 matplotlib 默认 tab10：`#1f77b4`（蓝）、`#ff7f0e`（橙）、`#2ca02c`（绿）、`#d62728`（红）
- **填充区域**：
  - 跟踪误差填充：浅蓝色 `fill_between(..., alpha=0.15, color='#1f77b4')`
  - 安全区域填充：浅绿色 `axvspan(..., alpha=0.08, color='green')` 或 `fill_between(..., alpha=0.1, color='green')`
  - CBF 触发区域：浅红色 `axvspan(..., alpha=0.1, color='red')`

### 图例规范

- **位置**：优先放在子图内部空白处（`loc='best'` 或手动指定），避免遮挡主曲线
- **排列**：多条曲线时可用 `ncol=2` 双列排列，节省纵向空间
- **标签**：全部使用中文，例如 `"参考功率"`、`"实际功率"`、`"槽1"`、`"槽2"`、`"安全限 (2%)"`
- **边框**：`frameon=True`，浅灰边框，`framealpha=0.9`

### 输出规范

- **保存格式**：仅输出 PNG（不保存 PDF），便于版本控制和团队协作
  ```python
  plt.savefig('paper/figures/xxx.png', dpi=600, bbox_inches='tight', facecolor='white')
  ```
- **背景**：纯白背景 `facecolor='white'`，无边距裁切 `bbox_inches='tight'`
- **脚本归档**：所有生成论文图表的脚本必须保存到 `paper/scripts/` 目录，命名格式为 `plot_<figure_name>.py`，确保图表可复现

### 子图拆分与LaTeX排版（SJTU模板标准）

多子图图表必须严格匹配 SJTU 本科毕业设计模板 (`中文毕业设计模板250928.pdf`) 中的 [A31]~[A33] 要求。子图标签 `(a)`、`(b)`、`(c)` 由 LaTeX 的 `subfigure` 环境自动生成，**不要**硬编码在 PNG 中。

#### 1. Matplotlib 脚本：提取单张子图 PNG

在保存整图后，使用 `ax.get_tightbbox()` 将每个子图裁剪为独立 PNG，供 LaTeX `\includegraphics` 单独引用：

```python
import matplotlib.pyplot as plt

# 1. 先保存整图（用于快速预览）
fig.savefig('paper/figures/xxx.png', dpi=600, bbox_inches='tight', facecolor='white')

# 2. 提取各子图为独立 PNG（供 LaTeX subfigure 使用）
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for idx, ax in enumerate(axes.flat):
    label = chr(ord('a') + idx)
    bbox = ax.get_tightbbox(renderer)
    bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
    bbox_inches = bbox_inches.expanded(1.02, 1.02)  # 留 2% 边距
    fig.savefig(f'paper/figures/xxx_{label}.png', dpi=600,
                bbox_inches=bbox_inches, facecolor='white')
plt.close()
```

> **要点**：
> - 子图 PNG 文件名为 `xxx_a.png`、`xxx_b.png`、...，与整图同名加后缀。
> - `expanded(1.02, 1.02)` 保留微小白边，避免坐标轴标签被裁切。
> - 子图内部**不加** `(a)`、`(b)` 标签文字，保持画面干净。

#### 2. LaTeX 排版：`subfigure` + 空 `\caption{}` + `\caption*{}`

在 `setup.tex` 中确保已加载 caption 居中配置（已默认添加）：

```latex
\captionsetup{justification=centering}
\captionsetup[subfigure]{justification=centering}
```

正文中使用 `subfigure` 环境排版（`sjtuthesis` 已自动加载 `subcaption` 包）：

```latex
\begin{figure}[!htbp]
  \centering
  \begin{subfigure}{0.48\textwidth}
    \centering
    \includegraphics[width=\linewidth]{figures/xxx_a.png}
    \caption{}
  \end{subfigure}
  \hfill
  \begin{subfigure}{0.48\textwidth}
    \centering
    \includegraphics[width=\linewidth]{figures/xxx_b.png}
    \caption{}
  \end{subfigure}
  % ... 更多子图 ...
  \caption{主图题：TCN Diffusion、Diffusion MLP、NMPC与MLP阶跃响应对比}
  \captionsetup{font=normalfont}
  \caption*{(a) 总功率跟踪；(b) 电解槽温度；(c) 氢氧杂质含量；(d) 电解槽电流；(e) 碱液流量；(f) 冷却水流量}
  \label{fig:xxx}
\end{figure}
```

> **排版规则**：
> - `\caption{}` 留空：只生成 `(a)`、`(b)` 编号，不输出文字，符合模板示例。
> - 主 `\caption{...}`：第一行，五号加粗居中，由 `sjtuthesis` 自动处理。
> - `\caption*{...}`：第二行，用于子图说明文字，使用 `\captionsetup{font=normalfont}` 取消加粗。
> - 子图说明格式：`(a) xxx；(b) xxx；...`，用全角分号 `；` 分隔。

### 典型子图类型速查

| 子图内容 | x 轴标签 | y 轴标签 | 参考线 | 备注 |
|---|---|---|---|---|
| 功率跟踪 | 时间 (min) | 功率 (MW) | 黑色虚线 `P_ref` | 可加浅蓝填充 |
| 温度 | 时间 (min) | 温度 (°C) | 黑色虚线 `T_ref`，红色虚线 `T_max` | — |
| HTO | 时间 (min) | HTO (%) | 红色虚线 2% | 安全区用浅绿填充 |
| 电流 | 时间 (min) | 电流 (A) 或 (kA) | 黑色虚线 `I_ref` | 注意单位统一 |
| 碱液流量 | 时间 (min) | 流量 (L/s) | — | 注意纵坐标范围 |
| 冷却剂流量 | 时间 (min) | 流量 (L/s) | 黑色虚线 `v_c_ref` | 注意纵坐标范围 |

## Development Notes

- Simulator dt is fixed at 0.2s for numerical stability
- Control dt is typically 60s (aligned with practical actuator limits)
- Horizon is typically 5 steps (5 minutes at 60s dt)
- Normalization: Conditions use dataset min-max; Actions use physical limits (I: 0-9360A, v_lye: 0-0.1, v_c: 0-1)
- GPU strongly recommended for model training (single-stack training ~30s/epoch on RTX 5070 vs minutes on CPU); inference works on CPU
- Single-stack training supports `--resume` to continue from `diffusion_tcn_policy_best.pth` checkpoint
- Current dev environment: NVIDIA GeForce RTX 5070, CUDA 13.0, PyTorch 2.10.0+cu130

## Zotero MCP Integration

This project uses Zotero as the literature source of truth via MCP (Model Context Protocol).

### Server Configuration

- **Server name**: `zotero-mcp`
- **Endpoint**: `http://127.0.0.1:23120/mcp`
- **Port**: 23120

### Setup Commands

```bash
# Add Zotero MCP server (local project scope)
claude mcp add --transport http zotero-mcp http://127.0.0.1:23120/mcp

# Add globally (all projects)
claude mcp add --transport http --scope user zotero-mcp http://127.0.0.1:23120/mcp
```

### Management Commands

```bash
claude mcp list          # List all MCP servers
claude mcp get zotero-mcp # Get server details
claude mcp remove zotero-mcp # Remove server
/mcp                     # Check status in Claude Code
```

### Available Tools

- `search_library` - Search Zotero library
- `get_item_details` - Get item metadata
- `get_item_fulltext` - Get PDF full text
- `search_fulltext` - Full-text search across papers
- `get_collections` - List Zotero collections
- `search_annotations` - Search annotations and highlights

### Prerequisites

1. Zotero desktop app must be running
2. Zotero MCP plugin/service enabled on port 23120
3. No firewall blocking localhost:23120
4. After adding, no Claude Code restart is needed
