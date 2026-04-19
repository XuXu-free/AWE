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
# Train Diffusion TCN on single-stack dataset (GPU recommended)
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

## Development Notes

- Simulator dt is fixed at 0.2s for numerical stability
- Control dt is typically 60s (aligned with practical actuator limits)
- Horizon is typically 5 steps (5 minutes at 60s dt)
- Normalization: Conditions use dataset min-max; Actions use physical limits (I: 0-9360A, v_lye: 0-0.1, v_c: 0-1)
- GPU strongly recommended for model training (single-stack training ~30s/epoch on RTX 5070 vs minutes on CPU); inference works on CPU
- Single-stack training supports `--resume` to continue from `diffusion_tcn_policy_best.pth` checkpoint
- Current dev environment: NVIDIA GeForce RTX 5070, CUDA 13.0, PyTorch 2.10.0+cu130
