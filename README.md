# AWE Multi-Stack Electrolysis Simulation & Control

This project implements a high-fidelity simulation and advanced control framework for a multi-stack Alkaline Water Electrolysis (AWE) system. It features both classical optimization-based control (NMPC) and modern generative AI-based control policies (Diffusion Models, Flow Matching).

## Features

- **Multi-Stack Simulator**: Detailed physics-based modeling of 4-stack AWE systems, including:
  - Electrochemical dynamics (voltage, current, efficiency)
  - Thermal dynamics (stack temperatures, cooling loops)
  - Gas purity (HTO - Hydrogen in Oxygen)
  - Balance of Plant (BoP) components

- **Advanced Controllers**:
  - **NMPC (Nonlinear Model Predictive Control)**: Optimization-based control for precise power tracking and temperature regulation.
  - **Generative AI Policies**:
    - **Diffusion Models**: Denoising Diffusion Probabilistic Models (DDPM) with MLP and TCN backbones.
    - **Flow Matching**: Continuous Normalizing Flows for efficient and high-quality control generation.
  - **Dynamic Controllers**: Model-based control leveraging learned system dynamics.

## Directory Structure

```
AWE/
├── controller/                 # Control algorithms
│   ├── multi_stack/
│   │   ├── nmpc_controller.py          # NMPC implementation
│   │   ├── model_controller.py         # Inference for AI policies (Diffusion/Flow Matching)
│   │   └── model_dynamic_controller.py # Dynamics-based control
├── diffusion/                  # Generative model architectures
│   ├── models/                 # Neural network definitions (MLP, TCN)
│   ├── ddpm.py                 # Diffusion model logic
│   └── flow_matching.py        # Flow Matching logic
├── plant/                      # System simulation models
│   ├── multi_stack_simulator.py # Core electrolyzer plant model
│   └── thermal_systems.py      # Thermal component models
├── scripts/                    # Execution scripts
│   └── multi_stack/
│       ├── run_multi_stack_test.py # Main testing script
│       ├── train_policy_model.py   # Training script for policies
│       └── generate_dataset.py     # Data generation script
└── output/                     # Results and model checkpoints
```

## Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install torch numpy scipy matplotlib tqdm
```

### Running Simulations

To run a simulation test with a specific controller:

1.  **NMPC Controller** (Baseline):
    ```bash
    python scripts/multi_stack/run_multi_stack_test.py --controller nmpc
    ```

2.  **Flow Matching TCN Policy** (Recommended AI Controller):
    ```bash
    python scripts/multi_stack/run_multi_stack_test.py --controller model --model_type flow_tcn
    ```

3.  **Other Model Types**:
    Supported `model_type` arguments:
    - `diffusion_mlp`: Diffusion model with MLP backbone
    - `diffusion_tcn`: Diffusion model with Temporal Convolutional Network backbone
    - `flow_mlp`: Flow Matching with MLP backbone
    - `flow_tcn`: Flow Matching with TCN backbone

### Training Policies

To train a new control policy model:

```bash
# Train a Flow Matching TCN model
python scripts/multi_stack/train_policy_model.py --model_type flow_tcn --epochs 100 --batch_size 64
```

### Generating Data

To generate a new dataset for training (using NMPC as the expert):

```bash
python scripts/multi_stack/generate_dataset.py
```

## Results

Simulation results, including CSV data logs and performance plots, are automatically saved in the `output/multi_stack/test/` directory.

## License

[MIT License](LICENSE)
