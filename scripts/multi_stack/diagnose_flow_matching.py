import os
import sys
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from diffusion.models import DiffusionTCN, FlowMatchingTCN, FlowMatchingMLP, DiffusionMLP
from diffusion.ddpm import DDPMScheduler
from diffusion.flow_matching import FlowMatchingScheduler

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
stats_path = os.path.join(project_root, 'output', 'multi_stack', 'diffusion_stats.npz')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load stats
stats = np.load(stats_path)
cond_min = torch.FloatTensor(stats['cond_min']).to(device)
cond_max = torch.FloatTensor(stats['cond_max']).to(device)
action_min = torch.FloatTensor(stats['action_min']).to(device)
action_max = torch.FloatTensor(stats['action_max']).to(device)
action_diff = action_max - action_min
action_diff[action_diff < 1e-6] = 1.0

# Test condition: use normalized range [0, 1] as expected by the model
obs_dim = 28
horizon = 5
action_dim = 9
num_samples = 128

# Random condition IN NORMALIZED SPACE [0, 1] (as seen during training)
cond_norm = torch.rand(1, obs_dim).to(device) * 0.4 + 0.3  # [0.3, 0.7] safe range
cond_batch = cond_norm.repeat(num_samples, 1)

print("=" * 60)
print("Diagnosing Flow Matching vs Diffusion Sampling")
print("Condition range: [0.3, 0.7] in normalized space")
print("=" * 60)

for model_name, ModelClass, SchedulerClass, model_type, extra_kwargs in [
    ('diffusion_tcn', DiffusionTCN, DDPMScheduler, 'diffusion_tcn', {'output_dim': action_dim, 'cond_dim': obs_dim, 'output_num': horizon, 'levels': 4}),
    ('diffusion_mlp', DiffusionMLP, DDPMScheduler, 'diffusion_mlp', {'action_dim': action_dim, 'obs_dim': obs_dim, 'horizon': horizon}),
    ('flow_tcn', FlowMatchingTCN, FlowMatchingScheduler, 'flow_tcn', {'action_dim': action_dim, 'obs_dim': obs_dim, 'horizon': horizon}),
    ('flow_mlp', FlowMatchingMLP, FlowMatchingScheduler, 'flow_mlp', {'action_dim': action_dim, 'obs_dim': obs_dim, 'horizon': horizon}),
]:
    print(f"\n--- {model_name} ---")
    model_path = os.path.join(project_root, 'output', 'multi_stack', 'policy', f'{model_name}_policy_best.pth')
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        continue

    if 'tcn' in model_name and 'diffusion' in model_name:
        model = ModelClass(**extra_kwargs).to(device)
    else:
        model = ModelClass(**extra_kwargs).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    scheduler = SchedulerClass(device=device)

    with torch.no_grad():
        if isinstance(scheduler, DDPMScheduler):
            samples = scheduler.sample(model, cond_batch, (num_samples, action_dim, horizon))
        else:
            samples = scheduler.sample(model, cond_batch, (num_samples, action_dim, horizon), steps=50, noise_scale=0.0)

    print(f"  Sample norm range: [{samples.min():.3f}, {samples.max():.3f}]")
    print(f"  Sample norm mean: {samples.mean():.3f}, std: {samples.std():.3f}")
    print(f"  Fraction outside [-1, 1]: {(torch.abs(samples) > 1).float().mean() * 100:.1f}%")

    # Denormalize
    action_diff_b = action_diff.view(1, -1, 1)
    action_min_b = action_min.view(1, -1, 1)
    actions = ((samples + 1) / 2) * action_diff_b + action_min_b

    # Clamp as in controller
    I_min, I_max = 0.0, 7500.0
    v_lye_min, v_lye_max = 0.01, 0.1
    v_c_min, v_c_max = 0.0, 1.0
    actions[:, 0:4, :] = torch.clamp(actions[:, 0:4, :], I_min, I_max)
    actions[:, 4:8, :] = torch.clamp(actions[:, 4:8, :], v_lye_min, v_lye_max)
    actions[:, 8, :] = torch.clamp(actions[:, 8, :], v_c_min, v_c_max)

    print(f"  Action I range (after clamp): [{actions[:, 0:4, :].min():.1f}, {actions[:, 0:4, :].max():.1f}]")
    print(f"  Action v_lye range (after clamp): [{actions[:, 4:8, :].min():.4f}, {actions[:, 4:8, :].max():.4f}]")
    print(f"  Action v_c range (after clamp): [{actions[:, 8, :].min():.4f}, {actions[:, 8, :].max():.4f}]")

    # Check first step actions
    first_step = actions[:, :, 0]
    print(f"  First-step I mean: {first_step[:, 0:4].mean():.1f}, std: {first_step[:, 0:4].std():.1f}")
    print(f"  First-step v_lye mean: {first_step[:, 4:8].mean():.4f}, std: {first_step[:, 4:8].std():.4f}")
    print(f"  First-step v_c mean: {first_step[:, 8].mean():.4f}, std: {first_step[:, 8].std():.4f}")
    print(f"  Fraction of clamped I values: {(first_step[:, 0:4] == I_min).float().mean() * 100:.1f}% at min, {(first_step[:, 0:4] == I_max).float().mean() * 100:.1f}% at max")
