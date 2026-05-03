"""
参考 run_20260426_165250 设置，运行 128 Serial vs 1024 JIT Batch 阶跃对比
设置: duration=14400, step_time=7200, P_initial=10MW, P_final=20MW, warmup=none
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from scripts.multi_stack.run_step_response_test import run_step_response

# Shared settings matching run_20260426_165250
duration = 14400
step_time = 7200
P_initial = 10.0e6
P_final = 20.0e6
warmup_controller = 'none'
base_subdir = 'run_128serial_vs_1024batch'
model_type = 'diffusion_tcn_l3'

# 1) 128 Serial
print("\n" + "=" * 70)
print("Running 128 Serial...")
print("=" * 70)

# Temporarily patch controller defaults
import controller.multi_stack.model_controller as mc_module
original_init = mc_module.MultiStackModelController.__init__

def patched_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self.num_candidates = 128
    self.use_batch_rollout = False

mc_module.MultiStackModelController.__init__ = patched_init

run_step_response(
    controller_type='model',
    model_type=model_type,
    duration=duration,
    step_time=step_time,
    P_initial=P_initial,
    P_final=P_final,
    warmup_controller=warmup_controller,
    output_subdir=os.path.join(base_subdir, 'serial_128')
)

# 2) 1024 JIT Batch
print("\n" + "=" * 70)
print("Running 1024 JIT Batch...")
print("=" * 70)

def patched_init_batch(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self.num_candidates = 1024
    self.use_batch_rollout = True

mc_module.MultiStackModelController.__init__ = patched_init_batch

run_step_response(
    controller_type='model',
    model_type=model_type,
    duration=duration,
    step_time=step_time,
    P_initial=P_initial,
    P_final=P_final,
    warmup_controller=warmup_controller,
    output_subdir=os.path.join(base_subdir, 'jit_batch_1024')
)

# Restore
mc_module.MultiStackModelController.__init__ = original_init

print("\n" + "=" * 70)
print("Both experiments completed.")
print("=" * 70)
