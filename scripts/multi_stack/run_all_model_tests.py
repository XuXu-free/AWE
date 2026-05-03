import os
import sys
import subprocess
import time
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(project_root)

# Create a unified subdirectory for this run
run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
run_subdir = f"run_{run_timestamp}"
output_base = os.path.join('output', 'multi_stack', 'test')
run_dir = os.path.join(output_base, run_subdir)
os.makedirs(run_dir, exist_ok=True)

print(f"Results will be saved to: {run_dir}")
sys.stdout.flush()

power_profile_path = os.path.join(project_root, 'output', 'power', 'wind', 'wind_power_2025-10_1min.csv')

models = [
    'diffusion_mlp',
    'diffusion_pure_mlp',
    'diffusion_tcn',
    'guided_diffusion_mlp',
    'guided_diffusion_tcn',
    'pure_mlp',
    'pure_tcn',
    'lstm',
]

max_concurrent = 3  # Limit concurrent processes to avoid OOM on Windows

all_tasks = []
for model in models:
    log_path = os.path.join(run_dir, f'log_{model}.txt')
    cmd = [
        'uv', 'run', 'python', 'scripts/multi_stack/run_multi_stack_test.py',
        '--controller', 'model',
        '--model_type', model,
        '--duration', '86400',
        '--warmup_controller', 'nmpc_simplified',
        '--output_subdir', run_subdir,
        '--power_profile', power_profile_path,
    ]
    all_tasks.append((model, cmd, log_path))

# NMPC simplified baseline
all_tasks.append(('nmpc_simplified', [
    'uv', 'run', 'python', 'scripts/multi_stack/run_multi_stack_test.py',
    '--controller', 'nmpc_simplified',
    '--duration', '86400',
    '--warmup_controller', 'none',
    '--output_subdir', run_subdir,
    '--power_profile', power_profile_path,
], os.path.join(run_dir, 'log_nmpc_simplified.txt')))

running = []  # list of (name, proc, log_file)
completed = []
task_iter = iter(all_tasks)

print(f"Running {len(all_tasks)} tests with max concurrency = {max_concurrent}")
sys.stdout.flush()

def start_next():
    try:
        name, cmd, log_path = next(task_iter)
        print(f"\n[START] {name}")
        sys.stdout.flush()
        log_file = open(log_path, 'w')
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        running.append((name, proc, log_file))
        return True
    except StopIteration:
        return False

# Seed initial batch
for _ in range(max_concurrent):
    if not start_next():
        break

# Poll loop
while running:
    time.sleep(10)
    finished = []
    for item in running:
        name, proc, log_file = item
        ret = proc.poll()
        if ret is not None:
            log_file.close()
            finished.append(item)
            completed.append((name, ret))
            print(f"[DONE] {name}: exit code {ret}")
            sys.stdout.flush()

    for item in finished:
        running.remove(item)

    # Start new tasks if slots available
    while len(running) < max_concurrent:
        if not start_next():
            break

print("\n" + "="*60)
print("All tests complete!")
for name, ret in completed:
    status = "OK" if ret == 0 else f"FAIL({ret})"
    print(f"  {name}: {status}")
