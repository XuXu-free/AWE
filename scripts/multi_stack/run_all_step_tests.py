import os
import sys
import subprocess
import time
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(project_root)

run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
run_subdir = f"run_{run_timestamp}"
output_base = os.path.join('output', 'multi_stack', 'test_step')
run_dir = os.path.join(output_base, run_subdir)
os.makedirs(run_dir, exist_ok=True)

print(f"Step response results will be saved to: {run_dir}")
sys.stdout.flush()

models = [
    'diffusion_mlp',
    'diffusion_pure_mlp',
    'diffusion_tcn',
    'guided_diffusion_mlp',
    'guided_diffusion_tcn',
    'pure_mlp',
    'pure_tcn',
    'flow_tcn',
    'flow_mlp',
    'lstm',
]

all_tasks = []
for model in models:
    log_path = os.path.join(run_dir, f'log_{model}.txt')
    cmd = [
        'uv', 'run', 'python', 'scripts/multi_stack/run_step_response_test.py',
        '--controller', 'model',
        '--model_type', model,
        '--duration', '7200',
        '--step_time', '3600',
        '--P_initial', '10.0e6',
        '--P_final', '20.0e6',
        '--warmup_controller', 'none',
        '--output_subdir', run_subdir
    ]
    all_tasks.append((model, cmd, log_path))

# NMPC simplified baseline
all_tasks.append(('nmpc_simplified', [
    'uv', 'run', 'python', 'scripts/multi_stack/run_step_response_test.py',
    '--controller', 'nmpc_simplified',
    '--duration', '7200',
    '--step_time', '3600',
    '--P_initial', '10.0e6',
    '--P_final', '20.0e6',
    '--warmup_controller', 'none',
    '--output_subdir', run_subdir
], os.path.join(run_dir, 'log_nmpc_simplified.txt')))

max_concurrent = 2
running = []
completed = []
task_iter = iter(all_tasks)

print(f"Running {len(all_tasks)} step response tests with max concurrency = {max_concurrent}")
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

for _ in range(max_concurrent):
    if not start_next():
        break

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

    while len(running) < max_concurrent:
        if not start_next():
            break

print("\n" + "="*60)
print("All step response tests complete!")
for name, ret in completed:
    status = "OK" if ret == 0 else f"FAIL({ret})"
    print(f"  {name}: {status}")
