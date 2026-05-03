import os
import sys
import subprocess
import time
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(project_root)

models = [
    'diffusion_mlp',
    'diffusion_pure_mlp',
    'diffusion_tcn',
    'guided_diffusion_mlp',
    'guided_diffusion_tcn',
    'deterministic_diffusion_mlp',
    'deterministic_diffusion_pure_mlp',
    'deterministic_diffusion_tcn',
    'deterministic_guided_diffusion_mlp',
    'deterministic_guided_diffusion_tcn',
    'early_stop_diffusion_mlp',
    'early_stop_diffusion_tcn',
    'early_stop_guided_diffusion_mlp',
    'early_stop_guided_diffusion_tcn',
    'pure_mlp',
    'pure_tcn',
    'flow_tcn',
    'flow_mlp',
    'lstm',
]

max_concurrent = 3

all_tasks = []
for model in models:
    log_path = os.path.join('output', 'multi_stack', 'test_step', f'log_{model}_2x.txt')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    cmd = [
        'python', 'scripts/multi_stack/run_step_response_test.py',
        '--controller', 'model',
        '--model_type', model,
        '--duration', '14400',
        '--step_time', '7200'
    ]
    all_tasks.append((model, cmd, log_path))

running = []
completed = []
task_iter = iter(all_tasks)

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

    while len(running) < max_concurrent:
        if not start_next():
            break

print("\n" + "="*60)
print("All step response tests complete!")
for name, ret in completed:
    status = "OK" if ret == 0 else f"FAIL({ret})"
    print(f"  {name}: {status}")
