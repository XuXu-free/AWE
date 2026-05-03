import json, glob, os, sys

path = sys.argv[1] if len(sys.argv) > 1 else 'output/multi_stack/test/run_20260426_121806'
files = glob.glob(os.path.join(path, '*_metrics.json'))
results = []
for f in files:
    basename = os.path.basename(f)
    name = basename.split('_data_')[0]
    if name.startswith('model_'):
        name = name[6:]
    with open(f, 'r') as fp:
        data = json.load(fp)
    duration = data.get('duration', 'N/A')
    power = data.get('rmse_power_mw', 'N/A')
    temp = data.get('rmse_temp_k', 'N/A')
    results.append((name, duration, power, temp))

results.sort(key=lambda x: x[2] if isinstance(x[2], (int, float)) else float('inf'))

print('%-40s %12s %16s %14s' % ('Model', 'Duration(s)', 'Power RMSE(MW)', 'Temp RMSE(K)'))
print('-'*84)
for name, dur, power, temp in results:
    p_str = '%.4f' % power if isinstance(power, (int, float)) else str(power)
    t_str = '%.4f' % temp if isinstance(temp, (int, float)) else str(temp)
    d_str = '%.0f' % dur if isinstance(dur, (int, float)) else str(dur)
    print('%-40s %12s %16s %14s' % (name, d_str, p_str, t_str))
