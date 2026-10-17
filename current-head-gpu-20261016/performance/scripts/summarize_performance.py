"""Summarize paired complete-Qwen timings without claiming full-state correctness."""

import csv
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    return json.loads((ROOT / path).read_text())

manifest = read('manifest.json')
assert manifest['status'] == 'measurements_completed'
assert manifest['source_commit'] == '87c22d1abae13abf375e8ed4ddfe62301cde87f4'
assert manifest['cleanup']['exit_code'] == 0
assert len(manifest['jobs']) == 12 and all(j['exit_code'] == 0 for j in manifest['jobs'])
rows, results = [], []
for fused in (0, 1):
    pairs, losses = [], []
    for repeat in range(3):
        arms = []
        for offload in (0, 1):
            name = f'f{fused}-r{repeat}-o{offload}'
            value = read(name + '/result.rank0.json')
            args = value['arguments']
            assert value['gpu_training_run'] and value['text_layers'] == 24 and value['rank'] == 0
            assert not value['states'] and len(value['samples']) == 10
            expected = {'tp': 1, 'sp': False, 'cp': 1, 'pp': 1, 'vp': None,
                        'seq_length': 4096, 'microbatches': 4, 'warmup': 3, 'iterations': 10,
                        'check_state': False, 'recompute_norm': False,
                        'min_offloaded_tensor_size': 1048576, 'fused_pre_gdr': bool(fused),
                        'fraction': 1.0 if offload else None}
            assert all(args[k] == v for k, v in expected.items()), name
            occupancy = read(name + '/occupancy.json')
            assert occupancy['gpu'] == manifest['gpu'] and not occupancy['external']
            assert occupancy['samples'] and any(s['processes'] for s in occupancy['samples'])
            for sample in value['samples']:
                assert math.isfinite(sample['seconds']) and sample['seconds'] > 0
                assert (sample['selected_transfer_bytes'] > 0) if offload else (sample['selected_transfer_bytes'] == 0)
                rows.append({'fusion': fused, 'repeat': repeat, 'offload': offload, **sample})
            log = next((ROOT / name / 'ranks').glob('**/stderr.log')).read_text()
            records = re.findall(r'Rank 0 step (\d+): losses=(\[[^\n]*?\]) grad_norm=', log)
            assert [int(step) for step, _ in records] == list(range(13)), name
            arm_losses = [json.loads(text) for _, text in records]
            assert all(len(step) == 4 and all(math.isfinite(x) for x in step) for step in arm_losses)
            losses.append(arm_losses)
            arms.append({'median_seconds': statistics.median(s['seconds'] for s in value['samples']),
                         'max_peak_allocated_bytes': max(s['peak_allocated_bytes'] for s in value['samples']),
                         'max_peak_reserved_bytes': max(s['peak_reserved_bytes'] for s in value['samples']),
                         'selected_transfer_bytes': value['offload_status']['selected_transfer_bytes']})
        pairs.append({'disabled': arms[0], 'offload': arms[1],
                      'runtime_change_pct': 100 * (arms[1]['median_seconds'] / arms[0]['median_seconds'] - 1)})
    baseline = statistics.median(p['disabled']['max_peak_allocated_bytes'] for p in pairs)
    enabled = statistics.median(p['offload']['max_peak_allocated_bytes'] for p in pairs)
    max_loss_delta = max(abs(x - y)
                         for arm in losses[1:] for a, b in zip(losses[0], arm, strict=True)
                         for x, y in zip(a, b, strict=True))
    results.append({'fused_pre_gdr': bool(fused), 'per_repeat': pairs,
                    'memory_saved_mib': (baseline - enabled) / 2**20,
                    'memory_saved_pct': 100 * (baseline - enabled) / baseline,
                    'disabled_peak_allocated_mib': baseline / 2**20,
                    'enabled_peak_allocated_mib': enabled / 2**20,
                    'paired_runtime_change_pct': statistics.median(p['runtime_change_pct'] for p in pairs),
                    'paired_runtime_change_range_pct': [min(p['runtime_change_pct'] for p in pairs),
                                                       max(p['runtime_change_pct'] for p in pairs)],
                    'logged_losses_exact_across_six_runs': losses[1:] == [losses[0]] * 5,
                    'max_logged_loss_delta': max_loss_delta})
summary = {'source_commit': manifest['source_commit'], 'gpu': manifest['gpu'],
           'runs': 12, 'measured_steps': len(rows), 'results': results,
           'statistics': 'Median of three paired per-run median-time changes; memory uses median per-run maximum peaks.',
           'correctness_limit': 'Finite full-precision logged losses are compared; timed runs do not record full gradient/Adam state.',
           'performance_limit': 'Same-revision offload on/off. No cross-version or controlled fusion-to-fusion claim.'}
assert len(rows) == 120
(ROOT / 'performance-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
with (ROOT / 'performance-samples.csv').open('w') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(summary, indent=2))
