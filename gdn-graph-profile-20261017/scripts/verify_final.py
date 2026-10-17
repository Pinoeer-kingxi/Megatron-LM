"""Verify recorded GPU results offline; this script does not execute GPU work."""
from collections import Counter
import json
import math
from pathlib import Path
import sqlite3
import statistics

from analyze_profiles import analyze

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text())


def records(label):
    return read(f"{label}/result.json")["records"]


def sequence(rows):
    return [(x["input_sha256"], x["loss"], x["grad_norm"]) for x in rows]


def verify_records(rows):
    assert all(math.isfinite(x["grad_norm"]) and x["grad_norm"] > 0 for x in rows)
    assert all(math.isfinite(v) for x in rows for v in x["loss"])
    assert all(x["pool"]["current_in_use"] == 0 for x in rows)
    replay = [x for x in rows if x["replay"]]
    assert replay
    assert all(x["pool"] == replay[0]["pool"] for x in replay)


def memory_and_time(rows):
    measured = [x for x in rows if x["measured"]]
    assert len(measured) == 10
    return dict(
        median_ms=statistics.median(x["seconds"] for x in measured) * 1000,
        peak_allocated_bytes=max(x["peak_allocated_bytes"] for x in measured),
        peak_reserved_bytes=max(x["peak_reserved_bytes"] for x in measured),
        selected_bytes=measured[0]["selected_bytes"],
        pinned_buffers=measured[0]["pool"]["total_allocated"],
    )


def kernel_counts(path):
    with sqlite3.connect(path) as c:
        start, end = c.execute(
            "select start,end from NVTX_EVENTS where text='training_step:7'"
        ).fetchone()
        return Counter(dict(c.execute(
            'select s.value,count(*) from CUPTI_ACTIVITY_KIND_KERNEL k '
            'join StringIds s on s.id=k.demangledName '
            'where k.start>=? and k.end<=? group by s.value', (start, end)
        )))


manifest = read('final-manifest.json')
assert manifest['status'] == 'completed_pending_verification'
assert len(manifest['jobs']) == 15
assert manifest['source']['source'] == '367d014c7bf74ba3553c9db58da47fa5bcba906e'
assert manifest['cleanup']['exit_code'] == 0
assert all(j['exit_code'] == j['expected_exit'] for j in manifest['jobs'])
for name, expected in (('profile', 4), ('probe', 8), ('threshold', 3), ('equal-heads', 1)):
    phase = read(name+'-manifest.json')
    assert len(phase['jobs']) == expected
    assert phase['cleanup']['exit_code'] == 0
    assert all(j['exit_code'] == 0 for j in phase['jobs'])
cpu = read('cpu-checks.json')
assert cpu['source'] == manifest['source']['source']
assert cpu['files'] == manifest['source']['files']
assert all(c['exit_code'] == 0 for c in cpu['checks'])

for model in ('gdn2', 'qwen'):
    reference = records(model+'-probe-disabled')
    for suffix in ('probe-disabled', 'probe-full', 'probe-full-copy8', 'probe-half', 'threshold4m'):
        rows = records(model+'-'+suffix)
        assert len(rows) == 16 and sequence(rows) == sequence(reference)
        verify_records(rows)
for label, gates in (('gdn2-saved-tensors', ('w',)), ('gdn2-saved-tensors-equal-heads', ('b', 'w'))):
    lifetime = read(label+'/result.json')['profile']
    for gate in gates:
        selected = [r for r in lifetime['rows'] if r['name'] == gate and r['phase'].startswith('gdn_core_attn')]
        assert len(selected) == 1 and selected[0]['tensor_bytes'] == 8*2**20
        storage = lifetime['storages'][str(selected[0]['storage'])]
        assert 'outside_recurrence' in storage['phases']
red = read('regression-red-control/reports/rank-0.json')
assert red['collected'] == 1 and red['exit_code'] == 1
assert len(red['outcomes']) == 1
assert red['outcomes'][0]['outcome'] == 'failed'
assert 'assert ("w" in names) == frozen_gates' in red['outcomes'][0]['detail']

pytest_summary = {}
for label, expected in (('final-gdn2-scopes', 20), ('final-gdn2-native', 18), ('final-gdn2-graphs', 9)):
    pytest_summary[label] = []
    for rank in (0, 1):
        x = read(f'{label}/reports/rank-{rank}.json')
        assert x['rank'] == rank and x['exit_code'] == 0
        assert x['collected'] == expected and len(x['outcomes']) == expected, (label, rank, x['collected'])
        assert all(o['outcome'] == 'passed' and o['when'] == 'call' for o in x['outcomes'])
        pytest_summary[label].append(dict(rank=rank, passed=expected))

state_checks = {}
for heads in (8, 16):
    left = records(f'final-state-k{heads}-disabled')
    right = records(f'final-state-k{heads}-enabled')
    assert len(left) == len(right) == 6
    assert sequence(left) == sequence(right)
    for rows in (left, right):
        verify_records(rows)
    for x, y in zip(left, right):
        for field in ('gradients', 'weights', 'optimizer'):
            assert x[field] == y[field], (heads, x['step'], field)
    state_checks[str(heads)] = 'six steps: exact losses, grad norms, all gradients, weights and complete optimizer fingerprints'

pairs = []
for repeat in range(3):
    left = records(f'gdn2-paired-{repeat}-previous')
    right = records(f'gdn2-paired-{repeat}-final')
    assert len(left) == len(right) == 16
    assert sequence(left) == sequence(right)
    for rows in (left, right):
        verify_records(rows)
    before, after = memory_and_time(left), memory_and_time(right)
    assert before['selected_bytes'] == 400 * 2**20
    assert after['selected_bytes'] == 360 * 2**20
    assert before['pinned_buffers'] == 54 and after['pinned_buffers'] == 48
    pairs.append(dict(repeat=repeat, before=before, after=after,
                      step_time_change_pct=(after['median_ms']/before['median_ms']-1)*100,
                      reserved_saved_bytes=before['peak_reserved_bytes']-after['peak_reserved_bytes']))

profiles = {}
for label in ('gdn2-profile-disabled', 'gdn2-profile-enabled', 'qwen-profile-disabled', 'qwen-profile-enabled', 'gdn2-final-profile'):
    path = ROOT/label/'trace.sqlite'
    profiles[label] = analyze(path)
    with sqlite3.connect(path) as c:
        diagnostic = [x[0] for x in c.execute('select text from DIAGNOSTIC_EVENT')]
        assert not any(word in text.lower() for text in diagnostic for word in ('dropped', 'lost records', 'overflow', 'error:'))
    verify_records(records(label))
for model in ('gdn2', 'qwen'):
    assert sequence(records(model+'-profile-disabled')) == sequence(records(model+'-profile-enabled'))
    assert kernel_counts(ROOT/f'{model}-profile-disabled/trace.sqlite') == kernel_counts(ROOT/f'{model}-profile-enabled/trace.sqlite')
assert kernel_counts(ROOT/'gdn2-profile-enabled/trace.sqlite') == kernel_counts(ROOT/'gdn2-final-profile/trace.sqlite')
assert sequence(records('gdn2-profile-enabled')) == sequence(records('gdn2-final-profile'))
for x in profiles['gdn2-profile-enabled']:
    for direction in ('Host-to-Device', 'Device-to-Host'):
        assert x['copies'][direction]['bytes'] == 1600 * 2**20
for x in profiles['gdn2-final-profile']:
    for direction in ('Host-to-Device', 'Device-to-Host'):
        assert x['copies'][direction]['bytes'] == 1440 * 2**20

result = dict(status='passed', source=manifest['source'], pytest=pytest_summary,
              red_control='old production source fails the targeted live-write-gate copy assertion',
              complete_state=state_checks, pairs=pairs,
              paired_median_step_change_pct=statistics.median(x['step_time_change_pct'] for x in pairs),
              paired_step_change_range_pct=[min(x['step_time_change_pct'] for x in pairs),max(x['step_time_change_pct'] for x in pairs)],
              median_reserved_saved_bytes=statistics.median(x['reserved_saved_bytes'] for x in pairs),
              profiles=profiles)
(ROOT/'verified-results.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='profiles'},indent=2))
