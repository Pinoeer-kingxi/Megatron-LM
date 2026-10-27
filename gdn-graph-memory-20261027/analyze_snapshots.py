"""Summarize the live allocations at the peak of one recorded replay step."""
import gzip
import pickle
from pathlib import Path

for mode in ('off', 'on', 'gdn2-off', 'gdn2-on'):
    with gzip.open(Path(__file__).parent / f'{mode}.pickle.gz', 'rb') as f:
        snapshot = pickle.load(f)
    live, peak_live = {}, {}
    current = peak = 0
    for event in snapshot['device_traces'][0]:
        if event['action'] == 'alloc':
            live[event['addr']] = event
            current += event['size']
        elif event['action'] == 'free_requested' and event['addr'] in live:
            current -= live.pop(event['addr'])['size']
        if current > peak:
            peak, peak_live = current, live.copy()
    print(f'{mode}: tracked incremental peak {peak / 2**20:.2f} MiB')
    for event in sorted(peak_live.values(), key=lambda e: -e['size'])[:3]:
        frames = [(f['name'], Path(f['filename']).name, f['line'])
                  for f in event['frames'] if f['filename'].endswith('.py')][:3]
        print(f"  {event['size'] / 2**20:.2f} MiB {frames}")
    for pool in sorted({tuple(s['segment_pool_id']) for s in snapshot['segments']}):
        segments = [s for s in snapshot['segments'] if tuple(s['segment_pool_id']) == pool]
        print(f"  pool {pool}: reserved {sum(s['total_size'] for s in segments) / 2**20:.2f} MiB, "
              f"allocated {sum(s['allocated_size'] for s in segments) / 2**20:.2f} MiB")
