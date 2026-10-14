"""Check full pretrained text training after rebasing, with fresh disabled baselines."""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path('/home/songbinbin/gdn-offload-artifacts')
OUT = ROOT / 'qwen-training/publication-20261011/tp_sp'
META = json.loads((ROOT / 'publication-20261011/community-container.json').read_text())
records = []
for fused in (False, True):
    directory = OUT / ('fused' if fused else 'unfused')
    directory.mkdir(parents=True, exist_ok=True)
    baseline = None
    for arm in ('disabled', '1'):
        occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
        assert not any(uuid in occupied for uuid in META['uuid_order']), occupied
        outputs = [directory / f'{arm}.rank{rank}.json' for rank in range(2)]
        assert not any(path.exists() for path in outputs)
        command = [
            'docker', 'exec', '-e', 'PYTHONPATH=/artifacts/causal-conv1d-python:/artifacts/bridge-python:/artifacts/Megatron-Bridge/src:/workspace',
            META['name'], '/artifacts/uv-container-bin', 'run', '--no-project', '--active',
            'python', '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2',
            '/artifacts/qwen_pool_lifetime_check.py',
            '--weights', '/artifacts/models/Qwen3.5-0.8B', '--data', '/artifacts/training-data/ks-item-sft.jsonl',
            '--tp', '2', '--sp', '--recompute-norm', '--check-state', '--seq-length', '128',
            '--min-offloaded-tensor-size', '1024', '--warmup', '1', '--iterations', '2',
            '--output', str(directory / f'{arm}.json').replace(str(ROOT), '/artifacts'),
        ]
        if fused:
            command.append('--fused-pre-gdr')
        if arm != 'disabled':
            command.extend(['--fraction', arm])
        print(f'Start fused={fused} arm={arm}', flush=True)
        with (directory / f'{arm}.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        results = [json.loads(path.read_text()) for path in outputs]
        states = [result['states'] for result in results]
        if baseline is None:
            baseline = states
        else:
            assert states == baseline, fused
        for rank, result in enumerate(results):
            assert result['gpu_training_run'] and not result['samples']
            assert (result['text_layers'], result['text_hidden_size'], result['dataset_tokens']) == (24, 1024, 30727)
            assert len(result['states']) == 3 and all(s['grads'] and s['weights'] and s['optimizer'] for s in result['states'])
            assert result['pool_verified_before_reset']
            assert result['pool_in_use_before_resets'] == [0] * (2 if arm == 'disabled' else 5)
            selected = result['offload_status']['selected_transfer_bytes']
            assert (selected == 0) if arm == 'disabled' else (selected > 0)
            records.append({'fused': fused, 'arm': arm, 'rank': rank, 'exact_steps': 3,
                            'file': str(outputs[rank].relative_to(ROOT)),
                            'sha256': hashlib.sha256(outputs[rank].read_bytes()).hexdigest(),
                            'selected_transfer_bytes': selected})
        print(f'Passed fused={fused} arm={arm}', flush=True)
report = {'status': 'passed', 'scope': 'Post-rebase complete Qwen TP=2/SP, norm recomputation enabled, disabled/full offload, fused/unfused',
          'fresh_two_rank_runs': 4, 'rank_files': 8, 'exact_complete_rank_step_states': 24,
          'all_pre_reset_pool_counts_zero': True, 'timed_samples': 0, 'results': records}
(ROOT / 'publication-20261011/qwen-rebased-validation.json').write_text(json.dumps(report, indent=2) + '\n')
print('Passed four fresh full-Qwen runs, eight rank files, 24 exact complete rank-step states.', flush=True)
