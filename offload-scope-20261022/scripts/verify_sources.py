"""Record exact-source scope, negative-control equivalence, signatures and merge preview."""

import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
repo = Path('/data/songbinbin/Proj/Proj_LWX/Megatron-LM')
worktrees = Path('/home/songbinbin/gdn-offload-artifacts/scope-20261022')
sources = json.loads((root / 'sources.json').read_text())
base = '0a9f3e55459175d7e5621a99efb2d6f5b8378a8e'
prerequisite = '73ac08c254a0ff8997a0ff559799a4d7f073248b'
manager = 'megatron/core/pipeline_parallel/fine_grained_activation_offload.py'
gpt = 'tests/unit_tests/pipeline_parallel/test_fine_grained_activation_offloading.py'
native = 'tests/unit_tests/ssm/test_gdn_offload_training.py'


def git(*args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def blob(revision, path):
    return subprocess.check_output(['git', '-C', str(repo), 'show', f'{revision}:{path}'])


def without_chunk_completion(content):
    start = content.index(b'    def finish_all_groups(')
    end = content.index(b'    def find_next_group(', start)
    return content[:start] + content[end:]


checks, signatures, merges = [], [], []
main = git('rev-parse', 'origin/main')
for name in ('shared', 'feature', 'extensions'):
    revision = sources[name]['revision']
    assert subprocess.check_output(['git', '-C', str(worktrees / name), 'rev-parse', 'HEAD'], text=True).strip() == revision
    assert not subprocess.check_output(['git', '-C', str(worktrees / name), 'status', '--porcelain'], text=True)
    subprocess.run(['git', '-C', str(repo), 'merge-base', '--is-ancestor', prerequisite, revision], check=True)
    assert without_chunk_completion(blob(base, manager)) == without_chunk_completion(blob(revision, manager))
    for path in ('megatron/core/pipeline_parallel/schedules.py', 'tests/unit_tests/pipeline_parallel/test_schedules.py'):
        assert blob(base, path) == blob(revision, path)
    kernel = 'megatron/core/fusions/fused_pre_gated_delta_rule.py'
    assert blob(revision, kernel) == blob(main, kernel)
    checks.append(dict(source=name, revision=revision,
                       only_manager_function_changed_from_prerequisite='finish_all_groups',
                       schedules_unchanged=True, native_fused_kernel_unchanged=True))
    predecessor = {'shared': base, 'feature': sources['shared']['revision'],
                   'extensions': sources['feature']['revision']}[name]
    assert (root / f'{name}-own.patch').read_bytes() == subprocess.check_output(
        ['git', '-C', str(repo), 'diff', predecessor, revision])
    result = subprocess.run(['git', '-C', str(repo), '-c',
        'gpg.ssh.allowedSignersFile=/data/songbinbin/Proj/Proj_LWX/.codex-artifacts/offload-review-20261021/allowed-signers',
        'verify-commit', revision], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    signatures.append(dict(revision=revision, exit_code=result.returncode, output=result.stdout + result.stderr))
    result = subprocess.run(['git', '-C', str(repo), 'merge-tree', '--write-tree', revision, main],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    tree = result.stdout.splitlines()[0]
    assert git('rev-parse', f'{tree}:megatron/core') == git('rev-parse', f'{revision}:megatron/core')
    merges.append(dict(source=name, revision=revision, main=main, merged_tree=tree,
                       clean_merge=True, core_unchanged=True))

assert blob(sources['control']['revision'], manager) == blob(base, manager)
assert blob(sources['control']['revision'], gpt) == blob(sources['shared']['revision'], gpt)
assert blob(sources['feature-control']['revision'], native) == blob(sources['feature']['revision'], native)
assert blob(sources['feature-control']['revision'], manager) == blob(base, manager)
changed = git('diff', '--name-only', base, sources['shared']['revision']).splitlines()
assert changed == [manager, gpt], changed
scope = dict(checks=checks, own_shared_files=changed,
             identical_fixed_gpt_regression_blob=git('rev-parse', f"{sources['shared']['revision']}:{gpt}"),
             identical_native_gdn_test_blob=git('rev-parse', f"{sources['feature']['revision']}:{native}"))
for filename, record in [('scope-verification.json', scope),
                         ('signature-verification.json', signatures),
                         ('latest-main-merge-check.json', merges)]:
    (root / filename).write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(scope, indent=2))
