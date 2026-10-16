set -eu
# Run once for this comparison; preserve this exported root between arms.
export TRITON_CACHE_DIR="$(mktemp -d /tmp/qwen-offload-triton.XXXXXX)"

export DOC_TEST_ARM=baseline
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  --log-dir /home/songbinbin/gdn-offload-artifacts/qwen-doc-cache-fix-20261016/ranks-baseline --redirects 3 --tee 3 --no-python \
  bash -c 'export TRITON_CACHE_DIR="${TRITON_CACHE_DIR}/rank-${RANK}"; exec python "$@"' \
  qwen-training /home/songbinbin/gdn-offload-artifacts/qwen-doc-cache-fix-20261016/probe.py \
  --weights /path/to/Qwen3.5-0.8B --data /path/to/messages.jsonl \
  --tp 2 --sp --pp 2 --seq-length 128 --min-offloaded-tensor-size 1024 \
  --warmup 1 --iterations 2 --check-state --output /home/songbinbin/gdn-offload-artifacts/qwen-doc-cache-fix-20261016/baseline.json --review-argument 'two words'
export DOC_TEST_ARM=repeat
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  --log-dir /home/songbinbin/gdn-offload-artifacts/qwen-doc-cache-fix-20261016/ranks-repeat --redirects 3 --tee 3 --no-python \
  bash -c 'export TRITON_CACHE_DIR="${TRITON_CACHE_DIR}/rank-${RANK}"; exec python "$@"' \
  qwen-training /home/songbinbin/gdn-offload-artifacts/qwen-doc-cache-fix-20261016/probe.py \
  --weights /path/to/Qwen3.5-0.8B --data /path/to/messages.jsonl \
  --tp 2 --sp --pp 2 --seq-length 128 --min-offloaded-tensor-size 1024 \
  --warmup 1 --iterations 2 --check-state --output /home/songbinbin/gdn-offload-artifacts/qwen-doc-cache-fix-20261016/repeat.json --review-argument 'two words'
export DOC_TEST_ARM=offload
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  --log-dir /home/songbinbin/gdn-offload-artifacts/qwen-doc-cache-fix-20261016/ranks-offload --redirects 3 --tee 3 --no-python \
  bash -c 'export TRITON_CACHE_DIR="${TRITON_CACHE_DIR}/rank-${RANK}"; exec python "$@"' \
  qwen-training /home/songbinbin/gdn-offload-artifacts/qwen-doc-cache-fix-20261016/probe.py \
  --weights /path/to/Qwen3.5-0.8B --data /path/to/messages.jsonl \
  --tp 2 --sp --pp 2 --seq-length 128 --min-offloaded-tensor-size 1024 \
  --warmup 1 --iterations 2 --check-state --output /home/songbinbin/gdn-offload-artifacts/qwen-doc-cache-fix-20261016/offload.json --fraction 1 --review-argument 'two words'
