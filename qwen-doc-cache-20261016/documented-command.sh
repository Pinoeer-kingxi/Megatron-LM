# Run once for this comparison; preserve this exported root between arms.
export TRITON_CACHE_DIR="$(mktemp -d /tmp/qwen-offload-triton.XXXXXX)"
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  --log-dir /tmp/qwen-disabled-ranks --redirects 3 --tee 3 --no-python \
  bash -c 'export TRITON_CACHE_DIR="${TRITON_CACHE_DIR}/rank-${RANK}"; exec python "$@"' \
  qwen-training tools/ssm/qwen_gdn_offload_training.py \
  --weights /path/to/Qwen3.5-0.8B --data /path/to/messages.jsonl \
  --tp 2 --sp --pp 2 --seq-length 128 --min-offloaded-tensor-size 1024 \
  --warmup 1 --iterations 2 --check-state --output /tmp/qwen-disabled.json
