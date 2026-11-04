#!/usr/bin/env bash
# Serve Qwen3.8-27B QSA with GSQ-RCO IQ3_S weights at 262K context with vision on one 24 GB GPU. Stop with Ctrl-C.
#   QSA_MODEL_DIR    directory holding the GGUF files (default: ./models)
#   QSA_DRAFTER      dflash (default) or mtp. mtp uses about 1.1 GiB less VRAM and decodes slower
#   QSA_DRAFTS       draft tokens per step (default: 4)
#   QSA_GRAPH_CACHE  1 keeps the CUDA graph cache: about 0.3% faster for 122 MiB more
#   QSA_HOST, QSA_PORT   default 127.0.0.1 and 8091
#   Choose the GPU with CUDA_VISIBLE_DEVICES. Extra arguments go to llama-server.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BIN=${QSA_BIN:-$ROOT/build/bin}
DIR=${QSA_MODEL_DIR:-./models}

export LLAMA_QSA_RESERVE_BLK=1 LLAMA_QSA_BLK_REBUILD=1 LLAMA_QSA27B_MASKED_PREFILL_MAX=8192 GGML_CUDA_TOPK_DET=1
export LLAMA_MTP_KV_ONLY=1 LLAMA_MTP_PAD=4 LLAMA_MTP_N_UBATCH=64 LLAMA_MTP_LAG=0 LLAMA_MTP_ASYNC=0
export LLAMA_REUSE_OUT_ORDER=1 LLAMA_KV_PAD_GEO=2 LLAMA_CKPT_TAIL_UBATCH=1
export GGML_FT_GRAPH_CACHE=${QSA_GRAPH_CACHE:-0} GGML_FT_GRAPH_CACHE_MIN_NODES=32 CUDA_SCALE_LAUNCH_QUEUES=4x
export LLAMA_QSA_RAW_RING=8192 MTMD_ON_DEMAND=1 GGML_CUDA_EVICT_MIB=2048 LLAMA_SAMPLE_TOPK=1
unset GGML_CUDA_GRAPH_OPT

case "${QSA_DRAFTER:-dflash}" in
  dflash)
    export LLAMA_SPEC_PREFILL_TAIL=2048 LLAMA_GPU_ARGMAX=1
    spec=(-md "$DIR/Qwen3.8-27B-DFlash2-IQ4_XS.gguf" -ngld 99 --spec-type draft-dflash --spec-draft-p-min 0
          --spec-draft-n-max "${QSA_DRAFTS:-4}" --spec-draft-sampling probabilistic) ;;
  mtp)
    export LLAMA_MTP_WINDOW=8192
    spec=(--spec-type draft-mtp --spec-draft-p-min 0 --spec-draft-n-max "${QSA_DRAFTS:-4}") ;;
  *) echo "serve.sh: QSA_DRAFTER must be dflash or mtp" >&2; exit 1 ;;
esac

exec "$BIN/llama-server" -m "$DIR/Qwen3.8-27B-QSA-GSQ-RCO-IQ3_S-mtp.gguf" --mmproj "$DIR/mmproj-Qwen3.8-27B-QSA-F16.gguf" \
  --alias qwen3.8-27b-qsa -ngl 99 -fa on -np 1 --fit off -t 8 -c 262144 -b 2048 -ub 256 \
  -ctk q8_0 -ctv q8_0 -ctkd f16 -ctvd f16 --no-context-shift --jinja --cache-ram 0 --ctx-checkpoints 8 --checkpoint-min-step 1024 \
  --host "${QSA_HOST:-127.0.0.1}" --port "${QSA_PORT:-8091}" "${spec[@]}" "$@"
