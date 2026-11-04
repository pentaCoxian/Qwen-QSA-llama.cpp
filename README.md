# Qwen-QSA-llama.cpp

Run Qwen3.8-27B with Qwen Sparse Attention (QSA) on one 24 GB GPU, at 262K context with vision.

This fork adds three things to the published QSA engine: memory changes that make 262K fit in 24 GB, kernel changes that speed up decode and prefill, and rejection sampling for the DFlash2 drafter.

## Results

Measured on one RTX 4090 at 262K context, with the settings in [`serve.sh`](deploy/qsa27b-24gb/serve.sh). Weights are GSQ-RCO IQ3_S and the drafter is DFlash2 IQ4_XS. The default is 4 drafts, but the rows marked below were measured at 5, before that default changed. "Before" is this engine without the speed changes, on the same four tasks.

| Harmonic mean of 4 tasks | Before | After |
|---|---:|---:|
| Sampled decode, 60K input, 10 seeds | 97.8 tok/s | 118.8 tok/s |
| Greedy decode, 60K input | 115.0 tok/s | 132.0 tok/s |
| Greedy decode, 255K input, 5 drafts | 103.2 tok/s | 118.0 tok/s |
| Cold prefill, 255K input | 2,041 tok/s (125 s) | 2,377 tok/s (107 s) |
| Peak VRAM, 255K input, 5 drafts | 23.16 GiB | 22.94 GiB |

Sampled means temperature 1.0, top-p 0.95, top-k 20. It averages over seeds, so it is the fairer decode comparison. Greedy runs one fixed text per build, and its speed moves with how well drafts match that text.

Draft count at 60K, where sampled figures leave out outputs that stopped early:

| Drafts | Sampled | Greedy | VRAM against 5 drafts |
|---:|---:|---:|---:|
| 3 | 115.4 tok/s | 120.2 tok/s | −298 MiB |
| **4** | 118.8 tok/s | 132.0 tok/s | −148 MiB |
| 5 | 115.0 tok/s | 130.4 tok/s | |
| 6 | 118.6 tok/s | 132.5 tok/s | +150 MiB |

The gaps between 4, 5 and 6 are close to sampling noise. Set `QSA_DRAFTS` to pick another count.

Speed depends on the task, because code accepts far more drafts than prose:

| Task, greedy at 255K, 5 drafts | tok/s | Draft acceptance |
|---|---:|---:|
| `sse_collector` (code) | 169.1 | 0.679 |
| `collector_tests` (code) | 155.9 | 0.609 |
| `english_peer_review` (prose) | 96.6 | 0.301 |
| `japanese_editorial` (prose) | 89.2 | 0.264 |

Quality at 255K, on the same machine, at 5 drafts:

| Check | Result |
|---|---|
| Fact recall, single fact | 24/24 (dense control 24/24) |
| Fact recall, two-hop | 11/16 (dense control 12/16) |
| Vision, image 255K tokens before the question | 14/14 |
| KL divergence from BF16, prose / code, at ubatch 8 | 0.0393 / 0.1520 (0.0394 / 0.1525 before the speed changes) |

The recall set is 40 questions, so treat it as a smoke test.

## Quick start

You need the NVIDIA driver, the CUDA toolkit, CMake and the `hf` command. The build below is for RTX 40-series cards. Other cards need a different `CMAKE_CUDA_ARCHITECTURES` and are untested.

```sh
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=$(which nvcc) -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_BUILD_UI=OFF
cmake --build build --target llama-server --parallel

hf download pentacoxian-dev/Qwen3.8-27B-QSA-GSQ-RCO-GGUF --local-dir models   # about 14.3 GB
deploy/qsa27b-24gb/serve.sh
```

Nothing else can share the GPU, and the server listens on `127.0.0.1:8091`. The script's header lists its options, including a lower-memory MTP drafter (about 21.9 GiB, 99.7 tok/s greedy at 255K).

### Making the model files yourself

The [model repo](https://huggingface.co/pentacoxian-dev/Qwen3.8-27B-QSA-GSQ-RCO-GGUF) holds the result of these steps. They need `llama-quantize` built as well, a 17 GB download and about 31 GB of disk:

```sh
hf download ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf --local-dir models
hf download higa2515/Qwen3.8-27B-QSA-BF16 qsa-indexers.safetensors --revision d61281f4626729e4cf138476a704c7fedc08f891 --local-dir models
hf download higa2515/Qwen3.8-27B-QSA-GGUF mmproj-Qwen3.8-27B-QSA-F16.gguf --local-dir models
hf download z-lab/Qwen3.8-27B-DFlash2-GGUF Qwen3.8-27B-DFlash2-BF16.gguf --local-dir models

python3 deploy/qsa27b-24gb/gguf_qsa_graft.py models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf \
  models/qsa-indexers.safetensors models/Qwen3.8-27B-QSA-GSQ-RCO-IQ3_S-mtp.gguf
build/bin/llama-quantize models/Qwen3.8-27B-DFlash2-BF16.gguf models/Qwen3.8-27B-DFlash2-IQ4_XS.gguf IQ4_XS
```

## What changed

Compared with the published engine:

| Change | Switch | Effect on an RTX 4090 |
|---|---|---|
| Raw indexer keys kept only for recent rows | `LLAMA_QSA_RAW_RING=8192` | Saves about 2 GiB. Needed to fit 262K |
| MTP draft cache keeps only recent cells | `LLAMA_MTP_WINDOW=8192` | Saves about 0.9 GiB with the MTP drafter |
| Small F16/BF16 GEMV skips cuBLAS, from upstream | `GGML_CUDA_NO_MMVF_BATCH=1` turns it off | 6-row step 20.2 to 18.0 ms |
| DeltaNet kernel handles 4 columns per warp, from upstream | `GGML_CUDA_NO_GDN_COLS4=1` turns it off | Prefill at 8K 2,876 to 3,062 tok/s |
| DeltaNet tensor-core kernel for prefill, from upstream | `GGML_CUDA_NO_GDN_MMA=1` turns it off | Prefill at 8K 3,062 to 3,115 tok/s |
| Unused packed-MoE branch compiled out of `mul_mat_q` | Define `GGML_CUDA_MMQ_MOE_PACK_BUILD` to keep it | Prefill at 8K 3,115 to 3,215 tok/s |
| Drafter skips prompt rows outside its window | `LLAMA_SPEC_PREFILL_TAIL=2048` | Prefill at 60K 2,389 to 2,542 tok/s, and about 1% more from skipping the layer copy |
| Rejection sampling for DFlash2 | `--spec-draft-sampling probabilistic` | Sampled decode 108.6 to 114.8 tok/s at 60K |

`serve.sh` also sets two existing switches. `LLAMA_SAMPLE_TOPK=1` adds about 2% to sampled decode and leaves samples identical. `LLAMA_GPU_ARGMAX=1` adds about 1.6% to greedy decode and leaves text identical.

Upstream's tensor-core DeltaNet kernel only writes the newest rollback snapshot. Here it handles the first part of each batch and the recurrent kernel finishes the last tokens, so all snapshots stay valid. It runs on Ada, Blackwell and DGX Spark only.

## Limits

- **Hardware:** tested on one RTX 4090 with this one model. Nothing else has been run.
- **State saving:** `LLAMA_QSA_RAW_RING` makes full-state save and restore throw, so `serve.sh` sets `--cache-ram 0` and a prompt cache is not available.
- **Editing after an image:** normal chat turns reprocess under 20 tokens, but editing a question right after an image reprocesses the whole prompt, about 110 s at 255K.
- **Output:** text is not bit-identical to builds without the speed changes, because rounding differs. About 1% of tokens pick a different top token. Runs of one build repeat exactly when the seed is fixed.
- **Early stops:** at temperature 1.0 the model occasionally ends at once, for example `No response requested.` after 5 tokens. It happens with and without rejection sampling.
- **Rejection sampling:** token distributions were compared over 1,500 samples on each of two prompts, with and without the drafter, and showed no detectable bias. A small bias could still escape a test of that size.
- **Packed MoE:** `GGML_CUDA_MMQ_MOE_PACK=1` is silently ignored in this build. It matters on V100, so use [Qwen-V100-llama.cpp](https://github.com/pentaCoxian/Qwen-V100-llama.cpp) for V100 cards.
- **Not tested:** video, images larger than 3200x1800, task benchmarks such as GSM8K, and sparse against dense KL at long context.

## Tests

From a clean build with GPUs hidden, 60 of 61 of llama.cpp's tests pass, including rollback and save-load. `test-jinja-py` fails only because the test machine has no `jinja2` Python module. With GPUs hidden these tests don't run the GPU kernels, which were checked on the real model instead: editing the last 1, 2, 3 or 5 tokens after a prefill matched a cold run on every token.

## Where this comes from

- [llama.cpp](https://github.com/ggml-org/llama.cpp), the upstream project.
- [Qwen-V100-llama.cpp](https://github.com/pentaCoxian/Qwen-V100-llama.cpp), the base of this fork, at `6209237`. Its V100 tuning code is still in the tree and is not used on current GPUs.
- [higa2515's Qwen3.8-27B QSA](https://huggingface.co/higa2515/Qwen3.8-27B-QSA-GGUF): the QSA runtime patch, indexers and vision projector. Its notes are in [`qsa27b/RUNTIME.md`](qsa27b/RUNTIME.md) and describe its two-V100 setup.
- [ISTA-DASLab's GSQ-RCO](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF) quantized weights.
- [z-lab's DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF) drafter.

MIT and Apache-2.0 notices are kept in `LICENSE`, `LICENSE-freetoken`, `LICENSE-pcie-patches` and `licenses/`. The fork's earlier README is in [`README.upstream.md`](README.upstream.md).
