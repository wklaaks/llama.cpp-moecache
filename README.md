# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## 🍴 About this fork — `wklaaks/llama.cpp-moecache`

This is a feature branch of [llama.cpp](https://github.com/ggml-org/llama.cpp)
focused on the **Qwen4Exp / Qwen3.8-Flash-Next** model family. It layers three
capabilities on top of current upstream (merged through PR #27742), none of which
are available in stock llama.cpp:

1. **MoE expert hot-cache (VRAM).** Keeps the most-recently-used MoE expert
   weights resident in GPU memory instead of re-streaming them from host on every
   token. For large Mixture-of-Experts models this removes the dominant
   decode-time memory-bandwidth bottleneck and substantially raises tokens/sec
   at the cost of a bounded VRAM budget.
2. **Native MTP / NextN speculative decoding.** A first-class draft-model path for
   Multi-Token-Prediction heads (`--spec-type draft-mtp`, `-md <sidecar.gguf>`).
   The converter can emit a compact MTP *sidecar* GGUF from the Hugging Face
   safetensors source — see below.
3. **Graph reuse + QSA fixes.** Enables CUDA graph reuse for the qwen4exp decode
   path and shares the sparse-attention (QSA) input across layers, plus assorted
   correctness/perf fixes to the indexer and PLE caches.

### Build the MTP sidecar

A self-contained helper lives in [`tools/mtp-sidecar/`](tools/mtp-sidecar/) —
it downloads **only** the MTP/nextn tensors from a Hugging Face safetensors
model (HTTP Range reads, no full-model download), checks disk space, runs this
repo's `convert_hf_to_gguf.py --mtp --remote`, and validates the resulting
sidecar GGUF:

```bash
./tools/mtp-sidecar/make_mtp_sidecar.py Qwen/Qwen3.8-Flash-Next \
    --out-dir ~/models --fork-dir "$(pwd)"
```

Then serve with the sidecar as the draft model:

```bash
./build/bin/llama-server -m <base.gguf> -md mtp-Qwen3.8-Flash-Next-Q8_0.gguf \
    --spec-type draft-mtp --spec-draft-n-max 3
```

Full instructions, options, and troubleshooting:
[`tools/mtp-sidecar/README.md`](tools/mtp-sidecar/README.md).

---

## Serving

Production serve command for the Qwen3.8-Flash-Next model family with MoE hot-cache
and MTP speculative decoding:

```bash
ionice -c2 -n0 env \
  LLAMA_QSA_GATHER=1 \
  GGML_CUDA_MOE_CACHE_RESERVE_MB=2500 \
  ~/llama.cpp-moecache/build/bin/llama-server \
  -m ~/models/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf \
  -md ~/models/mtp-Qwen3.8-Flash-Next-Q8_0.gguf \
  --alias Qwen3.8-Flash-Next \
  --mlock \
  -b 4096 -ub 2048 \
  --threads 64 \
  --threads-batch 48 \
  --threads-http 4 \
  --mmap \
  --cache-prompt \
  --slot-save-path ~/cache/Qwen3.8-Flash-Next \
  --cache-ram 32000 \
  --ctx-size 151000 \
  --device CUDA0 \
  --split-mode layer \
  --moe-cache on \
  --cpu-moe \
  --n-gpu-layers 999 \
  --spec-type draft-mtp \
  --spec-draft-n-max 5 \
  --device-draft CUDA0 \
  --n-gpu-layers-draft 99 \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --jinja \
  --reasoning-format deepseek \
  --reasoning on \
  --chat-template-kwargs '{"reasoning_effort":"low"}' \
  --reasoning-preserve \
  --reasoning-budget -1 \
  --temp 0.7 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.0 \
  --presence_penalty 0.0 \
  --parallel 1 \
  --slots \
  --cont-batching \
  --verbose \
  -lv 5 \
  --port ${PORT}
```

### Key flags explained

| Flag | Purpose |
|------|---------|
| `--cpu-moe` | Keeps **all** MoE expert weights in CPU RAM (not VRAM). Frees a large amount of VRAM on MoE models where experts dominate weight size. |
| `--moe-cache on` | Adaptively caches the hottest CPU-resident MoE experts into a bounded VRAM slab so repeated hits avoid the PCIe round-trip. Combined with `--cpu-moe`: full expert set stays in RAM, working set gets GPU speed. |
| `--spec-type draft-mtp` | Enables the native MTP (Multi-Token-Prediction) speculative-decoding path. |
| `-md <sidecar>` | The MTP sidecar GGUF (built via `tools/mtp-sidecar/make_mtp_sidecar.py`). |
| `--spec-draft-n-max 5` | Max tokens to draft per step. |
| `--n-gpu-layers 999` | Offload all layers to GPU (non-expert weights). |
| `--flash-attn on` | Required for the QSA gather fast path and generally faster attention. |
| `--cache-type-k/v q8_0` | Quantized KV cache — halves VRAM vs F16 at negligible quality loss. |

### Env vars

| Variable | Default | Effect |
|----------|---------|--------|
| `LLAMA_QSA_GATHER` | `32768` | Minimum KV length for the sparse-attention decode gather fast path. `1` = always on; `0` = always off; `<int>` = custom threshold. |
| `GGML_CUDA_MOE_CACHE_RESERVE_MB` | `3072` | Per-device **OOM headroom** (MiB): free VRAM the cache leaves untouched so KV cache/activations are never starved. The cache may only spend `(free VRAM − reserve)`. Raise to protect against OOM crashes; lower to give the cache more room. |
| `GGML_CUDA_MOE_CACHE_STATS` | `0` (off) | Log MoE-cache hit-rate/budget stats every N collect cycles. Diagnostics only. |

---

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon [In Progress]](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
