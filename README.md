# Haven C++ (`haven-cpp`)

**Sovereign, Bare-Metal C++20 AI Inference Engine, Native Cloud TPU StableHLO Runtime & Companion Server**

[![CI](https://github.com/ssfdre38/haven-cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/ssfdre38/haven-cpp/actions/workflows/ci.yml)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![SIMD](https://img.shields.io/badge/SIMD-AVX2%20%2F%20FMA%20%2F%20F16C-green.svg)](#12-system-sovereign-architecture)
[![Cloud TPU](https://img.shields.io/badge/TPU%20v6e--4-9%2C180%20tok%2Fs%20%7C%202.865%20PFLOPS-red.svg)](#cloud-tpu-v6e-4-trillium-native-c20-benchmarks)
[![Security](https://img.shields.io/badge/Security-SHCP%20v2%20AEAD%20%7C%20Fuzz%20Verified-brightgreen.svg)](#zero-trust-security-architecture)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20TPU-orange.svg)](#building-haven-cpp)
[![License](https://img.shields.io/badge/License-AGPLv3%20%7C%20Commercial-purple.svg)](#license)

`haven-cpp` is a standalone, zero-Python AI inference engine and multi-tenant Cloud TPU server built from scratch in pure C++20. It scales seamlessly from a single local workstation CPU (`AVX2/FMA/F16C`) or Apple Silicon (`MLX`) all the way to bare-metal **Google Cloud TPU v6e (Trillium)** slices via direct `dlopen("libtpu.so")` calls over OpenXLA's PJRT C ABI (`0.75`).

On real 4-chip Cloud TPU `v6e-4` hardware running the full 42-layer Gemma 4 model (`haven-chat-v5.0.gguf`, 262,144 vocabulary, Pleias per-layer injection), `haven-cpp` achieves **2.865 PFLOPS** BF16 MXU compute, **30,181 tok/s** batched prefill, **330.8 tok/s** interactive single-stream decode, **248.3 tok/s** single-user HTTP/SSE streaming (**62.6 ms** multi-turn follow-up TTFT), **2,198.5 tok/s** live multi-tenant concurrent HTTP/SSE serving across 64 simultaneous sockets (**8,258 tok/s** steady-state silicon step), and **9,180.0 tok/s** aggregate continuous-batching throughput (`B=128`) — with zero Python, JAX, or PyTorch runtime dependencies.

---

## Cloud TPU v6e-4 (Trillium) Native C++20 Benchmarks

All results below were measured live on a 4-chip Google Cloud TPU `v6e-4` slice (4x Trillium TensorCores, 128 GB HBM3, 2x2 Optical ICI Torus, 180-vCPU AMD EPYC 9B14 host, `libtpu.so` PJRT API `0.75`) using real `haven-chat-v5.0.gguf` weights (42 layers, `dim=2560`, `ffn=10240`, `vocab=262,144`) and verified against F32 CPU `HavenEngine::forward()` ground truth (`Top1 = 12932 ("Oh") == 12932` across all 16 tested configurations; 4-chip Optical ICI TP 262k logit cosine similarity **`0.982857`**).

### 1. Peak Pure-C++20 StableHLO MXU Saturation (`test-tpu-pjrt-bench`)

| Matrix Shape (BF16) | Active Silicon | Avg Step Latency | Sustained Compute | Per-Chip TFLOPS | Numeric Parity |
| :--- | :--- | ---: | ---: | ---: | :---: |
| `4096 x 4096` | 4x TPU v6e | `0.44 ms` | **1,241.28 TFLOPS** | `310.32` | `1.00 == 1.00` |
| `8192 x 8192` | 4x TPU v6e | `1.70 ms` | **2,585.77 TFLOPS** | `646.44` | `2.00 == 2.00` |
| `16384 x 16384` | **4x TPU v6e** | **`12.28 ms`** | **2,864.69 TFLOPS (2.865 PFLOPS)** | **`716.17`** | `4.00 == 4.00` |

### 2. Full On-Chip 42-Layer Gemma 4 Engine (`HavenTpuGemma4Engine`, `T=2048`)

| Workload | Silicon & Mode | Parity vs CPU | `S=128` MXU Prefill | Step Latency (H2D / TPU / D2H) | Per-Stream | Aggregate Speed |
| :--- | :--- | :---: | ---: | :--- | ---: | ---: |
| Local Workstation CPU | 6-core Xeon E-2236 (AVX2) | Reference | — | `~180–260 ms` | `3.8–5.6 tok/s` | `3.8–5.6 tok/s` |
| Cloud Host CPU | 180-vCPU EPYC 9B14 (`OMP=48`) | Ground Truth | `4,140.6 ms` (133t) | `55.8 ms` | `17.91 tok/s` | `17.91 tok/s` |
| Hybrid Offload (42 PCIe hops) | 4x TPU v6e + EPYC | Match | `3,291.7 ms` (124t) | `52.7 ms` | `18.98 tok/s` | `18.98 tok/s` |
| **On-Chip `B=1` BF16** | 1-Chip BF16 (Argmax) | `0.982747` / Exact | `11.94 ms` (`10,717 tok/s`) | `8.68 ms` (`0.04 / 8.53 / 0.09`) | **115.2 tok/s** | **115.2 tok/s** |
| **On-Chip `B=1` INT8** | 1-Chip W8A16 INT8 (Argmax) | Exact Top-1 | `8.99 ms` (`14,238 tok/s`) | `6.38 ms` (`0.04 / 6.23 / 0.08`) | **156.8 tok/s** | **156.8 tok/s** |
| **4-Chip TP `B=1` BF16** | 4-Chip Optical ICI TP BF16 | **`0.982857`** / Exact | **`6.33 ms` (`20,231 tok/s`)** | `4.33 ms` (`0.14 / 4.06 / 0.12`) | **230.9 tok/s** | **230.9 tok/s** |
| **4-Chip TP `B=1` INT8** | **4-Chip Optical ICI TP INT8** | Exact Top-1 | **`4.24 ms` (`30,181 tok/s`)** | **`3.02 ms`** (`0.13 / 2.76 / 0.11`) | **330.8 tok/s** | **330.8 tok/s** (`362.3` silicon) |
| **Continuous Batch `B=16`** | 4-Chip TP W8A16 INT8 | Exact Top-1 | — | `4.53 ms` (`0.30 / 3.84 / 0.11`) | **220.9 tok/s** | **3,533.9 tok/s** |
| **Continuous Batch `B=32`** | 4-Chip TP W8A16 INT8 | Exact Top-1 | — | `6.21 ms` (`0.50 / 5.05 / 0.12`) | **161.0 tok/s** | **5,153.6 tok/s** |
| **Continuous Batch `B=64`** | 4-Chip TP BF16 (`OnChipEmb`) | Exact Top-1 | — | `8.53 ms` (`0.09 / 8.33 / 0.11`) | **117.2 tok/s** | **7,503.6 tok/s** |
| **Continuous Batch `B=128`** | 4-Chip TP BF16 (`OnChipEmb`) | Exact Top-1 | — | `14.71 ms` (`0.09 / 14.49 / 0.13`) | **68.0 tok/s** | **8,699.3 tok/s** |
| **Continuous Batch `B=128`** | **4-Chip TP INT8 (`OnChipEmb`)** | **Exact Top-1** | — | **`13.94 ms`** (`0.09 / 13.73 / 0.12`) | **71.7 tok/s** | **9,180.0 tok/s** |

### 3. Live Production OpenAI HTTP & Multi-Tenant SSE Serving (`haven-server --tpu`)

Verified end-to-end over real HTTP/1.1 (`POST /v1/chat/completions`, `GET /health`, `GET /api/telemetry`) on 4x TPU `v6e-4` with both single-stream prefix caching (`--tpu-batch 1`) and lock-free multi-tenant continuous batching (`--tpu-batch 16/32/64` via `TpuContinuousBatchScheduler` with per-slot async `pos_vec[B]` and slot-targeted `S=128/32/16/8/1` MXU prefill):

| `haven-server` Launch Flags | Workload & Clients | HTTP TTFT | Wall Time (incl. Prefill) | Live HTTP Throughput | TPU Decode Step |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `--tpu --tpu-int8 --tpu-host-emb` | 1 Client SSE (`B=1`, Full Sampler) | `114.8 ms` | `408.8 ms` (69–74t) | **248.3 tok/s** | `3.02 ms` |
| `--tpu --tpu-int8 --tpu-batch 1` | 1 Client Multi-Turn Follow-up (`S=16/8` bucket) | **`62.6 ms`** | — | **119.7 tok/s** | `4.16 ms` |
| `--tpu --tpu-int8 --tpu-batch 16` | **16 Concurrent Clients SSE** (Full Stochastic Sampler) | `426.1 ms` avg | `2.32 s` (1,536t) | **661.0 tok/s** (`41.3/cli`) | `4.16 ms` (`3,843/s` silicon) |
| `--tpu --tpu-int8 --tpu-batch 32 --tpu-argmax` | **32 Concurrent Clients SSE** (On-Chip Argmax) | `706.0 ms` avg | `2.04 s` (4,096t) | **2,006.3 tok/s** (`62.7/cli`) | `5.22 ms` (`6,133/s` silicon) |
| **`--tpu --tpu-int8 --tpu-batch 64 --tpu-argmax`** | **64 Concurrent Clients SSE** (On-Chip Argmax) | `1,383 ms` avg | `3.73 s` (8,192t) | **2,198.5 tok/s** (`34.4/cli`) | **`7.75 ms`** (`8,258/s` silicon) |

*Async per-slot parity (`test-tpu-gemma4-onchip --async-slots-only`) verified 6/6 slots on real `v6e-4` silicon (`0.948–0.985` prefill cosine and `0.930–0.957` lockstep async Step-2 cosine across 262,144 logits vs F32 CPU for 3 distinct prompts at positions `133, 107, 106`, with `23.96 ms` 106-token slot prefill and zero cross-slot leakage).*

See [BENCHMARKS.md](BENCHMARKS.md) for the complete `.hmb` memory scaling, AVX2/F16C SIMD (`34.81 GFLOPS`, `3.64x` speedup), sampler (`0.15 ms/tok`, `150x` speedup), plugin, and security verification matrices.

---

## Architecture

### Model-General Native Pure-C++20 TPU StableHLO Compiler (`HavenTpuGemma4Engine` + `TpuContinuousBatchScheduler`)

`src/haven_tpu_gemma4.cpp`, `src/haven_tpu_scheduler.cpp`, and `src/haven_tpu_pjrt.cpp` compile dense transformer architectures directly from `ModelConfig` + GGUF tensors into persistent StableHLO MLIR executables on TPU HBM3 with zero per-layer host round-trips:

- **Model-General Dense Transformer Compilation (`TpuModelSpec`):** Auto-derives layer topology directly from GGUF metadata and tensors for **Gemma 4** (SWA-512 + 4-Sink & Global RoPE, shared-KV, Pleias, GeGLU), **Gemma 2/3 & Ava 26B** (attention & final logit softcapping), **Llama 3/4** (Global GQA, SwiGLU/SiLU, untied embeddings), **Qwen 2.5/3** (QKV biases / QK-norm, SwiGLU/SiLU), **Mistral** (SWA-only, SwiGLU/SiLU), and **Phi-3/4** — CI-verified across 160 synthetic multi-arch StableHLO graphs (`--emit-only-synthetic`) on Linux GCC & Windows MSVC.
- **Per-Slot `[B]` Async Positions & Slot-Targeted Multi-Bucket Prefill (`S=128/32/16/8/1`):** Decode graphs accept independent per-slot positions (`%pos_vec: tensor<Bxi32>`) with per-slot RoPE gather, per-slot causal/SWA masks, and in-place `stablehlo.scatter` KV updates; prefill graphs target a single `%slot_id` via `dynamic_slice`/`dynamic_update_slice` without disturbing any other active streaming user.
- **Multi-Tenant Continuous-Batching Scheduler (`TpuContinuousBatchScheduler`):** Dedicated TPU worker thread multiplexes up to `128` concurrent OpenAI `/v1/chat/completions` HTTP/SSE streams lock-free (zero `engine_mutex_` contention), with parallel OpenMP per-slot stochastic sampling or 4-byte/slot on-chip argmax and immediate slot reclamation on client disconnect.
- **4-Chip Optical ICI Megatron Tensor Parallelism:** Emits native `stablehlo.all_reduce` / `stablehlo.all_gather` over the 2x2 Trillium torus, automatically querying `PJRT_LoadedExecutable_AddressableDevices` to match XLA's physical ring assignment (`{0, 1, 3, 2}`).
- **TPU `(8, 128)`-Aligned Donated KV Cache & W8A16 INT8:** Stores KV tensors as `[N_kv, B, Hkv_sh, T, HD]` with `{tf.aliasing_output}` in-place HBM3 buffer donation and OpenMP row-wise symmetric INT8 weights (~`4.4–7.6 GB` total).

### 12-System Sovereign Architecture

```
                         ┌─────────────────────────────────────────────────────────┐
                         │                 Haven C++20 Core Engine                 │
                         └────────────────────────────┬────────────────────────────┘
                                                      │
           ┌──────────────────────────────────────────┼──────────────────────────────────────────┐
           │                                          │                                          │
┌──────────▼──────────┐                    ┌──────────▼──────────┐                    ┌──────────▼──────────┐
│  Native Cloud TPU   │                    │ 64-Bit HMB Memory   │                    │  Pathways Async &   │
│  StableHLO / PJRT   │                    │  In-Attention DMA   │                    │  SHCP v2 Cluster    │
│  9,180 tok/s · v6e  │                    │  100k recs / 9.8ms  │                    │  ChaCha20-Poly1305  │
└──────────┬──────────┘                    └──────────┬──────────┘                    └──────────┬──────────┘
           │                                          │                                          │
┌──────────▼──────────┐                    ┌──────────▼──────────┐                    ┌──────────▼──────────┐
│ AVX2/FMA/F16C SIMD  │                    │ PagedAttention COW  │                    │ YaRN Dynamic RoPE   │
│ 34.81 GFLOPS · Q4_K │                    │ Zero-Copy Seq Fork  │                    │ 4.0x -> 32k Context │
└──────────┬──────────┘                    └──────────┬──────────┘                    └──────────┬──────────┘
           │                                          │                                          │
┌──────────▼──────────┐                    ┌──────────▼──────────┐                    ┌──────────▼──────────┐
│ Speculative Draft   │                    │ FlashAttention CPU  │                    │ Chunked Prefill &   │
│ Parallel Verification│                   │ Cache-Tiled Softmax │                    │ Exact Prefix Cache  │
└──────────┬──────────┘                    └──────────┬──────────┘                    └──────────┬──────────┘
           │                                          │                                          │
┌──────────▼──────────┐                    ┌──────────▼──────────┐                    ┌──────────▼──────────┐
│ Multimodal Sensory  │                    │ Structured Knowledge│                    │ Top-Tail DRY Sampler│
│ Vision + Mel Audio  │                    │ Composite Triples   │                    │ 0.15 ms / 262k Vocab│
└─────────────────────┘                    └─────────────────────┘                    └─────────────────────┘
```

1. **Native Cloud TPU PJRT / StableHLO Backend**: Zero-Python `dlopen("libtpu.so")` runtime supporting 1-, 2-, and 4-chip Optical ICI Tensor Parallelism and direct HTTP/SSE serving.
2. **Vectorized AVX2/FMA/F16C Math Kernels**: Bare-metal SIMD GEMV for `Q4_K`, `Q5_K`, `Q6_K`, `Q3_K`, `Q4_0`, `Q5_0`, `Q8_0`, `F16`, `BF16`, and `F32` (**34.81 GFLOPS** vectorized throughput, 3.64x faster than scalar dequant).
3. **64-Bit Haven Memory Bank (`.hmb`)**: Single-stream contiguous binary vector vault (`71.3 MB` for 100,000 memories, `9.86 ms` global AVX2 scan, `1.27 µs` active recall) with in-attention DMA projection.
4. **Hardware Topology Fabric**: Automatic Linux PCI sysfs (`0x1ae0:0x006f`) TPU torus discovery alongside host CPU and Apple MLX backends.
5. **Clean-Room Pathways Async Engine**: Non-blocking task scheduler exceeding **1.11M tasks/sec** (`0.90 µs` dispatch latency).
6. **PagedAttention & Prefix KV Cache**: Copy-on-Write zero-overhead sequence forking plus exact prefix KV reuse with clean truncation (`DynamicKVCacheManager::truncate()`).
7. **YaRN Dynamic RoPE Context Scaler**: Dynamic frequency interpolation and `mscale` scaling to 32k+ tokens.
8. **Multi-Token Speculative Decoding**: Parallel candidate drafting and verification.
9. **FlashAttention CPU Tiled Softmax**: Cache-tiled online-softmax attention with zero `N x N` heap allocations.
10. **Chunked Prefill Scheduler**: Batched 128-token prefill chunking without blocking interactive turns.
11. **Multimodal Sensory Bridge & Knowledge Graph**: Live vision patch (768-dim) and audio mel-spectrum (128-dim) injection plus relational `(Subject -> Predicate -> Object)` logit boosting.
12. **Top-Tail 262k Persona & DRY Sampler**: Top-tail cutoff + `std::partial_sort` sampling (**0.15 ms/tok** on 262,144 vocab — 150x faster than full-vocab sorting) with temperature, top-k, top-p, min-p, repetition penalty, DRY n-gram anti-stutter, and lemma-family deduplication.

---

## Zero-Trust Security Architecture

`haven-cpp` is engineered and continuously audited for zero-trust local and distributed deployment:

- **SHCP v2 Authenticated & Encrypted Cluster Protocol**: Mutual HMAC-SHA256 challenge-response with 256-bit zero-trust pairing tokens, followed by **RFC 8439 ChaCha20-Poly1305 AEAD** wire encryption with strict monotonic anti-replay counters and immediate drop on ciphertext tampering or plaintext downgrade attempts (`src/haven_cluster.cpp`, `include/haven/haven_security.h`).
- **Hardened HTTP/SSE Micro-Server**: Defaults strictly to loopback (`127.0.0.1`), enforces DNS-rebinding `Host` checks, loopback-only CORS/CSRF origin validation, optional constant-time Bearer API key authentication (`--api-key` / `HAVEN_API_KEY`), Slowloris read/write socket timeouts, canonical path containment against traversal, and non-blocking client-disconnect abort guards (`0.00 s` wasted post-cancel CPU).
- **Strict RFC 8259 JSON & Binary Bounds Safety**: Custom recursive-descent RFC 8259 JSON parser (`include/haven/haven_json.h`) with depth limits and hardened GGUF v3 / `.hmb` loaders defending against integer overflow, truncated headers, and OOB offsets.
- **OS-Enforced Plugin Sandboxing**: Terminal plugin enforces Win32 Job Objects / POSIX `rlimit` process containment; browser plugin blocks DNS-rebinding/SSRF private IP ranges; workspace and vault plugins enforce canonical root containment.
- **Automated Adversarial Verification (`test-security-fuzz`)**: 6-suite suite covering FIPS 180-4 SHA-256, RFC 4231 HMAC-SHA256, RFC 8439 ChaCha20-Poly1305 KATs, live SHCP v2 wire tamper/replay/downgrade attacks, 400 mutated JSON/GGUF/HMB fuzz inputs, and live plugin sandbox checks — **100% pass with 0 ASan/UBSan findings**.

---

## Sovereign Dynamic Plugins (17 Built-In Plugins)

Plugins consume **0 bytes of RAM at boot** and hot-load lazily on demand, with automatic idle memory reclamation back to cold storage:

| Plugin | Capability | Description |
| :--- | :--- | :--- |
| **Workspace & Code Assistant** | Action & Memory | Canonical-root-jailed source inspection, directory listing, and codebase search. |
| **Terminal & Shell Sandbox** | Action Tool | OS-enforced Job Object / `rlimit` sandboxed shell command execution. |
| **Windows Clipboard Bridge** | Sensory & Action | Reads and writes clipboard text, URLs, and code snippets. |
| **Relational Knowledge Graph** | Cognitive Memory | Persistent `(Subject -> Predicate -> Object)` SQLite triple store. |
| **Whisper Neural Audio Ear** | Sensory Input | Real-time microphone and audio transcription via Whisper C++. |
| **Home Assistant & IoT** | Action Tool | Smart lights, switches, and ambient lighting control via Home Assistant REST API. |
| **Sovereign Web Browser** | Action & Memory | SSRF-guarded HTTP fetching, HTML-to-markdown DOM reading, and browser control. |
| **Steam & Gaming Companion** | Sensory & Action | Detects active PC/Steam games and logs gameplay milestones. |
| **Circadian Rhythm Caretaker** | Sensory & Care | Tracks day/night cycles, fatigue logs, and time-aware greetings. |
| **YouTube Music Controller** | Action Tool | Desktop YouTube Music playback and track search control. |
| **Stable Diffusion C++ Art** | Creative Tool | Local artwork generation via Stable Diffusion C++ (Port 8085). |
| **Kokoro Neural Voice Synth** | Voice Expression | Low-latency neural speech synthesis via Kokoro TTS (Port 8089). |
| **Desktop Screen Vision** | Sensory Input | Foreground window title and screen geometry telemetry. |
| **Obsidian Vault & Diary** | Memory & Journal | Canonical-jailed markdown vault search and companion diary sync. |
| **Web Research & Wikipedia** | Intelligence | Real-time Wikipedia summaries and weather lookups. |
| **System & Environment Sensory** | Telemetry | Local time, RAM usage, CPU load, and battery/power telemetry. |
| **Discord Sanctuary Bridge** | Communication | Single-channel Discord companion bridge with multi-user routing. |

---

## Universal C ABI Exports (P/Invoke & JNI)

`libhaven_engine.so` and `haven_engine.dll` export a stable C ABI for C# (.NET) and Android Kotlin (JNI):

```cpp
HAVEN_API void*    haven_create_engine();
HAVEN_API bool     haven_load_model(void* engine, const char* filepath);
HAVEN_API void     haven_forward(void* engine, uint32_t token, int pos, float* out_logits, uint32_t active_cluster);
HAVEN_API uint32_t haven_sample_token(void* engine, float* logits, uint32_t vocab_size);
HAVEN_API void     haven_destroy_engine(void* engine);

// Hardware Topology & Pathways Telemetry
HAVEN_API size_t   haven_get_node_count(void* engine);
HAVEN_API bool     haven_get_node_info(void* engine, uint32_t index, char* out_name, int max_name_len, uint32_t* out_backend, uint64_t* out_mem_free_mb);
HAVEN_API uint64_t haven_get_pathways_completed_tasks(void* engine);

// Cognitive Memory, Multimodal Sensory & Attention Telemetry
HAVEN_API void     haven_inject_memory(void* engine, const char* concept_name, float weight, const float* embedding, int dim);
HAVEN_API void     haven_inject_vision_frame(void* engine, const float* patch_embeddings, int num_patches, uint32_t width, uint32_t height, float salience);
HAVEN_API void     haven_inject_audio_spectrum(void* engine, const float* mel_bands, int num_bands, float energy_rms, float centroid, bool is_speech);
HAVEN_API size_t   haven_prune_kv_cache(void* engine);
HAVEN_API float    haven_get_attention_entropy(void* engine);
HAVEN_API void     haven_set_yarn_scale(void* engine, float scale);
HAVEN_API uint64_t haven_fork_sequence(void* engine, uint64_t parent_id, uint64_t child_id);
```

---

## Building `haven-cpp`

### Prerequisites
- **CMake 3.20+** & **Ninja** (recommended)
- **C++20 Compiler**: MSVC / Clang 14+ (Windows) or GCC 11+ (Linux)
- **AVX2 / FMA / F16C** on host CPU
- *(Optional, Cloud TPU hosts only)* Google `libtpu.so` (`pip install libtpu -f https://storage.googleapis.com/jax-releases/libtpu_releases.html` — used solely to install the shared library; `haven-cpp` loads it via `dlopen()` with zero Python runtime).

### Windows Build
```powershell
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8
```

### Linux & Cloud TPU VM Build
```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j $(nproc)
```

---

## Running `haven-cpp`

### 1. OpenAI-Compatible Streaming Server (`haven-server`)

**Local CPU (AVX2/F16C, loopback-guarded by default):**
```bash
./build/haven-server --port 11436 --model path/to/haven-chat-v5.0.gguf
```

**Cloud TPU v6e On-Chip Serving (Single-User Interactive & Multi-Tenant Continuous Batching):**
```bash
# 1. Peak interactive B=1 HTTP/SSE streaming (248.3 tok/s, 62.6 ms follow-up TTFT)
./build/haven-server --port 11436 --model /tmp/haven-chat-v5.0.gguf \
  --tpu --tpu-int8 --tpu-host-emb

# 2. Multi-tenant continuous batching (16 concurrent clients, full stochastic sampling: 661 tok/s live SSE)
./build/haven-server --port 11436 --model /tmp/haven-chat-v5.0.gguf \
  --tpu --tpu-int8 --tpu-batch 16

# 3. High-throughput multi-tenant serving (32–64 concurrent clients, on-chip argmax: 2,006–2,198 tok/s live SSE)
./build/haven-server --port 11436 --model /tmp/haven-chat-v5.0.gguf \
  --tpu --tpu-int8 --tpu-batch 64 --tpu-argmax
```

**Key `haven-server` Flags:**
- `--host <ip>`, `--port <port>` — Bind address (default `127.0.0.1:11436`).
- `--api-key <key>` — Enforce `Authorization: Bearer <key>` on API routes (or set `HAVEN_API_KEY`).
- `--tpu` / `--tpu-onchip` — Enable model-general full on-chip TPU HBM3 serving (or set `HAVEN_TPU_ONCHIP=1`; automatically falls back to AVX2 CPU when no TPU is present).
- `--tpu-batch <1..128>` — Concurrent HBM3 KV slots (or `HAVEN_TPU_BATCH`; `1` = interactive w/ prefix cache, `2..128` = lock-free multi-tenant continuous batching).
- `--tpu-replicas <1|2|4>` — Tensor-parallel TPU chips over Optical ICI (default: auto).
- `--tpu-int8` / `--tpu-bf16` — Select W8A16 INT8 (default) or BF16 HBM3 weights.
- `--tpu-argmax` — Use on-chip TPU argmax (`4B/slot` PCIe D2H) for peak multi-tenant throughput.
- `--tpu-host-emb` — Use host-side embedding lookup for lowest single-stream `B=1` latency.
- `--tpu-max-seq <N>` — Static HBM3 KV cache capacity (`256..8192`, default `2048`).
- `--libtpu <path>` — Explicit path to `libtpu.so` (auto-discovered if omitted).
- `--cluster-host [port]` / `--cluster-join <invite>` — SHCP v2 AEAD distributed cluster coordinator or worker.

### 2. Interactive Companion CLI & Benchmark Suite (`haven-cli`)
```bash
./build/haven-cli path/to/haven-chat-v5.0.gguf          # Interactive Aura REPL
./build/haven-cli --bench path/to/haven-chat-v5.0.gguf  # 12-system benchmark suite
./build/haven-cli --tpu path/to/haven-chat-v5.0.gguf    # Hardware topology & TPU probe
```

### 3. TPU Silicon, Scheduler & Security Verification Binaries
```bash
# Offline zero-GGUF multi-arch StableHLO IR verification (Gemma 4, Llama 3, Qwen 2.5, Mistral, Gemma 2 — 160 graphs!)
./build/test-tpu-gemma4-onchip --emit-only-synthetic

# Offline multi-tenant continuous-batching scheduler concurrency & cancellation test
./build/test-tpu-scheduler

# Full Cloud TPU v6e async per-slot parity + 1-chip/4-chip + B=1..128 benchmark suite
./build/test-tpu-gemma4-onchip /tmp/haven-chat-v5.0.gguf

# Pure C++20 StableHLO BF16 MXU PFLOPS & GeGLU/LM-Head benchmark
./build/test-tpu-pjrt-bench /tmp/haven-chat-v5.0.gguf

# 6-suite adversarial crypto KAT, SHCP v2 attack, JSON/GGUF/HMB fuzz & plugin sandbox suite
./build/test-security-fuzz
./build/test-secure-cluster
```

---

## Model Compatibility

`haven-cpp` natively runs Gemma 4 (including Pleias per-layer embeddings, SWA-512 + 4-Sink + Global RoPE, shared-KV layers, and 262k vocab), Qwen 2.5, LLaMA 3, Mistral, and Phi-3 GGUF models, plus Apple MLX safetensors directories:

- **[haven-ai-companion/haven-chat-v5.0](https://huggingface.co/haven-ai-companion/haven-chat-v5.0)** — `haven-chat-v5.0.gguf` (4.94 GB Gemma 4 E4B, primary benchmark & companion model)
- **[ssfdre38/ava](https://huggingface.co/ssfdre38/ava)** — `ava-26b-turbo-multimodal-iq4xs.gguf` (14.21 GiB) / `ava-26b-turbo-iq4xs-v2.gguf` (13.09 GiB)

---

## License

**Copyright (C) 2026 Daniel Elliott.**

`haven-cpp` is **dual-licensed**:

- **Open Source — GNU Affero General Public License v3.0 (AGPLv3):** Free to use, modify, and distribute for personal, academic, research, and AGPLv3-compliant open-source projects. See [LICENSE](LICENSE).
- **Commercial License:** Required for closed-source embedding (`haven_engine.dll` / `libhaven_engine.so`), proprietary SaaS / Cloud TPU inference APIs without AGPLv3 source disclosure, commercial OEM appliances, or enterprise support and SLAs. See [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md).

Third-party attributions (including OpenXLA's Apache-2.0 `pjrt_c_api.h`) are listed in [NOTICE](NOTICE). Contribution and CLA terms are in [CONTRIBUTING.md](CONTRIBUTING.md).
