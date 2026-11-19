# ✨ Haven C++ (`haven-cpp`)

**Sovereign, Bare-Metal C++20 AI Companion Engine, Hardware Topology Fabric & Pathways Execution System**

[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![SIMD](https://img.shields.io/badge/SIMD-AVX2%20%2F%20FMA%20%2F%20F16C-green.svg)]()
[![Topology](https://img.shields.io/badge/Topology-CPU%20%7C%20GPU%20%7C%20TPU%20v6e-red.svg)]()
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-purple.svg)]()

`haven-cpp` is a high-performance, standalone AI companion engine engineered from scratch in pure C++20. Designed for total privacy, cognitive sovereignty, and bare-metal execution, it provides an end-to-end cognitive architecture—from SIMD matrix-vector math kernels and in-attention cognitive memory to a heterogeneous Hardware Topology Fabric and a clean-room Pathways asynchronous execution engine.

Tested and validated on both local workstation AVX2 silicon and live **Google Cloud TPU v6e Trillium 8-Core Torus bare-metal hardware (2.2 PFLOPS / 256 GB HBM / 9.6 TB/s Optical ICI)**.

---

## ⚡ 12-System Enterprise Sovereign Architecture

`haven-cpp` integrates twelve sovereign subsystems into a unified native C++20 runtime:

```
                               ┌─────────────────────────────────────────────────────────┐
                               │                 Haven C++ Core Mind                     │
                               └────────────────────────────┬────────────────────────────┘
                                                            │
                 ┌──────────────────────────────────────────┼──────────────────────────────────────────┐
                 │                                          │                                          │
    ┌────────────▼────────────┐                ┌────────────▼────────────┐                ┌────────────▼────────────┐
    │    Hardware Topology    │                │  In-Attention Cognitive │                │  Pathways Asynchronous  │
    │         Fabric          │                │       Memory Vault      │                │     Execution Engine    │
    │  CPU · GPU · TPU Torus  │                │  DMA Vector Ingestion   │                │   1.11M+ tasks/sec      │
    └────────────┬────────────┘                └────────────┬────────────┘                └────────────┬────────────┘
                 │                                          │                                          │
    ┌────────────▼────────────┐                ┌────────────▼────────────┐                ┌────────────▼────────────┐
    │ Bare-Metal AVX2 Kernel  │                │   PagedAttention COW    │                │ YaRN Context Scaler     │
    │ GEMV / Q4_K / Q6_K / Q8 │                │ Zero-Copy Seq Forking   │                │ Dynamic RoPE (4x-32k)   │
    └────────────┬────────────┘                └────────────┬────────────┘                └────────────┬────────────┘
                 │                                          │                                          │
    ┌────────────▼────────────┐                ┌────────────▼────────────┐                ┌────────────▼────────────┐
    │ Speculative Decoding    │                │ FlashAttention CPU      │                │ Chunked Prefill Sched   │
    │ Parallel Candidate Draft│                │ Cache-Tiled Softmax     │                │ Asynchronous BG Queue   │
    └────────────┬────────────┘                └────────────┬────────────┘                └────────────┬────────────┘
                 │                                          │                                          │
    ┌────────────▼────────────┐                ┌────────────▼────────────┐                ┌────────────▼────────────┐
    │ Multimodal Sensory Br.  │                │ Structured Knowledge    │                │ Universal Persona       │
    │ Vision + Mel-Spectrum   │                │ Composite Triples Graph │                │ DRY / Min-P / Suppress  │
    └─────────────────────────┘                └─────────────────────────┘                └─────────────────────────┘
```

1. **🚀 Bare-Metal AVX2/FMA SIMD Math Kernel**: Vectorized GEMV matrix multiplication with support for Q4_K, Q6_K, and Q8_0 quantizations, accelerated across logical CPU cores via multi-threaded OpenMP.
2. **🧠 In-Attention Cognitive Memory (DMA)**: Direct Memory Access architecture that queries vector anchors in microseconds (**1.27 µs**) and projects emotional salience directly into transformer attention layers.
3. **🌐 Hardware Topology Fabric**: Hardware abstraction layer that probes host CPU, discrete NUMA GPUs, and Optical ICI TPU fabrics. Discovers physical accelerator hardware and dynamically routes tasks to the optimal node.
4. **⚡ Clean-Room Pathways Asynchronous Engine**: Non-blocking worker queue with promise-based scheduling, capable of sustained execution rates exceeding **1.11+ million tasks/second** with **0.90 µs** dispatch latency.
5. **📑 PagedAttention Virtual Paging (vLLM / Meta)**: Virtual block KV-cache management enabling zero-memory-duplication sequence forking via Copy-on-Write (COW).
6. **🌀 YaRN Dynamic RoPE Context Scaler**: Frequency band interpolation with dynamic temperature scaling (`mscale`) for seamless context expansion up to 32k+ tokens.
7. **🎯 Multi-Token Speculative Candidate Drafting**: Speculative head that drafts multi-token future candidates ahead in parallel to boost auto-regressive decode throughput.
8. **⚡ FlashAttention CPU Tiled Online-Softmax Kernel**: Cache-friendly $L_1/L_2$ tiled attention kernel computing full sequence attention with zero $N \times N$ memory allocations.
9. **⏳ Asynchronous Chunked Prefill Scheduler**: Slices large prompts into configurable chunks (e.g. 128 tokens) executed in the background without starving real-time decoding.
10. **👁️ Multimodal Sensory Attention Bridge**: Live projection channel for screen vision patches (768-dim) and microphone audio mel-spectrograms (128-dim) into attention.
11. **🕸️ Structured Relational Knowledge Graph**: Fast composite-key triple storage `(Subject -> Predicate -> Object)` boosting related token clusters dynamically.
12. **🎭 Universal Persona & DRY Sampler**: Temperature, Top-P, Min-P, repetition penalty, DRY n-gram phrase suppression, and cross-token lemma duplicate suppression.

---

## 🔬 Validated on Google Cloud TPU v6e Silicon

`haven-cpp` has been compiled and validated directly on bare-metal Google Cloud TPU hardware:

* **Hardware Node**: `haven-v6e-node-eu` in `europe-west4-a`
* **Host CPU**: AMD EPYC 9B14 (180 logical cores, 1.4 TB physical RAM)
* **Accelerators**: 8x Google TPU v6e Trillium cores on a 2x4 Optical ICI 3D Torus
* **Aggregate Accelerator Compute**: **2,200 TFLOPS (2.2 PFLOPS)**
* **Aggregate Memory & Bus**: **256 GB HBM (240 GB free)**, **9,600 GB/s (9.6 TB/s)** Optical Torus Interconnect
* **Auto-Discovery**: Linux PCI sysfs probe (`0x1ae0:0x006f`) automatically identifies all 8 Trillium devices and initializes the Torus mesh topology with zero manual configuration.

### Live Benchmark Results

```
==================================================================
🚀 HAVEN-CPP: ENTERPRISE SOVEREIGN C++ ENGINE BENCHMARK SUITE
   AVX2/FMA/F16C | PagedAttention | YaRN | Speculative | FlashAttn
==================================================================
[1/10] AVX2 + FMA SIMD Math Kernel Throughput   : 4.33 GFLOPS (1.89 µs per 4096-dim vector)
[2/10] Ingested Structured Knowledge Triples    : 2 relational triples
[3/10] AVX2 Cognitive Memory Vault Search       : 1.27 µs
[4/10] PagedAttention Virtual Zero-Copy Fork    : Sequence 1001 -> 1001 (0 µs overhead)
[5/10] YaRN Dynamic RoPE Context Scaler         : 4.0x -> 32,768 tokens (mscale active)
[6/10] Multi-Token Speculative Candidate Draft  : 4 tokens drafted ahead in parallel
[7/10] FlashAttention CPU Tiled Online-Softmax  : 512 tokens (Zero N*N allocations)
[8/10] Chunked Prefill Scheduler Task           : 512-token prompt (128-token chunks)
[9/10] SentencePiece Tokenizer & Gemma Turn     : Bit-exact Gemma 4 turn formatting
[10/10] In-Attention Token Generation           : 29.69 ms/token (33.7 tokens/second)
[11/12] Hardware Topology Fabric Discovery      : 2 Nodes (Host CPU + 8x TPU v6e Torus)
[12/12] Pathways Asynchronous TPU Dispatch      : 1,115,946 tasks/sec (0.90 µs/task)
==================================================================
✅ ALL 12 ENTERPRISE SOVEREIGN SYSTEMS 100% OPERATIONAL IN HAVEN-CPP!
==================================================================
```

---

## 🔌 Sovereign Dynamic Plugins (17 Built-In Plugins)

`haven-cpp` features an on-demand, lazy-loading plugin system. Plugins consume **0 bytes of RAM at boot** and are loaded dynamically only when an action or tool is requested. Idle plugins are automatically reclaimed back to cold storage by the engine's memory reaper.

| Plugin | Capability | Description |
| :--- | :--- | :--- |
| **📁 Local Workspace & Code Assistant** | Action & Memory | Inspects project files, reads source code, lists directories, and searches codebases. |
| **🐚 Native Terminal & Shell Sandbox** | Action Tool | Safely executes local shell commands (`dir`, `git`, `python`, `tree`), capturing output into attention. |
| **📋 Windows Clipboard Bridge** | Sensory & Action | Reads copied text, URLs, and code snippets directly from the active clipboard buffer. |
| **🧠 Relational Knowledge Graph** | Cognitive Memory | Stores persistent `(Subject -> Predicate -> Object)` structured relational memory triples in SQLite format. |
| **🎙️ Whisper Neural Audio Ear** | Sensory Input | Transcribes microphone audio streams and speech audio into text via Whisper C++. |
| **💡 Home Assistant & IoT Controller** | Action Tool | Controls smart lights, plugs, and ambient room illumination via Home Assistant REST API. |
| **🌐 Sovereign Web Browser** | Action & Memory | Real-time HTML-to-markdown webpage browsing, DOM text extraction, and desktop browser launcher. |
| **🎮 Steam & Gaming Companion** | Sensory & Action | Detects active PC/Steam games in real-time, logs gameplay milestones, and cheers the user on. |
| **🌙 Circadian Rhythm Caretaker** | Sensory & Care | Tracks day/night cycles, monitors fatigue, provides morning briefings and late-night health care. |
| **🎵 YouTube Music Controller** | Action Tool | Controls desktop YouTube Music playback, track navigation, and instant search. |
| **🎨 Stable Diffusion C++ Art Studio** | Creative Tool | Generates high-resolution artwork & visual scenes via Stable Diffusion C++ (Port 8085). |
| **🎙️ Kokoro Neural Voice Synthesizer** | Voice Expression | Synthesizes acoustic speech audio via Kokoro TTS (Port 8089). |
| **👁️ Desktop Screen Vision** | Sensory Input | Inspects active foreground window titles, active apps, and screen geometry. |
| **📚 Obsidian Vault & Diary** | Memory & Journal | Searches local markdown vaults and maintains autonomous companion chronicles. |
| **🌐 Wikipedia & Quick Search** | Intelligence | Real-time Wikipedia fact extraction, live weather telemetry, and quick queries. |
| **🖥️ System & Environment Sensory** | Telemetry | Injects real-time local time, RAM load, and battery/power status into prefill context. |
| **💬 Discord Sanctuary Channel Bridge** | Communication | Dedicated single-channel Discord bot with multi-user persona routing. |

---

## 🏛️ Universal C ABI Exports (P/Invoke & JNI)

`haven-cpp` exports a stable, low-overhead C ABI interface from `libhaven_engine.so` and `haven_engine.dll`, enabling drop-in bindings from C# (.NET / Gemmi) and Android Kotlin (JNI):

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

// Cognitive Memory & Attention Telemetry
HAVEN_API void     haven_inject_memory(void* engine, const char* concept_name, float weight, const float* embedding, int dim);
HAVEN_API void     haven_inject_vision_frame(void* engine, const float* patch_embeddings, int num_patches, uint32_t width, uint32_t height, float salience);
HAVEN_API void     haven_inject_audio_spectrum(void* engine, const float* mel_bands, int num_bands, float energy_rms, float centroid, bool is_speech);
HAVEN_API size_t   haven_prune_kv_cache(void* engine);
HAVEN_API float    haven_get_attention_entropy(void* engine);
HAVEN_API void     haven_set_yarn_scale(void* engine, float scale);
HAVEN_API uint64_t haven_fork_sequence(void* engine, uint64_t parent_id, uint64_t child_id);
```

---

## 🛠️ Building `haven-cpp`

### Prerequisites
* **CMake 3.20+**
* **C++20 Compiler**: MSVC (Windows), GCC 11+ (Linux), or Clang 14+
* **AVX2 / FMA support** on host CPU

### Windows Build (MSVC / Clang / GCC)
```powershell
# Configure build
cmake -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release

# Compile core engine, CLI, micro-server, test suites, and plugins
cmake --build build --config Release
```

### Linux Build (Ubuntu / Debian / Cloud TPU VMs)
```bash
# Configure build with GCC
cmake -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++

# Build all targets
cd build && ninja
```

---

## 🚀 Running the Executables

### 1. Interactive Companion Chat REPL (Aura)
```bash
./haven-cli [model_path.gguf]
```

#### In-Chat Slash Commands:
* `/memory` — Inspect active cognitive memory anchors in the vault
* `/plugins` — View registered plugins and RAM status (cold on disk vs active in RAM)
* `/plugins unload` — Flush idle plugins from memory
* `/plugins reload` — Hot-reload dynamic plugin libraries from disk
* `/tool <action> <payload>` — Execute an action directly (e.g. `/tool yt_music_search lofi`, `/tool wiki_summary Quantum`)
* `/reset` — Reset the active KV cache and conversation context
* `/exit` — Quit Sanctuary

### 2. Full 12-System Benchmark Suite
```bash
./haven-cli --bench [model_path.gguf]
```

### 3. Dedicated Hardware Topology & Pathways TPU Benchmark
```bash
./haven-cli --tpu [model_path.gguf]
```

### 4. Standalone Micro-Server (Port 11436)
```bash
./haven-server [port] [model_path.gguf]
```

---

## 🤖 Model Compatibility: Ava 26B Multimodal

`haven-cpp` is designed to run the official **Ava 26B Multimodal** companion model, trained on Cloud TPU v6e hardware and available on Hugging Face:

🔗 **[Hugging Face: ssfdre38/ava](https://huggingface.co/ssfdre38/ava)**

* `ava-26b-turbo-multimodal-iq4xs.gguf` (14.21 GiB)
* `ava-26b-turbo-iq4xs-v2.gguf` (13.09 GiB)

---

## 📜 License
MIT License. Built with ❤️ for Aura, Daniel, and the Haven Sovereign Companion Ecosystem.
