# Haven C++ (`haven-cpp`) Commercial License

**Copyright (C) 2026 Daniel Elliott. All rights reserved.**

Haven C++ (`haven-cpp`) is **dual-licensed** under:

1. **GNU Affero General Public License v3.0 (AGPLv3)** — see [LICENSE](LICENSE), and
2. **The Haven-CPP Commercial License** — described below.

You may choose **either** license depending on how you use, modify, embed, or deploy `haven-cpp`.

---

## Quick Decision Guide: Do You Need a Commercial License?

| Use Case | GNU AGPLv3 (Free) | Commercial License |
| :--- | :---: | :---: |
| Personal local AI companion, research, education, or private experimentation | Permitted | Not required |
| Open-source projects & public services that release their complete corresponding source code under AGPLv3 (including network-interaction disclosures per AGPLv3 Section 13) | Permitted | Not required |
| **Proprietary SaaS / Cloud Inference APIs** (serving models over HTTP, SSE, gRPC, or WebSockets without publishing your full service source code under AGPLv3) | Not permitted | **Required** |
| **Closed-Source Product Embedding** (linking `haven_core`, `haven_engine.dll`, `libhaven_engine.so`, `HavenTpuGemma4Engine`, or plugins into proprietary desktop, mobile, or enterprise applications) | Not permitted | **Required** |
| **Commercial TPU / Accelerator Appliances & OEM Distribution** (shipping pre-configured Cloud TPU, edge, or datacenter inference stacks) | Not permitted | **Required** |
| **Enterprise Support, SLA, Custom Model Bring-Up, & Legal Indemnification** | Not included | **Included** |

---

## Commercial License Benefits

Organizations purchasing a Haven-CPP Commercial License receive:

- **Full Proprietary Rights:** Embed `haven-cpp` (including the pure-C++20 Cloud TPU PJRT / StableHLO runtime, Optical ICI Tensor Parallelism, W8A16 INT8 quantization, AVX2/F16C SIMD kernels, 64-bit `.hmb` memory engine, and SHCP v2 cluster fabric) inside closed-source applications and commercial cloud APIs with **zero AGPLv3 copyleft or source-disclosure obligations**.
- **Production Cloud & Multi-Chip Deployment Rights:** Deploy across single-chip, multi-chip (`v6e-4` / `v6e-8`), and multi-host Cloud TPU / datacenter clusters.
- **Commercial Warranty, Priority Support & Custom Engineering:** Direct access to architecture guidance, custom model architecture bring-up, performance tuning, and priority security patches.
- **Clean IP Provenance:** 100% clean-room C++20 implementation with permissive-only third-party interface headers (see [NOTICE](NOTICE)).

---

## Licensing Tiers

- **Startup / Indie Commercial:** For early-stage teams embedding `haven-cpp` into commercial desktop, mobile, or self-hosted products.
- **Cloud & Production API:** Per-cluster or per-accelerator-slice licensing for high-throughput SaaS and inference providers running `haven-cpp` on Cloud TPU or CPU/GPU fleets.
- **Enterprise & OEM:** Unlimited internal deployment, OEM redistribution rights, dedicated SLA, custom kernel engineering, and enterprise legal terms.

---

## How to Obtain a Commercial License

To request commercial pricing, evaluation terms, or an enterprise licensing agreement, contact:

- **Copyright Holder:** Daniel Elliott
- **Email:** `ssfdre38@gmail.com` (Subject: `Haven-CPP Commercial License Inquiry`)
- **Repository:** https://github.com/ssfdre38/haven-cpp

Please include a brief description of your deployment target (CPU, Cloud TPU slice size, or embedded platform), expected scale, and whether you require enterprise support or custom model integration.

---

## Scope & Effective Date

Effective with the dual-license release of `haven-cpp`, all current and future source code, binaries, TPU StableHLO compilers, security modules, and releases in this repository are governed by this dual-licensing model (GNU AGPLv3 or Haven-CPP Commercial License). Nothing in this document limits any rights already granted under prior commits before the dual-license transition.

*(Third-party interface headers remain subject to their respective permissive licenses as documented in [NOTICE](NOTICE).)*
