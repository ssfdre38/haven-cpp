#include "haven/haven_avx2.h"
#include "haven/haven_engine.h"
#include "haven/haven_flash_attention.h"
#ifdef HAVEN_HAS_TPU
#include "haven/haven_tpu_gemma4.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace haven;

#define HAVEN_CHECK(cond, msg)                                                                 \
    do {                                                                                       \
        if (!(cond)) {                                                                         \
            std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__ << " (" << #cond << "): "   \
                      << (msg) << "\n";                                                        \
            return 1;                                                                          \
        }                                                                                      \
    } while (0)

namespace {

static uint16_t f32_to_f16(float f) {
    __m128 v = _mm_set_ss(f);
    __m128i h = _mm_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT);
    return static_cast<uint16_t>(_mm_extract_epi16(h, 0));
}

static double cosine_f64_vs_f32(const double* ref, const float* got, size_t n) {
    double dot = 0.0, nr = 0.0, ng = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double r = ref[i];
        const double g = static_cast<double>(got[i]);
        dot += r * g;
        nr  += r * r;
        ng  += g * g;
    }
    const double den = std::sqrt(nr) * std::sqrt(ng);
    return (den > 1e-30) ? (dot / den) : 0.0;
}

static double cosine_f32(const float* a, const float* b, size_t n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(a[i]);
        const double y = static_cast<double>(b[i]);
        dot += x * y;
        na  += x * x;
        nb  += y * y;
    }
    const double den = std::sqrt(na) * std::sqrt(nb);
    return (den > 1e-30) ? (dot / den) : 0.0;
}

static double max_abs_f64_vs_f32(const double* ref, const float* got, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) {
        m = std::max(m, std::fabs(ref[i] - static_cast<double>(got[i])));
    }
    return m;
}

static double max_rel_f64_vs_f32(const double* ref, const float* got, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double denom = std::max(1e-6, std::fabs(ref[i]));
        m = std::max(m, std::fabs(ref[i] - static_cast<double>(got[i])) / denom);
    }
    return m;
}

static int test_rmsnorm_parity(bool strict) {
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Stress test: D = 5376 (Gemma 4.5 31B hidden dim) with massive ±900x Gemma-style outlier spikes
    const int D = 5376;
    const float eps = 1e-6f;
    std::vector<float> x(D), w(D), out(D);
    for (int i = 0; i < D; ++i) {
        x[i] = dist(rng) * 0.25f;
        w[i] = 1.0f + dist(rng) * 0.3f;
    }
    x[17]   =  900.0f;
    x[1024] = -875.5f;
    x[4095] =  450.25f;
    w[17]   =  0.002f; // Realistic Gemma 4 norm suppression on outlier channel

    std::vector<double> ref(D);
    double sum_sq = 0.0;
    for (int i = 0; i < D; ++i) {
        const double xi = static_cast<double>(x[i]);
        sum_sq += xi * xi;
    }
    const double scale = 1.0 / std::sqrt(sum_sq / static_cast<double>(D) + static_cast<double>(eps));
    for (int i = 0; i < D; ++i) {
        ref[i] = static_cast<double>(x[i]) * scale * static_cast<double>(w[i]);
    }

    Avx2Math::rms_norm(out.data(), x.data(), w.data(), D, eps);

    const double cos_v = cosine_f64_vs_f32(ref.data(), out.data(), D);
    const double max_e = max_abs_f64_vs_f32(ref.data(), out.data(), D);
    const double rel_e = max_rel_f64_vs_f32(ref.data(), out.data(), D);

    std::cout << "  [RMSNorm D=5376 w/ 900x spikes] cosine=" << std::fixed << std::setprecision(9) << cos_v
              << " | max_abs=" << std::scientific << std::setprecision(3) << max_e
              << " | max_rel=" << rel_e << "\n";

    HAVEN_CHECK(cos_v > 0.9999999, "RMSNorm cosine dropped below 0.9999999");
    if (strict) {
        HAVEN_CHECK(max_e < 2e-6, "Strict RMSNorm max_abs error exceeded 2e-6");
    }
    return 0;
}

static int test_rope_parity(bool strict) {
    std::mt19937 rng(108);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    const int head_dim = 512;
    const int num_heads = 4;
    const int num_kv_heads = 2;
    const float freq_base = 1000000.0f;
    const float freq_scale = 1.0f;

    std::vector<float> freq_factors(head_dim / 2);
    for (int i = 0; i < head_dim / 2; ++i) {
        freq_factors[i] = 1.0f + 0.01f * static_cast<float>(i % 7);
    }

    for (bool neox : {true, false}) {
        for (int pos : {7, 32768, 131072}) {
            std::vector<float> q(num_heads * head_dim), k(num_kv_heads * head_dim);
            for (float& v : q) v = dist(rng);
            for (float& v : k) v = dist(rng);

            std::vector<double> ref_q(q.begin(), q.end());
            std::vector<double> ref_k(k.begin(), k.end());

            const int num_pairs = head_dim / 2;
            const double inv_rope_dim = -2.0 / static_cast<double>(head_dim);
            auto ref_rot = [&](std::vector<double>& buf, int count) {
                for (int h = 0; h < count; ++h) {
                    double* hd = buf.data() + static_cast<size_t>(h) * static_cast<size_t>(head_dim);
                    for (int i = 0; i < num_pairs; ++i) {
                        double theta = static_cast<double>(pos) * static_cast<double>(freq_scale)
                                     * std::pow(static_cast<double>(freq_base), static_cast<double>(i) * inv_rope_dim);
                        theta /= static_cast<double>(freq_factors[i]);
                        const double c = std::cos(theta);
                        const double s = std::sin(theta);
                        const int i0 = neox ? i : (2 * i);
                        const int i1 = neox ? (i + num_pairs) : (2 * i + 1);
                        const double v0 = hd[i0];
                        const double v1 = hd[i1];
                        hd[i0] = v0 * c - v1 * s;
                        hd[i1] = v0 * s + v1 * c;
                    }
                }
            };
            ref_rot(ref_q, num_heads);
            ref_rot(ref_k, num_kv_heads);

            Avx2Math::apply_rope(q.data(), k.data(), head_dim, num_heads, num_kv_heads,
                                 pos, freq_base, freq_scale, freq_factors.data(), head_dim, neox);

            const double cq = cosine_f64_vs_f32(ref_q.data(), q.data(), q.size());
            const double eq = max_abs_f64_vs_f32(ref_q.data(), q.data(), q.size());
            const double ek = max_abs_f64_vs_f32(ref_k.data(), k.data(), k.size());

            std::cout << "  [RoPE " << (neox ? "NeoX" : "Intv") << " pos=" << std::setw(6) << pos
                      << " base=1e6] cosine=" << std::fixed << std::setprecision(9) << cq
                      << " | max_abs_q=" << std::scientific << std::setprecision(3) << eq
                      << " | max_abs_k=" << ek << "\n";

            if (strict) {
                HAVEN_CHECK(cq > 0.99999999, "Strict RoPE cosine fell below 0.99999999");
                HAVEN_CHECK(eq < 2e-6 && ek < 2e-6, "Strict RoPE max_abs error exceeded 2e-6 at long context");
            } else {
                HAVEN_CHECK(cq > 0.9999, "Baseline RoPE cosine fell below 0.9999");
            }
        }
    }
    return 0;
}

static int test_softmax_and_flash_attention(bool strict) {
    std::mt19937 rng(777);
    std::normal_distribution<float> dist(0.0f, 12.0f);

    // Softmax extreme dynamic range (N = 4096)
    {
        const int N = 4096;
        std::vector<float> x(N);
        for (int i = 0; i < N; ++i) x[i] = dist(rng);
        x[123] = 75.0f;
        x[999] = 74.95f;
        x[2048] = -80.0f;

        std::vector<double> ref(N);
        double max_v = -std::numeric_limits<double>::infinity();
        for (int i = 0; i < N; ++i) max_v = std::max(max_v, static_cast<double>(x[i]));
        double sum = 0.0;
        for (int i = 0; i < N; ++i) {
            ref[i] = std::exp(static_cast<double>(x[i]) - max_v);
            sum += ref[i];
        }
        for (int i = 0; i < N; ++i) ref[i] /= sum;

        Avx2Math::softmax(x.data(), N);
        double got_sum = 0.0;
        for (float v : x) got_sum += static_cast<double>(v);
        const double max_e = max_abs_f64_vs_f32(ref.data(), x.data(), N);
        const double sum_err = std::fabs(got_sum - 1.0);

        std::cout << "  [Softmax N=4096 dynamic]    sum_err=" << std::scientific << std::setprecision(3) << sum_err
                  << " | max_abs=" << max_e << "\n";

        HAVEN_CHECK(max_e < 1e-6, "Softmax max_abs exceeded 1e-6");
        if (strict) {
            HAVEN_CHECK(sum_err < 1e-6, "Strict Softmax sum deviation exceeded 1e-6");
        }
    }

    // FlashAttention online-softmax (N = 512, HD = 256, with & without softcap = 50.0)
    {
        std::normal_distribution<float> attn_dist(0.0f, 0.5f);
        const uint32_t N = 512;
        const uint32_t HD = 256;
        std::vector<float> q(HD), K(N * HD), V(N * HD), out(HD, 0.0f);
        for (float& v : q) v = attn_dist(rng);
        for (float& v : K) v = attn_dist(rng);
        for (float& v : V) v = attn_dist(rng);

        for (float softcap : {0.0f, 50.0f}) {
            const float scale = (softcap > 0.0f) ? 1.0f : (1.0f / std::sqrt(static_cast<float>(HD)));

            std::vector<double> scores(N);
            double max_s = -std::numeric_limits<double>::infinity();
            for (uint32_t t = 0; t < N; ++t) {
                double dot = 0.0;
                for (uint32_t d = 0; d < HD; ++d) {
                    dot += static_cast<double>(q[d]) * static_cast<double>(K[t * HD + d]);
                }
                double sc = dot * static_cast<double>(scale);
                if (softcap > 0.0f) sc = static_cast<double>(softcap) * std::tanh(sc / static_cast<double>(softcap));
                scores[t] = sc;
                if (sc > max_s) max_s = sc;
            }
            double sum_e = 0.0;
            std::vector<double> probs(N);
            for (uint32_t t = 0; t < N; ++t) {
                probs[t] = std::exp(scores[t] - max_s);
                sum_e += probs[t];
            }
            std::vector<double> ref(HD, 0.0);
            for (uint32_t t = 0; t < N; ++t) {
                const double p = probs[t] / sum_e;
                for (uint32_t d = 0; d < HD; ++d) {
                    ref[d] += p * static_cast<double>(V[t * HD + d]);
                }
            }

            FlashAttentionCpu::compute_tiled_attention(
                q.data(), K.data(), V.data(), nullptr, N, HD, HD, 0, scale, softcap, out.data());

            const double cos_v = cosine_f64_vs_f32(ref.data(), out.data(), HD);
            const double max_e = max_abs_f64_vs_f32(ref.data(), out.data(), HD);

            std::cout << "  [FlashAttn N=512 HD=256 cap=" << std::fixed << std::setprecision(0) << std::setw(2) << softcap
                      << "] cosine=" << std::setprecision(9) << cos_v
                      << " | max_abs=" << std::scientific << std::setprecision(3) << max_e << "\n";

            HAVEN_CHECK(cos_v > 0.999999, "FlashAttention cosine below 0.999999");
            if (strict) {
                HAVEN_CHECK(max_e < 2e-6, "Strict FlashAttention max_abs exceeded 2e-6");
            }
        }
    }
    return 0;
}

static int test_fwht_and_activations() {
    std::mt19937 rng(2026);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // FWHT self-inverse & norm preservation with ±900x outlier spike
    const int N = 2560;
    std::vector<float> orig(N), v(N);
    for (int i = 0; i < N; ++i) orig[i] = dist(rng);
    orig[73] = 900.0f;
    orig[1337] = -650.0f;
    v = orig;

    double n0 = 0.0;
    for (float x : orig) n0 += static_cast<double>(x) * static_cast<double>(x);
    n0 = std::sqrt(n0);

    Avx2Math::fwht_block_inplace(v.data(), N, 256);
    double n1 = 0.0;
    for (float x : v) n1 += static_cast<double>(x) * static_cast<double>(x);
    n1 = std::sqrt(n1);

    Avx2Math::fwht_block_inplace(v.data(), N, 256);
    double inv_err = 0.0;
    for (int i = 0; i < N; ++i) {
        inv_err = std::max(inv_err, std::fabs(static_cast<double>(orig[i]) - static_cast<double>(v[i])));
    }
    std::cout << "  [FWHT B=256 N=2560 spikes]  norm_rel_err=" << std::scientific << std::setprecision(3)
              << (std::fabs(n1 - n0) / n0) << " | roundtrip_max_abs=" << inv_err << "\n";
    HAVEN_CHECK(std::fabs(n1 - n0) / n0 < 1e-6, "FWHT norm preservation exceeded 1e-6");
    HAVEN_CHECK(inv_err < 1e-4, "FWHT roundtrip error exceeded 1e-4");

    // GELU & SiLU accuracy over [-12, +12]
    const int M = 2401;
    std::vector<float> g_in(M), s_in(M);
    std::vector<double> g_ref(M), s_ref(M);
    const double s2pi = std::sqrt(2.0 / 3.14159265358979323846);
    for (int i = 0; i < M; ++i) {
        const double x = -12.0 + 0.01 * static_cast<double>(i);
        g_in[i] = static_cast<float>(x);
        s_in[i] = static_cast<float>(x);
        g_ref[i] = 0.5 * x * (1.0 + std::tanh(s2pi * (x + 0.044715 * x * x * x)));
        s_ref[i] = x / (1.0 + std::exp(-x));
    }
    Avx2Math::gelu(g_in.data(), M);
    Avx2Math::silu(s_in.data(), M);
    const double eg = max_abs_f64_vs_f32(g_ref.data(), g_in.data(), M);
    const double es = max_abs_f64_vs_f32(s_ref.data(), s_in.data(), M);
    std::cout << "  [Activations [-12,+12]]     gelu_max_abs=" << std::scientific << std::setprecision(3) << eg
              << " | silu_max_abs=" << es << "\n";
    HAVEN_CHECK(eg < 2e-6 && es < 2e-6, "GELU/SiLU accuracy exceeded 2e-6");
    return 0;
}

static int test_quant_gemv_parity() {
    std::mt19937 rng(99);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    const int rows = 64;
    const int cols = 512;
    std::vector<float> x(cols);
    for (int c = 0; c < cols; ++c) x[c] = dist(rng);
    // Inject a realistic Gemma 4 activation outlier to expose Q8 activation quantization loss!
    x[19]  =  42.0f;
    x[277] = -38.5f;

    // Build 11-token activation matrix (exercises both 1x8 full tiles and 1x4 tail tiles!)
    constexpr int S_batch = 11;
    std::vector<float> X(static_cast<size_t>(S_batch) * static_cast<size_t>(cols));
    for (int t = 0; t < S_batch; ++t) {
        float* xt = X.data() + static_cast<size_t>(t) * static_cast<size_t>(cols);
        for (int c = 0; c < cols; ++c) xt[c] = (t == 0) ? x[c] : dist(rng);
        // Inject extreme outlier spikes into every token (42x .. 662x)
        xt[(19 + t * 37) % cols]  = (t % 2 == 0) ? 42.0f : -662.0f;
        xt[(277 + t * 53) % cols] = -38.5f;
    }

    auto run_format = [&](const char* name, QuantType qtype, const void* data, size_t nbytes) -> int {
        TensorDesc td;
        td.name = name;
        td.type = qtype;
        td.data = data;
        td.size_bytes = nbytes;
        td.shape = {static_cast<uint64_t>(cols), static_cast<uint64_t>(rows)};

        // Golden FP64 reference across all S_batch tokens via exact dequantize_row
        std::vector<double> ref_all(static_cast<size_t>(S_batch) * static_cast<size_t>(rows), 0.0);
        std::vector<float> row_f32(cols);
        for (int r = 0; r < rows; ++r) {
            Avx2Math::dequantize_row(row_f32.data(), td, r, cols);
            for (int t = 0; t < S_batch; ++t) {
                const float* xt = X.data() + static_cast<size_t>(t) * static_cast<size_t>(cols);
                double acc = 0.0;
                for (int c = 0; c < cols; ++c) {
                    acc += static_cast<double>(row_f32[c]) * static_cast<double>(xt[c]);
                }
                ref_all[static_cast<size_t>(t) * static_cast<size_t>(rows) + static_cast<size_t>(r)] = acc;
            }
        }

        // 1. Fast Q8 + Outlier-Split GEMV (token 0) and GEMM (S=3 1x4 & S=11 1x8+1x4)
        std::vector<float> y_fast(rows, 0.0f);
        Avx2Math::gemv(y_fast.data(), td, x.data(), rows, cols);
        std::vector<float> Y_fast(static_cast<size_t>(S_batch) * static_cast<size_t>(rows), 0.0f);
        Avx2Math::gemm(Y_fast.data(), rows, td, X.data(), cols, S_batch, rows, cols);

        // 2. Exact FP32-activation GEMV (token 0) and 1x4/1x8 register-tiled GEMM (S=3 & S=11)
        std::vector<float> y_f32(rows, 0.0f);
        Avx2Math::gemv_f32(y_f32.data(), td, x.data(), rows, cols);
        std::vector<float> Y_f32_3(3ULL * static_cast<size_t>(rows), 0.0f);
        std::vector<float> Y_f32(static_cast<size_t>(S_batch) * static_cast<size_t>(rows), 0.0f);
        Avx2Math::gemm_f32(Y_f32_3.data(), rows, td, X.data(), cols, 3, rows, cols);
        Avx2Math::gemm_f32(Y_f32.data(), rows, td, X.data(), cols, S_batch, rows, cols);

        const double cos_fast    = cosine_f64_vs_f32(ref_all.data(), y_fast.data(), rows);
        const double err_fast    = max_abs_f64_vs_f32(ref_all.data(), y_fast.data(), rows);
        const double cos_f32     = cosine_f64_vs_f32(ref_all.data(), y_f32.data(), rows);
        const double err_f32     = max_abs_f64_vs_f32(ref_all.data(), y_f32.data(), rows);
        const double cos_gemm3   = cosine_f64_vs_f32(ref_all.data(), Y_f32_3.data(), 3ULL * static_cast<size_t>(rows));
        const double cos_gemm11  = cosine_f64_vs_f32(ref_all.data(), Y_f32.data(), static_cast<size_t>(S_batch) * static_cast<size_t>(rows));
        const double cos_q8_eq   = cosine_f32(y_fast.data(), Y_fast.data(), rows);
        const double cos_q8_all  = cosine_f64_vs_f32(ref_all.data(), Y_fast.data(), static_cast<size_t>(S_batch) * static_cast<size_t>(rows));

        std::cout << "  [GEMV/M " << std::left << std::setw(6) << name << std::right
                  << " 42-662x] Q8=" << std::fixed << std::setprecision(7) << cos_fast
                  << " (err=" << std::scientific << std::setprecision(2) << err_fast << ")"
                  << " | Q8_S11=" << std::fixed << std::setprecision(7) << cos_q8_all
                  << " | F32_1x8=" << std::setprecision(9) << cos_gemm11
                  << " (err=" << std::scientific << std::setprecision(2) << err_f32 << ")"
                  << " | v==m:" << std::fixed << std::setprecision(7) << cos_q8_eq << "\n";

        HAVEN_CHECK(cos_f32 > 0.9999999, "Exact gemv_f32 cosine fell below 0.9999999");
        HAVEN_CHECK(cos_gemm3 > 0.9999999, "Exact gemm_f32 (1x4 tile) cosine fell below 0.9999999");
        HAVEN_CHECK(cos_gemm11 > 0.9999999, "Exact gemm_f32 (1x8+1x4 tile) cosine fell below 0.9999999");
        HAVEN_CHECK(cos_fast > 0.99995, "Outlier-split Q8 GEMV cosine fell below 0.99995 under 42x spike");
        HAVEN_CHECK(cos_q8_all > 0.99995, "Outlier-split Q8 GEMM cosine fell below 0.99995 under 662x spike");
        HAVEN_CHECK(cos_q8_eq > 0.9999999, "Fast Q8 GEMV vs GEMM token-0 diverged!");
        return 0;
    };

    // Synthesize valid blocks for ALL 9 integer-Q8/FP32-tiled formats: IQ4_XS, IQ4_NL, Q6_K, Q5_K, Q4_K, Q5_1, Q5_0, Q4_0, Q8_0
    {
        std::vector<block_iq4_xs> blk(rows * (cols / 256));
        for (auto& b : blk) {
            b.d = f32_to_f16(0.015f);
            b.scales_h = static_cast<uint16_t>(rng() & 0xFFFFu);
            for (int i = 0; i < 4; ++i) b.scales_l[i] = static_cast<uint8_t>(rng() & 0xFFu);
            for (int i = 0; i < 128; ++i) b.qs[i] = static_cast<uint8_t>(rng() & 0xFFu);
        }
        if (run_format("IQ4_XS", QuantType::IQ4_XS, blk.data(), blk.size() * sizeof(block_iq4_xs)) != 0) return 1;
    }
    {
        std::vector<block_iq4_nl> blk(rows * (cols / 32));
        for (auto& b : blk) {
            b.d = f32_to_f16(0.012f);
            for (int i = 0; i < 16; ++i) b.qs[i] = static_cast<uint8_t>(rng() & 0xFFu);
        }
        if (run_format("IQ4_NL", QuantType::IQ4_NL, blk.data(), blk.size() * sizeof(block_iq4_nl)) != 0) return 1;
    }
    {
        std::vector<block_q6_K> blk(rows * (cols / 256));
        for (auto& b : blk) {
            b.d = f32_to_f16(0.008f);
            for (int i = 0; i < 128; ++i) b.ql[i] = static_cast<uint8_t>(rng() & 0xFFu);
            for (int i = 0; i < 64; ++i) b.qh[i] = static_cast<uint8_t>(rng() & 0xFFu);
            for (int i = 0; i < 16; ++i) b.scales[i] = static_cast<int8_t>((static_cast<int>(rng() % 63)) - 31);
        }
        if (run_format("Q6_K", QuantType::Q6_K, blk.data(), blk.size() * sizeof(block_q6_K)) != 0) return 1;
    }
    {
        std::vector<block_q5_K> blk(rows * (cols / 256));
        for (auto& b : blk) {
            b.d = f32_to_f16(0.014f);
            b.dmin = f32_to_f16(0.009f);
            for (int i = 0; i < 12; ++i) b.scales[i] = static_cast<uint8_t>(rng() & 0xFFu);
            for (int i = 0; i < 32; ++i) b.qh[i] = static_cast<uint8_t>(rng() & 0xFFu);
            for (int i = 0; i < 128; ++i) b.qs[i] = static_cast<uint8_t>(rng() & 0xFFu);
        }
        if (run_format("Q5_K", QuantType::Q5_K, blk.data(), blk.size() * sizeof(block_q5_K)) != 0) return 1;
    }
    {
        std::vector<block_q4_K> blk(rows * (cols / 256));
        for (auto& b : blk) {
            b.d = f32_to_f16(0.02f);
            b.dmin = f32_to_f16(0.01f);
            for (int i = 0; i < 12; ++i) b.scales[i] = static_cast<uint8_t>(rng() & 0xFFu);
            for (int i = 0; i < 128; ++i) b.qs[i] = static_cast<uint8_t>(rng() & 0xFFu);
        }
        if (run_format("Q4_K", QuantType::Q4_K, blk.data(), blk.size() * sizeof(block_q4_K)) != 0) return 1;
    }
    {
        std::vector<block_q5_1> blk(rows * (cols / 32));
        for (auto& b : blk) {
            b.d = f32_to_f16(0.018f);
            b.m = f32_to_f16(-0.08f);
            for (int i = 0; i < 4; ++i) b.qh[i] = static_cast<uint8_t>(rng() & 0xFFu);
            for (int i = 0; i < 16; ++i) b.qs[i] = static_cast<uint8_t>(rng() & 0xFFu);
        }
        if (run_format("Q5_1", QuantType::Q5_1, blk.data(), blk.size() * sizeof(block_q5_1)) != 0) return 1;
    }
    {
        std::vector<block_q5_0> blk(rows * (cols / 32));
        for (auto& b : blk) {
            b.d = f32_to_f16(0.02f);
            for (int i = 0; i < 4; ++i) b.qh[i] = static_cast<uint8_t>(rng() & 0xFFu);
            for (int i = 0; i < 16; ++i) b.qs[i] = static_cast<uint8_t>(rng() & 0xFFu);
        }
        if (run_format("Q5_0", QuantType::Q5_0, blk.data(), blk.size() * sizeof(block_q5_0)) != 0) return 1;
    }
    {
        std::vector<block_q4_0> blk(rows * (cols / 32));
        for (auto& b : blk) {
            b.d = f32_to_f16(0.025f);
            for (int i = 0; i < 16; ++i) b.qs[i] = static_cast<uint8_t>(rng() & 0xFFu);
        }
        if (run_format("Q4_0", QuantType::Q4_0, blk.data(), blk.size() * sizeof(block_q4_0)) != 0) return 1;
    }
    {
        std::vector<block_q8_0> blk(rows * (cols / 32));
        for (auto& b : blk) {
            b.d = f32_to_f16(0.01f);
            for (int i = 0; i < 32; ++i) b.qs[i] = static_cast<int8_t>((static_cast<int>(rng() % 255)) - 127);
        }
        if (run_format("Q8_0", QuantType::Q8_0, blk.data(), blk.size() * sizeof(block_q8_0)) != 0) return 1;
    }
    return 0;
}

static int test_long_context_swa_and_sinks_parity() {
    // 2048-token Sliding Window (win=512 + 4 Attention Sinks) + FP64 RoPE at pos=2047 vs Golden FP64 reference
    constexpr int TOTAL_SEQ = 2048;
    constexpr int WIN = 512;
    constexpr int SINKS = 4;
    constexpr int HD = 256;
    std::mt19937 rng(98765);
    std::normal_distribution<float> dist(0.0f, 0.35f);

    std::vector<float> q(HD);
    std::vector<float> K(static_cast<size_t>(TOTAL_SEQ) * HD);
    std::vector<float> V(static_cast<size_t>(TOTAL_SEQ) * HD);
    for (float& v : q) v = dist(rng);
    for (float& v : K) v = dist(rng);
    for (float& v : V) v = dist(rng);

    // Apply RoPE at long positions 0..2047 (testing sinks at 0..3 + far window 1536..2047)
    Avx2Math::apply_rope(q.data(), nullptr, HD, 1, 0, TOTAL_SEQ - 1, 1000000.0f, 1.0f, nullptr, HD, true);
    for (int p = 0; p < TOTAL_SEQ; ++p) {
        if (p < SINKS || p >= TOTAL_SEQ - WIN) {
            Avx2Math::apply_rope(K.data() + static_cast<size_t>(p) * HD, nullptr, HD, 1, 0, p, 1000000.0f, 1.0f, nullptr, HD, true);
        }
    }

    std::vector<int> indices;
    indices.reserve(SINKS + WIN);
    for (int p = 0; p < SINKS; ++p) indices.push_back(p);
    for (int p = TOTAL_SEQ - WIN; p < TOTAL_SEQ; ++p) indices.push_back(p);

    const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(HD)));
    const float softcap = 50.0f;

    std::vector<float> out_prod(HD, 0.0f);
    FlashAttentionCpu::compute_tiled_attention(
        q.data(), K.data(), V.data(), indices.data(),
        static_cast<uint32_t>(indices.size()), HD, HD, 0, scale, softcap, out_prod.data());

    // Golden FP64 reference over the exact same sink+window index set
    std::vector<double> scores(indices.size(), 0.0);
    double max_s = -1e300;
    for (size_t idx = 0; idx < indices.size(); ++idx) {
        const float* kt = K.data() + static_cast<size_t>(indices[idx]) * HD;
        double dot = 0.0;
        for (int d = 0; d < HD; ++d) dot += static_cast<double>(q[d]) * static_cast<double>(kt[d]);
        dot *= static_cast<double>(scale);
        dot = static_cast<double>(softcap) * std::tanh(dot / static_cast<double>(softcap));
        scores[idx] = dot;
        if (dot > max_s) max_s = dot;
    }
    double denom = 0.0;
    for (double& s : scores) { s = std::exp(s - max_s); denom += s; }
    std::vector<double> out_ref(HD, 0.0);
    for (size_t idx = 0; idx < indices.size(); ++idx) {
        const double w = scores[idx] / denom;
        const float* vt = V.data() + static_cast<size_t>(indices[idx]) * HD;
        for (int d = 0; d < HD; ++d) out_ref[d] += w * static_cast<double>(vt[d]);
    }

    const double cos_v = cosine_f64_vs_f32(out_ref.data(), out_prod.data(), HD);
    const double err_v = max_abs_f64_vs_f32(out_ref.data(), out_prod.data(), HD);
    std::cout << "  [LongCtx SWA pos=2047 +4sinks] cosine=" << std::fixed << std::setprecision(9) << cos_v
              << " | max_abs=" << std::scientific << std::setprecision(3) << err_v << "\n";
    HAVEN_CHECK(cos_v > 0.9999999 && err_v < 2e-6, "Long-context SWA + sinks attention drifted from FP64 reference");
    return 0;
}

static int test_multi_arch_e2e_exact_vs_fast() {
    // End-to-end quantized GGUF parity across non-Gemma architectures:
    //   1. Llama-3 (Interleaved RoPE, SiLU, untied output, Q8_0 weights, 55x norm spike)
    //   2. Qwen-2.5 (NeoX RoPE, QKV biases, SiLU, Q4_0 weights, 55x norm spike)
    //   3. Mistral (Sliding Window SWA=8 + sinks, SiLU, Q8_0 weights, 55x norm spike)
    struct ArchSpec {
        const char* label;
        const char* arch;
        uint32_t gguf_qtype; // 8=Q8_0, 2=Q4_0
        bool qkv_bias;
        uint32_t swa_win;
    };
    const ArchSpec specs[3] = {
        {"Llama-3-Q8_0",  "llama",   8u, false, 0u},
        {"Qwen-2.5-Q4_0", "qwen2",   2u, true,  0u},
        {"Mistral-SWA",   "mistral", 8u, false, 8u},
    };

    constexpr uint32_t V = 64, D = 256, F = 256, L = 2, NH = 4, NKV = 2, HD = 64;
    constexpr uint32_t QDIM = NH * HD, KVDIM = NKV * HD;
    const std::vector<uint32_t> prompt = {1, 5, 12, 7, 31, 44, 19, 28, 3, 52, 14, 9}; // S=12 (exercises 1x8 + 1x4 tiles + SWA>8!)

    auto quantize_matrix = [](const std::vector<float>& src, uint32_t qtype) -> std::vector<uint8_t> {
        const size_t nb = src.size() / 32;
        if (qtype == 8u) {
            std::vector<uint8_t> buf(nb * sizeof(block_q8_0));
            auto* dst = reinterpret_cast<block_q8_0*>(buf.data());
            for (size_t b = 0; b < nb; ++b) {
                float ma = 0.0f;
                for (int i = 0; i < 32; ++i) ma = std::max(ma, std::fabs(src[b * 32 + i]));
                const float d = (ma > 1e-12f) ? (ma / 127.0f) : 0.0f;
                dst[b].d = f32_to_f16(d);
                const float inv_d = (d > 0.0f) ? (1.0f / d) : 0.0f;
                for (int i = 0; i < 32; ++i) {
                    int q = std::clamp(static_cast<int>(std::lrint(src[b * 32 + i] * inv_d)), -127, 127);
                    dst[b].qs[i] = static_cast<int8_t>(q);
                }
            }
            return buf;
        } else {
            std::vector<uint8_t> buf(nb * sizeof(block_q4_0));
            auto* dst = reinterpret_cast<block_q4_0*>(buf.data());
            for (size_t b = 0; b < nb; ++b) {
                float ma = 0.0f;
                for (int i = 0; i < 32; ++i) ma = std::max(ma, std::fabs(src[b * 32 + i]));
                const float d = (ma > 1e-12f) ? (ma / 7.0f) : 0.0f;
                dst[b].d = f32_to_f16(d);
                const float inv_d = (d > 0.0f) ? (1.0f / d) : 0.0f;
                for (int j = 0; j < 16; ++j) {
                    int q0 = std::clamp(static_cast<int>(std::lrint(src[b * 32 + j] * inv_d)) + 8, 0, 15);
                    int q1 = std::clamp(static_cast<int>(std::lrint(src[b * 32 + 16 + j] * inv_d)) + 8, 0, 15);
                    dst[b].qs[j] = static_cast<uint8_t>((q0 & 0x0F) | ((q1 & 0x0F) << 4));
                }
            }
            return buf;
        }
    };

    for (const auto& sp : specs) {
        const auto tmp_path = std::filesystem::temp_directory_path() / (std::string("haven_parity_") + sp.arch + ".gguf");
        {
            struct TOut { std::string name; std::vector<uint64_t> dims; uint32_t type; std::vector<uint8_t> bytes; };
            std::vector<TOut> ts;
            auto gen_f32 = [](size_t n, int seed, float sc) {
                std::mt19937 grng(static_cast<uint32_t>(seed) * 2654435761u);
                std::vector<float> v(n);
                for (size_t i = 0; i < n; ++i) {
                    const double u1 = (static_cast<double>(grng()) + 0.5) / 4294967296.0;
                    const double u2 = (static_cast<double>(grng()) + 0.5) / 4294967296.0;
                    const double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(6.28318530717958647692 * u2);
                    v[i] = static_cast<float>(z * static_cast<double>(sc));
                }
                return v;
            };
            auto push_f32 = [&](const std::string& name, std::vector<uint64_t> dims, const std::vector<float>& v) {
                std::vector<uint8_t> b(v.size() * sizeof(float));
                std::memcpy(b.data(), v.data(), b.size());
                ts.push_back({name, std::move(dims), 0u, std::move(b)});
            };
            auto push_q = [&](const std::string& name, std::vector<uint64_t> dims, const std::vector<float>& v) {
                ts.push_back({name, std::move(dims), sp.gguf_qtype, quantize_matrix(v, sp.gguf_qtype)});
            };

            push_q("token_embd.weight", {D, V}, gen_f32(static_cast<size_t>(D) * V, 3, 0.25f));
            push_f32("output_norm.weight", {D}, std::vector<float>(D, 1.0f));
            push_q("output.weight", {D, V}, gen_f32(static_cast<size_t>(D) * V, 5, 0.12f));

            for (uint32_t l = 0; l < L; ++l) {
                const std::string p = "blk." + std::to_string(l) + ".";
                std::vector<float> an(D, 1.0f), fn(D, 1.0f);
                for (uint32_t i = 0; i < D; ++i) {
                    // Inject 55x channel spikes on un-baked stock weights to test Outlier-Split Q8 end-to-end
                    an[i] = (i == 19) ? 55.0f : (0.9f + 0.15f * std::cos(static_cast<float>(i + l)));
                    fn[i] = (i == 73) ? 55.0f : (1.0f + 0.15f * std::sin(static_cast<float>(i + l)));
                }
                push_f32(p + "attn_norm.weight", {D}, an);
                push_q(p + "attn_q.weight", {D, QDIM},  gen_f32(static_cast<size_t>(D) * QDIM,  11 + static_cast<int>(l), 0.05f));
                push_q(p + "attn_k.weight", {D, KVDIM}, gen_f32(static_cast<size_t>(D) * KVDIM, 17 + static_cast<int>(l), 0.05f));
                push_q(p + "attn_v.weight", {D, KVDIM}, gen_f32(static_cast<size_t>(D) * KVDIM, 23 + static_cast<int>(l), 0.05f));
                if (sp.qkv_bias) {
                    push_f32(p + "attn_q.bias", {QDIM},  gen_f32(QDIM,  43 + static_cast<int>(l), 0.04f));
                    push_f32(p + "attn_k.bias", {KVDIM}, gen_f32(KVDIM, 47 + static_cast<int>(l), 0.04f));
                    push_f32(p + "attn_v.bias", {KVDIM}, gen_f32(KVDIM, 53 + static_cast<int>(l), 0.04f));
                }
                push_q(p + "attn_output.weight", {QDIM, D}, gen_f32(static_cast<size_t>(QDIM) * D, 29 + static_cast<int>(l), 0.05f));
                push_f32(p + "ffn_norm.weight", {D}, fn);
                push_q(p + "ffn_gate.weight", {D, F}, gen_f32(static_cast<size_t>(D) * F, 31 + static_cast<int>(l), 0.05f));
                push_q(p + "ffn_up.weight",   {D, F}, gen_f32(static_cast<size_t>(D) * F, 37 + static_cast<int>(l), 0.05f));
                push_q(p + "ffn_down.weight", {F, D}, gen_f32(static_cast<size_t>(F) * D, 41 + static_cast<int>(l), 0.05f));
            }

            std::ofstream f(tmp_path, std::ios::binary | std::ios::trunc);
            auto wu32 = [&](uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };
            auto wu64 = [&](uint64_t v) { f.write(reinterpret_cast<const char*>(&v), 8); };
            auto wstr = [&](const std::string& s) { wu64(s.size()); f.write(s.data(), static_cast<std::streamsize>(s.size())); };
            auto kv_str = [&](const std::string& k, const std::string& v) { wstr(k); wu32(8); wstr(v); };
            auto kv_u32 = [&](const std::string& k, uint32_t v) { wstr(k); wu32(4); wu32(v); };
            auto kv_f32 = [&](const std::string& k, float v) { wstr(k); wu32(6); f.write(reinterpret_cast<const char*>(&v), 4); };

            const std::string ap(sp.arch);
            const uint64_t n_kv = 11 + (sp.swa_win > 0 ? 1 : 0);
            wu32(0x46554747u); wu32(3); wu64(ts.size()); wu64(n_kv);
            kv_str("general.architecture", ap);
            kv_u32(ap + ".block_count", L);
            kv_u32(ap + ".context_length", 128);
            kv_u32(ap + ".embedding_length", D);
            kv_u32(ap + ".feed_forward_length", F);
            kv_u32(ap + ".attention.head_count", NH);
            kv_u32(ap + ".attention.head_count_kv", NKV);
            kv_u32(ap + ".attention.key_length", HD);
            kv_f32(ap + ".attention.layer_norm_rms_epsilon", 1e-5f);
            kv_f32(ap + ".rope.freq_base", 500000.0f);
            if (sp.swa_win > 0) kv_u32(ap + ".attention.sliding_window", sp.swa_win);
            wstr("tokenizer.ggml.tokens"); wu32(9); wu32(8); wu64(V);
            for (uint32_t i = 0; i < V; ++i) wstr("t" + std::to_string(i));

            uint64_t off = 0;
            for (const auto& t : ts) {
                wstr(t.name);
                wu32(static_cast<uint32_t>(t.dims.size()));
                for (uint64_t d : t.dims) wu64(d);
                wu32(t.type);
                wu64(off);
                off += ((t.bytes.size() + 31ULL) / 32ULL) * 32ULL;
            }
            auto pad32 = [&]() {
                uint64_t p = static_cast<uint64_t>(f.tellp());
                uint64_t pad = (32ULL - (p % 32ULL)) % 32ULL;
                if (pad) {
                    std::vector<uint8_t> z(pad, 0);
                    f.write(reinterpret_cast<const char*>(z.data()), static_cast<std::streamsize>(pad));
                }
            };
            pad32();
            for (const auto& t : ts) {
                f.write(reinterpret_cast<const char*>(t.bytes.data()), static_cast<std::streamsize>(t.bytes.size()));
                pad32();
            }
        }

        int status = 0;
        {
            HavenEngine eng;
            if (!eng.load_model(tmp_path.string())) {
                std::cerr << "[FAIL] Failed to load synthetic GGUF: " << sp.label << "\n";
                status = 1;
            } else {
                std::vector<float> lg_es(V), lg_eb(V), lg_fs(V), lg_fb(V);
                eng.set_exact_math(true);
                eng.get_kv_cache().reset();
                for (size_t i = 0; i < prompt.size(); ++i)
                    eng.forward(prompt[i], static_cast<int>(i), (i + 1 == prompt.size()) ? lg_es.data() : nullptr);
                eng.get_kv_cache().reset();
                eng.forward_batch(prompt.data(), static_cast<int>(prompt.size()), 0, lg_eb.data());

                eng.set_exact_math(false);
                eng.get_kv_cache().reset();
                for (size_t i = 0; i < prompt.size(); ++i)
                    eng.forward(prompt[i], static_cast<int>(i), (i + 1 == prompt.size()) ? lg_fs.data() : nullptr);
                eng.get_kv_cache().reset();
                eng.forward_batch(prompt.data(), static_cast<int>(prompt.size()), 0, lg_fb.data());

                const double cos_eb  = cosine_f32(lg_es.data(), lg_eb.data(), V);
                const double cos_fs  = cosine_f32(lg_es.data(), lg_fs.data(), V);
                const double cos_fb  = cosine_f32(lg_es.data(), lg_fb.data(), V);
                const double cos_f_s_b = cosine_f32(lg_fs.data(), lg_fb.data(), V);
                const auto t_es = std::max_element(lg_es.begin(), lg_es.end()) - lg_es.begin();
                const auto t_eb = std::max_element(lg_eb.begin(), lg_eb.end()) - lg_eb.begin();
                const auto t_fb = std::max_element(lg_fb.begin(), lg_fb.end()) - lg_fb.begin();

                std::cout << "  [MultiArch " << std::left << std::setw(13) << sp.label << std::right
                          << " 55x] exact_s==b:" << std::fixed << std::setprecision(9) << cos_eb
                          << " | fast_s:" << std::setprecision(7) << cos_fs
                          << " | fast_b:" << cos_fb
                          << " | fs==fb:" << cos_f_s_b
                          << " | Top1:" << (t_es == t_eb && t_es == t_fb ? "MATCH" : "DIFF") << "\n";
                if (cos_eb <= 0.9999999 || cos_fs <= 0.9995 || cos_fb <= 0.9995 || cos_f_s_b <= 0.999999 || t_es != t_eb || t_es != t_fb) {
                    std::cerr << "[FAIL] Multi-arch parity regression on " << sp.label << "\n";
                    status = 1;
                }
            }
        }
        std::error_code ec;
        std::filesystem::remove(tmp_path, ec);
        if (status != 0) return status;
    }
    return 0;
}

#ifdef HAVEN_HAS_TPU
static int test_tpu_w4_quantizer_quality(bool strict) {
    std::mt19937 rng(31415);
    std::normal_distribution<float> norm(0.0f, 0.04f);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    const int rows = 128;
    const int cols = 512;
    const int G = HavenTpuGemma4Engine::compute_w4_group_size(cols);
    auto eval_w4 = [&](const char* label, const std::vector<float>& src, double& out_snr, double& out_cos, double& out_naive_snr) {
        std::vector<int8_t> packed(rows * (cols / 2));
        std::vector<float> scales(rows * (cols / G));
        std::vector<float> recon(rows * cols);
        HavenTpuGemma4Engine::quantize_f32_to_s4_packed_group(src.data(), rows, cols, G, packed.data(), scales.data());
        HavenTpuGemma4Engine::dequantize_s4_packed_group_to_f32(packed.data(), scales.data(), rows, cols, G, recon.data());

        double mse = 0.0, sig_pow = 0.0, naive_mse = 0.0;
        for (int r = 0; r < rows; ++r) {
            for (int g = 0; g < cols / G; ++g) {
                const float* grp = src.data() + static_cast<size_t>(r) * cols + static_cast<size_t>(g) * G;
                float ma = 0.0f;
                for (int i = 0; i < G; ++i) ma = std::max(ma, std::fabs(grp[i]));
                const float s_naive = (ma > 1e-12f) ? (ma / 7.0f) : 1.0f;
                for (int i = 0; i < G; ++i) {
                    int q = std::clamp(static_cast<int>(std::lrint(grp[i] / s_naive)), -7, 7);
                    const double dn = static_cast<double>(grp[i]) - static_cast<double>(q) * static_cast<double>(s_naive);
                    naive_mse += dn * dn;
                }
            }
        }
        for (size_t i = 0; i < src.size(); ++i) {
            const double s = static_cast<double>(src[i]);
            const double d = s - static_cast<double>(recon[i]);
            mse += d * d;
            sig_pow += s * s;
        }
        const double rmse = std::sqrt(mse / static_cast<double>(src.size()));
        out_snr = 10.0 * std::log10(sig_pow / std::max(1e-30, mse));
        out_naive_snr = 10.0 * std::log10(sig_pow / std::max(1e-30, naive_mse));
        out_cos = cosine_f32(src.data(), recon.data(), src.size());
        std::cout << "  [TPU W4A16 G=" << G << " " << label << "] RMSE=" << std::scientific << std::setprecision(4) << rmse
                  << " | SNR=" << std::fixed << std::setprecision(2) << out_snr << " dB (naive=" << out_naive_snr << " dB, +"
                  << (out_snr - out_naive_snr) << " dB)"
                  << " | cosine=" << std::setprecision(6) << out_cos << "\n";
    };

    std::vector<float> src_gauss(rows * cols);
    std::vector<float> src_heavy(rows * cols);
    for (int i = 0; i < rows * cols; ++i) {
        float v = norm(rng);
        src_gauss[i] = v;
        if (uni(rng) < 0.01f) v *= 6.0f;
        src_heavy[i] = v;
    }

    double g_snr = 0.0, g_cos = 0.0, g_naive = 0.0;
    double h_snr = 0.0, h_cos = 0.0, h_naive = 0.0;
    eval_w4("gaussian  ", src_gauss, g_snr, g_cos, g_naive);
    eval_w4("heavy-tail", src_heavy, h_snr, h_cos, h_naive);

    HAVEN_CHECK(g_cos > 0.992, "TPU W4A16 Gaussian cosine fell below 0.992");
    HAVEN_CHECK(h_cos > 0.975, "TPU W4A16 heavy-tail cosine fell below 0.975");
    if (strict) {
        HAVEN_CHECK(g_snr > 18.5, "Strict TPU W4A16 Gaussian SNR fell below 18.5 dB");
        HAVEN_CHECK(g_snr >= g_naive + 0.5, "Strict TPU W4A16 MSE-opt must beat naive Gaussian quantizer by >= +0.5 dB");
        HAVEN_CHECK(h_snr > 14.0, "Strict TPU W4A16 heavy-tail SNR fell below 14.0 dB");
        HAVEN_CHECK(h_snr >= h_naive + 1.2, "Strict TPU W4A16 MSE-opt must beat naive heavy-tail quantizer by >= +1.2 dB");
    }
    return 0;
}
#endif

static int run_live_gguf_parity(const std::string& gguf_path) {
    std::cout << "\n--- Live GGUF End-to-End Parity Audit: " << gguf_path << " ---\n";
    HavenEngine eng;
    if (!eng.load_model(gguf_path)) {
        std::cerr << "[FAIL] Could not load GGUF: " << gguf_path << "\n";
        return 1;
    }
    const auto& cfg = eng.get_config();
    const std::string prompt = eng.format_chat_prompt("Reply with three words: precision math works.");
    auto toks = eng.tokenize(prompt, true);
    std::cout << "  Model: " << cfg.architecture << " (" << cfg.num_layers << "L, D=" << cfg.embedding_dim
              << ", V=" << cfg.vocab_size << ") | Prompt tokens=" << toks.size() << "\n";

    std::vector<float> lg_exact_s_pre(cfg.vocab_size), lg_exact_b_pre(cfg.vocab_size);
    std::vector<float> lg_exact_s_dec(cfg.vocab_size), lg_exact_b_dec(cfg.vocab_size);
    std::vector<float> lg_fast_s_pre(cfg.vocab_size),  lg_fast_b_pre(cfg.vocab_size);

    auto measure_diff = [&](const std::vector<float>& a, const std::vector<float>& b, double& out_max) -> double {
        out_max = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            out_max = std::max(out_max, std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
        }
        return cosine_f32(a.data(), b.data(), a.size());
    };

    // 1. Exact FP32-activation mode: single forward() vs batched forward_batch()
    eng.set_exact_math(true);
    eng.get_kv_cache().reset();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < toks.size(); ++i) {
        eng.forward(toks[i], static_cast<int>(i), (i + 1 == toks.size()) ? lg_exact_s_pre.data() : nullptr);
    }
    const uint32_t top1_es = static_cast<uint32_t>(std::max_element(lg_exact_s_pre.begin(), lg_exact_s_pre.end()) - lg_exact_s_pre.begin());
    eng.forward(top1_es, static_cast<int>(toks.size()), lg_exact_s_dec.data());
    const uint32_t top2_es = static_cast<uint32_t>(std::max_element(lg_exact_s_dec.begin(), lg_exact_s_dec.end()) - lg_exact_s_dec.begin());
    auto t1 = std::chrono::high_resolution_clock::now();

    eng.get_kv_cache().reset();
    auto t2 = std::chrono::high_resolution_clock::now();
    eng.forward_batch(toks.data(), static_cast<int>(toks.size()), 0, lg_exact_b_pre.data());
    const uint32_t top1_eb = static_cast<uint32_t>(std::max_element(lg_exact_b_pre.begin(), lg_exact_b_pre.end()) - lg_exact_b_pre.begin());
    eng.forward(top1_es, static_cast<int>(toks.size()), lg_exact_b_dec.data());
    const uint32_t top2_eb = static_cast<uint32_t>(std::max_element(lg_exact_b_dec.begin(), lg_exact_b_dec.end()) - lg_exact_b_dec.begin());
    auto t3 = std::chrono::high_resolution_clock::now();

    // 2. Fast Q8 mode: single forward() vs batched forward_batch() (and vs Exact FP32 ground truth!)
    eng.set_exact_math(false);
    eng.get_kv_cache().reset();
    auto t4 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < toks.size(); ++i) {
        eng.forward(toks[i], static_cast<int>(i), (i + 1 == toks.size()) ? lg_fast_s_pre.data() : nullptr);
    }
    const uint32_t top1_fs = static_cast<uint32_t>(std::max_element(lg_fast_s_pre.begin(), lg_fast_s_pre.end()) - lg_fast_s_pre.begin());
    auto t5 = std::chrono::high_resolution_clock::now();

    eng.get_kv_cache().reset();
    auto t6 = std::chrono::high_resolution_clock::now();
    eng.forward_batch(toks.data(), static_cast<int>(toks.size()), 0, lg_fast_b_pre.data());
    const uint32_t top1_fb = static_cast<uint32_t>(std::max_element(lg_fast_b_pre.begin(), lg_fast_b_pre.end()) - lg_fast_b_pre.begin());
    auto t7 = std::chrono::high_resolution_clock::now();

    double max_ep = 0.0, max_ed = 0.0, max_fvs_e = 0.0, max_fb_vs_e = 0.0;
    const double cos_ep     = measure_diff(lg_exact_s_pre, lg_exact_b_pre, max_ep);
    const double cos_ed     = measure_diff(lg_exact_s_dec, lg_exact_b_dec, max_ed);
    const double cos_fs_e   = measure_diff(lg_exact_s_pre, lg_fast_s_pre, max_fvs_e);
    const double cos_fb_e   = measure_diff(lg_exact_s_pre, lg_fast_b_pre, max_fb_vs_e);

    std::cout << "  [EXACT single vs batch prefill] cosine=" << std::fixed << std::setprecision(9) << cos_ep
              << " | max_abs=" << std::scientific << std::setprecision(3) << max_ep
              << " | Top1=" << top1_eb << " vs " << top1_es << " (" << (top1_eb == top1_es ? "MATCH" : "DIFF") << ")"
              << " | single=" << std::fixed << std::setprecision(0) << std::chrono::duration<double, std::milli>(t1 - t0).count()
              << " ms, batch=" << std::chrono::duration<double, std::milli>(t3 - t2).count() << " ms\n";
    std::cout << "  [EXACT decode after batch]      cosine=" << std::fixed << std::setprecision(9) << cos_ed
              << " | max_abs=" << std::scientific << std::setprecision(3) << max_ed
              << " | Top1=" << top2_eb << " vs " << top2_es << " (" << (top2_eb == top2_es ? "MATCH" : "DIFF") << ")\n";
    std::cout << "  [FAST  single vs EXACT truth]   cosine=" << std::fixed << std::setprecision(7) << cos_fs_e
              << " | max_abs=" << std::scientific << std::setprecision(3) << max_fvs_e
              << " | Top1=" << top1_fs << " vs " << top1_es << " (" << (top1_fs == top1_es ? "MATCH" : "DIFF") << ")"
              << " | " << std::fixed << std::setprecision(0) << std::chrono::duration<double, std::milli>(t5 - t4).count() << " ms\n";
    std::cout << "  [FAST  batch  vs EXACT truth]   cosine=" << std::fixed << std::setprecision(7) << cos_fb_e
              << " | max_abs=" << std::scientific << std::setprecision(3) << max_fb_vs_e
              << " | Top1=" << top1_fb << " vs " << top1_es << " (" << (top1_fb == top1_es ? "MATCH" : "DIFF") << ")"
              << " | " << std::fixed << std::setprecision(0) << std::chrono::duration<double, std::milli>(t7 - t6).count() << " ms\n";
    HAVEN_CHECK(cos_ep > 0.9999999 && cos_ed > 0.9999999, "Exact single vs batch prefill/decode diverged below 0.9999999!");
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    std::cout << std::unitbuf;
    std::string gguf_path;
    bool strict = true;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--gguf" && i + 1 < argc) gguf_path = argv[++i];
        else if (a == "--strict") strict = true;
        else if (a == "--relaxed") strict = false;
    }

    std::cout << "======================================================================\n";
    std::cout << "  HAVEN-CPP GOLDEN FP64/FP32 MATH PARITY PROFILER SUITE\n";
    std::cout << "======================================================================\n";

    if (test_rmsnorm_parity(strict) != 0) return 1;
    if (test_rope_parity(strict) != 0) return 1;
    if (test_softmax_and_flash_attention(strict) != 0) return 1;
    if (test_fwht_and_activations() != 0) return 1;
    if (test_quant_gemv_parity() != 0) return 1;
    if (test_long_context_swa_and_sinks_parity() != 0) return 1;
    if (test_multi_arch_e2e_exact_vs_fast() != 0) return 1;
#ifdef HAVEN_HAS_TPU
    if (test_tpu_w4_quantizer_quality(strict) != 0) return 1;
#endif

    if (!gguf_path.empty()) {
        if (run_live_gguf_parity(gguf_path) != 0) return 1;
    }

    std::cout << "======================================================================\n";
    std::cout << "  ALL MATH PARITY CHECKS PASSED!\n";
    std::cout << "======================================================================\n";
    return 0;
}
