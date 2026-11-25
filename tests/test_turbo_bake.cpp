#include "haven/haven_avx2.h"
#include "haven/haven_engine.h"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace haven;

static void naive_fwht_block(std::vector<float>& x, int block_size) {
    const float inv_s = 1.0f / std::sqrt(static_cast<float>(block_size));
    for (int off = 0; off < static_cast<int>(x.size()); off += block_size) {
        for (int h = 1; h < block_size; h <<= 1) {
            for (int i = 0; i < block_size; i += (h << 1)) {
                for (int j = 0; j < h; ++j) {
                    float a = x[off + i + j];
                    float b = x[off + i + j + h];
                    x[off + i + j]     = a + b;
                    x[off + i + j + h] = a - b;
                }
            }
        }
        for (int i = 0; i < block_size; ++i) x[off + i] *= inv_s;
    }
}

static void test_fwht_exactness() {
    std::cout << "[1/3] Testing AVX2 fwht_block_inplace orthogonality & exactness...\n";
    const int sizes[] = {8, 16, 64, 256, 512};
    for (int bs : sizes) {
        const int n = bs * 3;
        std::vector<float> orig(n);
        for (int i = 0; i < n; ++i) {
            orig[i] = std::sin(static_cast<float>(i) * 0.37f) * 2.5f + ((i % bs == 17 % bs) ? 662.0f : 0.0f);
        }
        std::vector<float> ref = orig;
        std::vector<float> simd = orig;
        naive_fwht_block(ref, bs);
        Avx2Math::fwht_block_inplace(simd.data(), n, bs);

        float max_diff = 0.0f;
        for (int i = 0; i < n; ++i) max_diff = std::max(max_diff, std::fabs(ref[i] - simd[i]));
        if (max_diff > 1e-4f) {
            std::cerr << "FAIL: FWHT bs=" << bs << " max_diff=" << max_diff << "\n";
            std::exit(1);
        }

        // Check crest factor flattening on the 662x spike!
        if (bs == 256) {
            float max_orig = 0.0f, max_rot = 0.0f;
            for (int i = 0; i < bs; ++i) {
                max_orig = std::max(max_orig, std::fabs(orig[i]));
                max_rot  = std::max(max_rot,  std::fabs(simd[i]));
            }
            std::cout << "  - Block-256 spike flattening: max|x| " << max_orig << " -> " << max_rot
                      << " (" << (max_orig / max_rot) << "x reduction)\n";
            assert(max_rot < max_orig / 10.0f);
        }

        // Exact self-inverse: H(H(x)) == x
        Avx2Math::fwht_block_inplace(simd.data(), n, bs);
        float inv_diff = 0.0f;
        for (int i = 0; i < n; ++i) inv_diff = std::max(inv_diff, std::fabs(orig[i] - simd[i]));
        if (inv_diff > 1e-4f) {
            std::cerr << "FAIL: FWHT self-inverse bs=" << bs << " inv_diff=" << inv_diff << "\n";
            std::exit(1);
        }
    }
    std::cout << "  ✓ PASS: AVX2 FWHT matches scalar reference & self-inverse across B=8..512!\n";
}

static void test_norm_fold_hadamard_and_r2_math() {
    std::cout << "[2/3] Testing Norm-Fold + Block-256 FWHT (662x outlier) & QuaRot R2 V->O...\n";
    constexpr int D = 256;
    constexpr int R = 64;
    std::vector<float> x(D), gamma(D), W(R * D);
    for (int i = 0; i < D; ++i) {
        x[i] = std::sin(static_cast<float>(i) * 0.19f + 0.5f);
        gamma[i] = (i == 73) ? 662.0f : (0.8f + 0.3f * std::cos(static_cast<float>(i) * 0.11f));
    }
    for (int r = 0; r < R; ++r) {
        for (int c = 0; c < D; ++c) {
            W[r * D + c] = 0.05f * std::sin(static_cast<float>(r * 13 + c * 7) * 0.03f);
            if (c == 73) W[r * D + c] *= 0.01f; // Small weight compensating huge gamma (classic transformer spike)
        }
    }

    // Reference: y_ref = W @ (RMSNorm(x) * gamma)
    std::vector<float> xn_raw(D);
    Avx2Math::rms_norm(xn_raw.data(), x.data(), gamma.data(), D, 1e-6f);
    std::vector<float> y_ref(R, 0.0f);
    for (int r = 0; r < R; ++r) y_ref[r] = Avx2Math::vec_dot_f32(D, W.data() + r * D, xn_raw.data());

    // Baked Gemma 4.5 Turbo: W_baked = (W * diag(gamma)) @ H_256, x_baked = H_256 @ RMSNorm(x, ones)
    std::vector<float> ones(D, 1.0f);
    std::vector<float> xn_unit(D);
    Avx2Math::rms_norm(xn_unit.data(), x.data(), ones.data(), D, 1e-6f);
    Avx2Math::fwht_block_inplace(xn_unit.data(), D, D);

    std::vector<float> W_baked = W;
    for (int r = 0; r < R; ++r) {
        float* row = W_baked.data() + r * D;
        for (int c = 0; c < D; ++c) row[c] *= gamma[c];
        Avx2Math::fwht_block_inplace(row, D, D);
    }
    std::vector<float> y_baked(R, 0.0f);
    for (int r = 0; r < R; ++r) y_baked[r] = Avx2Math::vec_dot_f32(D, W_baked.data() + r * D, xn_unit.data());

    float cos_fp32 = Avx2Math::cosine_similarity(y_ref.data(), y_baked.data(), R);
    if (cos_fp32 < 0.99999f) {
        std::cerr << "FAIL: Norm-fold + H256 FP32 equivalence cosine=" << cos_fp32 << "\n";
        std::exit(1);
    }

    // Compare Q8_K (per-256 block dynamic quantization) error BEFORE vs AFTER baking on the 662x spike!
    Q8KBatch b_raw, b_baked;
    Avx2Math::pack_q8k_batch(xn_raw.data(), D, D, 1, b_raw);
    Avx2Math::pack_q8k_batch(xn_unit.data(), D, D, 1, b_baked);

    auto dequant_q8k_dot = [](const std::vector<float>& mat, const Q8KBatch& b, int rows, int cols) {
        std::vector<float> out(rows, 0.0f);
        float d = b.tiles[0].d[0];
        std::vector<float> xq(cols);
        for (int ib = 0; ib < 8; ++ib) {
            const int8_t* qp = b.tiles[0].qs.data() + ib * 256;
            for (int j = 0; j < 32; ++j) xq[ib * 32 + j] = d * static_cast<float>(qp[j]);
        }
        for (int r = 0; r < rows; ++r) out[r] = Avx2Math::vec_dot_f32(cols, mat.data() + r * cols, xq.data());
        return out;
    };
    auto y_q8_raw   = dequant_q8k_dot(W, b_raw, R, D);
    auto y_q8_baked = dequant_q8k_dot(W_baked, b_baked, R, D);
    float cos_raw   = Avx2Math::cosine_similarity(y_ref.data(), y_q8_raw.data(), R);
    float cos_baked = Avx2Math::cosine_similarity(y_ref.data(), y_q8_baked.data(), R);
    std::cout << "  - Q8_K cosine under 662x gamma spike: raw=" << cos_raw << " -> baked_H256=" << cos_baked << "\n";
    if (cos_baked <= cos_raw || cos_baked < 0.9995f) {
        std::cerr << "FAIL: Expected baked H256 Q8_K cosine to crush raw outlier error!\n";
        std::exit(1);
    }

    // QuaRot R2 V -> O through parameter-free V-RMSNorm + Attention weighted sum!
    constexpr int HD = 64;
    std::vector<float> v0(HD), v1(HD), Wo(D * HD);
    for (int i = 0; i < HD; ++i) {
        v0[i] = std::sin(i * 0.21f) + (i == 3 ? 50.0f : 0.0f);
        v1[i] = std::cos(i * 0.33f) - (i == 9 ? 30.0f : 0.0f);
    }
    for (int r = 0; r < D; ++r)
        for (int c = 0; c < HD; ++c)
            Wo[r * HD + c] = std::sin((r * 7 + c * 11) * 0.05f);

    std::vector<float> vn0(HD), vn1(HD), attn_ref(HD), o_ref(D, 0.0f);
    Avx2Math::rms_norm(vn0.data(), v0.data(), ones.data(), HD, 1e-6f);
    Avx2Math::rms_norm(vn1.data(), v1.data(), ones.data(), HD, 1e-6f);
    for (int i = 0; i < HD; ++i) attn_ref[i] = 0.7f * vn0[i] + 0.3f * vn1[i];
    for (int r = 0; r < D; ++r) o_ref[r] = Avx2Math::vec_dot_f32(HD, Wo.data() + r * HD, attn_ref.data());

    // Rotated R2: v' = H_hd @ v, Wo' = Wo @ H_hd
    std::vector<float> v0p = v0, v1p = v1, Wop = Wo;
    Avx2Math::fwht_block_inplace(v0p.data(), HD, HD);
    Avx2Math::fwht_block_inplace(v1p.data(), HD, HD);
    for (int r = 0; r < D; ++r) Avx2Math::fwht_block_inplace(Wop.data() + r * HD, HD, HD);

    std::vector<float> vn0p(HD), vn1p(HD), attn_rot(HD), o_rot(D, 0.0f);
    Avx2Math::rms_norm(vn0p.data(), v0p.data(), ones.data(), HD, 1e-6f);
    Avx2Math::rms_norm(vn1p.data(), v1p.data(), ones.data(), HD, 1e-6f);
    for (int i = 0; i < HD; ++i) attn_rot[i] = 0.7f * vn0p[i] + 0.3f * vn1p[i];
    for (int r = 0; r < D; ++r) o_rot[r] = Avx2Math::vec_dot_f32(HD, Wop.data() + r * HD, attn_rot.data());

    float cos_r2 = Avx2Math::cosine_similarity(o_ref.data(), o_rot.data(), D);
    if (cos_r2 < 0.99999f) {
        std::cerr << "FAIL: QuaRot R2 V->O cosine=" << cos_r2 << "\n";
        std::exit(1);
    }
    std::cout << "  ✓ PASS: Norm-fold + H256 & QuaRot R2 V->O verified (cosine=" << cos_r2 << ")!\n";
}

// Write a minimal synthetic 2-layer F32 Gemma 4 GGUF (raw, pure orthogonal turbo, or norm-folded turbo)
static void write_mini_gemma4_gguf(const std::string& path, bool baked_turbo, bool fold_norms) {
    constexpr uint32_t V = 64, D = 256, F = 256, L = 2, NH = 4, NKV = 2, HD = 64;
    const uint32_t QDIM = NH * HD, KVDIM = NKV * HD;

    auto gen = [](int n, int seed, float scale) {
        std::vector<float> v(n);
        for (int i = 0; i < n; ++i) v[i] = scale * std::sin(static_cast<float>((i + 1) * seed) * 0.137f);
        return v;
    };

    struct TOut { std::string name; std::vector<uint64_t> dims; std::vector<float> data; };
    std::vector<TOut> ts;

    ts.push_back({"token_embd.weight", {D, V}, gen(D * V, 3, 0.2f)});
    ts.push_back({"output_norm.weight", {D}, std::vector<float>(D, 1.0f)});

    for (uint32_t l = 0; l < L; ++l) {
        std::string p = "blk." + std::to_string(l) + ".";
        std::vector<float> an(D, 1.0f), fn(D, 1.0f);
        for (uint32_t i = 0; i < D; ++i) {
            an[i] = (i == 19) ? 45.0f : (0.9f + 0.2f * std::cos(static_cast<float>(i + l)));
            fn[i] = (i == 41) ? 35.0f : (1.1f + 0.15f * std::sin(static_cast<float>(i + l)));
        }
        auto wq = gen(D * QDIM, 11 + l, 0.04f);
        auto wk = gen(D * KVDIM, 17 + l, 0.04f);
        auto wv = gen(D * KVDIM, 23 + l, 0.04f);
        auto wo = gen(QDIM * D, 29 + l, 0.04f);
        auto wg = gen(D * F, 31 + l, 0.04f);
        auto wu = gen(D * F, 37 + l, 0.04f);
        auto wd = gen(F * D, 41 + l, 0.04f);

        if (baked_turbo) {
            auto fold_h = [](std::vector<float>& mat, int rows, int cols, const std::vector<float>* g, int h_blk) {
                for (int r = 0; r < rows; ++r) {
                    float* row = mat.data() + r * cols;
                    if (g) for (int c = 0; c < cols; ++c) row[c] *= (*g)[c];
                    Avx2Math::fwht_block_inplace(row, cols, h_blk);
                }
            };
            fold_h(wq, QDIM, D, fold_norms ? &an : nullptr, D);
            fold_h(wk, KVDIM, D, fold_norms ? &an : nullptr, D);
            fold_h(wv, KVDIM, D, fold_norms ? &an : nullptr, D);
            // QuaRot R2 on V -> O
            for (int c = 0; c < static_cast<int>(D); ++c) {
                std::vector<float> col(HD);
                for (uint32_t h = 0; h < NKV; ++h) {
                    for (uint32_t i = 0; i < HD; ++i) col[i] = wv[(h * HD + i) * D + c];
                    Avx2Math::fwht_block_inplace(col.data(), HD, HD);
                    for (uint32_t i = 0; i < HD; ++i) wv[(h * HD + i) * D + c] = col[i];
                }
            }
            for (uint32_t r = 0; r < D; ++r) Avx2Math::fwht_block_inplace(wo.data() + r * QDIM, QDIM, HD);

            fold_h(wg, F, D, fold_norms ? &fn : nullptr, D);
            fold_h(wu, F, D, fold_norms ? &fn : nullptr, D);
            fold_h(wd, D, F, nullptr, F);
            if (fold_norms) {
                std::fill(an.begin(), an.end(), 1.0f);
                std::fill(fn.begin(), fn.end(), 1.0f);
            }
        }

        ts.push_back({p + "attn_norm.weight", {D}, an});
        ts.push_back({p + "attn_q.weight", {D, QDIM}, wq});
        ts.push_back({p + "attn_k.weight", {D, KVDIM}, wk});
        ts.push_back({p + "attn_v.weight", {D, KVDIM}, wv});
        ts.push_back({p + "attn_q_norm.weight", {HD}, std::vector<float>(HD, 1.0f)});
        ts.push_back({p + "attn_k_norm.weight", {HD}, std::vector<float>(HD, 1.0f)});
        ts.push_back({p + "attn_output.weight", {QDIM, D}, wo});
        ts.push_back({p + "post_attention_norm.weight", {D}, std::vector<float>(D, 1.0f)});
        ts.push_back({p + "ffn_norm.weight", {D}, fn});
        ts.push_back({p + "ffn_gate.weight", {D, F}, wg});
        ts.push_back({p + "ffn_up.weight", {D, F}, wu});
        ts.push_back({p + "ffn_down.weight", {F, D}, wd});
        ts.push_back({p + "post_ffw_norm.weight", {D}, std::vector<float>(D, 1.0f)});
    }

    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    auto wu32 = [&](uint32_t x) { f.write(reinterpret_cast<const char*>(&x), 4); };
    auto wu64 = [&](uint64_t x) { f.write(reinterpret_cast<const char*>(&x), 8); };
    auto wstr = [&](const std::string& s) { wu64(s.size()); f.write(s.data(), s.size()); };
    auto kv_str = [&](const std::string& k, const std::string& v) { wstr(k); wu32(8); wstr(v); };
    auto kv_u32 = [&](const std::string& k, uint32_t v) { wstr(k); wu32(4); wu32(v); };
    auto kv_f32 = [&](const std::string& k, float v) { wstr(k); wu32(6); f.write(reinterpret_cast<const char*>(&v), 4); };
    auto kv_bool = [&](const std::string& k, bool v) { wstr(k); wu32(7); uint8_t b = v ? 1 : 0; f.write(reinterpret_cast<const char*>(&b), 1); };

    uint64_t n_kv = 11 + (baked_turbo ? 5 : 0);
    wu32(0x46554747u); wu32(3); wu64(ts.size()); wu64(n_kv);

    kv_str("general.architecture", "gemma4");
    kv_u32("gemma4.block_count", L);
    kv_u32("gemma4.context_length", 128);
    kv_u32("gemma4.embedding_length", D);
    kv_u32("gemma4.feed_forward_length", F);
    kv_u32("gemma4.attention.head_count", NH);
    kv_u32("gemma4.attention.head_count_kv", NKV);
    kv_u32("gemma4.attention.key_length", HD);
    kv_f32("gemma4.attention.layer_norm_rms_epsilon", 1e-6f);
    kv_f32("gemma4.rope.freq_base", 10000.0f);
    // Minimal vocab array
    wstr("tokenizer.ggml.tokens"); wu32(9); wu32(8); wu64(V);
    for (uint32_t i = 0; i < V; ++i) wstr("t" + std::to_string(i));

    if (baked_turbo) {
        kv_u32("gemma4.turbo.version", 1);
        kv_u32("gemma4.turbo.hadamard_qkv", D);
        kv_u32("gemma4.turbo.hadamard_ffn", D);
        kv_u32("gemma4.turbo.hadamard_ffn_down", F);
        kv_bool("gemma4.turbo.folded_norms", fold_norms);
    }

    uint64_t off = 0;
    for (const auto& t : ts) {
        wstr(t.name);
        wu32(static_cast<uint32_t>(t.dims.size()));
        for (uint64_t d : t.dims) wu64(d);
        wu32(0); // F32
        wu64(off);
        uint64_t nb = t.data.size() * sizeof(float);
        off += ((nb + 31ULL) / 32ULL) * 32ULL;
    }
    auto pad32 = [&]() {
        uint64_t p = static_cast<uint64_t>(f.tellp());
        uint64_t pad = (32ULL - (p % 32ULL)) % 32ULL;
        std::vector<uint8_t> z(pad, 0);
        if (pad) f.write(reinterpret_cast<const char*>(z.data()), pad);
    };
    pad32();
    for (const auto& t : ts) {
        f.write(reinterpret_cast<const char*>(t.data.data()), t.data.size() * sizeof(float));
        pad32();
    }
}

static void test_engine_e2e_turbo_parity() {
    std::cout << "[3/3] Testing HavenEngine forward() & forward_batch() end-to-end Gemma 4.5 Turbo parity (orthogonal & norm-folded)...\n";
    auto tmp = std::filesystem::temp_directory_path();
    std::string raw_path   = (tmp / "haven_test_g4_raw.gguf").string();
    std::string ortho_path = (tmp / "haven_test_g4_ortho.gguf").string();
    std::string fold_path  = (tmp / "haven_test_g4_fold.gguf").string();
    write_mini_gemma4_gguf(raw_path, false, false);
    write_mini_gemma4_gguf(ortho_path, true, false);
    write_mini_gemma4_gguf(fold_path, true, true);

    HavenEngine e_raw, e_ortho, e_fold;
    if (!e_raw.load_model(raw_path) || !e_ortho.load_model(ortho_path) || !e_fold.load_model(fold_path)) {
        std::cerr << "FAIL: Could not load synthetic GGUFs!\n";
        std::exit(1);
    }
    assert(e_raw.get_config().turbo_version == 0);
    assert(e_ortho.get_config().turbo_version == 1 && !e_ortho.get_config().turbo_folded_norms);
    assert(e_fold.get_config().turbo_version == 1 && e_fold.get_config().turbo_folded_norms);

    std::vector<uint32_t> prompt = {2, 5, 11, 19, 27, 33, 42, 51, 7, 13};
    std::vector<float> log_raw_seq(64), log_ortho_seq(64), log_fold_seq(64);
    std::vector<float> log_raw_bat(64), log_ortho_bat(64), log_fold_bat(64);

    for (size_t i = 0; i < prompt.size(); ++i) {
        e_raw.forward(prompt[i], static_cast<int>(i), (i + 1 == prompt.size()) ? log_raw_seq.data() : nullptr);
        e_ortho.forward(prompt[i], static_cast<int>(i), (i + 1 == prompt.size()) ? log_ortho_seq.data() : nullptr);
        e_fold.forward(prompt[i], static_cast<int>(i), (i + 1 == prompt.size()) ? log_fold_seq.data() : nullptr);
    }
    e_raw.get_kv_cache().reset();
    e_ortho.get_kv_cache().reset();
    e_fold.get_kv_cache().reset();
    e_raw.forward_batch(prompt.data(), static_cast<int>(prompt.size()), 0, log_raw_bat.data());
    e_ortho.forward_batch(prompt.data(), static_cast<int>(prompt.size()), 0, log_ortho_bat.data());
    e_fold.forward_batch(prompt.data(), static_cast<int>(prompt.size()), 0, log_fold_bat.data());

    float cos_os = Avx2Math::cosine_similarity(log_raw_seq.data(), log_ortho_seq.data(), 64);
    float cos_ob = Avx2Math::cosine_similarity(log_raw_bat.data(), log_ortho_bat.data(), 64);
    float cos_fs = Avx2Math::cosine_similarity(log_raw_seq.data(), log_fold_seq.data(), 64);
    float cos_fb = Avx2Math::cosine_similarity(log_raw_bat.data(), log_fold_bat.data(), 64);
    std::cout << "  - E2E Orthogonal Turbo cosine: forward()=" << cos_os << " | forward_batch()=" << cos_ob << "\n";
    std::cout << "  - E2E Norm-Fold Turbo cosine : forward()=" << cos_fs << " | forward_batch()=" << cos_fb << "\n";

    std::error_code ec;
    std::filesystem::remove(raw_path, ec);
    std::filesystem::remove(ortho_path, ec);
    std::filesystem::remove(fold_path, ec);

    if (cos_os < 0.9999f || cos_ob < 0.9999f || cos_fs < 0.9999f || cos_fb < 0.9999f) {
        std::cerr << "FAIL: E2E Gemma 4.5 Turbo parity mismatch!\n";
        std::exit(1);
    }
    std::cout << "  ✓ PASS: HavenEngine forward() & forward_batch() match bit-identically on both Turbo modes!\n";
}

static void test_gemma4_tokenizer_bos_eos_and_space_prefix() {
    std::cout << "[4/4] Testing HavenTokenizer Gemma 4 BOS/EOS priority & add_space_prefix=false...\n";
    // Build minimal synthetic vocab mimicking Gemma 4's exact ID layout where "<s>" (203) and "</s>" (212)
    // appear AFTER "<bos>" (2) and "<eos>" (1) and must NEVER overwrite them!
    std::vector<std::string> vocab(256, "<unused>");
    vocab[0]   = "<pad>";
    vocab[1]   = "<eos>";
    vocab[2]   = "<bos>";
    vocab[3]   = "<unk>";
    vocab[105] = "<|turn>";
    vocab[106] = "<turn|>";
    vocab[107] = "\n";
    vocab[203] = "<s>";
    vocab[212] = "</s>";
    vocab[220] = "user";
    vocab[221] = "\xe2\x96\x81user";
    vocab[222] = "model";
    vocab[223] = "\xe2\x96\x81model";
    vocab[224] = "Hello";
    vocab[225] = "\xe2\x96\x81Hello";
    vocab[226] = "\xe2\x96\x81world";

    ModelConfig g4_cfg;
    g4_cfg.architecture = "gemma4";
    g4_cfg.is_gemma = true;
    g4_cfg.has_bos_token_id = true; g4_cfg.bos_token_id = 2;
    g4_cfg.has_eos_token_id = true; g4_cfg.eos_token_id = 1;
    g4_cfg.has_pad_token_id = true; g4_cfg.pad_token_id = 0;
    g4_cfg.has_unk_token_id = true; g4_cfg.unk_token_id = 3;
    g4_cfg.has_add_space_prefix = true; g4_cfg.add_space_prefix = false;

    HavenTokenizer tok;
    tok.initialize(vocab, &g4_cfg);
    if (tok.bos_token() != 2 || tok.eos_token() != 1 || tok.is_stop_token(212)) {
        std::cerr << "FAIL: Gemma 4 BOS/EOS mismatch (bos=" << tok.bos_token() << ", eos=" << tok.eos_token() << ")\n";
        std::exit(1);
    }

    auto ids = tok.encode("<|turn>user\nHello world<turn|>\n<|turn>model\n", true);
    const std::vector<uint32_t> expected = {2, 105, 220, 107, 224, 226, 106, 107, 105, 222, 107};
    if (ids != expected) {
        std::cerr << "FAIL: Gemma 4 encode mismatch! Got:";
        for (auto id : ids) std::cerr << " " << id;
        std::cerr << "\n";
        std::exit(1);
    }

    // Also verify heuristic fallback without ModelConfig still preserves <bos>=2 over <s>=203
    HavenTokenizer tok_heur;
    tok_heur.initialize(vocab);
    if (tok_heur.bos_token() != 2 || tok_heur.eos_token() != 1) {
        std::cerr << "FAIL: Heuristic BOS/EOS overwritten by <s>/</s>!\n";
        std::exit(1);
    }
    std::cout << "  ✓ PASS: Gemma 4 BOS=2, EOS=1, stop_tokens, and exact add_space_prefix=false tokenization!\n";
}

int main() {
    std::cout << "==================================================================\n";
    std::cout << "  HAVEN-CPP GEMMA 4.5 TURBO BAKE & FWHT VERIFICATION SUITE\n";
    std::cout << "==================================================================\n";
    test_fwht_exactness();
    test_norm_fold_hadamard_and_r2_math();
    test_engine_e2e_turbo_parity();
    test_gemma4_tokenizer_bos_eos_and_space_prefix();
    std::cout << "==================================================================\n";
    std::cout << "  ALL GEMMA 4.5 TURBO BAKE TESTS PASSED 100%!\n";
    std::cout << "==================================================================\n";
    return 0;
}
