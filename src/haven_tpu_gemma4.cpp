#include "haven/haven_tpu_gemma4.h"
#include "haven/haven_avx2.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace haven {

namespace {

static std::string fmt_f32(float v) {
    std::ostringstream ss;
    ss << std::scientific << std::setprecision(7) << v;
    return ss.str();
}

static std::string replica_group_attr(int R) {
    std::ostringstream ss;
    ss << "dense<[[";
    for (int r = 0; r < R; ++r) {
        if (r > 0) ss << ", ";
        ss << r;
    }
    ss << "]]> : tensor<1x" << R << "xi64>";
    return ss.str();
}

// Quantize a [rows, cols] float row-major slice into symmetric INT8 + per-row F32 scales
static void quantize_f32_to_s8_per_row(const float* src, int rows, int cols,
                                       int8_t* dst_q, float* dst_scale) {
    for (int r = 0; r < rows; ++r) {
        const float* row = src + static_cast<size_t>(r) * static_cast<size_t>(cols);
        int8_t* qrow = dst_q + static_cast<size_t>(r) * static_cast<size_t>(cols);
        float max_abs = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float a = std::fabs(row[c]);
            if (a > max_abs) max_abs = a;
        }
        const float scale = (max_abs > 1e-12f) ? (max_abs / 127.0f) : 1.0f;
        const float inv_scale = (max_abs > 1e-12f) ? (127.0f / max_abs) : 0.0f;
        dst_scale[r] = scale;
        for (int c = 0; c < cols; ++c) {
            int q = static_cast<int>(std::lrintf(row[c] * inv_scale));
            q = std::clamp(q, -127, 127);
            qrow[c] = static_cast<int8_t>(q);
        }
    }
}

// Dequantize a full GGUF matrix [total_rows, total_cols] into F32 host buffer in parallel
static void dequantize_full_matrix_f32(const TensorDesc* t, int rows, int cols, std::vector<float>& out) {
    out.assign(static_cast<size_t>(rows) * static_cast<size_t>(cols), 0.0f);
    if (!t || !t->data) return;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) {
        Avx2Math::dequantize_row(out.data() + static_cast<size_t>(r) * static_cast<size_t>(cols), *t, r, cols);
    }
}

static std::vector<float> extract_f32_1d(const TensorDesc* t, int count, float fallback_val = 1.0f) {
    std::vector<float> out(static_cast<size_t>(count), fallback_val);
    if (t && t->data && t->type == QuantType::F32 &&
        (t->size_bytes == 0 || t->size_bytes >= static_cast<size_t>(count) * sizeof(float))) {
        std::memcpy(out.data(), t->data, static_cast<size_t>(count) * sizeof(float));
    }
    return out;
}

// Helper for clean StableHLO SSA emission
struct MlirEmitter {
    std::ostringstream ss;
    int next_id = 0;

    std::string tmp(const char* prefix = "v") {
        return "%" + std::string(prefix) + std::to_string(next_id++);
    }

    // Emit RMSNorm on 3D tensor [B, S, D] (F32) with optional 1D weight [D] (F32)
    std::string emit_rms_norm_3d(const std::string& x, const std::string& w_1d,
                                 int B, int S, int D, float eps, const std::string& c_zero) {
        const std::string t3 = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(D) + "xf32>";
        const std::string t2 = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "xf32>";
        const std::string t1 = "tensor<" + std::to_string(D) + "xf32>";

        std::string sq = tmp("sq");
        ss << "    " << sq << " = stablehlo.multiply " << x << ", " << x << " : " << t3 << "\n";
        std::string sum = tmp("rsum");
        ss << "    " << sum << " = stablehlo.reduce(" << sq << " init: " << c_zero << ") across dimensions = [2] : ("
           << t3 << ", tensor<f32>) -> " << t2 << "\n"
           << "      reducer(%ra: tensor<f32>, %rb: tensor<f32>) {\n"
           << "        %rc = stablehlo.add %ra, %rb : tensor<f32>\n"
           << "        stablehlo.return %rc : tensor<f32>\n"
           << "      }\n";
        std::string c_inv = tmp("invd");
        std::string c_eps = tmp("eps");
        ss << "    " << c_inv << " = stablehlo.constant dense<" << fmt_f32(1.0f / static_cast<float>(D)) << "> : " << t2 << "\n";
        ss << "    " << c_eps << " = stablehlo.constant dense<" << fmt_f32(eps) << "> : " << t2 << "\n";
        std::string mean = tmp("mean");
        std::string var  = tmp("var");
        std::string rsq  = tmp("rsq");
        std::string rsqb = tmp("rsqb");
        std::string xn   = tmp("xn");
        ss << "    " << mean << " = stablehlo.multiply " << sum << ", " << c_inv << " : " << t2 << "\n";
        ss << "    " << var  << " = stablehlo.add " << mean << ", " << c_eps << " : " << t2 << "\n";
        ss << "    " << rsq  << " = stablehlo.rsqrt " << var << " : " << t2 << "\n";
        ss << "    " << rsqb << " = stablehlo.broadcast_in_dim " << rsq << ", dims = [0, 1] : (" << t2 << ") -> " << t3 << "\n";
        ss << "    " << xn   << " = stablehlo.multiply " << x << ", " << rsqb << " : " << t3 << "\n";
        if (w_1d.empty()) return xn;
        std::string wb  = tmp("wb");
        std::string out = tmp("rn");
        ss << "    " << wb  << " = stablehlo.broadcast_in_dim " << w_1d << ", dims = [2] : (" << t1 << ") -> " << t3 << "\n";
        ss << "    " << out << " = stablehlo.multiply " << xn << ", " << wb << " : " << t3 << "\n";
        return out;
    }

    // Emit RMSNorm on 4D per-head tensor [B, S, H, HD] (F32) with optional 1D weight [HD] (F32)
    std::string emit_rms_norm_4d(const std::string& x, const std::string& w_1d,
                                 int B, int S, int H, int HD, float eps, const std::string& c_zero) {
        const std::string t4 = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(H) + "x" + std::to_string(HD) + "xf32>";
        const std::string t3 = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(H) + "xf32>";
        const std::string t1 = "tensor<" + std::to_string(HD) + "xf32>";

        std::string sq = tmp("hsq");
        ss << "    " << sq << " = stablehlo.multiply " << x << ", " << x << " : " << t4 << "\n";
        std::string sum = tmp("hsum");
        ss << "    " << sum << " = stablehlo.reduce(" << sq << " init: " << c_zero << ") across dimensions = [3] : ("
           << t4 << ", tensor<f32>) -> " << t3 << "\n"
           << "      reducer(%ra: tensor<f32>, %rb: tensor<f32>) {\n"
           << "        %rc = stablehlo.add %ra, %rb : tensor<f32>\n"
           << "        stablehlo.return %rc : tensor<f32>\n"
           << "      }\n";
        std::string c_inv = tmp("hinv");
        std::string c_eps = tmp("heps");
        ss << "    " << c_inv << " = stablehlo.constant dense<" << fmt_f32(1.0f / static_cast<float>(HD)) << "> : " << t3 << "\n";
        ss << "    " << c_eps << " = stablehlo.constant dense<" << fmt_f32(eps) << "> : " << t3 << "\n";
        std::string mean = tmp("hmean");
        std::string var  = tmp("hvar");
        std::string rsq  = tmp("hrsq");
        std::string rsqb = tmp("hrsqb");
        std::string xn   = tmp("hxn");
        ss << "    " << mean << " = stablehlo.multiply " << sum << ", " << c_inv << " : " << t3 << "\n";
        ss << "    " << var  << " = stablehlo.add " << mean << ", " << c_eps << " : " << t3 << "\n";
        ss << "    " << rsq  << " = stablehlo.rsqrt " << var << " : " << t3 << "\n";
        ss << "    " << rsqb << " = stablehlo.broadcast_in_dim " << rsq << ", dims = [0, 1, 2] : (" << t3 << ") -> " << t4 << "\n";
        ss << "    " << xn   << " = stablehlo.multiply " << x << ", " << rsqb << " : " << t4 << "\n";
        if (w_1d.empty()) return xn;
        std::string wb  = tmp("hwb");
        std::string out = tmp("hrn");
        ss << "    " << wb  << " = stablehlo.broadcast_in_dim " << w_1d << ", dims = [3] : (" << t1 << ") -> " << t4 << "\n";
        ss << "    " << out << " = stablehlo.multiply " << xn << ", " << wb << " : " << t4 << "\n";
        return out;
    }

    // Slice 2D layer matrix [N, K] from 3D stack [L, N, K] and multiply: [B, S, K] x [N, K]^T -> [B, S, N] (F32)
    std::string emit_linear_3d(const std::string& x_bf16,
                               const std::string& w_stack, const std::string& s_stack,
                               int layer_idx, int L, int B, int S, int N, int K, bool use_int8) {
        const std::string x_t  = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(K) + "xbf16>";
        const std::string y_t  = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(N) + "xf32>";
        const std::string el   = use_int8 ? "i8" : "bf16";
        const std::string ws_t = "tensor<" + std::to_string(L) + "x" + std::to_string(N) + "x" + std::to_string(K) + "x" + el + ">";
        const std::string w1_t = "tensor<1x" + std::to_string(N) + "x" + std::to_string(K) + "x" + el + ">";
        const std::string w2_t = "tensor<" + std::to_string(N) + "x" + std::to_string(K) + "x" + el + ">";
        const std::string wb_t = "tensor<" + std::to_string(N) + "x" + std::to_string(K) + "xbf16>";

        std::string w_sl = tmp("wsl");
        std::string w_2d = tmp("w2d");
        ss << "    " << w_sl << " = stablehlo.slice " << w_stack << " [" << layer_idx << ":" << (layer_idx + 1)
           << ", 0:" << N << ", 0:" << K << "] : (" << ws_t << ") -> " << w1_t << "\n";
        ss << "    " << w_2d << " = stablehlo.reshape " << w_sl << " : (" << w1_t << ") -> " << w2_t << "\n";

        std::string w_mat = w_2d;
        if (use_int8) {
            w_mat = tmp("wbf");
            ss << "    " << w_mat << " = stablehlo.convert " << w_2d << " : (" << w2_t << ") -> " << wb_t << "\n";
        }

        std::string dot = tmp("dot");
        ss << "    " << dot << " = stablehlo.dot_general " << x_bf16 << ", " << w_mat
           << ", contracting_dims = [2] x [1] : (" << x_t << ", " << wb_t << ") -> " << y_t << "\n";

        if (!use_int8) return dot;

        const std::string ss_t = "tensor<" + std::to_string(L) + "x" + std::to_string(N) + "xf32>";
        const std::string s1_t = "tensor<1x" + std::to_string(N) + "xf32>";
        const std::string s0_t = "tensor<" + std::to_string(N) + "xf32>";
        std::string s_sl = tmp("ssl");
        std::string s_1d = tmp("s1d");
        std::string s_bc = tmp("sbc");
        std::string y_sc = tmp("ysc");
        ss << "    " << s_sl << " = stablehlo.slice " << s_stack << " [" << layer_idx << ":" << (layer_idx + 1)
           << ", 0:" << N << "] : (" << ss_t << ") -> " << s1_t << "\n";
        ss << "    " << s_1d << " = stablehlo.reshape " << s_sl << " : (" << s1_t << ") -> " << s0_t << "\n";
        ss << "    " << s_bc << " = stablehlo.broadcast_in_dim " << s_1d << ", dims = [2] : (" << s0_t << ") -> " << y_t << "\n";
        ss << "    " << y_sc << " = stablehlo.multiply " << dot << ", " << s_bc << " : " << y_t << "\n";
        return y_sc;
    }

    // Exact Gemma 4 tanh-approx GELU on 3D F32 tensor [B, S, M]
    std::string emit_gelu_3d(const std::string& g, int B, int S, int M) {
        const std::string t = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(M) + "xf32>";
        std::string ch = tmp("ch"), c1 = tmp("c1"), cs = tmp("cs"), cc = tmp("cc");
        std::string g2 = tmp("g2"), g3 = tmp("g3"), cg3 = tmp("cg3"), inn = tmp("inn");
        std::string sc = tmp("sc"), th = tmp("th"), op = tmp("op"), cdf = tmp("cdf"), out = tmp("gelu");
        ss << "    " << ch << " = stablehlo.constant dense<5.0000000e-01> : " << t << "\n"
           << "    " << c1 << " = stablehlo.constant dense<1.0000000e+00> : " << t << "\n"
           << "    " << cs << " = stablehlo.constant dense<7.9788458e-01> : " << t << "\n"
           << "    " << cc << " = stablehlo.constant dense<4.4715002e-02> : " << t << "\n"
           << "    " << g2 << " = stablehlo.multiply " << g << ", " << g << " : " << t << "\n"
           << "    " << g3 << " = stablehlo.multiply " << g2 << ", " << g << " : " << t << "\n"
           << "    " << cg3 << " = stablehlo.multiply " << cc << ", " << g3 << " : " << t << "\n"
           << "    " << inn << " = stablehlo.add " << g << ", " << cg3 << " : " << t << "\n"
           << "    " << sc << " = stablehlo.multiply " << cs << ", " << inn << " : " << t << "\n"
           << "    " << th << " = stablehlo.tanh " << sc << " : " << t << "\n"
           << "    " << op << " = stablehlo.add " << c1 << ", " << th << " : " << t << "\n"
           << "    " << cdf << " = stablehlo.multiply " << ch << ", " << op << " : " << t << "\n"
           << "    " << out << " = stablehlo.multiply " << g << ", " << cdf << " : " << t << "\n";
        return out;
    }

    // Half-split RoPE on [B, S, H, HD] (F32) matching Avx2Math::apply_rope exactly
    std::string emit_rope_4d(const std::string& t, const std::string& cos_s, const std::string& sin_s,
                             int B, int S, int H, int HD) {
        const int HD2 = HD / 2;
        const std::string t4  = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(H) + "x" + std::to_string(HD) + "xf32>";
        const std::string t4h = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(H) + "x" + std::to_string(HD2) + "xf32>";
        const std::string tc  = "tensor<" + std::to_string(S) + "x" + std::to_string(HD2) + "xf32>";

        std::string t0 = tmp("rp0"), t1 = tmp("rp1"), cb = tmp("cb"), sb = tmp("sb");
        std::string m00 = tmp("m"), m11 = tmp("m"), m01 = tmp("m"), m10 = tmp("m");
        std::string o0 = tmp("ro0"), o1 = tmp("ro1"), out = tmp("rot");
        ss << "    " << t0 << " = stablehlo.slice " << t << " [0:" << B << ", 0:" << S << ", 0:" << H << ", 0:" << HD2 << "] : (" << t4 << ") -> " << t4h << "\n"
           << "    " << t1 << " = stablehlo.slice " << t << " [0:" << B << ", 0:" << S << ", 0:" << H << ", " << HD2 << ":" << HD << "] : (" << t4 << ") -> " << t4h << "\n"
           << "    " << cb << " = stablehlo.broadcast_in_dim " << cos_s << ", dims = [1, 3] : (" << tc << ") -> " << t4h << "\n"
           << "    " << sb << " = stablehlo.broadcast_in_dim " << sin_s << ", dims = [1, 3] : (" << tc << ") -> " << t4h << "\n"
           << "    " << m00 << " = stablehlo.multiply " << t0 << ", " << cb << " : " << t4h << "\n"
           << "    " << m11 << " = stablehlo.multiply " << t1 << ", " << sb << " : " << t4h << "\n"
           << "    " << o0  << " = stablehlo.subtract " << m00 << ", " << m11 << " : " << t4h << "\n"
           << "    " << m01 << " = stablehlo.multiply " << t0 << ", " << sb << " : " << t4h << "\n"
           << "    " << m10 << " = stablehlo.multiply " << t1 << ", " << cb << " : " << t4h << "\n"
           << "    " << o1  << " = stablehlo.add " << m01 << ", " << m10 << " : " << t4h << "\n"
           << "    " << out << " = stablehlo.concatenate " << o0 << ", " << o1 << ", dim = 3 : (" << t4h << ", " << t4h << ") -> " << t4 << "\n";
        return out;
    }

    // Cross-replica AllReduce sum across R chips over Optical ICI (no-op if R == 1)
    std::string emit_all_reduce_sum_3d(const std::string& x, int B, int S, int D, int R) {
        if (R <= 1) return x;
        const std::string t = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(D) + "xf32>";
        std::string out = tmp("ar");
        ss << "    " << out << " = \"stablehlo.all_reduce\"(" << x << ") ({\n"
           << "    ^bb0(%ra: tensor<f32>, %rb: tensor<f32>):\n"
           << "      %rs = stablehlo.add %ra, %rb : tensor<f32>\n"
           << "      stablehlo.return %rs : tensor<f32>\n"
           << "    }) {replica_groups = " << replica_group_attr(R) << "} : (" << t << ") -> " << t << "\n";
        return out;
    }

    // Slice 1D vector [D] from 2D stack [L, D] (F32)
    std::string slice_1d_from_2d(const std::string& stack, int idx, int L, int D) {
        const std::string t2 = "tensor<" + std::to_string(L) + "x" + std::to_string(D) + "xf32>";
        const std::string t1 = "tensor<1x" + std::to_string(D) + "xf32>";
        const std::string t0 = "tensor<" + std::to_string(D) + "xf32>";
        std::string s = tmp("nsl"), r = tmp("n1d");
        ss << "    " << s << " = stablehlo.slice " << stack << " [" << idx << ":" << (idx + 1)
           << ", 0:" << D << "] : (" << t2 << ") -> " << t1 << "\n"
           << "    " << r << " = stablehlo.reshape " << s << " : (" << t1 << ") -> " << t0 << "\n";
        return r;
    }
};

} // namespace

std::string HavenTpuGemma4Engine::emit_stablehlo_mlir(const ModelConfig& mc,
                                                      const TpuGemma4Config& tc,
                                                      int seq_chunk_len) {
    const int L = static_cast<int>(mc.num_layers); // 42
    const int D = static_cast<int>(mc.embedding_dim); // 2560
    const int F = static_cast<int>(mc.hidden_dim); // 10240
    const int V = static_cast<int>(mc.vocab_size); // 262144
    const int H = static_cast<int>(mc.num_heads); // 8
    const int Hkv = static_cast<int>(mc.num_kv_heads); // 2
    const int R = std::max(1, tc.num_replicas);
    const int B = std::max(1, tc.batch_size);
    const int S = std::max(1, seq_chunk_len);
    const int T = std::max(S, tc.max_seq_len);
    const bool i8 = tc.use_int8_weights;
    const std::string wel = i8 ? "i8" : "bf16";

    const int F_sh = F / R;
    const int V_sh = V / R;
    const int H_sh = H / R;
    const int Hkv_sh = (R == 1) ? Hkv : 1;
    const int G_sh = H_sh / Hkv_sh; // GQA group size per replica

    const int HD_swa = 256;
    const int HD_glob = 512;
    const int Q_swa_sh = H_sh * HD_swa;
    const int KV_swa_sh = Hkv_sh * HD_swa;
    const int Q_glob_sh = H_sh * HD_glob;
    const int KV_glob_sh = Hkv_sh * HD_glob;
    const int P_dim = 256;

    // Count SWA / Global & KV-owning layers (matching Gemma 4 E4B: l % 6 != 5 is SWA, l < L - 18 owns KV)
    const int kv_owner_limit = (L > 20) ? (L - 18) : L; // 24
    int N_swa = 0, N_glob = 0, N_swa_kv = 0, N_glob_kv = 0;
    std::vector<int> layer_type_idx(static_cast<size_t>(L), 0);
    std::vector<int> layer_kv_idx(static_cast<size_t>(L), -1);
    int last_swa_kv_idx = 0, last_glob_kv_idx = 0;

    for (int l = 0; l < L; ++l) {
        bool is_swa = (l % 6 != 5);
        bool has_kv = (l < kv_owner_limit);
        if (is_swa) {
            layer_type_idx[static_cast<size_t>(l)] = N_swa++;
            if (has_kv) {
                layer_kv_idx[static_cast<size_t>(l)] = N_swa_kv;
                last_swa_kv_idx = N_swa_kv++;
            }
        } else {
            layer_type_idx[static_cast<size_t>(l)] = N_glob++;
            if (has_kv) {
                layer_kv_idx[static_cast<size_t>(l)] = N_glob_kv;
                last_glob_kv_idx = N_glob_kv++;
            }
        }
    }

    MlirEmitter em;
    const std::string kv_ks_t = "tensor<" + std::to_string(N_swa_kv)  + "x" + std::to_string(B) + "x" + std::to_string(T) + "x" + std::to_string(Hkv_sh) + "x" + std::to_string(HD_swa)  + "xbf16>";
    const std::string kv_kg_t = "tensor<" + std::to_string(N_glob_kv) + "x" + std::to_string(B) + "x" + std::to_string(T) + "x" + std::to_string(Hkv_sh) + "x" + std::to_string(HD_glob) + "xbf16>";
    const std::string out0_t  = tc.return_full_logits
        ? ("tensor<" + std::to_string(B) + "x" + std::to_string(V) + "xf32>")
        : ("tensor<" + std::to_string(B) + "xi32>");

    auto donate_attr = [&](int out_slot) -> std::string {
        if (!tc.donate_kv_cache) return "";
        return " {tf.aliasing_output = " + std::to_string(out_slot) + " : i32}";
    };

    em.ss << "module @haven_gemma4_onchip {\n"
          << "  func.func @main(\n"
          // Dynamic per-step inputs (0..2)
          << "    %x_emb_in: tensor<" << B << "x" << S << "x" << D << "xbf16>,\n"
          << "    %pleias_tok_in: tensor<" << B << "x" << S << "x" << L << "x" << P_dim << "xbf16>,\n"
          << "    %pos: tensor<i32>,\n"
          // Stacked projection weights (3..14)
          << "    %w_ffn_gate: tensor<" << L << "x" << F_sh << "x" << D << "x" << wel << ">,\n"
          << "    %w_ffn_up: tensor<" << L << "x" << F_sh << "x" << D << "x" << wel << ">,\n"
          << "    %w_ffn_down: tensor<" << L << "x" << D << "x" << F_sh << "x" << wel << ">,\n"
          << "    %w_swa_q: tensor<" << N_swa << "x" << Q_swa_sh << "x" << D << "x" << wel << ">,\n"
          << "    %w_swa_o: tensor<" << N_swa << "x" << D << "x" << Q_swa_sh << "x" << wel << ">,\n"
          << "    %w_swa_k: tensor<" << N_swa_kv << "x" << KV_swa_sh << "x" << D << "x" << wel << ">,\n"
          << "    %w_swa_v: tensor<" << N_swa_kv << "x" << KV_swa_sh << "x" << D << "x" << wel << ">,\n"
          << "    %w_glob_q: tensor<" << N_glob << "x" << Q_glob_sh << "x" << D << "x" << wel << ">,\n"
          << "    %w_glob_o: tensor<" << N_glob << "x" << D << "x" << Q_glob_sh << "x" << wel << ">,\n"
          << "    %w_glob_k: tensor<" << N_glob_kv << "x" << KV_glob_sh << "x" << D << "x" << wel << ">,\n"
          << "    %w_glob_v: tensor<" << N_glob_kv << "x" << KV_glob_sh << "x" << D << "x" << wel << ">,\n"
          << "    %w_lm_head: tensor<" << V_sh << "x" << D << "x" << wel << ">,\n";

    // Optional INT8 per-row F32 scales (12 scale tensors right after weights when use_int8_weights=true)
    if (i8) {
        em.ss << "    %s_ffn_gate: tensor<" << L << "x" << F_sh << "xf32>,\n"
              << "    %s_ffn_up: tensor<" << L << "x" << F_sh << "xf32>,\n"
              << "    %s_ffn_down: tensor<" << L << "x" << D << "xf32>,\n"
              << "    %s_swa_q: tensor<" << N_swa << "x" << Q_swa_sh << "xf32>,\n"
              << "    %s_swa_o: tensor<" << N_swa << "x" << D << "xf32>,\n"
              << "    %s_swa_k: tensor<" << N_swa_kv << "x" << KV_swa_sh << "xf32>,\n"
              << "    %s_swa_v: tensor<" << N_swa_kv << "x" << KV_swa_sh << "xf32>,\n"
              << "    %s_glob_q: tensor<" << N_glob << "x" << Q_glob_sh << "xf32>,\n"
              << "    %s_glob_o: tensor<" << N_glob << "x" << D << "xf32>,\n"
              << "    %s_glob_k: tensor<" << N_glob_kv << "x" << KV_glob_sh << "xf32>,\n"
              << "    %s_glob_v: tensor<" << N_glob_kv << "x" << KV_glob_sh << "xf32>,\n"
              << "    %s_lm_head: tensor<" << V_sh << "xf32>,\n";
    }

    // Replicated BF16 Pleias weights, F32 norms, RoPE tables
    em.ss << "    %w_inp_gate: tensor<" << L << "x" << P_dim << "x" << D << "xbf16>,\n"
          << "    %w_proj: tensor<" << L << "x" << D << "x" << P_dim << "xbf16>,\n"
          << "    %w_pl_mproj: tensor<" << (L * P_dim) << "x" << D << "xbf16>,\n"
          << "    %w_pl_pnorm: tensor<" << P_dim << "xf32>,\n"
          << "    %w_attn_norm: tensor<" << L << "x" << D << "xf32>,\n"
          << "    %w_post_attn_norm: tensor<" << L << "x" << D << "xf32>,\n"
          << "    %w_ffn_norm: tensor<" << L << "x" << D << "xf32>,\n"
          << "    %w_post_ffn_norm: tensor<" << L << "x" << D << "xf32>,\n"
          << "    %w_post_norm: tensor<" << L << "x" << D << "xf32>,\n"
          << "    %w_layer_scale: tensor<" << L << "xf32>,\n"
          << "    %w_swa_q_norm: tensor<" << N_swa << "x" << HD_swa << "xf32>,\n"
          << "    %w_swa_k_norm: tensor<" << N_swa_kv << "x" << HD_swa << "xf32>,\n"
          << "    %w_glob_q_norm: tensor<" << N_glob << "x" << HD_glob << "xf32>,\n"
          << "    %w_glob_k_norm: tensor<" << N_glob_kv << "x" << HD_glob << "xf32>,\n"
          << "    %w_out_norm: tensor<" << D << "xf32>,\n"
          << "    %cos_swa: tensor<" << T << "x" << (HD_swa / 2) << "xf32>,\n"
          << "    %sin_swa: tensor<" << T << "x" << (HD_swa / 2) << "xf32>,\n"
          << "    %cos_glob: tensor<" << T << "x" << (HD_glob / 2) << "xf32>,\n"
          << "    %sin_glob: tensor<" << T << "x" << (HD_glob / 2) << "xf32>,\n"
          // Mutable donated HBM3 KV Cache buffers
          << "    %kv_ks_in: " << kv_ks_t << donate_attr(1) << ",\n"
          << "    %kv_vs_in: " << kv_ks_t << donate_attr(2) << ",\n"
          << "    %kv_kg_in: " << kv_kg_t << donate_attr(3) << ",\n"
          << "    %kv_vg_in: " << kv_kg_t << donate_attr(4) << "\n"
          << "  ) -> (" << out0_t << ", " << kv_ks_t << ", " << kv_ks_t << ", " << kv_kg_t << ", " << kv_kg_t << ") {\n";

    // Common scalar constants
    const std::string c0_f32 = "%c0_f32";
    const std::string c0_i32 = "%c0_i32";
    const std::string c_ninf = "%c_ninf";
    em.ss << "    " << c0_f32 << " = stablehlo.constant dense<0.0000000e+00> : tensor<f32>\n"
          << "    " << c0_i32 << " = stablehlo.constant dense<0> : tensor<i32>\n"
          << "    " << c_ninf << " = stablehlo.constant dense<-1.0000000e+09> : tensor<f32>\n";

    // Pre-slice RoPE cos/sin for [pos : pos + S] once for all SWA & Global layers!
    em.ss << "    %cos_swa_s = stablehlo.dynamic_slice %cos_swa, %pos, " << c0_i32
          << ", sizes = [" << S << ", " << (HD_swa / 2) << "] : (tensor<" << T << "x" << (HD_swa / 2)
          << "xf32>, tensor<i32>, tensor<i32>) -> tensor<" << S << "x" << (HD_swa / 2) << "xf32>\n"
          << "    %sin_swa_s = stablehlo.dynamic_slice %sin_swa, %pos, " << c0_i32
          << ", sizes = [" << S << ", " << (HD_swa / 2) << "] : (tensor<" << T << "x" << (HD_swa / 2)
          << "xf32>, tensor<i32>, tensor<i32>) -> tensor<" << S << "x" << (HD_swa / 2) << "xf32>\n"
          << "    %cos_glob_s = stablehlo.dynamic_slice %cos_glob, %pos, " << c0_i32
          << ", sizes = [" << S << ", " << (HD_glob / 2) << "] : (tensor<" << T << "x" << (HD_glob / 2)
          << "xf32>, tensor<i32>, tensor<i32>) -> tensor<" << S << "x" << (HD_glob / 2) << "xf32>\n"
          << "    %sin_glob_s = stablehlo.dynamic_slice %sin_glob, %pos, " << c0_i32
          << ", sizes = [" << S << ", " << (HD_glob / 2) << "] : (tensor<" << T << "x" << (HD_glob / 2)
          << "xf32>, tensor<i32>, tensor<i32>) -> tensor<" << S << "x" << (HD_glob / 2) << "xf32>\n";

    // Build exact Causal [S, T] and SWA-512 + 4-Sink [S, T] boolean attention masks once!
    const std::string t_st_i32 = "tensor<" + std::to_string(S) + "x" + std::to_string(T) + "xi32>";
    const std::string t_st_i1  = "tensor<" + std::to_string(S) + "x" + std::to_string(T) + "xi1>";
    em.ss << "    %k_iota = stablehlo.iota dim = 0 : tensor<" << T << "xi32>\n"
          << "    %q_iota = stablehlo.iota dim = 0 : tensor<" << S << "xi32>\n"
          << "    %pos_s = stablehlo.broadcast_in_dim %pos, dims = [] : (tensor<i32>) -> tensor<" << S << "xi32>\n"
          << "    %q_pos = stablehlo.add %q_iota, %pos_s : tensor<" << S << "xi32>\n"
          << "    %k_st = stablehlo.broadcast_in_dim %k_iota, dims = [1] : (tensor<" << T << "xi32>) -> " << t_st_i32 << "\n"
          << "    %q_st = stablehlo.broadcast_in_dim %q_pos, dims = [0] : (tensor<" << S << "xi32>) -> " << t_st_i32 << "\n"
          << "    %mask_glob = stablehlo.compare LE, %k_st, %q_st, SIGNED : (" << t_st_i32 << ", " << t_st_i32 << ") -> " << t_st_i1 << "\n"
          << "    %c_win = stablehlo.constant dense<511> : " << t_st_i32 << "\n"
          << "    %c_sink = stablehlo.constant dense<4> : " << t_st_i32 << "\n"
          << "    %win_lo = stablehlo.subtract %q_st, %c_win : " << t_st_i32 << "\n"
          << "    %in_win = stablehlo.compare GE, %k_st, %win_lo, SIGNED : (" << t_st_i32 << ", " << t_st_i32 << ") -> " << t_st_i1 << "\n"
          << "    %is_sink = stablehlo.compare LT, %k_st, %c_sink, SIGNED : (" << t_st_i32 << ", " << t_st_i32 << ") -> " << t_st_i1 << "\n"
          << "    %win_or_sink = stablehlo.or %in_win, %is_sink : " << t_st_i1 << "\n"
          << "    %mask_swa = stablehlo.and %mask_glob, %win_or_sink : " << t_st_i1 << "\n";

    // Step 1: Convert x_emb_in [B, S, D] to F32 and compute Pleias per-layer embeddings [B, S, L, 256]
    const std::string t_bsd_f32 = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(D) + "xf32>";
    const std::string t_bsd_bf  = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(D) + "xbf16>";
    const std::string t_pl_f32  = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(L) + "x" + std::to_string(P_dim) + "xf32>";
    const std::string t_pl_bf   = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(L) + "x" + std::to_string(P_dim) + "xbf16>";
    const std::string t_pl_flat = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(L * P_dim) + "xf32>";

    em.ss << "    %x0 = stablehlo.convert %x_emb_in : (" << t_bsd_bf << ") -> " << t_bsd_f32 << "\n"
          << "    %pl_proj_raw = stablehlo.dot_general %x_emb_in, %w_pl_mproj, contracting_dims = [2] x [1] : ("
          << t_bsd_bf << ", tensor<" << (L * P_dim) << "x" << D << "xbf16>) -> " << t_pl_flat << "\n"
          << "    %c_pl_pscale = stablehlo.constant dense<" << fmt_f32(1.0f / std::sqrt(static_cast<float>(D))) << "> : " << t_pl_flat << "\n"
          << "    %pl_proj_sc = stablehlo.multiply %pl_proj_raw, %c_pl_pscale : " << t_pl_flat << "\n"
          << "    %pl_proj_4d = stablehlo.reshape %pl_proj_sc : (" << t_pl_flat << ") -> " << t_pl_f32 << "\n";
    std::string pl_proj_norm = em.emit_rms_norm_4d("%pl_proj_4d", "%w_pl_pnorm", B, S, L, P_dim, mc.rms_norm_eps, c0_f32);
    em.ss << "    %pl_tok_f32 = stablehlo.convert %pleias_tok_in : (" << t_pl_bf << ") -> " << t_pl_f32 << "\n"
          << "    %pl_sum = stablehlo.add " << pl_proj_norm << ", %pl_tok_f32 : " << t_pl_f32 << "\n"
          << "    %c_inv_s2 = stablehlo.constant dense<" << fmt_f32(1.0f / std::sqrt(2.0f)) << "> : " << t_pl_f32 << "\n"
          << "    %pleias_all = stablehlo.multiply %pl_sum, %c_inv_s2 : " << t_pl_f32 << "\n";

    std::string x_cur = "%x0";
    std::string kv_ks = "%kv_ks_in";
    std::string kv_vs = "%kv_vs_in";
    std::string kv_kg = "%kv_kg_in";
    std::string kv_vg = "%kv_vg_in";

    // Step 2: Unroll all 42 Gemma 4 layers with exact static shapes & zero overhead
    for (int l = 0; l < L; ++l) {
        const bool is_swa = (l % 6 != 5);
        const bool has_kv = (l < kv_owner_limit);
        const int ti = layer_type_idx[static_cast<size_t>(l)];
        const int kvi = has_kv ? layer_kv_idx[static_cast<size_t>(l)] : (is_swa ? last_swa_kv_idx : last_glob_kv_idx);
        const int HD = is_swa ? HD_swa : HD_glob;
        const int Q_sh = is_swa ? Q_swa_sh : Q_glob_sh;
        const int KV_sh = is_swa ? KV_swa_sh : KV_glob_sh;
        const int N_type = is_swa ? N_swa : N_glob;
        const int N_kv_type = is_swa ? N_swa_kv : N_glob_kv;

        em.ss << "    // --- Layer " << l << (is_swa ? " (SWA" : " (Global")
              << (has_kv ? ", KV-owner)" : ", KV-shared)") << " ---\n";

        // 2a. Pre-Attention RMSNorm
        std::string attn_nw = em.slice_1d_from_2d("%w_attn_norm", l, L, D);
        std::string x_an_f32 = em.emit_rms_norm_3d(x_cur, attn_nw, B, S, D, mc.rms_norm_eps, c0_f32);
        std::string x_an_bf = em.tmp("anbf");
        em.ss << "    " << x_an_bf << " = stablehlo.convert " << x_an_f32 << " : (" << t_bsd_f32 << ") -> " << t_bsd_bf << "\n";

        // 2b. Q Projection + Q-Norm + RoPE
        std::string q_flat = em.emit_linear_3d(x_an_bf,
            is_swa ? "%w_swa_q" : "%w_glob_q",
            is_swa ? "%s_swa_q" : "%s_glob_q",
            ti, N_type, B, S, Q_sh, D, i8);
        const std::string t_q3 = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(Q_sh) + "xf32>";
        const std::string t_q4 = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(H_sh) + "x" + std::to_string(HD) + "xf32>";
        std::string q_4d = em.tmp("q4");
        em.ss << "    " << q_4d << " = stablehlo.reshape " << q_flat << " : (" << t_q3 << ") -> " << t_q4 << "\n";
        std::string q_nw = em.slice_1d_from_2d(is_swa ? "%w_swa_q_norm" : "%w_glob_q_norm", ti, N_type, HD);
        std::string q_n = em.emit_rms_norm_4d(q_4d, q_nw, B, S, H_sh, HD, mc.rms_norm_eps, c0_f32);
        std::string q_rot = em.emit_rope_4d(q_n, is_swa ? "%cos_swa_s" : "%cos_glob_s", is_swa ? "%sin_swa_s" : "%sin_glob_s", B, S, H_sh, HD);

        // 2c. K, V Projections + K-Norm + V-Norm (unit weights) + RoPE + In-place KV Cache Update
        const std::string& kv_k_stack = is_swa ? kv_ks : kv_kg;
        const std::string& kv_v_stack = is_swa ? kv_vs : kv_vg;
        const std::string& kv_stack_t = is_swa ? kv_ks_t : kv_kg_t;

        if (has_kv) {
            std::string k_flat = em.emit_linear_3d(x_an_bf,
                is_swa ? "%w_swa_k" : "%w_glob_k",
                is_swa ? "%s_swa_k" : "%s_glob_k",
                kvi, N_kv_type, B, S, KV_sh, D, i8);
            std::string v_flat = em.emit_linear_3d(x_an_bf,
                is_swa ? "%w_swa_v" : "%w_glob_v",
                is_swa ? "%s_swa_v" : "%s_glob_v",
                kvi, N_kv_type, B, S, KV_sh, D, i8);
            const std::string t_kv3 = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(KV_sh) + "xf32>";
            const std::string t_kv4 = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(Hkv_sh) + "x" + std::to_string(HD) + "xf32>";
            const std::string t_kv4_bf = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(Hkv_sh) + "x" + std::to_string(HD) + "xbf16>";
            const std::string t_kv5_bf = "tensor<1x" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(Hkv_sh) + "x" + std::to_string(HD) + "xbf16>";

            std::string k_4d = em.tmp("k4"), v_4d = em.tmp("v4");
            em.ss << "    " << k_4d << " = stablehlo.reshape " << k_flat << " : (" << t_kv3 << ") -> " << t_kv4 << "\n"
                  << "    " << v_4d << " = stablehlo.reshape " << v_flat << " : (" << t_kv3 << ") -> " << t_kv4 << "\n";
            std::string k_nw = em.slice_1d_from_2d(is_swa ? "%w_swa_k_norm" : "%w_glob_k_norm", kvi, N_kv_type, HD);
            std::string k_n = em.emit_rms_norm_4d(k_4d, k_nw, B, S, Hkv_sh, HD, mc.rms_norm_eps, c0_f32);
            std::string v_n = em.emit_rms_norm_4d(v_4d, "", B, S, Hkv_sh, HD, mc.rms_norm_eps, c0_f32);
            std::string k_rot = em.emit_rope_4d(k_n, is_swa ? "%cos_swa_s" : "%cos_glob_s", is_swa ? "%sin_swa_s" : "%sin_glob_s", B, S, Hkv_sh, HD);

            std::string k_bf = em.tmp("kbf"), v_bf = em.tmp("vbf"), k_5 = em.tmp("k5"), v_5 = em.tmp("v5");
            std::string c_kvi = em.tmp("kvi"), nk = em.tmp("kvk"), nv = em.tmp("kvv");
            em.ss << "    " << k_bf << " = stablehlo.convert " << k_rot << " : (" << t_kv4 << ") -> " << t_kv4_bf << "\n"
                  << "    " << v_bf << " = stablehlo.convert " << v_n   << " : (" << t_kv4 << ") -> " << t_kv4_bf << "\n"
                  << "    " << k_5  << " = stablehlo.reshape " << k_bf << " : (" << t_kv4_bf << ") -> " << t_kv5_bf << "\n"
                  << "    " << v_5  << " = stablehlo.reshape " << v_bf << " : (" << t_kv4_bf << ") -> " << t_kv5_bf << "\n"
                  << "    " << c_kvi << " = stablehlo.constant dense<" << kvi << "> : tensor<i32>\n"
                  << "    " << nk << " = stablehlo.dynamic_update_slice " << kv_k_stack << ", " << k_5 << ", "
                  << c_kvi << ", " << c0_i32 << ", %pos, " << c0_i32 << ", " << c0_i32
                  << " : (" << kv_stack_t << ", " << t_kv5_bf << ", tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> " << kv_stack_t << "\n"
                  << "    " << nv << " = stablehlo.dynamic_update_slice " << kv_v_stack << ", " << v_5 << ", "
                  << c_kvi << ", " << c0_i32 << ", %pos, " << c0_i32 << ", " << c0_i32
                  << " : (" << kv_stack_t << ", " << t_kv5_bf << ", tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> " << kv_stack_t << "\n";
            if (is_swa) { kv_ks = nk; kv_vs = nv; } else { kv_kg = nk; kv_vg = nv; }
        }

        // 2e. GQA Attention directly on HBM3 KV Cache (zero KV head duplication!)
        const std::string& cur_k_stack = is_swa ? kv_ks : kv_kg;
        const std::string& cur_v_stack = is_swa ? kv_vs : kv_vg;
        const std::string t_kv1_t = "tensor<1x" + std::to_string(B) + "x" + std::to_string(T) + "x" + std::to_string(Hkv_sh) + "x" + std::to_string(HD) + "xbf16>";
        const std::string t_kv4_t = "tensor<" + std::to_string(B) + "x" + std::to_string(T) + "x" + std::to_string(Hkv_sh) + "x" + std::to_string(HD) + "xbf16>";
        const std::string t_q5_bf = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(Hkv_sh) + "x" + std::to_string(G_sh) + "x" + std::to_string(HD) + "xbf16>";
        const std::string t_sc5   = "tensor<" + std::to_string(B) + "x" + std::to_string(Hkv_sh) + "x" + std::to_string(S) + "x" + std::to_string(G_sh) + "x" + std::to_string(T) + "xf32>";
        const std::string t_sc5_i1 = "tensor<" + std::to_string(B) + "x" + std::to_string(Hkv_sh) + "x" + std::to_string(S) + "x" + std::to_string(G_sh) + "x" + std::to_string(T) + "xi1>";
        const std::string t_sc5_bf = "tensor<" + std::to_string(B) + "x" + std::to_string(Hkv_sh) + "x" + std::to_string(S) + "x" + std::to_string(G_sh) + "x" + std::to_string(T) + "xbf16>";
        const std::string t_sc4   = "tensor<" + std::to_string(B) + "x" + std::to_string(Hkv_sh) + "x" + std::to_string(S) + "x" + std::to_string(G_sh) + "xf32>";
        const std::string t_ctx5  = "tensor<" + std::to_string(B) + "x" + std::to_string(Hkv_sh) + "x" + std::to_string(S) + "x" + std::to_string(G_sh) + "x" + std::to_string(HD) + "xf32>";
        const std::string t_ctx5t = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(Hkv_sh) + "x" + std::to_string(G_sh) + "x" + std::to_string(HD) + "xf32>";
        const std::string t_ao_bf = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(Q_sh) + "xbf16>";

        std::string ks1 = em.tmp("ks1"), klay = em.tmp("klay"), vs1 = em.tmp("vs1"), vlay = em.tmp("vlay");
        std::string qbf4 = em.tmp("qbf"), q5 = em.tmp("q5"), sc = em.tmp("sc"), mb = em.tmp("mb");
        std::string nib = em.tmp("nib"), msc = em.tmp("msc"), mx = em.tmp("mx"), mxb = em.tmp("mxb");
        std::string sh = em.tmp("sh"), ex = em.tmp("ex"), se = em.tmp("se"), seb = em.tmp("seb");
        std::string pr = em.tmp("pr"), prbf = em.tmp("prbf"), ctx = em.tmp("ctx"), ctxt = em.tmp("ctxt");
        std::string ao_f32 = em.tmp("aof"), ao_bf = em.tmp("aob");

        em.ss << "    " << ks1  << " = stablehlo.slice " << cur_k_stack << " [" << kvi << ":" << (kvi + 1)
              << ", 0:" << B << ", 0:" << T << ", 0:" << Hkv_sh << ", 0:" << HD << "] : (" << kv_stack_t << ") -> " << t_kv1_t << "\n"
              << "    " << klay << " = stablehlo.reshape " << ks1 << " : (" << t_kv1_t << ") -> " << t_kv4_t << "\n"
              << "    " << vs1  << " = stablehlo.slice " << cur_v_stack << " [" << kvi << ":" << (kvi + 1)
              << ", 0:" << B << ", 0:" << T << ", 0:" << Hkv_sh << ", 0:" << HD << "] : (" << kv_stack_t << ") -> " << t_kv1_t << "\n"
              << "    " << vlay << " = stablehlo.reshape " << vs1 << " : (" << t_kv1_t << ") -> " << t_kv4_t << "\n"
              << "    " << qbf4 << " = stablehlo.convert " << q_rot << " : (" << t_q4 << ") -> tensor<" << B << "x" << S << "x" << H_sh << "x" << HD << "xbf16>\n"
              << "    " << q5   << " = stablehlo.reshape " << qbf4 << " : (tensor<" << B << "x" << S << "x" << H_sh << "x" << HD << "xbf16>) -> " << t_q5_bf << "\n"
              << "    " << sc   << " = stablehlo.dot_general " << q5 << ", " << klay
              << ", batching_dims = [0, 2] x [0, 2], contracting_dims = [4] x [3] : (" << t_q5_bf << ", " << t_kv4_t << ") -> " << t_sc5 << "\n"
              << "    " << mb   << " = stablehlo.broadcast_in_dim " << (is_swa ? "%mask_swa" : "%mask_glob")
              << ", dims = [2, 4] : (" << t_st_i1 << ") -> " << t_sc5_i1 << "\n"
              << "    " << nib  << " = stablehlo.broadcast_in_dim " << c_ninf << ", dims = [] : (tensor<f32>) -> " << t_sc5 << "\n"
              << "    " << msc  << " = stablehlo.select " << mb << ", " << sc << ", " << nib << " : " << t_sc5_i1 << ", " << t_sc5 << "\n"
              << "    " << mx   << " = stablehlo.reduce(" << msc << " init: " << c_ninf << ") across dimensions = [4] : ("
              << t_sc5 << ", tensor<f32>) -> " << t_sc4 << "\n"
              << "      reducer(%ra: tensor<f32>, %rb: tensor<f32>) {\n"
              << "        %rc = stablehlo.maximum %ra, %rb : tensor<f32>\n"
              << "        stablehlo.return %rc : tensor<f32>\n"
              << "      }\n"
              << "    " << mxb  << " = stablehlo.broadcast_in_dim " << mx << ", dims = [0, 1, 2, 3] : (" << t_sc4 << ") -> " << t_sc5 << "\n"
              << "    " << sh   << " = stablehlo.subtract " << msc << ", " << mxb << " : " << t_sc5 << "\n"
              << "    " << ex   << " = stablehlo.exponential " << sh << " : " << t_sc5 << "\n"
              << "    " << se   << " = stablehlo.reduce(" << ex << " init: " << c0_f32 << ") across dimensions = [4] : ("
              << t_sc5 << ", tensor<f32>) -> " << t_sc4 << "\n"
              << "      reducer(%ra: tensor<f32>, %rb: tensor<f32>) {\n"
              << "        %rc = stablehlo.add %ra, %rb : tensor<f32>\n"
              << "        stablehlo.return %rc : tensor<f32>\n"
              << "      }\n"
              << "    " << seb  << " = stablehlo.broadcast_in_dim " << se << ", dims = [0, 1, 2, 3] : (" << t_sc4 << ") -> " << t_sc5 << "\n"
              << "    " << pr   << " = stablehlo.divide " << ex << ", " << seb << " : " << t_sc5 << "\n"
              << "    " << prbf << " = stablehlo.convert " << pr << " : (" << t_sc5 << ") -> " << t_sc5_bf << "\n"
              << "    " << ctx  << " = stablehlo.dot_general " << prbf << ", " << vlay
              << ", batching_dims = [0, 1] x [0, 2], contracting_dims = [4] x [1] : (" << t_sc5_bf << ", " << t_kv4_t << ") -> " << t_ctx5 << "\n"
              << "    " << ctxt << " = stablehlo.transpose " << ctx << ", dims = [0, 2, 1, 3, 4] : (" << t_ctx5 << ") -> " << t_ctx5t << "\n"
              << "    " << ao_f32 << " = stablehlo.reshape " << ctxt << " : (" << t_ctx5t << ") -> " << t_q3 << "\n"
              << "    " << ao_bf  << " = stablehlo.convert " << ao_f32 << " : (" << t_q3 << ") -> " << t_ao_bf << "\n";

        // 2f. Attention Output Projection + Optical ICI AllReduce (if R > 1) + Post-Attn Norm + Residual 1
        std::string attn_out_loc = em.emit_linear_3d(ao_bf,
            is_swa ? "%w_swa_o" : "%w_glob_o",
            is_swa ? "%s_swa_o" : "%s_glob_o",
            ti, N_type, B, S, D, Q_sh, i8);
        std::string attn_out = em.emit_all_reduce_sum_3d(attn_out_loc, B, S, D, R);
        std::string pa_nw = em.slice_1d_from_2d("%w_post_attn_norm", l, L, D);
        std::string attn_normed = em.emit_rms_norm_3d(attn_out, pa_nw, B, S, D, mc.rms_norm_eps, c0_f32);
        std::string x_res1 = em.tmp("res1");
        em.ss << "    " << x_res1 << " = stablehlo.add " << x_cur << ", " << attn_normed << " : " << t_bsd_f32 << "\n";

        // 2g-2i. Pre-FFN RMSNorm + Sharded GeGLU FFN + Optical ICI AllReduce (if R > 1) + Post-FFN Norm + Residual 2
        std::string fn_nw = em.slice_1d_from_2d("%w_ffn_norm", l, L, D);
        std::string x_fn_f32 = em.emit_rms_norm_3d(x_res1, fn_nw, B, S, D, mc.rms_norm_eps, c0_f32);
        std::string x_fn_bf = em.tmp("fnbf");
        em.ss << "    " << x_fn_bf << " = stablehlo.convert " << x_fn_f32 << " : (" << t_bsd_f32 << ") -> " << t_bsd_bf << "\n";

        std::string f_gate = em.emit_linear_3d(x_fn_bf, "%w_ffn_gate", "%s_ffn_gate", l, L, B, S, F_sh, D, i8);
        std::string f_up   = em.emit_linear_3d(x_fn_bf, "%w_ffn_up",   "%s_ffn_up",   l, L, B, S, F_sh, D, i8);
        std::string f_gelu = em.emit_gelu_3d(f_gate, B, S, F_sh);
        const std::string t_ff_f32 = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(F_sh) + "xf32>";
        const std::string t_ff_bf  = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(F_sh) + "xbf16>";
        std::string f_act = em.tmp("fact"), f_act_bf = em.tmp("fabf");
        em.ss << "    " << f_act    << " = stablehlo.multiply " << f_gelu << ", " << f_up << " : " << t_ff_f32 << "\n"
              << "    " << f_act_bf << " = stablehlo.convert " << f_act << " : (" << t_ff_f32 << ") -> " << t_ff_bf << "\n";
        std::string f_down_loc = em.emit_linear_3d(f_act_bf, "%w_ffn_down", "%s_ffn_down", l, L, B, S, D, F_sh, i8);
        std::string f_down = em.emit_all_reduce_sum_3d(f_down_loc, B, S, D, R);
        std::string pf_nw = em.slice_1d_from_2d("%w_post_ffn_norm", l, L, D);
        std::string f_normed = em.emit_rms_norm_3d(f_down, pf_nw, B, S, D, mc.rms_norm_eps, c0_f32);
        std::string x_res2 = em.tmp("res2");
        em.ss << "    " << x_res2 << " = stablehlo.add " << x_res1 << ", " << f_normed << " : " << t_bsd_f32 << "\n";

        // 2-pleias. Per-Layer Token Injection (replicated tiny 256-dim branch, zero AllReduce!)
        std::string x_r2_bf = em.tmp("r2bf");
        em.ss << "    " << x_r2_bf << " = stablehlo.convert " << x_res2 << " : (" << t_bsd_f32 << ") -> " << t_bsd_bf << "\n";
        std::string pg = em.emit_linear_3d(x_r2_bf, "%w_inp_gate", "", l, L, B, S, P_dim, D, false);
        std::string pg_gelu = em.emit_gelu_3d(pg, B, S, P_dim);
        const std::string t_p4 = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x1x" + std::to_string(P_dim) + "xf32>";
        const std::string t_p3 = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(P_dim) + "xf32>";
        const std::string t_p3_bf = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(P_dim) + "xbf16>";
        std::string pl_sl = em.tmp("plsl"), pl_3 = em.tmp("pl3"), pg_mul = em.tmp("pgm"), pg_bf = em.tmp("pgbf");
        em.ss << "    " << pl_sl << " = stablehlo.slice %pleias_all [0:" << B << ", 0:" << S << ", " << l << ":" << (l + 1)
              << ", 0:" << P_dim << "] : (" << t_pl_f32 << ") -> " << t_p4 << "\n"
              << "    " << pl_3  << " = stablehlo.reshape " << pl_sl << " : (" << t_p4 << ") -> " << t_p3 << "\n"
              << "    " << pg_mul << " = stablehlo.multiply " << pg_gelu << ", " << pl_3 << " : " << t_p3 << "\n"
              << "    " << pg_bf  << " = stablehlo.convert " << pg_mul << " : (" << t_p3 << ") -> " << t_p3_bf << "\n";
        std::string pl_out = em.emit_linear_3d(pg_bf, "%w_proj", "", l, L, B, S, D, P_dim, false);
        std::string pn_nw = em.slice_1d_from_2d("%w_post_norm", l, L, D);
        std::string pl_normed = em.emit_rms_norm_3d(pl_out, pn_nw, B, S, D, mc.rms_norm_eps, c0_f32);
        std::string x_res3 = em.tmp("res3");
        em.ss << "    " << x_res3 << " = stablehlo.add " << x_res2 << ", " << pl_normed << " : " << t_bsd_f32 << "\n";

        // 2-scale. Layer Output Scale
        std::string ls_1 = em.tmp("ls1"), ls_0 = em.tmp("ls0"), ls_b = em.tmp("lsb"), x_next = em.tmp("xl");
        em.ss << "    " << ls_1 << " = stablehlo.slice %w_layer_scale [" << l << ":" << (l + 1)
              << "] : (tensor<" << L << "xf32>) -> tensor<1xf32>\n"
              << "    " << ls_0 << " = stablehlo.reshape " << ls_1 << " : (tensor<1xf32>) -> tensor<f32>\n"
              << "    " << ls_b << " = stablehlo.broadcast_in_dim " << ls_0 << ", dims = [] : (tensor<f32>) -> " << t_bsd_f32 << "\n"
              << "    " << x_next << " = stablehlo.multiply " << x_res3 << ", " << ls_b << " : " << t_bsd_f32 << "\n";
        x_cur = x_next;
    }

    // Step 3: Slice last token [B, 1, D] -> Final Output RMSNorm -> Sharded 262k LM Head -> Softcapping
    const std::string t_b1d_f32 = "tensor<" + std::to_string(B) + "x1x" + std::to_string(D) + "xf32>";
    const std::string t_bd_f32  = "tensor<" + std::to_string(B) + "x" + std::to_string(D) + "xf32>";
    const std::string t_bd_bf   = "tensor<" + std::to_string(B) + "x" + std::to_string(D) + "xbf16>";
    const std::string t_bvsh_f32 = "tensor<" + std::to_string(B) + "x" + std::to_string(V_sh) + "xf32>";

    std::string x_last = x_cur;
    if (S > 1) {
        x_last = em.tmp("xlast");
        em.ss << "    " << x_last << " = stablehlo.slice " << x_cur << " [0:" << B << ", " << (S - 1) << ":" << S
              << ", 0:" << D << "] : (" << t_bsd_f32 << ") -> " << t_b1d_f32 << "\n";
    }
    std::string x_final_n = em.emit_rms_norm_3d(x_last, "%w_out_norm", B, 1, D, mc.rms_norm_eps, c0_f32);
    em.ss << "    %xf_2d = stablehlo.reshape " << x_final_n << " : (" << t_b1d_f32 << ") -> " << t_bd_f32 << "\n"
          << "    %xf_bf = stablehlo.convert %xf_2d : (" << t_bd_f32 << ") -> " << t_bd_bf << "\n";

    std::string lm_w = "%w_lm_head";
    if (i8) {
        em.ss << "    %lm_wbf = stablehlo.convert %w_lm_head : (tensor<" << V_sh << "x" << D << "xi8>) -> tensor<" << V_sh << "x" << D << "xbf16>\n";
        lm_w = "%lm_wbf";
    }
    em.ss << "    %logits_raw = stablehlo.dot_general %xf_bf, " << lm_w << ", contracting_dims = [1] x [1] : ("
          << t_bd_bf << ", tensor<" << V_sh << "x" << D << "xbf16>) -> " << t_bvsh_f32 << "\n";

    std::string logits_sc = "%logits_raw";
    if (i8) {
        em.ss << "    %lm_sb = stablehlo.broadcast_in_dim %s_lm_head, dims = [1] : (tensor<" << V_sh << "xf32>) -> " << t_bvsh_f32 << "\n"
              << "    %logits_i8s = stablehlo.multiply %logits_raw, %lm_sb : " << t_bvsh_f32 << "\n";
        logits_sc = "%logits_i8s";
    }

    // Gemma 4 Final Logit Softcapping (30.0 * tanh(x / 30.0))
    std::string logits_loc = logits_sc;
    if (mc.final_logit_softcapping > 0.0f) {
        em.ss << "    %c_cap = stablehlo.constant dense<" << fmt_f32(mc.final_logit_softcapping) << "> : " << t_bvsh_f32 << "\n"
              << "    %c_icap = stablehlo.constant dense<" << fmt_f32(1.0f / mc.final_logit_softcapping) << "> : " << t_bvsh_f32 << "\n"
              << "    %lc_div = stablehlo.multiply " << logits_sc << ", %c_icap : " << t_bvsh_f32 << "\n"
              << "    %lc_th = stablehlo.tanh %lc_div : " << t_bvsh_f32 << "\n"
              << "    %logits_cap = stablehlo.multiply %lc_th, %c_cap : " << t_bvsh_f32 << "\n";
        logits_loc = "%logits_cap";
    }

    // Step 4: Output either full [B, V] F32 logits or ultra-fast on-chip Argmax [B] i32 (4 bytes/seq!)
    std::string final_out0;
    if (tc.return_full_logits) {
        if (R == 1) {
            final_out0 = logits_loc;
        } else {
            em.ss << "    %logits_full = \"stablehlo.all_gather\"(" << logits_loc << ") {\n"
                  << "      all_gather_dim = 1 : i64,\n"
                  << "      replica_groups = " << replica_group_attr(R) << "\n"
                  << "    } : (" << t_bvsh_f32 << ") -> tensor<" << B << "x" << V << "xf32>\n";
            final_out0 = "%logits_full";
        }
    } else {
        const std::string t_bvsh_i32 = "tensor<" + std::to_string(B) + "x" + std::to_string(V_sh) + "xi32>";
        em.ss << "    %loc_idx = stablehlo.iota dim = 1 : " << t_bvsh_i32 << "\n";
        std::string g_idx = "%loc_idx";
        if (R > 1) {
            em.ss << "    %rid_u32 = stablehlo.replica_id : tensor<ui32>\n"
                  << "    %rid_i32 = stablehlo.convert %rid_u32 : (tensor<ui32>) -> tensor<i32>\n"
                  << "    %c_vsh = stablehlo.constant dense<" << V_sh << "> : tensor<i32>\n"
                  << "    %v_off = stablehlo.multiply %rid_i32, %c_vsh : tensor<i32>\n"
                  << "    %v_off_b = stablehlo.broadcast_in_dim %v_off, dims = [] : (tensor<i32>) -> " << t_bvsh_i32 << "\n"
                  << "    %glob_idx = stablehlo.add %loc_idx, %v_off_b : " << t_bvsh_i32 << "\n";
            g_idx = "%glob_idx";
        }
        em.ss << "    %lmax_v, %lmax_i = stablehlo.reduce(" << logits_loc << " init: " << c_ninf
              << "), (" << g_idx << " init: " << c0_i32 << ") across dimensions = [1] : ("
              << t_bvsh_f32 << ", " << t_bvsh_i32 << ", tensor<f32>, tensor<i32>) -> (tensor<" << B << "xf32>, tensor<" << B << "xi32>)\n"
              << "      reducer(%va: tensor<f32>, %vb: tensor<f32>) (%ia: tensor<i32>, %ib: tensor<i32>) {\n"
              << "        %gt = stablehlo.compare GT, %va, %vb, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>\n"
              << "        %vo = stablehlo.select %gt, %va, %vb : tensor<i1>, tensor<f32>\n"
              << "        %io = stablehlo.select %gt, %ia, %ib : tensor<i1>, tensor<i32>\n"
              << "        stablehlo.return %vo, %io : tensor<f32>, tensor<i32>\n"
              << "      }\n";
        if (R == 1) {
            final_out0 = "%lmax_i";
        } else {
            em.ss << "    %lv_2d = stablehlo.reshape %lmax_v : (tensor<" << B << "xf32>) -> tensor<" << B << "x1xf32>\n"
                  << "    %li_2d = stablehlo.reshape %lmax_i : (tensor<" << B << "xi32>) -> tensor<" << B << "x1xi32>\n"
                  << "    %ag_v = \"stablehlo.all_gather\"(%lv_2d) {\n"
                  << "      all_gather_dim = 1 : i64,\n"
                  << "      replica_groups = " << replica_group_attr(R) << "\n"
                  << "    } : (tensor<" << B << "x1xf32>) -> tensor<" << B << "x" << R << "xf32>\n"
                  << "    %ag_i = \"stablehlo.all_gather\"(%li_2d) {\n"
                  << "      all_gather_dim = 1 : i64,\n"
                  << "      replica_groups = " << replica_group_attr(R) << "\n"
                  << "    } : (tensor<" << B << "x1xi32>) -> tensor<" << B << "x" << R << "xi32>\n"
                  << "    %gmax_v, %gmax_i = stablehlo.reduce(%ag_v init: " << c_ninf
                  << "), (%ag_i init: " << c0_i32 << ") across dimensions = [1] : (tensor<" << B << "x" << R << "xf32>, tensor<"
                  << B << "x" << R << "xi32>, tensor<f32>, tensor<i32>) -> (tensor<" << B << "xf32>, tensor<" << B << "xi32>)\n"
                  << "      reducer(%va: tensor<f32>, %vb: tensor<f32>) (%ia: tensor<i32>, %ib: tensor<i32>) {\n"
                  << "        %gt = stablehlo.compare GT, %va, %vb, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>\n"
                  << "        %vo = stablehlo.select %gt, %va, %vb : tensor<i1>, tensor<f32>\n"
                  << "        %io = stablehlo.select %gt, %ia, %ib : tensor<i1>, tensor<i32>\n"
                  << "        stablehlo.return %vo, %io : tensor<f32>, tensor<i32>\n"
                  << "      }\n";
            final_out0 = "%gmax_i";
        }
    }

    em.ss << "    return " << final_out0 << ", " << kv_ks << ", " << kv_vs << ", " << kv_kg << ", " << kv_vg
          << " : " << out0_t << ", " << kv_ks_t << ", " << kv_ks_t << ", " << kv_kg_t << ", " << kv_kg_t << "\n"
          << "  }\n"
          << "}\n";

    return em.ss.str();
}

HavenTpuGemma4Engine::HavenTpuGemma4Engine() = default;
HavenTpuGemma4Engine::~HavenTpuGemma4Engine() { shutdown(); }

void HavenTpuGemma4Engine::free_kv_cache() {
    if (backend_) {
        for (auto& rep : replica_kv_bufs_) {
            for (auto*& b : rep) {
                backend_->destroy_buffer(b);
                b = nullptr;
            }
        }
    }
    replica_kv_bufs_.clear();
}

void HavenTpuGemma4Engine::shutdown() {
    free_kv_cache();
    if (backend_) {
        for (auto& rep : replica_weight_bufs_) {
            for (auto*& b : rep) {
                backend_->destroy_buffer(b);
                b = nullptr;
            }
        }
        backend_->destroy_executable(decode_exec_);
        backend_->destroy_executable(prefill_exec_);
    }
    replica_weight_bufs_.clear();
    decode_exec_ = nullptr;
    prefill_exec_ = nullptr;
    backend_ = nullptr;
    host_engine_ = nullptr;
    initialized_ = false;
}

bool HavenTpuGemma4Engine::build_and_upload_weights(const HavenEngine& host_engine) {
    const int L = static_cast<int>(model_cfg_.num_layers);
    const int D = static_cast<int>(model_cfg_.embedding_dim);
    const int F = static_cast<int>(model_cfg_.hidden_dim);
    const int V = static_cast<int>(model_cfg_.vocab_size);
    const int H = static_cast<int>(model_cfg_.num_heads);
    const int Hkv = static_cast<int>(model_cfg_.num_kv_heads);
    const int R = cfg_.num_replicas;
    const bool i8 = cfg_.use_int8_weights;
    const int T = cfg_.max_seq_len;

    const int V_sh = V / R;
    const int H_sh = H / R;
    const int Hkv_sh = (R == 1) ? Hkv : 1;
    const int gqa_ratio = std::max(1, H / Hkv);
    const int HD_swa = 256, HD_glob = 512, P_dim = 256;

    const int kv_owner_limit = (L > 20) ? (L - 18) : L;
    std::vector<int> swa_layers, glob_layers, swa_kv_layers, glob_kv_layers;
    for (int l = 0; l < L; ++l) {
        if (l % 6 != 5) {
            swa_layers.push_back(l);
            if (l < kv_owner_limit) swa_kv_layers.push_back(l);
        } else {
            glob_layers.push_back(l);
            if (l < kv_owner_limit) glob_kv_layers.push_back(l);
        }
    }
    const int N_swa = static_cast<int>(swa_layers.size());
    const int N_glob = static_cast<int>(glob_layers.size());
    const int N_swa_kv = static_cast<int>(swa_kv_layers.size());
    const int N_glob_kv = static_cast<int>(glob_kv_layers.size());

    replica_weight_bufs_.assign(static_cast<size_t>(R), {});
    size_t total_bytes = 0;

    // Temporary storage for per-replica INT8 scale buffers so argument order matches MLIR signature
    std::vector<std::vector<PJRT_Buffer*>> replica_scale_bufs(static_cast<size_t>(R));

    // Helper: stack layers into [count, rows_sh, cols_sh] per replica (row-sharded or col-sharded)
    auto upload_sharded_stack = [&](const std::vector<int>& layer_list,
                                    const std::function<const TensorDesc*(int)>& getter,
                                    int full_rows, int full_cols,
                                    bool shard_rows, bool shard_cols,
                                    bool is_kv_head_shard) -> bool {
        const int cnt = static_cast<int>(layer_list.size());
        const int r_sh = shard_rows ? (is_kv_head_shard ? (full_rows / Hkv * Hkv_sh) : (full_rows / R)) : full_rows;
        const int c_sh = shard_cols ? (full_cols / R) : full_cols;

        const size_t shard_elems = static_cast<size_t>(cnt) * static_cast<size_t>(r_sh) * static_cast<size_t>(c_sh);
        std::vector<std::vector<uint16_t>> bf16_shards(i8 ? 0 : static_cast<size_t>(R));
        std::vector<std::vector<int8_t>>   s8_shards(i8 ? static_cast<size_t>(R) : 0);
        std::vector<std::vector<float>>    sc_shards(i8 ? static_cast<size_t>(R) : 0);
        for (int r = 0; r < R; ++r) {
            if (!i8) bf16_shards[static_cast<size_t>(r)].resize(shard_elems);
            else {
                s8_shards[static_cast<size_t>(r)].resize(shard_elems);
                sc_shards[static_cast<size_t>(r)].resize(static_cast<size_t>(cnt) * static_cast<size_t>(r_sh));
            }
        }

        std::vector<float> full_f32;
        std::vector<float> slice_f32(static_cast<size_t>(r_sh) * static_cast<size_t>(c_sh));
        for (int idx = 0; idx < cnt; ++idx) {
            const TensorDesc* td = getter(layer_list[static_cast<size_t>(idx)]);
            dequantize_full_matrix_f32(td, full_rows, full_cols, full_f32);

            for (int rep = 0; rep < R; ++rep) {
                int r_off = 0;
                if (shard_rows) {
                    if (is_kv_head_shard) {
                        int kv_head_start = (rep * H_sh) / gqa_ratio;
                        r_off = kv_head_start * (full_rows / Hkv);
                    } else {
                        r_off = rep * r_sh;
                    }
                }
                int c_off = shard_cols ? (rep * c_sh) : 0;

                for (int rr = 0; rr < r_sh; ++rr) {
                    const float* src_row = full_f32.data() + static_cast<size_t>(r_off + rr) * static_cast<size_t>(full_cols) + static_cast<size_t>(c_off);
                    std::memcpy(slice_f32.data() + static_cast<size_t>(rr) * static_cast<size_t>(c_sh),
                                src_row, static_cast<size_t>(c_sh) * sizeof(float));
                }

                const size_t dst_off = static_cast<size_t>(idx) * static_cast<size_t>(r_sh) * static_cast<size_t>(c_sh);
                if (!i8) {
                    HavenTpuPjrtBackend::convert_f32_to_bf16(slice_f32.data(), bf16_shards[static_cast<size_t>(rep)].data() + dst_off, slice_f32.size());
                } else {
                    quantize_f32_to_s8_per_row(slice_f32.data(), r_sh, c_sh,
                        s8_shards[static_cast<size_t>(rep)].data() + dst_off,
                        sc_shards[static_cast<size_t>(rep)].data() + static_cast<size_t>(idx) * static_cast<size_t>(r_sh));
                }
            }
        }

        for (int rep = 0; rep < R; ++rep) {
            PJRT_Buffer* wb = nullptr;
            if (!i8) {
                wb = backend_->upload_bf16_nd(bf16_shards[static_cast<size_t>(rep)].data(), {cnt, r_sh, c_sh}, rep);
                total_bytes += shard_elems * sizeof(uint16_t);
            } else {
                wb = backend_->upload_s8_nd(s8_shards[static_cast<size_t>(rep)].data(), {cnt, r_sh, c_sh}, rep);
                PJRT_Buffer* sb = backend_->upload_f32_nd(sc_shards[static_cast<size_t>(rep)].data(), {cnt, r_sh}, rep);
                if (!sb) { backend_->destroy_buffer(wb); return false; }
                replica_scale_bufs[static_cast<size_t>(rep)].push_back(sb);
                total_bytes += shard_elems + static_cast<size_t>(cnt) * static_cast<size_t>(r_sh) * sizeof(float);
            }
            if (!wb) return false;
            replica_weight_bufs_[static_cast<size_t>(rep)].push_back(wb);
        }
        return true;
    };

    std::vector<int> all_layers(static_cast<size_t>(L));
    for (int l = 0; l < L; ++l) all_layers[static_cast<size_t>(l)] = l;

    // 1. Upload the 11 stacked layer projection matrices in exact MLIR parameter order!
    if (!upload_sharded_stack(all_layers,     [&](int l){ return host_engine.get_layer_tensors(l).ffn_gate; },    F, D, true, false, false)) return false;
    if (!upload_sharded_stack(all_layers,     [&](int l){ return host_engine.get_layer_tensors(l).ffn_up; },      F, D, true, false, false)) return false;
    if (!upload_sharded_stack(all_layers,     [&](int l){ return host_engine.get_layer_tensors(l).ffn_down; },    D, F, false, true, false)) return false;
    if (!upload_sharded_stack(swa_layers,     [&](int l){ return host_engine.get_layer_tensors(l).attn_q; },      H * HD_swa, D, true, false, false)) return false;
    if (!upload_sharded_stack(swa_layers,     [&](int l){ return host_engine.get_layer_tensors(l).attn_output; }, D, H * HD_swa, false, true, false)) return false;
    if (!upload_sharded_stack(swa_kv_layers,  [&](int l){ return host_engine.get_layer_tensors(l).attn_k; },      Hkv * HD_swa, D, (R > 1), false, true)) return false;
    if (!upload_sharded_stack(swa_kv_layers,  [&](int l){ return host_engine.get_layer_tensors(l).attn_v; },      Hkv * HD_swa, D, (R > 1), false, true)) return false;
    if (!upload_sharded_stack(glob_layers,    [&](int l){ return host_engine.get_layer_tensors(l).attn_q; },      H * HD_glob, D, true, false, false)) return false;
    if (!upload_sharded_stack(glob_layers,    [&](int l){ return host_engine.get_layer_tensors(l).attn_output; }, D, H * HD_glob, false, true, false)) return false;
    if (!upload_sharded_stack(glob_kv_layers, [&](int l){ return host_engine.get_layer_tensors(l).attn_k; },      Hkv * HD_glob, D, (R > 1), false, true)) return false;
    if (!upload_sharded_stack(glob_kv_layers, [&](int l){ return host_engine.get_layer_tensors(l).attn_v; },      Hkv * HD_glob, D, (R > 1), false, true)) return false;

    // 2. Upload sharded 262k LM Head [V_sh, D]
    {
        const TensorDesc* lm = host_engine.get_output_tensor();
        if (!lm) lm = host_engine.get_token_embd_tensor();
        std::vector<float> lm_f32;
        dequantize_full_matrix_f32(lm, V, D, lm_f32);
        for (int rep = 0; rep < R; ++rep) {
            const float* shard_ptr = lm_f32.data() + static_cast<size_t>(rep) * static_cast<size_t>(V_sh) * static_cast<size_t>(D);
            const size_t elems = static_cast<size_t>(V_sh) * static_cast<size_t>(D);
            PJRT_Buffer* wb = nullptr;
            if (!i8) {
                std::vector<uint16_t> bf(elems);
                HavenTpuPjrtBackend::convert_f32_to_bf16(shard_ptr, bf.data(), elems);
                wb = backend_->upload_bf16_nd(bf.data(), {V_sh, D}, rep);
                total_bytes += elems * sizeof(uint16_t);
            } else {
                std::vector<int8_t> sq(elems);
                std::vector<float> sc(static_cast<size_t>(V_sh));
                quantize_f32_to_s8_per_row(shard_ptr, V_sh, D, sq.data(), sc.data());
                wb = backend_->upload_s8_nd(sq.data(), {V_sh, D}, rep);
                PJRT_Buffer* sb = backend_->upload_f32_nd(sc.data(), {V_sh}, rep);
                if (!sb) { backend_->destroy_buffer(wb); return false; }
                replica_scale_bufs[static_cast<size_t>(rep)].push_back(sb);
                total_bytes += elems + static_cast<size_t>(V_sh) * sizeof(float);
            }
            if (!wb) return false;
            replica_weight_bufs_[static_cast<size_t>(rep)].push_back(wb);
        }
    }

    // Append INT8 scales immediately after the 12 main weights (matching MLIR arg order!)
    if (i8) {
        for (int rep = 0; rep < R; ++rep) {
            for (auto* sb : replica_scale_bufs[static_cast<size_t>(rep)]) {
                replica_weight_bufs_[static_cast<size_t>(rep)].push_back(sb);
            }
        }
    }

    // 3. Upload replicated small BF16 Pleias weights: w_inp_gate [42, 256, 2560], w_proj [42, 2560, 256], w_pl_mproj [10752, 2560]
    auto upload_replicated_bf16_stack = [&](const std::vector<int>& layer_list,
                                            const std::function<const TensorDesc*(int)>& getter,
                                            int rows, int cols) -> bool {
        const int cnt = static_cast<int>(layer_list.size());
        std::vector<uint16_t> bf(static_cast<size_t>(cnt) * static_cast<size_t>(rows) * static_cast<size_t>(cols));
        std::vector<float> mat;
        for (int idx = 0; idx < cnt; ++idx) {
            dequantize_full_matrix_f32(getter(layer_list[static_cast<size_t>(idx)]), rows, cols, mat);
            HavenTpuPjrtBackend::convert_f32_to_bf16(mat.data(), bf.data() + static_cast<size_t>(idx) * mat.size(), mat.size());
        }
        for (int rep = 0; rep < R; ++rep) {
            PJRT_Buffer* b = backend_->upload_bf16_nd(bf.data(), {cnt, rows, cols}, rep);
            if (!b) return false;
            replica_weight_bufs_[static_cast<size_t>(rep)].push_back(b);
            total_bytes += bf.size() * sizeof(uint16_t);
        }
        return true;
    };

    if (!upload_replicated_bf16_stack(all_layers, [&](int l){ return host_engine.get_layer_tensors(l).inp_gate; }, P_dim, D)) return false;
    if (!upload_replicated_bf16_stack(all_layers, [&](int l){ return host_engine.get_layer_tensors(l).proj; },     D, P_dim)) return false;

    {
        std::vector<float> mp_f32;
        dequantize_full_matrix_f32(host_engine.get_per_layer_model_proj_tensor(), L * P_dim, D, mp_f32);
        std::vector<uint16_t> mp_bf(mp_f32.size());
        HavenTpuPjrtBackend::convert_f32_to_bf16(mp_f32.data(), mp_bf.data(), mp_f32.size());
        for (int rep = 0; rep < R; ++rep) {
            PJRT_Buffer* b = backend_->upload_bf16_nd(mp_bf.data(), {L * P_dim, D}, rep);
            if (!b) return false;
            replica_weight_bufs_[static_cast<size_t>(rep)].push_back(b);
            total_bytes += mp_bf.size() * sizeof(uint16_t);
        }
    }

    // 4. Upload all F32 norm stacks & layer scales (replicated across replicas)
    auto upload_rep_f32 = [&](const std::vector<float>& data, const std::vector<int64_t>& dims) -> bool {
        for (int rep = 0; rep < R; ++rep) {
            PJRT_Buffer* b = backend_->upload_f32_nd(data.data(), dims, rep);
            if (!b) return false;
            replica_weight_bufs_[static_cast<size_t>(rep)].push_back(b);
            total_bytes += data.size() * sizeof(float);
        }
        return true;
    };

    auto stack_norms = [&](const std::vector<int>& layer_list,
                           const std::function<const TensorDesc*(int)>& getter,
                           int dim, float fallback = 1.0f) -> std::vector<float> {
        std::vector<float> out(layer_list.size() * static_cast<size_t>(dim), fallback);
        for (size_t i = 0; i < layer_list.size(); ++i) {
            auto v = extract_f32_1d(getter(layer_list[i]), dim, fallback);
            std::memcpy(out.data() + i * static_cast<size_t>(dim), v.data(), static_cast<size_t>(dim) * sizeof(float));
        }
        return out;
    };

    if (!upload_rep_f32(extract_f32_1d(host_engine.get_per_layer_proj_norm_tensor(), P_dim, 1.0f), {P_dim})) return false;
    if (!upload_rep_f32(stack_norms(all_layers, [&](int l){ return host_engine.get_layer_tensors(l).attn_norm; }, D), {L, D})) return false;
    if (!upload_rep_f32(stack_norms(all_layers, [&](int l){ return host_engine.get_layer_tensors(l).post_attention_norm; }, D), {L, D})) return false;
    if (!upload_rep_f32(stack_norms(all_layers, [&](int l){ return host_engine.get_layer_tensors(l).ffn_norm; }, D), {L, D})) return false;
    if (!upload_rep_f32(stack_norms(all_layers, [&](int l){ return host_engine.get_layer_tensors(l).post_ffw_norm; }, D), {L, D})) return false;
    if (!upload_rep_f32(stack_norms(all_layers, [&](int l){ return host_engine.get_layer_tensors(l).post_norm; }, D), {L, D})) return false;

    {
        std::vector<float> lscales(static_cast<size_t>(L), 1.0f);
        for (int l = 0; l < L; ++l) {
            auto v = extract_f32_1d(host_engine.get_layer_tensors(l).layer_output_scale, 1, 1.0f);
            lscales[static_cast<size_t>(l)] = v[0];
        }
        if (!upload_rep_f32(lscales, {L})) return false;
    }

    if (!upload_rep_f32(stack_norms(swa_layers,     [&](int l){ return host_engine.get_layer_tensors(l).attn_q_norm; }, HD_swa),  {N_swa, HD_swa})) return false;
    if (!upload_rep_f32(stack_norms(swa_kv_layers,  [&](int l){ return host_engine.get_layer_tensors(l).attn_k_norm; }, HD_swa),  {N_swa_kv, HD_swa})) return false;
    if (!upload_rep_f32(stack_norms(glob_layers,    [&](int l){ return host_engine.get_layer_tensors(l).attn_q_norm; }, HD_glob), {N_glob, HD_glob})) return false;
    if (!upload_rep_f32(stack_norms(glob_kv_layers, [&](int l){ return host_engine.get_layer_tensors(l).attn_k_norm; }, HD_glob), {N_glob_kv, HD_glob})) return false;
    if (!upload_rep_f32(extract_f32_1d(host_engine.get_output_norm_tensor(), D, 1.0f), {D})) return false;

    // 5. Precompute exact RoPE cos/sin tables matching Avx2Math::apply_rope bit-for-bit!
    auto build_rope_tables = [&](int head_dim, float freq_base, const float* freq_factors,
                                 std::vector<float>& cos_tbl, std::vector<float>& sin_tbl) {
        const int pairs = head_dim / 2;
        cos_tbl.resize(static_cast<size_t>(T) * static_cast<size_t>(pairs));
        sin_tbl.resize(static_cast<size_t>(T) * static_cast<size_t>(pairs));
        for (int p = 0; p < T; ++p) {
            for (int i = 0; i < pairs; ++i) {
                float theta = static_cast<float>(p) * std::pow(freq_base, -2.0f * static_cast<float>(i) / static_cast<float>(head_dim)) * model_cfg_.rope_freq_scale;
                if (freq_factors && freq_factors[i] > 0.0f) {
                    theta /= freq_factors[i];
                }
                cos_tbl[static_cast<size_t>(p) * static_cast<size_t>(pairs) + static_cast<size_t>(i)] = std::cos(theta);
                sin_tbl[static_cast<size_t>(p) * static_cast<size_t>(pairs) + static_cast<size_t>(i)] = std::sin(theta);
            }
        }
    };

    const TensorDesc* rf_td = host_engine.get_rope_freqs_tensor();
    const float* rf_ptr = (rf_td && rf_td->data && rf_td->type == QuantType::F32 &&
        (rf_td->size_bytes == 0 || rf_td->size_bytes >= 256ULL * sizeof(float)))
        ? reinterpret_cast<const float*>(rf_td->data) : nullptr;

    std::vector<float> cos_swa, sin_swa, cos_glob, sin_glob;
    build_rope_tables(HD_swa,  model_cfg_.rope_freq_base_swa, nullptr, cos_swa, sin_swa);
    build_rope_tables(HD_glob, model_cfg_.rope_freq_base,     rf_ptr,  cos_glob, sin_glob);

    if (!upload_rep_f32(cos_swa,  {T, HD_swa / 2})) return false;
    if (!upload_rep_f32(sin_swa,  {T, HD_swa / 2})) return false;
    if (!upload_rep_f32(cos_glob, {T, HD_glob / 2})) return false;
    if (!upload_rep_f32(sin_glob, {T, HD_glob / 2})) return false;

    metrics_.total_hbm_weight_bytes = total_bytes;
    return true;
}

bool HavenTpuGemma4Engine::allocate_zero_kv_cache() {
    free_kv_cache();
    const int L = static_cast<int>(model_cfg_.num_layers);
    const int R = cfg_.num_replicas;
    const int B = cfg_.batch_size;
    const int T = cfg_.max_seq_len;
    const int Hkv = static_cast<int>(model_cfg_.num_kv_heads);
    const int Hkv_sh = (R == 1) ? Hkv : 1;
    const int kv_owner_limit = (L > 20) ? (L - 18) : L;
    int N_swa_kv = 0, N_glob_kv = 0;
    for (int l = 0; l < kv_owner_limit; ++l) {
        if (l % 6 != 5) N_swa_kv++; else N_glob_kv++;
    }

    const std::vector<int64_t> swa_dims  = {N_swa_kv,  B, T, Hkv_sh, 256};
    const std::vector<int64_t> glob_dims = {N_glob_kv, B, T, Hkv_sh, 512};
    const size_t swa_elems  = static_cast<size_t>(N_swa_kv)  * B * T * Hkv_sh * 256ULL;
    const size_t glob_elems = static_cast<size_t>(N_glob_kv) * B * T * Hkv_sh * 512ULL;

    std::vector<uint16_t> z_swa(swa_elems, 0);
    std::vector<uint16_t> z_glob(glob_elems, 0);

    replica_kv_bufs_.assign(static_cast<size_t>(R), {});
    size_t kv_bytes = 0;
    for (int rep = 0; rep < R; ++rep) {
        PJRT_Buffer* ks = backend_->upload_bf16_nd(z_swa.data(),  swa_dims,  rep);
        PJRT_Buffer* vs = backend_->upload_bf16_nd(z_swa.data(),  swa_dims,  rep);
        PJRT_Buffer* kg = backend_->upload_bf16_nd(z_glob.data(), glob_dims, rep);
        PJRT_Buffer* vg = backend_->upload_bf16_nd(z_glob.data(), glob_dims, rep);
        if (!ks || !vs || !kg || !vg) {
            backend_->destroy_buffer(ks); backend_->destroy_buffer(vs);
            backend_->destroy_buffer(kg); backend_->destroy_buffer(vg);
            free_kv_cache();
            return false;
        }
        replica_kv_bufs_[static_cast<size_t>(rep)] = {ks, vs, kg, vg};
        kv_bytes += (2 * swa_elems + 2 * glob_elems) * sizeof(uint16_t);
    }
    metrics_.total_hbm_kv_bytes = kv_bytes;
    return true;
}

bool HavenTpuGemma4Engine::reset_kv_cache() {
    if (!initialized_) return false;
    return allocate_zero_kv_cache();
}

bool HavenTpuGemma4Engine::initialize(HavenTpuPjrtBackend* backend,
                                      const HavenEngine& host_engine,
                                      const TpuGemma4Config& cfg) {
    shutdown();
    if (!backend || !backend->is_initialized()) {
        last_error_ = "PJRT backend is not initialized.";
        return false;
    }
    const auto& mc = host_engine.get_config();
    if (mc.num_layers == 0 || mc.embedding_dim == 0 || mc.hidden_dim == 0 || mc.vocab_size == 0) {
        last_error_ = "Host engine model configuration is invalid.";
        return false;
    }
    if (cfg.num_replicas != 1 && cfg.num_replicas != 2 && cfg.num_replicas != 4) {
        last_error_ = "num_replicas must be 1, 2, or 4.";
        return false;
    }
    if (static_cast<size_t>(cfg.num_replicas) > backend->get_device_count()) {
        last_error_ = "Requested num_replicas exceeds addressable TPU devices.";
        return false;
    }

    backend_ = backend;
    host_engine_ = &host_engine;
    model_cfg_ = mc;
    cfg_ = cfg;
    metrics_ = {};

    // 1. Upload all 42 layers + LM head + norms + RoPE tables to TPU HBM3
    auto t0 = std::chrono::high_resolution_clock::now();
    if (!build_and_upload_weights(host_engine)) {
        last_error_ = "Weight upload failed: " + backend_->get_last_error();
        shutdown();
        return false;
    }
    if (!allocate_zero_kv_cache()) {
        last_error_ = "KV cache HBM3 allocation failed: " + backend_->get_last_error();
        shutdown();
        return false;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    metrics_.weight_upload_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 2. Compile full 42-layer decode graph (S = 1)
    auto dc0 = std::chrono::high_resolution_clock::now();
    std::string decode_mlir = emit_stablehlo_mlir(model_cfg_, cfg_, 1);
    decode_exec_ = (cfg_.num_replicas == 1)
        ? backend_->compile_stablehlo(decode_mlir, -1)
        : backend_->compile_stablehlo_replicated(decode_mlir, cfg_.num_replicas);
    auto dc1 = std::chrono::high_resolution_clock::now();
    metrics_.decode_compile_ms = std::chrono::duration<double, std::milli>(dc1 - dc0).count();
    if (!decode_exec_) {
        last_error_ = "Decode StableHLO compile failed: " + backend_->get_last_error();
        shutdown();
        return false;
    }

    // 3. Optionally compile batched prefill graph (S = prefill_chunk_size)
    if (cfg_.prefill_chunk_size > 1) {
        auto pc0 = std::chrono::high_resolution_clock::now();
        std::string prefill_mlir = emit_stablehlo_mlir(model_cfg_, cfg_, cfg_.prefill_chunk_size);
        prefill_exec_ = (cfg_.num_replicas == 1)
            ? backend_->compile_stablehlo(prefill_mlir, -1)
            : backend_->compile_stablehlo_replicated(prefill_mlir, cfg_.num_replicas);
        auto pc1 = std::chrono::high_resolution_clock::now();
        metrics_.prefill_compile_ms = std::chrono::duration<double, std::milli>(pc1 - pc0).count();
        if (!prefill_exec_) {
            last_error_ = "Prefill StableHLO compile failed: " + backend_->get_last_error();
            shutdown();
            return false;
        }
    }

    initialized_ = true;
    return true;
}

bool HavenTpuGemma4Engine::execute_graph(PJRT_LoadedExecutable* exec,
                                         int S,
                                         const std::vector<uint32_t>& flat_tokens,
                                         int start_pos,
                                         std::vector<float>* out_logits,
                                         std::vector<uint32_t>& out_next_tokens) {
    if (!initialized_ || !exec) return false;
    const int B = cfg_.batch_size;
    const int R = cfg_.num_replicas;
    const int L = static_cast<int>(model_cfg_.num_layers);
    const int D = static_cast<int>(model_cfg_.embedding_dim);
    const int P_dim = 256;
    const int V = static_cast<int>(model_cfg_.vocab_size);
    if (static_cast<int>(flat_tokens.size()) != B * S) {
        last_error_ = "flat_tokens size must equal B * S.";
        return false;
    }

    // Fast host lookup of [B, S, D] token embeddings and [B, S, L, 256] Pleias token embeddings (~5 us for B=1, S=1!)
    const size_t emb_n = static_cast<size_t>(B) * static_cast<size_t>(S) * static_cast<size_t>(D);
    const size_t pl_n  = static_cast<size_t>(B) * static_cast<size_t>(S) * static_cast<size_t>(L) * static_cast<size_t>(P_dim);
    emb_f32_scratch_.resize(emb_n);
    emb_bf16_scratch_.resize(emb_n);
    pleias_f32_scratch_.resize(pl_n);
    pleias_bf16_scratch_.resize(pl_n);

    const TensorDesc* tok_td = host_engine_->get_token_embd_tensor();
    const TensorDesc* pl_td  = host_engine_->get_per_layer_token_embd_tensor();
    const float emb_scale = model_cfg_.is_gemma ? std::sqrt(static_cast<float>(D)) : 1.0f;
    const float tok_pl_scale = std::sqrt(static_cast<float>(P_dim));

    for (int idx = 0; idx < B * S; ++idx) {
        const int tok = static_cast<int>(std::min<uint32_t>(flat_tokens[static_cast<size_t>(idx)], model_cfg_.vocab_size - 1));
        float* edst = emb_f32_scratch_.data() + static_cast<size_t>(idx) * static_cast<size_t>(D);
        if (tok_td) {
            Avx2Math::dequantize_row(edst, *tok_td, tok, D);
            for (int c = 0; c < D; ++c) edst[c] *= emb_scale;
        } else {
            std::memset(edst, 0, static_cast<size_t>(D) * sizeof(float));
        }

        float* pdst = pleias_f32_scratch_.data() + static_cast<size_t>(idx) * static_cast<size_t>(L * P_dim);
        if (pl_td) {
            Avx2Math::dequantize_row(pdst, *pl_td, tok, L * P_dim);
            for (int c = 0; c < L * P_dim; ++c) pdst[c] *= tok_pl_scale;
        } else {
            std::memset(pdst, 0, static_cast<size_t>(L * P_dim) * sizeof(float));
        }
    }

    HavenTpuPjrtBackend::convert_f32_to_bf16(emb_f32_scratch_.data(), emb_bf16_scratch_.data(), emb_n);
    HavenTpuPjrtBackend::convert_f32_to_bf16(pleias_f32_scratch_.data(), pleias_bf16_scratch_.data(), pl_n);

    int32_t pos_i32 = static_cast<int32_t>(std::clamp(start_pos, 0, std::max(0, cfg_.max_seq_len - S)));

    // Upload tiny per-step inputs (26 KB for B=1, S=1!) to each replica
    std::vector<PJRT_Buffer*> step_emb(static_cast<size_t>(R), nullptr);
    std::vector<PJRT_Buffer*> step_pl(static_cast<size_t>(R), nullptr);
    std::vector<PJRT_Buffer*> step_pos(static_cast<size_t>(R), nullptr);
    std::vector<std::vector<PJRT_Buffer*>> args_per_rep(static_cast<size_t>(R));

    for (int rep = 0; rep < R; ++rep) {
        step_emb[static_cast<size_t>(rep)] = backend_->upload_bf16_nd(emb_bf16_scratch_.data(), {B, S, D}, rep);
        step_pl[static_cast<size_t>(rep)]  = backend_->upload_bf16_nd(pleias_bf16_scratch_.data(), {B, S, L, P_dim}, rep);
        step_pos[static_cast<size_t>(rep)] = backend_->upload_i32_nd(&pos_i32, {}, rep);
        if (!step_emb[static_cast<size_t>(rep)] || !step_pl[static_cast<size_t>(rep)] || !step_pos[static_cast<size_t>(rep)]) {
            for (int k = 0; k <= rep; ++k) {
                backend_->destroy_buffer(step_emb[static_cast<size_t>(k)]);
                backend_->destroy_buffer(step_pl[static_cast<size_t>(k)]);
                backend_->destroy_buffer(step_pos[static_cast<size_t>(k)]);
            }
            last_error_ = "Per-step H2D upload failed: " + backend_->get_last_error();
            return false;
        }
        auto& a = args_per_rep[static_cast<size_t>(rep)];
        a.reserve(3 + replica_weight_bufs_[static_cast<size_t>(rep)].size() + 4);
        a.push_back(step_emb[static_cast<size_t>(rep)]);
        a.push_back(step_pl[static_cast<size_t>(rep)]);
        a.push_back(step_pos[static_cast<size_t>(rep)]);
        for (auto* wb : replica_weight_bufs_[static_cast<size_t>(rep)]) a.push_back(wb);
        for (auto* kv : replica_kv_bufs_[static_cast<size_t>(rep)]) a.push_back(kv);
    }

    // Execute ALL 42 layers + HBM3 KV update + LM head + Argmax in ONE SINGLE TPU LAUNCH!
    std::vector<std::vector<PJRT_Buffer*>> outs_per_rep;
    bool ok = false;
    if (R == 1) {
        outs_per_rep.resize(1);
        ok = backend_->execute_1d(exec, 0, args_per_rep[0], 5, outs_per_rep[0]);
    } else {
        ok = backend_->execute_replicated(exec, R, args_per_rep, 5, outs_per_rep);
    }

    for (int rep = 0; rep < R; ++rep) {
        backend_->destroy_buffer(step_emb[static_cast<size_t>(rep)]);
        backend_->destroy_buffer(step_pl[static_cast<size_t>(rep)]);
        backend_->destroy_buffer(step_pos[static_cast<size_t>(rep)]);
    }

    if (!ok) {
        last_error_ = "On-chip Gemma 4 execute failed: " + backend_->get_last_error();
        return false;
    }

    // Swap mutated/donated KV cache handles in-place (destroying old handles safely!)
    for (int rep = 0; rep < R; ++rep) {
        for (size_t k = 0; k < 4; ++k) {
            backend_->destroy_buffer(replica_kv_bufs_[static_cast<size_t>(rep)][k]);
            replica_kv_bufs_[static_cast<size_t>(rep)][k] = outs_per_rep[static_cast<size_t>(rep)][1 + k];
        }
    }

    // Read output 0 from replica 0 (all replicas hold identical gathered output!)
    out_next_tokens.assign(static_cast<size_t>(B), 0);
    if (cfg_.return_full_logits) {
        std::vector<float> local_logits(static_cast<size_t>(B) * static_cast<size_t>(V));
        ok = backend_->download_f32(outs_per_rep[0][0], local_logits.data(), local_logits.size());
        if (ok) {
            for (int b = 0; b < B; ++b) {
                const float* row = local_logits.data() + static_cast<size_t>(b) * static_cast<size_t>(V);
                out_next_tokens[static_cast<size_t>(b)] = static_cast<uint32_t>(std::max_element(row, row + V) - row);
            }
            if (out_logits) {
                *out_logits = std::move(local_logits);
            }
        }
    } else {
        std::vector<int32_t> argmax_ids(static_cast<size_t>(B), 0);
        ok = backend_->download_i32(outs_per_rep[0][0], argmax_ids.data(), argmax_ids.size());
        if (ok) {
            for (int b = 0; b < B; ++b) {
                out_next_tokens[static_cast<size_t>(b)] = static_cast<uint32_t>(std::max(0, argmax_ids[static_cast<size_t>(b)]));
            }
        }
    }

    for (int rep = 0; rep < R; ++rep) {
        backend_->destroy_buffer(outs_per_rep[static_cast<size_t>(rep)][0]);
    }
    return ok;
}

bool HavenTpuGemma4Engine::prefill_prompt(const std::vector<uint32_t>& tokens,
                                          int start_pos,
                                          std::vector<float>* out_logits,
                                          uint32_t* out_next_token) {
    if (!initialized_ || tokens.empty()) return false;
    const int B = cfg_.batch_size;
    const int P = cfg_.prefill_chunk_size;
    int pos = start_pos;
    size_t cursor = 0;
    std::vector<uint32_t> next_toks;

    // Chunked batched prefill across full 42-layer TPU MXU graph
    while (prefill_exec_ && P > 1 && cursor + static_cast<size_t>(P) <= tokens.size()) {
        std::vector<uint32_t> chunk(static_cast<size_t>(B) * static_cast<size_t>(P));
        for (int b = 0; b < B; ++b) {
            std::memcpy(chunk.data() + static_cast<size_t>(b) * static_cast<size_t>(P),
                        tokens.data() + cursor, static_cast<size_t>(P) * sizeof(uint32_t));
        }
        bool is_last = (cursor + static_cast<size_t>(P) == tokens.size());
        if (!execute_graph(prefill_exec_, P, chunk, pos, is_last ? out_logits : nullptr, next_toks)) {
            return false;
        }
        cursor += static_cast<size_t>(P);
        pos += P;
    }

    // Remainder tail via fast on-chip decode step
    while (cursor < tokens.size()) {
        std::vector<uint32_t> step_toks(static_cast<size_t>(B), tokens[cursor]);
        bool is_last = (cursor + 1 == tokens.size());
        if (!execute_graph(decode_exec_, 1, step_toks, pos, is_last ? out_logits : nullptr, next_toks)) {
            return false;
        }
        cursor++;
        pos++;
    }

    if (out_next_token && !next_toks.empty()) {
        *out_next_token = next_toks[0];
    }
    return true;
}

bool HavenTpuGemma4Engine::decode_step(const std::vector<uint32_t>& tokens_per_batch,
                                       int pos,
                                       std::vector<float>* out_logits,
                                       std::vector<uint32_t>& out_next_tokens) {
    return execute_graph(decode_exec_, 1, tokens_per_batch, pos, out_logits, out_next_tokens);
}

} // namespace haven
