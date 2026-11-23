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

static int compute_hkv_shard(int Hkv, int R) {
    if (R <= 1) return Hkv;
    if (Hkv >= R && (Hkv % R == 0)) return Hkv / R;
    return 1;
}

// Quantize a [rows, cols] float row-major slice into symmetric INT8 + per-row F32 scales (OpenMP parallel)
static void quantize_f32_to_s8_per_row(const float* src, int rows, int cols,
                                       int8_t* dst_q, float* dst_scale) {
    #pragma omp parallel for schedule(static)
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
    if (t && t->data) {
        if (t->type == QuantType::F32 &&
            (t->size_bytes == 0 || t->size_bytes >= static_cast<size_t>(count) * sizeof(float))) {
            std::memcpy(out.data(), t->data, static_cast<size_t>(count) * sizeof(float));
        } else {
            Avx2Math::dequantize_row(out.data(), *t, 0, count);
        }
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

    // Add 1D bias sliced from [L, N] onto [B, S, N] F32 tensor (Qwen 2 / 2.5 QKV bias)
    std::string emit_add_bias_3d(const std::string& x, const std::string& b_stack,
                                 int layer_idx, int L, int B, int S, int N) {
        const std::string y_t = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(N) + "xf32>";
        const std::string b0_t = "tensor<" + std::to_string(N) + "xf32>";
        std::string b_1d = slice_1d_from_2d(b_stack, layer_idx, L, N);
        std::string b_bc = tmp("bbc");
        std::string out  = tmp("badd");
        ss << "    " << b_bc << " = stablehlo.broadcast_in_dim " << b_1d << ", dims = [2] : (" << b0_t << ") -> " << y_t << "\n"
           << "    " << out  << " = stablehlo.add " << x << ", " << b_bc << " : " << y_t << "\n";
        return out;
    }

    // Exact Gemma tanh-approx GELU on 3D F32 tensor [B, S, M]
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

    // Exact SiLU (SwiGLU gate) on 3D F32 tensor [B, S, M] for Llama 3/4, Qwen 2.5/3, Mistral, Phi-3/4
    std::string emit_silu_3d(const std::string& g, int B, int S, int M) {
        const std::string t = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(M) + "xf32>";
        std::string sig = tmp("sig"), out = tmp("silu");
        ss << "    " << sig << " = stablehlo.logistic " << g << " : " << t << "\n"
           << "    " << out << " = stablehlo.multiply " << g << ", " << sig << " : " << t << "\n";
        return out;
    }

    // Half-split RoPE on [B, S, H, HD] (F32) matching Avx2Math::apply_rope exactly.
    // When per_slot_pos == true (S == 1, B > 1), cos_s/sin_s have shape [B, HD/2] broadcast on dims [0, 3].
    // Otherwise cos_s/sin_s have shape [S, HD/2] broadcast on dims [1, 3].
    std::string emit_rope_4d(const std::string& t, const std::string& cos_s, const std::string& sin_s,
                             int B, int S, int H, int HD, bool per_slot_pos) {
        const int HD2 = HD / 2;
        const std::string t4  = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(H) + "x" + std::to_string(HD) + "xf32>";
        const std::string t4h = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(H) + "x" + std::to_string(HD2) + "xf32>";
        const std::string tc  = "tensor<" + std::to_string(per_slot_pos ? B : S) + "x" + std::to_string(HD2) + "xf32>";
        const char* bdims = per_slot_pos ? "[0, 3]" : "[1, 3]";

        std::string t0 = tmp("rp0"), t1 = tmp("rp1"), cb = tmp("cb"), sb = tmp("sb");
        std::string m00 = tmp("m"), m11 = tmp("m"), m01 = tmp("m"), m10 = tmp("m");
        std::string o0 = tmp("ro0"), o1 = tmp("ro1"), out = tmp("rot");
        ss << "    " << t0 << " = stablehlo.slice " << t << " [0:" << B << ", 0:" << S << ", 0:" << H << ", 0:" << HD2 << "] : (" << t4 << ") -> " << t4h << "\n"
           << "    " << t1 << " = stablehlo.slice " << t << " [0:" << B << ", 0:" << S << ", 0:" << H << ", " << HD2 << ":" << HD << "] : (" << t4 << ") -> " << t4h << "\n"
           << "    " << cb << " = stablehlo.broadcast_in_dim " << cos_s << ", dims = " << bdims << " : (" << tc << ") -> " << t4h << "\n"
           << "    " << sb << " = stablehlo.broadcast_in_dim " << sin_s << ", dims = " << bdims << " : (" << tc << ") -> " << t4h << "\n"
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
    std::string emit_all_reduce_sum_3d(const std::string& x, int B, int S, int D, int R, const char* elem_type = "f32") {
        if (R <= 1) return x;
        const std::string et(elem_type);
        const std::string t = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(D) + "x" + et + ">";
        const std::string st = "tensor<" + et + ">";
        std::string out = tmp("ar");
        ss << "    " << out << " = \"stablehlo.all_reduce\"(" << x << ") ({\n"
           << "    ^bb0(%ra: " << st << ", %rb: " << st << "):\n"
           << "      %rs = stablehlo.add %ra, %rb : " << st << "\n"
           << "      stablehlo.return %rs : " << st << "\n"
           << "    }) {replica_groups = " << replica_group_attr(R) << "} : (" << t << ") -> " << t << "\n";
        return out;
    }

    // Gather rows from a 2D [V_sh, K] table (bf16 or i8) using [B, S, 1] i32 indices -> [B, S, K] f32
    std::string emit_table_gather_2d(const std::string& table, const std::string& scale_1d,
                                     const std::string& idx_3d, int V_sh, int K, int B, int S, bool use_int8) {
        const std::string wel = use_int8 ? "i8" : "bf16";
        const std::string tab_t = "tensor<" + std::to_string(V_sh) + "x" + std::to_string(K) + "x" + wel + ">";
        const std::string idx_t = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x1xi32>";
        const std::string raw_t = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(K) + "x" + wel + ">";
        const std::string f32_t = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(K) + "xf32>";

        std::string g_raw = tmp("graw");
        std::string g_f32 = tmp("gf32");
        ss << "    " << g_raw << " = \"stablehlo.gather\"(" << table << ", " << idx_3d << ") {\n"
           << "      dimension_numbers = #stablehlo.gather<\n"
           << "        offset_dims = [2],\n"
           << "        collapsed_slice_dims = [0],\n"
           << "        start_index_map = [0],\n"
           << "        index_vector_dim = 2\n"
           << "      >,\n"
           << "      indices_are_sorted = false,\n"
           << "      slice_sizes = array<i64: 1, " << K << ">\n"
           << "    } : (" << tab_t << ", " << idx_t << ") -> " << raw_t << "\n"
           << "    " << g_f32 << " = stablehlo.convert " << g_raw << " : (" << raw_t << ") -> " << f32_t << "\n";

        if (!use_int8) return g_f32;

        const std::string s1_t = "tensor<" + std::to_string(V_sh) + "xf32>";
        const std::string s2_t = "tensor<" + std::to_string(V_sh) + "x1xf32>";
        const std::string gs_t = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x1xf32>";
        const std::string g2_t = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "xf32>";
        std::string s2 = tmp("gs2"), gs = tmp("gsc"), g2 = tmp("gsr"), sb = tmp("gsb"), out = tmp("gdq");
        ss << "    " << s2 << " = stablehlo.reshape " << scale_1d << " : (" << s1_t << ") -> " << s2_t << "\n"
           << "    " << gs << " = \"stablehlo.gather\"(" << s2 << ", " << idx_3d << ") {\n"
           << "      dimension_numbers = #stablehlo.gather<\n"
           << "        offset_dims = [2],\n"
           << "        collapsed_slice_dims = [0],\n"
           << "        start_index_map = [0],\n"
           << "        index_vector_dim = 2\n"
           << "      >,\n"
           << "      indices_are_sorted = false,\n"
           << "      slice_sizes = array<i64: 1, 1>\n"
           << "    } : (" << s2_t << ", " << idx_t << ") -> " << gs_t << "\n"
           << "    " << g2  << " = stablehlo.reshape " << gs << " : (" << gs_t << ") -> " << g2_t << "\n"
           << "    " << sb  << " = stablehlo.broadcast_in_dim " << g2 << ", dims = [0, 1] : (" << g2_t << ") -> " << f32_t << "\n"
           << "    " << out << " = stablehlo.multiply " << g_f32 << ", " << sb << " : " << f32_t << "\n";
        return out;
    }

    // Gather [B, HD2] RoPE rows from [T, HD2] using [B, 1] i32 indices
    std::string emit_rope_gather_2d(const std::string& table, const std::string& pos_idx,
                                    int T, int HD2, int B) {
        const std::string tab_t = "tensor<" + std::to_string(T) + "x" + std::to_string(HD2) + "xf32>";
        const std::string idx_t = "tensor<" + std::to_string(B) + "x1xi32>";
        const std::string out_t = "tensor<" + std::to_string(B) + "x" + std::to_string(HD2) + "xf32>";
        std::string out = tmp("rg");
        ss << "    " << out << " = \"stablehlo.gather\"(" << table << ", " << pos_idx << ") {\n"
           << "      dimension_numbers = #stablehlo.gather<\n"
           << "        offset_dims = [1],\n"
           << "        collapsed_slice_dims = [0],\n"
           << "        start_index_map = [0],\n"
           << "        index_vector_dim = 1\n"
           << "      >,\n"
           << "      indices_are_sorted = false,\n"
           << "      slice_sizes = array<i64: 1, " << HD2 << ">\n"
           << "    } : (" << tab_t << ", " << idx_t << ") -> " << out_t << "\n";
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

static std::vector<int> resolve_prefill_buckets(const TpuGemma4Config& cfg) {
    std::vector<int> buckets;
    if (!cfg.prefill_buckets.empty()) {
        for (int b : cfg.prefill_buckets) {
            if (b >= 1 && b <= cfg.max_seq_len) buckets.push_back(b);
        }
    } else if (cfg.prefill_chunk_size > 1) {
        if (cfg.prefill_chunk_size >= 128) {
            for (int b : {cfg.prefill_chunk_size, 32, 16, 8}) {
                if (b <= cfg.max_seq_len) buckets.push_back(b);
            }
        } else {
            buckets.push_back(std::min(cfg.prefill_chunk_size, cfg.max_seq_len));
            if (cfg.prefill_chunk_size > 16) buckets.push_back(16);
            if (cfg.prefill_chunk_size > 8) buckets.push_back(8);
        }
    }
    // For multi-tenant continuous batching (batch_size > 1), always include an isolated
    // slot-targeted S=1 tail bucket when prefill is enabled so prefill_slot() never touches other slots!
    if (!buckets.empty() && cfg.batch_size > 1) {
        buckets.push_back(1);
    }
    std::sort(buckets.begin(), buckets.end(), std::greater<int>());
    buckets.erase(std::unique(buckets.begin(), buckets.end()), buckets.end());
    return buckets;
}

} // namespace

TpuModelSpec TpuModelSpec::from_model_config(const ModelConfig& mc, const HavenEngine* host_engine) {
    TpuModelSpec s;
    s.architecture = mc.architecture;
    s.num_layers = static_cast<int>(mc.num_layers);
    s.embedding_dim = static_cast<int>(mc.embedding_dim);
    s.hidden_dim = static_cast<int>(mc.hidden_dim);
    s.vocab_size = static_cast<int>(mc.vocab_size);
    s.num_heads = static_cast<int>(mc.num_heads);
    s.num_kv_heads = static_cast<int>(mc.num_kv_heads);
    s.rms_norm_eps = mc.rms_norm_eps;
    s.attn_logit_softcapping = mc.attn_logit_softcapping;
    s.final_logit_softcapping = mc.final_logit_softcapping;
    s.is_gemma = mc.is_gemma;
    s.emb_scale = mc.is_gemma ? std::sqrt(static_cast<float>(std::max(1, s.embedding_dim))) : 1.0f;
    s.ffn_activation = mc.ffn_activation;
    s.sliding_window = (mc.sliding_window > 0) ? static_cast<int>(mc.sliding_window) : 512;

    const bool has_swa_arch = (mc.sliding_window > 0) && (mc.is_gemma || mc.architecture == "mistral");
    s.layer_is_swa.assign(static_cast<size_t>(s.num_layers), false);
    s.layer_has_kv.assign(static_cast<size_t>(s.num_layers), true);
    s.layer_type_idx.assign(static_cast<size_t>(s.num_layers), 0);
    s.layer_kv_idx.assign(static_cast<size_t>(s.num_layers), -1);

    const int default_hd = (mc.head_dim > 0) ? static_cast<int>(mc.head_dim)
                         : ((s.num_heads > 0) ? (s.embedding_dim / s.num_heads) : 128);
    s.head_dim_swa = default_hd;
    s.head_dim_glob = default_hd;

    if (host_engine != nullptr && s.num_layers > 0) {
        const auto& lt0 = host_engine->get_layer_tensors(0);
        s.has_pleias = (host_engine->get_per_layer_model_proj_tensor() != nullptr && lt0.inp_gate != nullptr);
        s.has_q_norm = (lt0.attn_q_norm != nullptr);
        s.has_k_norm = (lt0.attn_k_norm != nullptr);
        s.has_v_norm = (mc.is_gemma && s.has_q_norm && s.has_k_norm);
        s.has_post_attn_norm = (lt0.post_attention_norm != nullptr);
        s.has_post_ffn_norm = (lt0.post_ffw_norm != nullptr);
        s.has_post_norm = (lt0.post_norm != nullptr);
        s.has_layer_scale = (lt0.layer_output_scale != nullptr);
        s.has_qkv_bias = (lt0.attn_q_bias != nullptr || lt0.attn_k_bias != nullptr || lt0.attn_v_bias != nullptr);
        s.has_tied_embeddings = (host_engine->get_output_tensor() == nullptr ||
                                 host_engine->get_output_tensor() == host_engine->get_token_embd_tensor());
        if (s.has_pleias && lt0.inp_gate && lt0.inp_gate->shape.size() > 1) {
            s.pleias_dim = static_cast<int>(lt0.inp_gate->shape[1]);
        }

        bool found_swa_hd = false, found_glob_hd = false;
        for (int l = 0; l < s.num_layers; ++l) {
            const auto& lt = host_engine->get_layer_tensors(static_cast<uint32_t>(l));
            bool is_swa = has_swa_arch && (mc.architecture == "mistral" || (l % 6 != 5));
            bool has_kv = (lt.attn_k != nullptr);
            s.layer_is_swa[static_cast<size_t>(l)] = is_swa;
            s.layer_has_kv[static_cast<size_t>(l)] = has_kv;

            if (lt.attn_q && lt.attn_q->shape.size() > 1 && s.num_heads > 0) {
                int hd = static_cast<int>(lt.attn_q->shape[1]) / s.num_heads;
                if (hd > 0) {
                    if (is_swa && !found_swa_hd) { s.head_dim_swa = hd; found_swa_hd = true; }
                    if (!is_swa && !found_glob_hd) { s.head_dim_glob = hd; found_glob_hd = true; }
                }
            }
        }
    } else {
        // Synthetic specification from ModelConfig (for offline CI & IR inspection)
        const bool is_gemma4_42 = (mc.is_gemma && s.num_layers == 42);
        const bool is_gemma3_or_4 = (mc.is_gemma && (mc.architecture.find("gemma3") != std::string::npos ||
                                                     mc.architecture.find("gemma4") != std::string::npos ||
                                                     is_gemma4_42));
        s.has_pleias = is_gemma4_42;
        s.has_q_norm = is_gemma3_or_4 || (mc.architecture == "qwen3");
        s.has_k_norm = is_gemma3_or_4 || (mc.architecture == "qwen3");
        s.has_v_norm = is_gemma4_42;
        s.has_post_attn_norm = mc.is_gemma;
        s.has_post_ffn_norm = mc.is_gemma;
        s.has_post_norm = is_gemma4_42;
        s.has_layer_scale = is_gemma4_42;
        s.has_qkv_bias = (mc.architecture == "qwen2");
        s.has_tied_embeddings = mc.is_gemma;
        if (is_gemma4_42) {
            s.head_dim_swa = 256;
            s.head_dim_glob = 512;
        }
        const int kv_owner_limit = (is_gemma4_42 && s.num_layers > 20)
            ? (s.num_layers - static_cast<int>(mc.shared_kv_layers > 0 ? mc.shared_kv_layers : 18))
            : s.num_layers;
        for (int l = 0; l < s.num_layers; ++l) {
            bool is_swa = has_swa_arch && (mc.architecture == "mistral" || (l % 6 != 5));
            bool has_kv = (l < kv_owner_limit);
            s.layer_is_swa[static_cast<size_t>(l)] = is_swa;
            s.layer_has_kv[static_cast<size_t>(l)] = has_kv;
        }
    }

    s.num_swa_layers = 0;
    s.num_glob_layers = 0;
    s.num_swa_kv_layers = 0;
    s.num_glob_kv_layers = 0;
    s.last_swa_kv_idx = 0;
    s.last_glob_kv_idx = 0;

    for (int l = 0; l < s.num_layers; ++l) {
        bool is_swa = s.layer_is_swa[static_cast<size_t>(l)];
        bool has_kv = s.layer_has_kv[static_cast<size_t>(l)];
        if (is_swa) {
            s.layer_type_idx[static_cast<size_t>(l)] = s.num_swa_layers++;
            if (has_kv) {
                s.layer_kv_idx[static_cast<size_t>(l)] = s.num_swa_kv_layers;
                s.last_swa_kv_idx = s.num_swa_kv_layers++;
            }
        } else {
            s.layer_type_idx[static_cast<size_t>(l)] = s.num_glob_layers++;
            if (has_kv) {
                s.layer_kv_idx[static_cast<size_t>(l)] = s.num_glob_kv_layers;
                s.last_glob_kv_idx = s.num_glob_kv_layers++;
            }
        }
    }
    return s;
}

bool HavenTpuGemma4Engine::is_model_supported(const HavenEngine& host_engine, std::string& reason_out) {
    const auto& mc = host_engine.get_config();
    if (mc.num_layers == 0 || mc.embedding_dim == 0 || mc.hidden_dim == 0 ||
        mc.vocab_size == 0 || mc.num_heads == 0 || mc.num_kv_heads == 0) {
        reason_out = "Invalid model dimensions (zero layers, dims, heads, or vocab).";
        return false;
    }
    if (mc.num_heads % mc.num_kv_heads != 0) {
        reason_out = "num_heads must be divisible by num_kv_heads for GQA.";
        return false;
    }
    const auto& lt0 = host_engine.get_layer_tensors(0);
    if (!lt0.attn_q || !lt0.attn_k || !lt0.attn_v || !lt0.attn_output ||
        !lt0.ffn_gate || !lt0.ffn_up || !lt0.ffn_down) {
        reason_out = "Layer 0 is missing required dense attention/FFN tensors (MoE/SSM models not yet supported).";
        return false;
    }
    TpuModelSpec spec = TpuModelSpec::from_model_config(mc, &host_engine);
    if (spec.num_kv_buffers() == 0) {
        reason_out = "Model has no KV-owning layers.";
        return false;
    }
    if ((spec.num_swa_layers > 0 && (spec.head_dim_swa <= 0 || (spec.head_dim_swa % 2) != 0)) ||
        (spec.num_glob_layers > 0 && (spec.head_dim_glob <= 0 || (spec.head_dim_glob % 2) != 0))) {
        reason_out = "Head dimension must be positive and even for RoPE.";
        return false;
    }
    reason_out.clear();
    return true;
}

std::string HavenTpuGemma4Engine::emit_stablehlo_mlir(const ModelConfig& mc,
                                                      const TpuGemma4Config& tc,
                                                      int seq_chunk_len,
                                                      bool slot_targeted) {
    TpuModelSpec spec = TpuModelSpec::from_model_config(mc, nullptr);
    return emit_stablehlo_mlir(spec, tc, seq_chunk_len, slot_targeted);
}

std::string HavenTpuGemma4Engine::emit_stablehlo_mlir(const TpuModelSpec& spec,
                                                      const TpuGemma4Config& tc,
                                                      int seq_chunk_len,
                                                      bool slot_targeted) {
    const int L = spec.num_layers;
    const int D = spec.embedding_dim;
    const int F = spec.hidden_dim;
    const int V = spec.vocab_size;
    const int H = spec.num_heads;
    const int Hkv = spec.num_kv_heads;
    const int R = std::max(1, tc.num_replicas);
    const int B_cap = std::max(1, tc.batch_size);
    const int B = slot_targeted ? 1 : B_cap;
    const int S = std::max(1, seq_chunk_len);
    const int T = std::max(S, tc.max_seq_len);
    const bool i8 = tc.use_int8_weights;
    const bool onchip_emb = tc.use_onchip_embeddings;
    const bool per_slot_pos = (!slot_targeted && S == 1 && B_cap > 1);
    const std::string wel = i8 ? "i8" : "bf16";

    const int F_sh = F / R;
    const int V_sh = V / R;
    const int H_sh = H / R;
    const int Hkv_sh = compute_hkv_shard(Hkv, R);
    const int G_sh = std::max(1, H_sh / Hkv_sh);

    const int HD_swa = spec.head_dim_swa;
    const int HD_glob = spec.head_dim_glob;
    const int Q_swa_sh = H_sh * HD_swa;
    const int KV_swa_sh = Hkv_sh * HD_swa;
    const int Q_glob_sh = H_sh * HD_glob;
    const int KV_glob_sh = Hkv_sh * HD_glob;
    const int P_dim = spec.pleias_dim;

    const int N_swa = spec.num_swa_layers;
    const int N_glob = spec.num_glob_layers;
    const int N_swa_kv = spec.num_swa_kv_layers;
    const int N_glob_kv = spec.num_glob_kv_layers;
    const bool has_swa = (N_swa > 0);
    const bool has_glob = (N_glob > 0);
    const bool has_swa_kv = (N_swa_kv > 0);
    const bool has_glob_kv = (N_glob_kv > 0);

    MlirEmitter em;
    // Zero-padding TPU (8,128)-aligned KV Cache layout: [N_kv, B_cap, Hkv_sh, T, HD]
    const std::string kv_ks_t = "tensor<" + std::to_string(N_swa_kv)  + "x" + std::to_string(B_cap) + "x" + std::to_string(Hkv_sh) + "x" + std::to_string(T) + "x" + std::to_string(HD_swa)  + "xbf16>";
    const std::string kv_kg_t = "tensor<" + std::to_string(N_glob_kv) + "x" + std::to_string(B_cap) + "x" + std::to_string(Hkv_sh) + "x" + std::to_string(T) + "x" + std::to_string(HD_glob) + "xbf16>";
    const std::string out0_t  = tc.return_full_logits
        ? ("tensor<" + std::to_string(B) + "x" + std::to_string(V) + "xf32>")
        : ("tensor<" + std::to_string(B) + "xi32>");

    auto donate_attr = [&](int out_slot) -> std::string {
        if (!tc.donate_kv_cache) return "";
        return " {tf.aliasing_output = " + std::to_string(out_slot) + " : i32}";
    };

    em.ss << "module @haven_tpu_transformer {\n"
          << "  func.func @main(\n";

    // Dynamic per-step inputs
    if (onchip_emb) {
        em.ss << "    %tok_ids_in: tensor<" << B << "x" << S << "xi32>,\n";
    } else {
        em.ss << "    %x_emb_in: tensor<" << B << "x" << S << "x" << D << "xbf16>,\n";
        if (spec.has_pleias) {
            em.ss << "    %pleias_tok_in: tensor<" << B << "x" << S << "x" << L << "x" << P_dim << "xbf16>,\n";
        }
    }
    if (per_slot_pos) {
        em.ss << "    %pos_vec: tensor<" << B_cap << "xi32>,\n";
    } else {
        em.ss << "    %pos: tensor<i32>,\n";
    }
    if (slot_targeted) {
        em.ss << "    %slot_id: tensor<i32>,\n";
    }

    // 1. Stacked projection weights
    em.ss << "    %w_ffn_gate: tensor<" << L << "x" << F_sh << "x" << D << "x" << wel << ">,\n"
          << "    %w_ffn_up: tensor<" << L << "x" << F_sh << "x" << D << "x" << wel << ">,\n"
          << "    %w_ffn_down: tensor<" << L << "x" << D << "x" << F_sh << "x" << wel << ">,\n";
    if (has_swa) {
        em.ss << "    %w_swa_q: tensor<" << N_swa << "x" << Q_swa_sh << "x" << D << "x" << wel << ">,\n"
              << "    %w_swa_o: tensor<" << N_swa << "x" << D << "x" << Q_swa_sh << "x" << wel << ">,\n";
        if (has_swa_kv) {
            em.ss << "    %w_swa_k: tensor<" << N_swa_kv << "x" << KV_swa_sh << "x" << D << "x" << wel << ">,\n"
                  << "    %w_swa_v: tensor<" << N_swa_kv << "x" << KV_swa_sh << "x" << D << "x" << wel << ">,\n";
        }
    }
    if (has_glob) {
        em.ss << "    %w_glob_q: tensor<" << N_glob << "x" << Q_glob_sh << "x" << D << "x" << wel << ">,\n"
              << "    %w_glob_o: tensor<" << N_glob << "x" << D << "x" << Q_glob_sh << "x" << wel << ">,\n";
        if (has_glob_kv) {
            em.ss << "    %w_glob_k: tensor<" << N_glob_kv << "x" << KV_glob_sh << "x" << D << "x" << wel << ">,\n"
                  << "    %w_glob_v: tensor<" << N_glob_kv << "x" << KV_glob_sh << "x" << D << "x" << wel << ">,\n";
        }
    }
    em.ss << "    %w_lm_head: tensor<" << V_sh << "x" << D << "x" << wel << ">,\n";
    if (onchip_emb && !spec.has_tied_embeddings) {
        em.ss << "    %w_tok_emb: tensor<" << V_sh << "x" << D << "x" << wel << ">,\n";
    }
    if (onchip_emb && spec.has_pleias) {
        em.ss << "    %w_pl_tok_emb: tensor<" << V_sh << "x" << (L * P_dim) << "x" << wel << ">,\n";
    }

    // 2. Optional INT8 per-row F32 scales (exact 1:1 mirror of projection weights)
    if (i8) {
        em.ss << "    %s_ffn_gate: tensor<" << L << "x" << F_sh << "xf32>,\n"
              << "    %s_ffn_up: tensor<" << L << "x" << F_sh << "xf32>,\n"
              << "    %s_ffn_down: tensor<" << L << "x" << D << "xf32>,\n";
        if (has_swa) {
            em.ss << "    %s_swa_q: tensor<" << N_swa << "x" << Q_swa_sh << "xf32>,\n"
                  << "    %s_swa_o: tensor<" << N_swa << "x" << D << "xf32>,\n";
            if (has_swa_kv) {
                em.ss << "    %s_swa_k: tensor<" << N_swa_kv << "x" << KV_swa_sh << "xf32>,\n"
                      << "    %s_swa_v: tensor<" << N_swa_kv << "x" << KV_swa_sh << "xf32>,\n";
            }
        }
        if (has_glob) {
            em.ss << "    %s_glob_q: tensor<" << N_glob << "x" << Q_glob_sh << "xf32>,\n"
                  << "    %s_glob_o: tensor<" << N_glob << "x" << D << "xf32>,\n";
            if (has_glob_kv) {
                em.ss << "    %s_glob_k: tensor<" << N_glob_kv << "x" << KV_glob_sh << "xf32>,\n"
                      << "    %s_glob_v: tensor<" << N_glob_kv << "x" << KV_glob_sh << "xf32>,\n";
            }
        }
        em.ss << "    %s_lm_head: tensor<" << V_sh << "xf32>,\n";
        if (onchip_emb && !spec.has_tied_embeddings) {
            em.ss << "    %s_tok_emb: tensor<" << V_sh << "xf32>,\n";
        }
        if (onchip_emb && spec.has_pleias) {
            em.ss << "    %s_pl_tok_emb: tensor<" << V_sh << "xf32>,\n";
        }
    }

    // 3. Optional QKV Biases (Qwen 2 / 2.5)
    if (spec.has_qkv_bias) {
        if (has_swa) {
            em.ss << "    %b_swa_q: tensor<" << N_swa << "x" << Q_swa_sh << "xf32>,\n";
            if (has_swa_kv) {
                em.ss << "    %b_swa_k: tensor<" << N_swa_kv << "x" << KV_swa_sh << "xf32>,\n"
                      << "    %b_swa_v: tensor<" << N_swa_kv << "x" << KV_swa_sh << "xf32>,\n";
            }
        }
        if (has_glob) {
            em.ss << "    %b_glob_q: tensor<" << N_glob << "x" << Q_glob_sh << "xf32>,\n";
            if (has_glob_kv) {
                em.ss << "    %b_glob_k: tensor<" << N_glob_kv << "x" << KV_glob_sh << "xf32>,\n"
                      << "    %b_glob_v: tensor<" << N_glob_kv << "x" << KV_glob_sh << "xf32>,\n";
            }
        }
    }

    // 4. Optional Pleias per-layer injection weights (Gemma 4 E4B)
    if (spec.has_pleias) {
        em.ss << "    %w_inp_gate: tensor<" << L << "x" << P_dim << "x" << D << "xbf16>,\n"
              << "    %w_proj: tensor<" << L << "x" << D << "x" << P_dim << "xbf16>,\n"
              << "    %w_pl_mproj: tensor<" << (L * P_dim) << "x" << D << "xbf16>,\n"
              << "    %w_pl_pnorm: tensor<" << P_dim << "xf32>,\n";
    }

    // 5. F32 Norms & Layer Scales
    em.ss << "    %w_attn_norm: tensor<" << L << "x" << D << "xf32>,\n";
    if (spec.has_post_attn_norm) {
        em.ss << "    %w_post_attn_norm: tensor<" << L << "x" << D << "xf32>,\n";
    }
    em.ss << "    %w_ffn_norm: tensor<" << L << "x" << D << "xf32>,\n";
    if (spec.has_post_ffn_norm) {
        em.ss << "    %w_post_ffn_norm: tensor<" << L << "x" << D << "xf32>,\n";
    }
    if (spec.has_post_norm) {
        em.ss << "    %w_post_norm: tensor<" << L << "x" << D << "xf32>,\n";
    }
    if (spec.has_layer_scale) {
        em.ss << "    %w_layer_scale: tensor<" << L << "xf32>,\n";
    }
    if (spec.has_q_norm) {
        if (has_swa)  em.ss << "    %w_swa_q_norm: tensor<" << N_swa << "x" << HD_swa << "xf32>,\n";
        if (has_glob) em.ss << "    %w_glob_q_norm: tensor<" << N_glob << "x" << HD_glob << "xf32>,\n";
    }
    if (spec.has_k_norm) {
        if (has_swa_kv)  em.ss << "    %w_swa_k_norm: tensor<" << N_swa_kv << "x" << HD_swa << "xf32>,\n";
        if (has_glob_kv) em.ss << "    %w_glob_k_norm: tensor<" << N_glob_kv << "x" << HD_glob << "xf32>,\n";
    }
    em.ss << "    %w_out_norm: tensor<" << D << "xf32>,\n";

    // 6. Precomputed RoPE tables
    if (has_swa) {
        em.ss << "    %cos_swa: tensor<" << T << "x" << (HD_swa / 2) << "xf32>,\n"
              << "    %sin_swa: tensor<" << T << "x" << (HD_swa / 2) << "xf32>,\n";
    }
    if (has_glob) {
        em.ss << "    %cos_glob: tensor<" << T << "x" << (HD_glob / 2) << "xf32>,\n"
              << "    %sin_glob: tensor<" << T << "x" << (HD_glob / 2) << "xf32>,\n";
    }

    // 7. Mutable donated HBM3 KV Cache buffers
    int alias_slot = 1;
    std::vector<std::string> kv_ret_types;
    if (has_swa_kv) {
        const int a_ks = alias_slot++;
        const int a_vs = alias_slot++;
        em.ss << "    %kv_ks_in: " << kv_ks_t << donate_attr(a_ks) << ",\n"
              << "    %kv_vs_in: " << kv_ks_t << donate_attr(a_vs) << (has_glob_kv ? ",\n" : "\n");
        kv_ret_types.push_back(kv_ks_t);
        kv_ret_types.push_back(kv_ks_t);
    }
    if (has_glob_kv) {
        const int a_kg = alias_slot++;
        const int a_vg = alias_slot++;
        em.ss << "    %kv_kg_in: " << kv_kg_t << donate_attr(a_kg) << ",\n"
              << "    %kv_vg_in: " << kv_kg_t << donate_attr(a_vg) << "\n";
        kv_ret_types.push_back(kv_kg_t);
        kv_ret_types.push_back(kv_kg_t);
    }

    em.ss << "  ) -> (" << out0_t;
    for (const auto& kt : kv_ret_types) em.ss << ", " << kt;
    em.ss << ") {\n";

    // Common scalar constants
    const std::string c0_f32 = "%c0_f32";
    const std::string c0_i32 = "%c0_i32";
    const std::string c_ninf = "%c_ninf";
    em.ss << "    " << c0_f32 << " = stablehlo.constant dense<0.0000000e+00> : tensor<f32>\n"
          << "    " << c0_i32 << " = stablehlo.constant dense<0> : tensor<i32>\n"
          << "    " << c_ninf << " = stablehlo.constant dense<-1.0000000e+09> : tensor<f32>\n";

    // Pre-slice/gather RoPE cos/sin AND build Causal + SWA attention masks once!
    std::string cos_swa_var, sin_swa_var, cos_glob_var, sin_glob_var;
    std::string mask_t_i1;
    if (per_slot_pos) {
        // Independent per-slot positions: %pos_vec: tensor<B_cap x i32>
        const std::string t_b1_i32 = "tensor<" + std::to_string(B_cap) + "x1xi32>";
        const std::string t_bt_i32 = "tensor<" + std::to_string(B_cap) + "x" + std::to_string(T) + "xi32>";
        mask_t_i1 = "tensor<" + std::to_string(B_cap) + "x" + std::to_string(T) + "xi1>";

        em.ss << "    %pos_col = stablehlo.reshape %pos_vec : (tensor<" << B_cap << "xi32>) -> " << t_b1_i32 << "\n"
              << "    %b_iota_1d = stablehlo.iota dim = 0 : tensor<" << B_cap << "xi32>\n"
              << "    %b_iota_col = stablehlo.reshape %b_iota_1d : (tensor<" << B_cap << "xi32>) -> " << t_b1_i32 << "\n";

        if (has_swa) {
            cos_swa_var = em.emit_rope_gather_2d("%cos_swa", "%pos_col", T, HD_swa / 2, B_cap);
            sin_swa_var = em.emit_rope_gather_2d("%sin_swa", "%pos_col", T, HD_swa / 2, B_cap);
        }
        if (has_glob) {
            cos_glob_var = em.emit_rope_gather_2d("%cos_glob", "%pos_col", T, HD_glob / 2, B_cap);
            sin_glob_var = em.emit_rope_gather_2d("%sin_glob", "%pos_col", T, HD_glob / 2, B_cap);
        }

        em.ss << "    %k_iota = stablehlo.iota dim = 0 : tensor<" << T << "xi32>\n"
              << "    %k_bt = stablehlo.broadcast_in_dim %k_iota, dims = [1] : (tensor<" << T << "xi32>) -> " << t_bt_i32 << "\n"
              << "    %q_bt = stablehlo.broadcast_in_dim %pos_vec, dims = [0] : (tensor<" << B_cap << "xi32>) -> " << t_bt_i32 << "\n"
              << "    %mask_glob = stablehlo.compare LE, %k_bt, %q_bt, SIGNED : (" << t_bt_i32 << ", " << t_bt_i32 << ") -> " << mask_t_i1 << "\n";
        if (has_swa) {
            em.ss << "    %c_win = stablehlo.constant dense<" << std::max(0, spec.sliding_window - 1) << "> : " << t_bt_i32 << "\n"
                  << "    %c_sink = stablehlo.constant dense<4> : " << t_bt_i32 << "\n"
                  << "    %win_lo = stablehlo.subtract %q_bt, %c_win : " << t_bt_i32 << "\n"
                  << "    %in_win = stablehlo.compare GE, %k_bt, %win_lo, SIGNED : (" << t_bt_i32 << ", " << t_bt_i32 << ") -> " << mask_t_i1 << "\n"
                  << "    %is_sink = stablehlo.compare LT, %k_bt, %c_sink, SIGNED : (" << t_bt_i32 << ", " << t_bt_i32 << ") -> " << mask_t_i1 << "\n"
                  << "    %win_or_sink = stablehlo.or %in_win, %is_sink : " << mask_t_i1 << "\n"
                  << "    %mask_swa = stablehlo.and %mask_glob, %win_or_sink : " << mask_t_i1 << "\n";
        }
    } else {
        // Uniform scalar %pos (B=1 decode or slot-targeted S-token MXU prefill)
        if (has_swa) {
            em.ss << "    %cos_swa_s = stablehlo.dynamic_slice %cos_swa, %pos, " << c0_i32
                  << ", sizes = [" << S << ", " << (HD_swa / 2) << "] : (tensor<" << T << "x" << (HD_swa / 2)
                  << "xf32>, tensor<i32>, tensor<i32>) -> tensor<" << S << "x" << (HD_swa / 2) << "xf32>\n"
                  << "    %sin_swa_s = stablehlo.dynamic_slice %sin_swa, %pos, " << c0_i32
                  << ", sizes = [" << S << ", " << (HD_swa / 2) << "] : (tensor<" << T << "x" << (HD_swa / 2)
                  << "xf32>, tensor<i32>, tensor<i32>) -> tensor<" << S << "x" << (HD_swa / 2) << "xf32>\n";
            cos_swa_var = "%cos_swa_s";
            sin_swa_var = "%sin_swa_s";
        }
        if (has_glob) {
            em.ss << "    %cos_glob_s = stablehlo.dynamic_slice %cos_glob, %pos, " << c0_i32
                  << ", sizes = [" << S << ", " << (HD_glob / 2) << "] : (tensor<" << T << "x" << (HD_glob / 2)
                  << "xf32>, tensor<i32>, tensor<i32>) -> tensor<" << S << "x" << (HD_glob / 2) << "xf32>\n"
                  << "    %sin_glob_s = stablehlo.dynamic_slice %sin_glob, %pos, " << c0_i32
                  << ", sizes = [" << S << ", " << (HD_glob / 2) << "] : (tensor<" << T << "x" << (HD_glob / 2)
                  << "xf32>, tensor<i32>, tensor<i32>) -> tensor<" << S << "x" << (HD_glob / 2) << "xf32>\n";
            cos_glob_var = "%cos_glob_s";
            sin_glob_var = "%sin_glob_s";
        }

        const std::string t_st_i32 = "tensor<" + std::to_string(S) + "x" + std::to_string(T) + "xi32>";
        mask_t_i1 = "tensor<" + std::to_string(S) + "x" + std::to_string(T) + "xi1>";
        em.ss << "    %k_iota = stablehlo.iota dim = 0 : tensor<" << T << "xi32>\n"
              << "    %q_iota = stablehlo.iota dim = 0 : tensor<" << S << "xi32>\n"
              << "    %pos_s = stablehlo.broadcast_in_dim %pos, dims = [] : (tensor<i32>) -> tensor<" << S << "xi32>\n"
              << "    %q_pos = stablehlo.add %q_iota, %pos_s : tensor<" << S << "xi32>\n"
              << "    %k_st = stablehlo.broadcast_in_dim %k_iota, dims = [1] : (tensor<" << T << "xi32>) -> " << t_st_i32 << "\n"
              << "    %q_st = stablehlo.broadcast_in_dim %q_pos, dims = [0] : (tensor<" << S << "xi32>) -> " << t_st_i32 << "\n"
              << "    %mask_glob = stablehlo.compare LE, %k_st, %q_st, SIGNED : (" << t_st_i32 << ", " << t_st_i32 << ") -> " << mask_t_i1 << "\n";
        if (has_swa) {
            em.ss << "    %c_win = stablehlo.constant dense<" << std::max(0, spec.sliding_window - 1) << "> : " << t_st_i32 << "\n"
                  << "    %c_sink = stablehlo.constant dense<4> : " << t_st_i32 << "\n"
                  << "    %win_lo = stablehlo.subtract %q_st, %c_win : " << t_st_i32 << "\n"
                  << "    %in_win = stablehlo.compare GE, %k_st, %win_lo, SIGNED : (" << t_st_i32 << ", " << t_st_i32 << ") -> " << mask_t_i1 << "\n"
                  << "    %is_sink = stablehlo.compare LT, %k_st, %c_sink, SIGNED : (" << t_st_i32 << ", " << t_st_i32 << ") -> " << mask_t_i1 << "\n"
                  << "    %win_or_sink = stablehlo.or %in_win, %is_sink : " << mask_t_i1 << "\n"
                  << "    %mask_swa = stablehlo.and %mask_glob, %win_or_sink : " << mask_t_i1 << "\n";
        }
    }

    // Step 1: Obtain x_emb [B, S, D] and optional Pleias per-layer embeddings [B, S, L, P_dim]
    const std::string t_bsd_f32    = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(D) + "xf32>";
    const std::string t_bsd_bf     = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(D) + "xbf16>";
    const std::string t_pl_f32     = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(L) + "x" + std::to_string(P_dim) + "xf32>";
    const std::string t_pl_bf      = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(L) + "x" + std::to_string(P_dim) + "xbf16>";
    const std::string t_pl_flat    = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(L * P_dim) + "xf32>";
    const std::string t_pl_flat_bf = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(L * P_dim) + "xbf16>";

    std::string x_emb_bf_var;
    std::string x0_f32_var;
    std::string pl_tok_f32_var;

    if (onchip_emb) {
        const std::string t_bs_i32 = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "xi32>";
        const std::string t_bs_i1  = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "xi1>";
        const std::string t_idx3   = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x1xi32>";
        em.ss << "    %c_z_bs = stablehlo.constant dense<0> : " << t_bs_i32 << "\n"
              << "    %c_vmax_bs = stablehlo.constant dense<" << (V_sh - 1) << "> : " << t_bs_i32 << "\n";

        std::string rel_ids = "%tok_ids_in";
        if (R > 1) {
            em.ss << "    %rid_u32_e = stablehlo.replica_id : tensor<ui32>\n"
                  << "    %rid_i32_e = stablehlo.convert %rid_u32_e : (tensor<ui32>) -> tensor<i32>\n"
                  << "    %c_vsh_e = stablehlo.constant dense<" << V_sh << "> : tensor<i32>\n"
                  << "    %v_lo = stablehlo.multiply %rid_i32_e, %c_vsh_e : tensor<i32>\n"
                  << "    %v_lo_b = stablehlo.broadcast_in_dim %v_lo, dims = [] : (tensor<i32>) -> " << t_bs_i32 << "\n"
                  << "    %rel_ids = stablehlo.subtract %tok_ids_in, %v_lo_b : " << t_bs_i32 << "\n"
                  << "    %ge_lo = stablehlo.compare GE, %rel_ids, %c_z_bs, SIGNED : (" << t_bs_i32 << ", " << t_bs_i32 << ") -> " << t_bs_i1 << "\n"
                  << "    %le_hi = stablehlo.compare LE, %rel_ids, %c_vmax_bs, SIGNED : (" << t_bs_i32 << ", " << t_bs_i32 << ") -> " << t_bs_i1 << "\n"
                  << "    %in_shard = stablehlo.and %ge_lo, %le_hi : " << t_bs_i1 << "\n";
            rel_ids = "%rel_ids";
        }
        em.ss << "    %cl_hi = stablehlo.minimum " << rel_ids << ", %c_vmax_bs : " << t_bs_i32 << "\n"
              << "    %cl_id = stablehlo.maximum %cl_hi, %c_z_bs : " << t_bs_i32 << "\n"
              << "    %idx_3d = stablehlo.reshape %cl_id : (" << t_bs_i32 << ") -> " << t_idx3 << "\n";

        const std::string emb_tbl = spec.has_tied_embeddings ? "%w_lm_head" : "%w_tok_emb";
        const std::string emb_scl = spec.has_tied_embeddings ? "%s_lm_head" : "%s_tok_emb";
        std::string emb_loc = em.emit_table_gather_2d(emb_tbl, emb_scl, "%idx_3d", V_sh, D, B, S, i8);
        std::string emb_sc_loc = emb_loc;
        if (std::fabs(spec.emb_scale - 1.0f) > 1e-6f) {
            em.ss << "    %c_emb_sc = stablehlo.constant dense<" << fmt_f32(spec.emb_scale) << "> : " << t_bsd_f32 << "\n"
                  << "    %emb_sc_loc = stablehlo.multiply " << emb_loc << ", %c_emb_sc : " << t_bsd_f32 << "\n";
            emb_sc_loc = "%emb_sc_loc";
        }
        std::string emb_f32 = emb_sc_loc;
        if (R > 1) {
            const std::string t_bsd_i1 = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(D) + "xi1>";
            em.ss << "    %in_sh_d = stablehlo.broadcast_in_dim %in_shard, dims = [0, 1] : (" << t_bs_i1 << ") -> " << t_bsd_i1 << "\n"
                  << "    %z_bsd = stablehlo.constant dense<0.0000000e+00> : " << t_bsd_f32 << "\n"
                  << "    %emb_masked = stablehlo.select %in_sh_d, " << emb_sc_loc << ", %z_bsd : " << t_bsd_i1 << ", " << t_bsd_f32 << "\n";
            emb_f32 = em.emit_all_reduce_sum_3d("%emb_masked", B, S, D, R, "f32");
        }
        em.ss << "    %x_emb_bf = stablehlo.convert " << emb_f32 << " : (" << t_bsd_f32 << ") -> " << t_bsd_bf << "\n";
        x0_f32_var = emb_f32;
        x_emb_bf_var = "%x_emb_bf";

        if (spec.has_pleias) {
            std::string pl_loc = em.emit_table_gather_2d("%w_pl_tok_emb", "%s_pl_tok_emb", "%idx_3d", V_sh, L * P_dim, B, S, i8);
            const float tok_pl_scale = std::sqrt(static_cast<float>(P_dim));
            em.ss << "    %c_pl_tsc = stablehlo.constant dense<" << fmt_f32(tok_pl_scale) << "> : " << t_pl_flat << "\n"
                  << "    %pl_sc_loc = stablehlo.multiply " << pl_loc << ", %c_pl_tsc : " << t_pl_flat << "\n"
                  << "    %pl_loc_bf = stablehlo.convert %pl_sc_loc : (" << t_pl_flat << ") -> " << t_pl_flat_bf << "\n";
            std::string pl_flat_bf = "%pl_loc_bf";
            if (R > 1) {
                const std::string t_plf_i1 = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(L * P_dim) + "xi1>";
                em.ss << "    %in_sh_pl = stablehlo.broadcast_in_dim %in_shard, dims = [0, 1] : (" << t_bs_i1 << ") -> " << t_plf_i1 << "\n"
                      << "    %z_plbf = stablehlo.constant dense<0.0000000e+00> : " << t_pl_flat_bf << "\n"
                      << "    %pl_masked_bf = stablehlo.select %in_sh_pl, %pl_loc_bf, %z_plbf : " << t_plf_i1 << ", " << t_pl_flat_bf << "\n";
                pl_flat_bf = em.emit_all_reduce_sum_3d("%pl_masked_bf", B, S, L * P_dim, R, "bf16");
            }
            em.ss << "    %pl_tok_4d_bf = stablehlo.reshape " << pl_flat_bf << " : (" << t_pl_flat_bf << ") -> " << t_pl_bf << "\n"
                  << "    %pl_tok_f32 = stablehlo.convert %pl_tok_4d_bf : (" << t_pl_bf << ") -> " << t_pl_f32 << "\n";
            pl_tok_f32_var = "%pl_tok_f32";
        }
    } else {
        em.ss << "    %x0 = stablehlo.convert %x_emb_in : (" << t_bsd_bf << ") -> " << t_bsd_f32 << "\n";
        x0_f32_var = "%x0";
        x_emb_bf_var = "%x_emb_in";
        if (spec.has_pleias) {
            em.ss << "    %pl_tok_f32 = stablehlo.convert %pleias_tok_in : (" << t_pl_bf << ") -> " << t_pl_f32 << "\n";
            pl_tok_f32_var = "%pl_tok_f32";
        }
    }

    if (spec.has_pleias) {
        em.ss << "    %pl_proj_raw = stablehlo.dot_general " << x_emb_bf_var << ", %w_pl_mproj, contracting_dims = [2] x [1] : ("
              << t_bsd_bf << ", tensor<" << (L * P_dim) << "x" << D << "xbf16>) -> " << t_pl_flat << "\n"
              << "    %c_pl_pscale = stablehlo.constant dense<" << fmt_f32(1.0f / std::sqrt(static_cast<float>(D))) << "> : " << t_pl_flat << "\n"
              << "    %pl_proj_sc = stablehlo.multiply %pl_proj_raw, %c_pl_pscale : " << t_pl_flat << "\n"
              << "    %pl_proj_4d = stablehlo.reshape %pl_proj_sc : (" << t_pl_flat << ") -> " << t_pl_f32 << "\n";
        std::string pl_proj_norm = em.emit_rms_norm_4d("%pl_proj_4d", "%w_pl_pnorm", B, S, L, P_dim, spec.rms_norm_eps, c0_f32);
        em.ss << "    %pl_sum = stablehlo.add " << pl_proj_norm << ", " << pl_tok_f32_var << " : " << t_pl_f32 << "\n"
              << "    %c_inv_s2 = stablehlo.constant dense<" << fmt_f32(1.0f / std::sqrt(2.0f)) << "> : " << t_pl_f32 << "\n"
              << "    %pleias_all = stablehlo.multiply %pl_sum, %c_inv_s2 : " << t_pl_f32 << "\n";
    }

    std::string x_cur = x0_f32_var;
    std::string kv_ks = "%kv_ks_in";
    std::string kv_vs = "%kv_vs_in";
    std::string kv_kg = "%kv_kg_in";
    std::string kv_vg = "%kv_vg_in";

    // Step 2: Unroll all L transformer layers with exact static shapes
    for (int l = 0; l < L; ++l) {
        const bool is_swa = spec.layer_is_swa[static_cast<size_t>(l)];
        const bool has_kv = spec.layer_has_kv[static_cast<size_t>(l)];
        const int ti = spec.layer_type_idx[static_cast<size_t>(l)];
        const int kvi = has_kv ? spec.layer_kv_idx[static_cast<size_t>(l)]
                               : (is_swa ? spec.last_swa_kv_idx : spec.last_glob_kv_idx);
        const int HD = is_swa ? HD_swa : HD_glob;
        const int Q_sh = is_swa ? Q_swa_sh : Q_glob_sh;
        const int KV_sh = is_swa ? KV_swa_sh : KV_glob_sh;
        const int N_type = is_swa ? N_swa : N_glob;
        const int N_kv_type = is_swa ? N_swa_kv : N_glob_kv;
        const std::string& cos_s = is_swa ? cos_swa_var : cos_glob_var;
        const std::string& sin_s = is_swa ? sin_swa_var : sin_glob_var;
        const float attn_scale = spec.is_gemma ? 1.0f : (1.0f / std::sqrt(static_cast<float>(HD)));

        em.ss << "    // --- Layer " << l << (is_swa ? " (SWA" : " (Global")
              << (has_kv ? ", KV-owner)" : ", KV-shared)") << " ---\n";

        // 2a. Pre-Attention RMSNorm
        std::string attn_nw = em.slice_1d_from_2d("%w_attn_norm", l, L, D);
        std::string x_an_f32 = em.emit_rms_norm_3d(x_cur, attn_nw, B, S, D, spec.rms_norm_eps, c0_f32);
        std::string x_an_bf = em.tmp("anbf");
        em.ss << "    " << x_an_bf << " = stablehlo.convert " << x_an_f32 << " : (" << t_bsd_f32 << ") -> " << t_bsd_bf << "\n";

        // 2b. Q Projection (+ optional bias) + optional Q-Norm + RoPE
        std::string q_flat = em.emit_linear_3d(x_an_bf,
            is_swa ? "%w_swa_q" : "%w_glob_q",
            is_swa ? "%s_swa_q" : "%s_glob_q",
            ti, N_type, B, S, Q_sh, D, i8);
        if (spec.has_qkv_bias) {
            q_flat = em.emit_add_bias_3d(q_flat, is_swa ? "%b_swa_q" : "%b_glob_q", ti, N_type, B, S, Q_sh);
        }
        const std::string t_q3 = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(Q_sh) + "xf32>";
        const std::string t_q4 = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(H_sh) + "x" + std::to_string(HD) + "xf32>";
        std::string q_4d = em.tmp("q4");
        em.ss << "    " << q_4d << " = stablehlo.reshape " << q_flat << " : (" << t_q3 << ") -> " << t_q4 << "\n";
        std::string q_pre_rope = q_4d;
        if (spec.has_q_norm) {
            std::string q_nw = em.slice_1d_from_2d(is_swa ? "%w_swa_q_norm" : "%w_glob_q_norm", ti, N_type, HD);
            q_pre_rope = em.emit_rms_norm_4d(q_4d, q_nw, B, S, H_sh, HD, spec.rms_norm_eps, c0_f32);
        }
        std::string q_rot = em.emit_rope_4d(q_pre_rope, cos_s, sin_s, B, S, H_sh, HD, per_slot_pos);

        // 2c. K, V Projections (+ optional bias) + optional K/V-Norm + RoPE + In-place KV Cache Update
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
            if (spec.has_qkv_bias) {
                k_flat = em.emit_add_bias_3d(k_flat, is_swa ? "%b_swa_k" : "%b_glob_k", kvi, N_kv_type, B, S, KV_sh);
                v_flat = em.emit_add_bias_3d(v_flat, is_swa ? "%b_swa_v" : "%b_glob_v", kvi, N_kv_type, B, S, KV_sh);
            }
            const std::string t_kv3    = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(KV_sh) + "xf32>";
            const std::string t_kv4    = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(Hkv_sh) + "x" + std::to_string(HD) + "xf32>";
            const std::string t_kv4_bf = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(Hkv_sh) + "x" + std::to_string(HD) + "xbf16>";

            std::string k_4d = em.tmp("k4"), v_4d = em.tmp("v4");
            em.ss << "    " << k_4d << " = stablehlo.reshape " << k_flat << " : (" << t_kv3 << ") -> " << t_kv4 << "\n"
                  << "    " << v_4d << " = stablehlo.reshape " << v_flat << " : (" << t_kv3 << ") -> " << t_kv4 << "\n";
            std::string k_pre_rope = k_4d;
            if (spec.has_k_norm) {
                std::string k_nw = em.slice_1d_from_2d(is_swa ? "%w_swa_k_norm" : "%w_glob_k_norm", kvi, N_kv_type, HD);
                k_pre_rope = em.emit_rms_norm_4d(k_4d, k_nw, B, S, Hkv_sh, HD, spec.rms_norm_eps, c0_f32);
            }
            std::string v_final = v_4d;
            if (spec.has_v_norm) {
                v_final = em.emit_rms_norm_4d(v_4d, "", B, S, Hkv_sh, HD, spec.rms_norm_eps, c0_f32);
            }
            std::string k_rot = em.emit_rope_4d(k_pre_rope, cos_s, sin_s, B, S, Hkv_sh, HD, per_slot_pos);

            std::string k_bf = em.tmp("kbf"), v_bf = em.tmp("vbf");
            em.ss << "    " << k_bf << " = stablehlo.convert " << k_rot   << " : (" << t_kv4 << ") -> " << t_kv4_bf << "\n"
                  << "    " << v_bf << " = stablehlo.convert " << v_final << " : (" << t_kv4 << ") -> " << t_kv4_bf << "\n";

            std::string nk = em.tmp("kvk"), nv = em.tmp("kvv");
            if (per_slot_pos) {
                // Async per-slot [B_cap] KV update via in-place unique_indices stablehlo.scatter!
                const std::string t_upd   = "tensor<" + std::to_string(B_cap) + "x" + std::to_string(Hkv_sh) + "x" + std::to_string(HD) + "xbf16>";
                const std::string t_b1    = "tensor<" + std::to_string(B_cap) + "x1xi32>";
                const std::string t_scidx = "tensor<" + std::to_string(B_cap) + "x3xi32>";
                std::string k_upd = em.tmp("kscu"), v_upd = em.tmp("vscu"), c_kvi_col = em.tmp("kvic"), sc_idx = em.tmp("scidx");
                em.ss << "    " << k_upd << " = stablehlo.reshape " << k_bf << " : (" << t_kv4_bf << ") -> " << t_upd << "\n"
                      << "    " << v_upd << " = stablehlo.reshape " << v_bf << " : (" << t_kv4_bf << ") -> " << t_upd << "\n"
                      << "    " << c_kvi_col << " = stablehlo.constant dense<" << kvi << "> : " << t_b1 << "\n"
                      << "    " << sc_idx << " = stablehlo.concatenate " << c_kvi_col << ", %b_iota_col, %pos_col, dim = 1 : ("
                      << t_b1 << ", " << t_b1 << ", " << t_b1 << ") -> " << t_scidx << "\n"
                      << "    " << nk << " = \"stablehlo.scatter\"(" << kv_k_stack << ", " << sc_idx << ", " << k_upd << ") ({\n"
                      << "    ^bb0(%old: tensor<bf16>, %new: tensor<bf16>):\n"
                      << "      stablehlo.return %new : tensor<bf16>\n"
                      << "    }) {\n"
                      << "      scatter_dimension_numbers = #stablehlo.scatter<\n"
                      << "        update_window_dims = [1, 2],\n"
                      << "        inserted_window_dims = [0, 1, 3],\n"
                      << "        scatter_dims_to_operand_dims = [0, 1, 3],\n"
                      << "        index_vector_dim = 1\n"
                      << "      >,\n"
                      << "      indices_are_sorted = false,\n"
                      << "      unique_indices = true\n"
                      << "    } : (" << kv_stack_t << ", " << t_scidx << ", " << t_upd << ") -> " << kv_stack_t << "\n"
                      << "    " << nv << " = \"stablehlo.scatter\"(" << kv_v_stack << ", " << sc_idx << ", " << v_upd << ") ({\n"
                      << "    ^bb0(%old: tensor<bf16>, %new: tensor<bf16>):\n"
                      << "      stablehlo.return %new : tensor<bf16>\n"
                      << "    }) {\n"
                      << "      scatter_dimension_numbers = #stablehlo.scatter<\n"
                      << "        update_window_dims = [1, 2],\n"
                      << "        inserted_window_dims = [0, 1, 3],\n"
                      << "        scatter_dims_to_operand_dims = [0, 1, 3],\n"
                      << "        index_vector_dim = 1\n"
                      << "      >,\n"
                      << "      indices_are_sorted = false,\n"
                      << "      unique_indices = true\n"
                      << "    } : (" << kv_stack_t << ", " << t_scidx << ", " << t_upd << ") -> " << kv_stack_t << "\n";
            } else {
                // Single-stream (B=1) or Slot-Targeted Prefill (%slot_id): direct dynamic_update_slice!
                const std::string t_kv4t_bf = "tensor<" + std::to_string(B) + "x" + std::to_string(Hkv_sh) + "x" + std::to_string(S) + "x" + std::to_string(HD) + "xbf16>";
                const std::string t_kv5_bf  = "tensor<1x" + std::to_string(B) + "x" + std::to_string(Hkv_sh) + "x" + std::to_string(S) + "x" + std::to_string(HD) + "xbf16>";
                std::string k_tr = em.tmp("ktr"), v_tr = em.tmp("vtr"), k_5 = em.tmp("k5"), v_5 = em.tmp("v5"), c_kvi = em.tmp("kvi");
                const std::string b_idx_var = slot_targeted ? "%slot_id" : c0_i32;
                em.ss << "    " << k_tr << " = stablehlo.transpose " << k_bf << ", dims = [0, 2, 1, 3] : (" << t_kv4_bf << ") -> " << t_kv4t_bf << "\n"
                      << "    " << v_tr << " = stablehlo.transpose " << v_bf << ", dims = [0, 2, 1, 3] : (" << t_kv4_bf << ") -> " << t_kv4t_bf << "\n"
                      << "    " << k_5  << " = stablehlo.reshape " << k_tr << " : (" << t_kv4t_bf << ") -> " << t_kv5_bf << "\n"
                      << "    " << v_5  << " = stablehlo.reshape " << v_tr << " : (" << t_kv4t_bf << ") -> " << t_kv5_bf << "\n"
                      << "    " << c_kvi << " = stablehlo.constant dense<" << kvi << "> : tensor<i32>\n"
                      << "    " << nk << " = stablehlo.dynamic_update_slice " << kv_k_stack << ", " << k_5 << ", "
                      << c_kvi << ", " << b_idx_var << ", " << c0_i32 << ", %pos, " << c0_i32
                      << " : (" << kv_stack_t << ", " << t_kv5_bf << ", tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> " << kv_stack_t << "\n"
                      << "    " << nv << " = stablehlo.dynamic_update_slice " << kv_v_stack << ", " << v_5 << ", "
                      << c_kvi << ", " << b_idx_var << ", " << c0_i32 << ", %pos, " << c0_i32
                      << " : (" << kv_stack_t << ", " << t_kv5_bf << ", tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> " << kv_stack_t << "\n";
            }
            if (is_swa) { kv_ks = nk; kv_vs = nv; } else { kv_kg = nk; kv_vg = nv; }
        }

        // 2e. GQA Attention directly on zero-padded [B, Hkv_sh, T, HD] HBM3 KV Cache
        const std::string& cur_k_stack = is_swa ? kv_ks : kv_kg;
        const std::string& cur_v_stack = is_swa ? kv_vs : kv_vg;
        const std::string t_kv1_t  = "tensor<1x" + std::to_string(B) + "x" + std::to_string(Hkv_sh) + "x" + std::to_string(T) + "x" + std::to_string(HD) + "xbf16>";
        const std::string t_kv4_t  = "tensor<" + std::to_string(B) + "x" + std::to_string(Hkv_sh) + "x" + std::to_string(T) + "x" + std::to_string(HD) + "xbf16>";
        const std::string t_q5_bf  = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(Hkv_sh) + "x" + std::to_string(G_sh) + "x" + std::to_string(HD) + "xbf16>";
        const std::string t_sc5    = "tensor<" + std::to_string(B) + "x" + std::to_string(Hkv_sh) + "x" + std::to_string(S) + "x" + std::to_string(G_sh) + "x" + std::to_string(T) + "xf32>";
        const std::string t_sc5_i1 = "tensor<" + std::to_string(B) + "x" + std::to_string(Hkv_sh) + "x" + std::to_string(S) + "x" + std::to_string(G_sh) + "x" + std::to_string(T) + "xi1>";
        const std::string t_sc5_bf = "tensor<" + std::to_string(B) + "x" + std::to_string(Hkv_sh) + "x" + std::to_string(S) + "x" + std::to_string(G_sh) + "x" + std::to_string(T) + "xbf16>";
        const std::string t_sc4    = "tensor<" + std::to_string(B) + "x" + std::to_string(Hkv_sh) + "x" + std::to_string(S) + "x" + std::to_string(G_sh) + "xf32>";
        const std::string t_ctx5   = "tensor<" + std::to_string(B) + "x" + std::to_string(Hkv_sh) + "x" + std::to_string(S) + "x" + std::to_string(G_sh) + "x" + std::to_string(HD) + "xf32>";
        const std::string t_ctx5t  = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(Hkv_sh) + "x" + std::to_string(G_sh) + "x" + std::to_string(HD) + "xf32>";
        const std::string t_ao_bf  = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(Q_sh) + "xbf16>";

        std::string ks1 = em.tmp("ks1"), klay = em.tmp("klay"), vs1 = em.tmp("vs1"), vlay = em.tmp("vlay");
        if (slot_targeted) {
            std::string c_kvir = em.tmp("kvir");
            em.ss << "    " << c_kvir << " = stablehlo.constant dense<" << kvi << "> : tensor<i32>\n"
                  << "    " << ks1 << " = stablehlo.dynamic_slice " << cur_k_stack << ", " << c_kvir << ", %slot_id, "
                  << c0_i32 << ", " << c0_i32 << ", " << c0_i32 << ", sizes = [1, 1, " << Hkv_sh << ", " << T << ", " << HD
                  << "] : (" << kv_stack_t << ", tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> " << t_kv1_t << "\n"
                  << "    " << klay << " = stablehlo.reshape " << ks1 << " : (" << t_kv1_t << ") -> " << t_kv4_t << "\n"
                  << "    " << vs1 << " = stablehlo.dynamic_slice " << cur_v_stack << ", " << c_kvir << ", %slot_id, "
                  << c0_i32 << ", " << c0_i32 << ", " << c0_i32 << ", sizes = [1, 1, " << Hkv_sh << ", " << T << ", " << HD
                  << "] : (" << kv_stack_t << ", tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> " << t_kv1_t << "\n"
                  << "    " << vlay << " = stablehlo.reshape " << vs1 << " : (" << t_kv1_t << ") -> " << t_kv4_t << "\n";
        } else {
            em.ss << "    " << ks1  << " = stablehlo.slice " << cur_k_stack << " [" << kvi << ":" << (kvi + 1)
                  << ", 0:" << B << ", 0:" << Hkv_sh << ", 0:" << T << ", 0:" << HD << "] : (" << kv_stack_t << ") -> " << t_kv1_t << "\n"
                  << "    " << klay << " = stablehlo.reshape " << ks1 << " : (" << t_kv1_t << ") -> " << t_kv4_t << "\n"
                  << "    " << vs1  << " = stablehlo.slice " << cur_v_stack << " [" << kvi << ":" << (kvi + 1)
                  << ", 0:" << B << ", 0:" << Hkv_sh << ", 0:" << T << ", 0:" << HD << "] : (" << kv_stack_t << ") -> " << t_kv1_t << "\n"
                  << "    " << vlay << " = stablehlo.reshape " << vs1 << " : (" << t_kv1_t << ") -> " << t_kv4_t << "\n";
        }

        std::string qbf4 = em.tmp("qbf"), q5 = em.tmp("q5"), sc = em.tmp("sc");
        em.ss << "    " << qbf4 << " = stablehlo.convert " << q_rot << " : (" << t_q4 << ") -> tensor<" << B << "x" << S << "x" << H_sh << "x" << HD << "xbf16>\n"
              << "    " << q5   << " = stablehlo.reshape " << qbf4 << " : (tensor<" << B << "x" << S << "x" << H_sh << "x" << HD << "xbf16>) -> " << t_q5_bf << "\n"
              << "    " << sc   << " = stablehlo.dot_general " << q5 << ", " << klay
              << ", batching_dims = [0, 2] x [0, 1], contracting_dims = [4] x [3] : (" << t_q5_bf << ", " << t_kv4_t << ") -> " << t_sc5 << "\n";

        std::string sc_cur = sc;
        if (std::fabs(attn_scale - 1.0f) > 1e-6f) {
            std::string c_as = em.tmp("asc"), sc_s = em.tmp("scs");
            em.ss << "    " << c_as << " = stablehlo.constant dense<" << fmt_f32(attn_scale) << "> : " << t_sc5 << "\n"
                  << "    " << sc_s << " = stablehlo.multiply " << sc_cur << ", " << c_as << " : " << t_sc5 << "\n";
            sc_cur = sc_s;
        }
        if (spec.attn_logit_softcapping > 0.0f) {
            std::string c_ac = em.tmp("acap"), c_iac = em.tmp("iacap"), ac_d = em.tmp("acd"), ac_t = em.tmp("act"), sc_c = em.tmp("scc");
            em.ss << "    " << c_ac  << " = stablehlo.constant dense<" << fmt_f32(spec.attn_logit_softcapping) << "> : " << t_sc5 << "\n"
                  << "    " << c_iac << " = stablehlo.constant dense<" << fmt_f32(1.0f / spec.attn_logit_softcapping) << "> : " << t_sc5 << "\n"
                  << "    " << ac_d  << " = stablehlo.multiply " << sc_cur << ", " << c_iac << " : " << t_sc5 << "\n"
                  << "    " << ac_t  << " = stablehlo.tanh " << ac_d << " : " << t_sc5 << "\n"
                  << "    " << sc_c  << " = stablehlo.multiply " << ac_t << ", " << c_ac << " : " << t_sc5 << "\n";
            sc_cur = sc_c;
        }

        const char* mask_bdims = per_slot_pos ? "[0, 4]" : "[2, 4]";
        std::string mb = em.tmp("mb"), nib = em.tmp("nib"), msc = em.tmp("msc"), mx = em.tmp("mx"), mxb = em.tmp("mxb");
        std::string sh = em.tmp("sh"), ex = em.tmp("ex"), se = em.tmp("se"), seb = em.tmp("seb");
        std::string pr = em.tmp("pr"), prbf = em.tmp("prbf"), ctx = em.tmp("ctx"), ctxt = em.tmp("ctxt");
        std::string ao_f32 = em.tmp("aof"), ao_bf = em.tmp("aob");

        em.ss << "    " << mb   << " = stablehlo.broadcast_in_dim " << (is_swa ? "%mask_swa" : "%mask_glob")
              << ", dims = " << mask_bdims << " : (" << mask_t_i1 << ") -> " << t_sc5_i1 << "\n"
              << "    " << nib  << " = stablehlo.broadcast_in_dim " << c_ninf << ", dims = [] : (tensor<f32>) -> " << t_sc5 << "\n"
              << "    " << msc  << " = stablehlo.select " << mb << ", " << sc_cur << ", " << nib << " : " << t_sc5_i1 << ", " << t_sc5 << "\n"
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
              << ", batching_dims = [0, 1] x [0, 1], contracting_dims = [4] x [2] : (" << t_sc5_bf << ", " << t_kv4_t << ") -> " << t_ctx5 << "\n"
              << "    " << ctxt << " = stablehlo.transpose " << ctx << ", dims = [0, 2, 1, 3, 4] : (" << t_ctx5 << ") -> " << t_ctx5t << "\n"
              << "    " << ao_f32 << " = stablehlo.reshape " << ctxt << " : (" << t_ctx5t << ") -> " << t_q3 << "\n"
              << "    " << ao_bf  << " = stablehlo.convert " << ao_f32 << " : (" << t_q3 << ") -> " << t_ao_bf << "\n";

        // 2f. Attention Output Projection + Optical ICI AllReduce + Optional Post-Attn Norm + Residual 1
        std::string attn_out_loc = em.emit_linear_3d(ao_bf,
            is_swa ? "%w_swa_o" : "%w_glob_o",
            is_swa ? "%s_swa_o" : "%s_glob_o",
            ti, N_type, B, S, D, Q_sh, i8);
        std::string attn_out = em.emit_all_reduce_sum_3d(attn_out_loc, B, S, D, R);
        std::string attn_final = attn_out;
        if (spec.has_post_attn_norm) {
            std::string pa_nw = em.slice_1d_from_2d("%w_post_attn_norm", l, L, D);
            attn_final = em.emit_rms_norm_3d(attn_out, pa_nw, B, S, D, spec.rms_norm_eps, c0_f32);
        }
        std::string x_res1 = em.tmp("res1");
        em.ss << "    " << x_res1 << " = stablehlo.add " << x_cur << ", " << attn_final << " : " << t_bsd_f32 << "\n";

        // 2g-2i. Pre-FFN RMSNorm + Sharded GeGLU/SwiGLU FFN + Optical ICI AllReduce + Optional Post-FFN Norm + Residual 2
        std::string fn_nw = em.slice_1d_from_2d("%w_ffn_norm", l, L, D);
        std::string x_fn_f32 = em.emit_rms_norm_3d(x_res1, fn_nw, B, S, D, spec.rms_norm_eps, c0_f32);
        std::string x_fn_bf = em.tmp("fnbf");
        em.ss << "    " << x_fn_bf << " = stablehlo.convert " << x_fn_f32 << " : (" << t_bsd_f32 << ") -> " << t_bsd_bf << "\n";

        std::string f_gate = em.emit_linear_3d(x_fn_bf, "%w_ffn_gate", "%s_ffn_gate", l, L, B, S, F_sh, D, i8);
        std::string f_up   = em.emit_linear_3d(x_fn_bf, "%w_ffn_up",   "%s_ffn_up",   l, L, B, S, F_sh, D, i8);
        std::string f_gate_act = (spec.ffn_activation == FfnActivation::GELU)
            ? em.emit_gelu_3d(f_gate, B, S, F_sh)
            : em.emit_silu_3d(f_gate, B, S, F_sh);
        const std::string t_ff_f32 = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(F_sh) + "xf32>";
        const std::string t_ff_bf  = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(F_sh) + "xbf16>";
        std::string f_act = em.tmp("fact"), f_act_bf = em.tmp("fabf");
        em.ss << "    " << f_act    << " = stablehlo.multiply " << f_gate_act << ", " << f_up << " : " << t_ff_f32 << "\n"
              << "    " << f_act_bf << " = stablehlo.convert " << f_act << " : (" << t_ff_f32 << ") -> " << t_ff_bf << "\n";
        std::string f_down_loc = em.emit_linear_3d(f_act_bf, "%w_ffn_down", "%s_ffn_down", l, L, B, S, D, F_sh, i8);
        std::string f_down = em.emit_all_reduce_sum_3d(f_down_loc, B, S, D, R);
        std::string f_final = f_down;
        if (spec.has_post_ffn_norm) {
            std::string pf_nw = em.slice_1d_from_2d("%w_post_ffn_norm", l, L, D);
            f_final = em.emit_rms_norm_3d(f_down, pf_nw, B, S, D, spec.rms_norm_eps, c0_f32);
        }
        std::string x_after_ffn = em.tmp("res2");
        em.ss << "    " << x_after_ffn << " = stablehlo.add " << x_res1 << ", " << f_final << " : " << t_bsd_f32 << "\n";

        // Optional Pleias Per-Layer Token Injection (Gemma 4 E4B)
        std::string x_after_pl = x_after_ffn;
        if (spec.has_pleias) {
            std::string x_r2_bf = em.tmp("r2bf");
            em.ss << "    " << x_r2_bf << " = stablehlo.convert " << x_after_ffn << " : (" << t_bsd_f32 << ") -> " << t_bsd_bf << "\n";
            std::string pg = em.emit_linear_3d(x_r2_bf, "%w_inp_gate", "", l, L, B, S, P_dim, D, false);
            std::string pg_gelu = em.emit_gelu_3d(pg, B, S, P_dim);
            const std::string t_p4    = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x1x" + std::to_string(P_dim) + "xf32>";
            const std::string t_p3    = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(P_dim) + "xf32>";
            const std::string t_p3_bf = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(P_dim) + "xbf16>";
            std::string pl_sl = em.tmp("plsl"), pl_3 = em.tmp("pl3"), pg_mul = em.tmp("pgm"), pg_bf = em.tmp("pgbf");
            em.ss << "    " << pl_sl << " = stablehlo.slice %pleias_all [0:" << B << ", 0:" << S << ", " << l << ":" << (l + 1)
                  << ", 0:" << P_dim << "] : (" << t_pl_f32 << ") -> " << t_p4 << "\n"
                  << "    " << pl_3  << " = stablehlo.reshape " << pl_sl << " : (" << t_p4 << ") -> " << t_p3 << "\n"
                  << "    " << pg_mul << " = stablehlo.multiply " << pg_gelu << ", " << pl_3 << " : " << t_p3 << "\n"
                  << "    " << pg_bf  << " = stablehlo.convert " << pg_mul << " : (" << t_p3 << ") -> " << t_p3_bf << "\n";
            std::string pl_out = em.emit_linear_3d(pg_bf, "%w_proj", "", l, L, B, S, D, P_dim, false);
            std::string pl_final = pl_out;
            if (spec.has_post_norm) {
                std::string pn_nw = em.slice_1d_from_2d("%w_post_norm", l, L, D);
                pl_final = em.emit_rms_norm_3d(pl_out, pn_nw, B, S, D, spec.rms_norm_eps, c0_f32);
            }
            x_after_pl = em.tmp("res3");
            em.ss << "    " << x_after_pl << " = stablehlo.add " << x_after_ffn << ", " << pl_final << " : " << t_bsd_f32 << "\n";
        }

        // Optional Layer Output Scale
        if (spec.has_layer_scale) {
            std::string ls_1 = em.tmp("ls1"), ls_0 = em.tmp("ls0"), ls_b = em.tmp("lsb"), x_next = em.tmp("xl");
            em.ss << "    " << ls_1 << " = stablehlo.slice %w_layer_scale [" << l << ":" << (l + 1)
                  << "] : (tensor<" << L << "xf32>) -> tensor<1xf32>\n"
                  << "    " << ls_0 << " = stablehlo.reshape " << ls_1 << " : (tensor<1xf32>) -> tensor<f32>\n"
                  << "    " << ls_b << " = stablehlo.broadcast_in_dim " << ls_0 << ", dims = [] : (tensor<f32>) -> " << t_bsd_f32 << "\n"
                  << "    " << x_next << " = stablehlo.multiply " << x_after_pl << ", " << ls_b << " : " << t_bsd_f32 << "\n";
            x_cur = x_next;
        } else {
            x_cur = x_after_pl;
        }
    }

    // Step 3: Slice last token [B, 1, D] -> Final Output RMSNorm -> Sharded LM Head -> Optional Softcapping
    const std::string t_b1d_f32  = "tensor<" + std::to_string(B) + "x1x" + std::to_string(D) + "xf32>";
    const std::string t_bd_f32   = "tensor<" + std::to_string(B) + "x" + std::to_string(D) + "xf32>";
    const std::string t_bd_bf    = "tensor<" + std::to_string(B) + "x" + std::to_string(D) + "xbf16>";
    const std::string t_bvsh_f32 = "tensor<" + std::to_string(B) + "x" + std::to_string(V_sh) + "xf32>";

    std::string x_last = x_cur;
    if (S > 1) {
        x_last = em.tmp("xlast");
        em.ss << "    " << x_last << " = stablehlo.slice " << x_cur << " [0:" << B << ", " << (S - 1) << ":" << S
              << ", 0:" << D << "] : (" << t_bsd_f32 << ") -> " << t_b1d_f32 << "\n";
    }
    std::string x_final_n = em.emit_rms_norm_3d(x_last, "%w_out_norm", B, 1, D, spec.rms_norm_eps, c0_f32);
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

    std::string logits_loc = logits_sc;
    if (spec.final_logit_softcapping > 0.0f) {
        em.ss << "    %c_cap = stablehlo.constant dense<" << fmt_f32(spec.final_logit_softcapping) << "> : " << t_bvsh_f32 << "\n"
              << "    %c_icap = stablehlo.constant dense<" << fmt_f32(1.0f / spec.final_logit_softcapping) << "> : " << t_bvsh_f32 << "\n"
              << "    %lc_div = stablehlo.multiply " << logits_sc << ", %c_icap : " << t_bvsh_f32 << "\n"
              << "    %lc_th = stablehlo.tanh %lc_div : " << t_bvsh_f32 << "\n"
              << "    %logits_cap = stablehlo.multiply %lc_th, %c_cap : " << t_bvsh_f32 << "\n";
        logits_loc = "%logits_cap";
    }

    // Step 4: Output either full [B, V] F32 logits or ultra-fast on-chip Argmax [B] i32
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

    em.ss << "    return " << final_out0;
    if (has_swa_kv)  em.ss << ", " << kv_ks << ", " << kv_vs;
    if (has_glob_kv) em.ss << ", " << kv_kg << ", " << kv_vg;
    em.ss << " : " << out0_t;
    for (const auto& kt : kv_ret_types) em.ss << ", " << kt;
    em.ss << "\n"
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
        for (auto& kv : slot_prefill_execs_) {
            backend_->destroy_executable(kv.second);
            kv.second = nullptr;
        }
    }
    replica_weight_bufs_.clear();
    slot_prefill_execs_.clear();
    decode_exec_ = nullptr;
    backend_ = nullptr;
    host_engine_ = nullptr;
    initialized_ = false;
}

bool HavenTpuGemma4Engine::build_and_upload_weights(const HavenEngine& host_engine) {
    const int L = spec_.num_layers;
    const int D = spec_.embedding_dim;
    const int F = spec_.hidden_dim;
    const int V = spec_.vocab_size;
    const int H = spec_.num_heads;
    const int Hkv = spec_.num_kv_heads;
    const int R = cfg_.num_replicas;
    const bool i8 = cfg_.use_int8_weights;
    const int T = cfg_.max_seq_len;

    const int V_sh = V / R;
    const int H_sh = H / R;
    const int Hkv_sh = compute_hkv_shard(Hkv, R);
    const int gqa_ratio = std::max(1, H / Hkv);
    const int HD_swa = spec_.head_dim_swa;
    const int HD_glob = spec_.head_dim_glob;
    const int P_dim = spec_.pleias_dim;

    std::vector<int> all_layers(static_cast<size_t>(L));
    std::vector<int> swa_layers, glob_layers, swa_kv_layers, glob_kv_layers;
    for (int l = 0; l < L; ++l) {
        all_layers[static_cast<size_t>(l)] = l;
        if (spec_.layer_is_swa[static_cast<size_t>(l)]) {
            swa_layers.push_back(l);
            if (spec_.layer_has_kv[static_cast<size_t>(l)]) swa_kv_layers.push_back(l);
        } else {
            glob_layers.push_back(l);
            if (spec_.layer_has_kv[static_cast<size_t>(l)]) glob_kv_layers.push_back(l);
        }
    }
    const int N_swa = static_cast<int>(swa_layers.size());
    const int N_glob = static_cast<int>(glob_layers.size());
    const int N_swa_kv = static_cast<int>(swa_kv_layers.size());
    const int N_glob_kv = static_cast<int>(glob_kv_layers.size());

    replica_weight_bufs_.assign(static_cast<size_t>(R), {});
    size_t total_bytes = 0;

    std::vector<std::vector<PJRT_Buffer*>> replica_scale_bufs(static_cast<size_t>(R));

    auto upload_sharded_stack = [&](const std::vector<int>& layer_list,
                                    const std::function<const TensorDesc*(int)>& getter,
                                    int full_rows, int full_cols,
                                    bool shard_rows, bool shard_cols,
                                    bool is_kv_head_shard) -> bool {
        const int cnt = static_cast<int>(layer_list.size());
        if (cnt <= 0) return true;
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
            const int dev = replica_dev_indices_[static_cast<size_t>(rep)];
            PJRT_Buffer* wb = nullptr;
            if (!i8) {
                wb = backend_->upload_bf16_nd(bf16_shards[static_cast<size_t>(rep)].data(), {cnt, r_sh, c_sh}, dev);
                total_bytes += shard_elems * sizeof(uint16_t);
            } else {
                wb = backend_->upload_s8_nd(s8_shards[static_cast<size_t>(rep)].data(), {cnt, r_sh, c_sh}, dev);
                PJRT_Buffer* sb = backend_->upload_f32_nd(sc_shards[static_cast<size_t>(rep)].data(), {cnt, r_sh}, dev);
                if (!sb) { backend_->destroy_buffer(wb); return false; }
                replica_scale_bufs[static_cast<size_t>(rep)].push_back(sb);
                total_bytes += shard_elems + static_cast<size_t>(cnt) * static_cast<size_t>(r_sh) * sizeof(float);
            }
            if (!wb) return false;
            replica_weight_bufs_[static_cast<size_t>(rep)].push_back(wb);
        }
        return true;
    };

    auto upload_sharded_vocab_table = [&](const TensorDesc* td, int cols) -> bool {
        const size_t shard_elems = static_cast<size_t>(V_sh) * static_cast<size_t>(cols);
        for (int rep = 0; rep < R; ++rep) {
            const int dev = replica_dev_indices_[static_cast<size_t>(rep)];
            const int row_base = rep * V_sh;
            PJRT_Buffer* wb = nullptr;
            if (!i8) {
                std::vector<uint16_t> bf(shard_elems, 0);
                if (td && td->data) {
                    #pragma omp parallel
                    {
                        std::vector<float> row_f32(static_cast<size_t>(cols));
                        #pragma omp for schedule(static)
                        for (int r = 0; r < V_sh; ++r) {
                            Avx2Math::dequantize_row(row_f32.data(), *td, row_base + r, cols);
                            HavenTpuPjrtBackend::convert_f32_to_bf16(
                                row_f32.data(),
                                bf.data() + static_cast<size_t>(r) * static_cast<size_t>(cols),
                                static_cast<size_t>(cols));
                        }
                    }
                }
                wb = backend_->upload_bf16_nd(bf.data(), {V_sh, cols}, dev);
                total_bytes += shard_elems * sizeof(uint16_t);
            } else {
                std::vector<int8_t> sq(shard_elems, 0);
                std::vector<float> sc(static_cast<size_t>(V_sh), 1.0f);
                if (td && td->data) {
                    #pragma omp parallel
                    {
                        std::vector<float> row_f32(static_cast<size_t>(cols));
                        #pragma omp for schedule(static)
                        for (int r = 0; r < V_sh; ++r) {
                            Avx2Math::dequantize_row(row_f32.data(), *td, row_base + r, cols);
                            float max_abs = 0.0f;
                            for (int c = 0; c < cols; ++c) {
                                float a = std::fabs(row_f32[static_cast<size_t>(c)]);
                                if (a > max_abs) max_abs = a;
                            }
                            const float scale = (max_abs > 1e-12f) ? (max_abs / 127.0f) : 1.0f;
                            const float inv_scale = (max_abs > 1e-12f) ? (127.0f / max_abs) : 0.0f;
                            sc[static_cast<size_t>(r)] = scale;
                            int8_t* dst_r = sq.data() + static_cast<size_t>(r) * static_cast<size_t>(cols);
                            for (int c = 0; c < cols; ++c) {
                                int q = static_cast<int>(std::lrintf(row_f32[static_cast<size_t>(c)] * inv_scale));
                                dst_r[c] = static_cast<int8_t>(std::clamp(q, -127, 127));
                            }
                        }
                    }
                }
                wb = backend_->upload_s8_nd(sq.data(), {V_sh, cols}, dev);
                PJRT_Buffer* sb = backend_->upload_f32_nd(sc.data(), {V_sh}, dev);
                if (!sb) { backend_->destroy_buffer(wb); return false; }
                replica_scale_bufs[static_cast<size_t>(rep)].push_back(sb);
                total_bytes += shard_elems + static_cast<size_t>(V_sh) * sizeof(float);
            }
            if (!wb) return false;
            replica_weight_bufs_[static_cast<size_t>(rep)].push_back(wb);
        }
        return true;
    };

    // 1. Stacked layer projection matrices (exact MLIR parameter order)
    if (!upload_sharded_stack(all_layers, [&](int l){ return host_engine.get_layer_tensors(l).ffn_gate; }, F, D, true, false, false)) return false;
    if (!upload_sharded_stack(all_layers, [&](int l){ return host_engine.get_layer_tensors(l).ffn_up; },   F, D, true, false, false)) return false;
    if (!upload_sharded_stack(all_layers, [&](int l){ return host_engine.get_layer_tensors(l).ffn_down; }, D, F, false, true, false)) return false;
    if (N_swa > 0) {
        if (!upload_sharded_stack(swa_layers,    [&](int l){ return host_engine.get_layer_tensors(l).attn_q; },      H * HD_swa, D, true, false, false)) return false;
        if (!upload_sharded_stack(swa_layers,    [&](int l){ return host_engine.get_layer_tensors(l).attn_output; }, D, H * HD_swa, false, true, false)) return false;
        if (N_swa_kv > 0) {
            if (!upload_sharded_stack(swa_kv_layers, [&](int l){ return host_engine.get_layer_tensors(l).attn_k; }, Hkv * HD_swa, D, (R > 1), false, true)) return false;
            if (!upload_sharded_stack(swa_kv_layers, [&](int l){ return host_engine.get_layer_tensors(l).attn_v; }, Hkv * HD_swa, D, (R > 1), false, true)) return false;
        }
    }
    if (N_glob > 0) {
        if (!upload_sharded_stack(glob_layers,    [&](int l){ return host_engine.get_layer_tensors(l).attn_q; },      H * HD_glob, D, true, false, false)) return false;
        if (!upload_sharded_stack(glob_layers,    [&](int l){ return host_engine.get_layer_tensors(l).attn_output; }, D, H * HD_glob, false, true, false)) return false;
        if (N_glob_kv > 0) {
            if (!upload_sharded_stack(glob_kv_layers, [&](int l){ return host_engine.get_layer_tensors(l).attn_k; }, Hkv * HD_glob, D, (R > 1), false, true)) return false;
            if (!upload_sharded_stack(glob_kv_layers, [&](int l){ return host_engine.get_layer_tensors(l).attn_v; }, Hkv * HD_glob, D, (R > 1), false, true)) return false;
        }
    }

    // 2. LM Head + optional untied Token Embeddings + optional Pleias per-layer token embeddings
    const TensorDesc* lm_td = host_engine.get_output_tensor();
    if (!lm_td) lm_td = host_engine.get_token_embd_tensor();
    if (!upload_sharded_vocab_table(lm_td, D)) return false;
    if (cfg_.use_onchip_embeddings && !spec_.has_tied_embeddings) {
        if (!upload_sharded_vocab_table(host_engine.get_token_embd_tensor(), D)) return false;
    }
    if (cfg_.use_onchip_embeddings && spec_.has_pleias) {
        if (!upload_sharded_vocab_table(host_engine.get_per_layer_token_embd_tensor(), L * P_dim)) return false;
    }

    // Append INT8 per-row scales immediately after projection weights (matching MLIR signature)
    if (i8) {
        for (int rep = 0; rep < R; ++rep) {
            for (auto* sb : replica_scale_bufs[static_cast<size_t>(rep)]) {
                replica_weight_bufs_[static_cast<size_t>(rep)].push_back(sb);
            }
        }
    }

    // 3. Optional QKV Biases (sharded along head dim matching Q/K/V)
    if (spec_.has_qkv_bias) {
        auto upload_sharded_bias = [&](const std::vector<int>& layer_list,
                                       const std::function<const TensorDesc*(int)>& getter,
                                       int full_dim, bool is_kv_head_shard) -> bool {
            const int cnt = static_cast<int>(layer_list.size());
            if (cnt <= 0) return true;
            const int d_sh = is_kv_head_shard ? (full_dim / Hkv * Hkv_sh) : (full_dim / R);
            std::vector<std::vector<float>> rep_data(static_cast<size_t>(R),
                std::vector<float>(static_cast<size_t>(cnt) * static_cast<size_t>(d_sh), 0.0f));
            for (int idx = 0; idx < cnt; ++idx) {
                auto full_b = extract_f32_1d(getter(layer_list[static_cast<size_t>(idx)]), full_dim, 0.0f);
                for (int rep = 0; rep < R; ++rep) {
                    int off = is_kv_head_shard ? (((rep * H_sh) / gqa_ratio) * (full_dim / Hkv)) : (rep * d_sh);
                    std::memcpy(rep_data[static_cast<size_t>(rep)].data() + static_cast<size_t>(idx) * static_cast<size_t>(d_sh),
                                full_b.data() + off, static_cast<size_t>(d_sh) * sizeof(float));
                }
            }
            for (int rep = 0; rep < R; ++rep) {
                const int dev = replica_dev_indices_[static_cast<size_t>(rep)];
                PJRT_Buffer* b = backend_->upload_f32_nd(rep_data[static_cast<size_t>(rep)].data(), {cnt, d_sh}, dev);
                if (!b) return false;
                replica_weight_bufs_[static_cast<size_t>(rep)].push_back(b);
                total_bytes += rep_data[static_cast<size_t>(rep)].size() * sizeof(float);
            }
            return true;
        };
        if (N_swa > 0) {
            if (!upload_sharded_bias(swa_layers, [&](int l){ return host_engine.get_layer_tensors(l).attn_q_bias; }, H * HD_swa, false)) return false;
            if (N_swa_kv > 0) {
                if (!upload_sharded_bias(swa_kv_layers, [&](int l){ return host_engine.get_layer_tensors(l).attn_k_bias; }, Hkv * HD_swa, true)) return false;
                if (!upload_sharded_bias(swa_kv_layers, [&](int l){ return host_engine.get_layer_tensors(l).attn_v_bias; }, Hkv * HD_swa, true)) return false;
            }
        }
        if (N_glob > 0) {
            if (!upload_sharded_bias(glob_layers, [&](int l){ return host_engine.get_layer_tensors(l).attn_q_bias; }, H * HD_glob, false)) return false;
            if (N_glob_kv > 0) {
                if (!upload_sharded_bias(glob_kv_layers, [&](int l){ return host_engine.get_layer_tensors(l).attn_k_bias; }, Hkv * HD_glob, true)) return false;
                if (!upload_sharded_bias(glob_kv_layers, [&](int l){ return host_engine.get_layer_tensors(l).attn_v_bias; }, Hkv * HD_glob, true)) return false;
            }
        }
    }

    auto upload_rep_f32 = [&](const std::vector<float>& data, const std::vector<int64_t>& dims) -> bool {
        for (int rep = 0; rep < R; ++rep) {
            const int dev = replica_dev_indices_[static_cast<size_t>(rep)];
            PJRT_Buffer* b = backend_->upload_f32_nd(data.data(), dims, dev);
            if (!b) return false;
            replica_weight_bufs_[static_cast<size_t>(rep)].push_back(b);
            total_bytes += data.size() * sizeof(float);
        }
        return true;
    };

    // 4. Optional Pleias weights (Gemma 4 E4B)
    if (spec_.has_pleias) {
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
                const int dev = replica_dev_indices_[static_cast<size_t>(rep)];
                PJRT_Buffer* b = backend_->upload_bf16_nd(bf.data(), {cnt, rows, cols}, dev);
                if (!b) return false;
                replica_weight_bufs_[static_cast<size_t>(rep)].push_back(b);
                total_bytes += bf.size() * sizeof(uint16_t);
            }
            return true;
        };
        if (!upload_replicated_bf16_stack(all_layers, [&](int l){ return host_engine.get_layer_tensors(l).inp_gate; }, P_dim, D)) return false;
        if (!upload_replicated_bf16_stack(all_layers, [&](int l){ return host_engine.get_layer_tensors(l).proj; },     D, P_dim)) return false;

        std::vector<float> mp_f32;
        dequantize_full_matrix_f32(host_engine.get_per_layer_model_proj_tensor(), L * P_dim, D, mp_f32);
        std::vector<uint16_t> mp_bf(mp_f32.size());
        HavenTpuPjrtBackend::convert_f32_to_bf16(mp_f32.data(), mp_bf.data(), mp_f32.size());
        for (int rep = 0; rep < R; ++rep) {
            const int dev = replica_dev_indices_[static_cast<size_t>(rep)];
            PJRT_Buffer* b = backend_->upload_bf16_nd(mp_bf.data(), {L * P_dim, D}, dev);
            if (!b) return false;
            replica_weight_bufs_[static_cast<size_t>(rep)].push_back(b);
            total_bytes += mp_bf.size() * sizeof(uint16_t);
        }
        if (!upload_rep_f32(extract_f32_1d(host_engine.get_per_layer_proj_norm_tensor(), P_dim, 1.0f), {P_dim})) return false;
    }

    // 5. F32 Norm stacks & layer scales (replicated)
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

    if (!upload_rep_f32(stack_norms(all_layers, [&](int l){ return host_engine.get_layer_tensors(l).attn_norm; }, D), {L, D})) return false;
    if (spec_.has_post_attn_norm) {
        if (!upload_rep_f32(stack_norms(all_layers, [&](int l){ return host_engine.get_layer_tensors(l).post_attention_norm; }, D), {L, D})) return false;
    }
    if (!upload_rep_f32(stack_norms(all_layers, [&](int l){ return host_engine.get_layer_tensors(l).ffn_norm; }, D), {L, D})) return false;
    if (spec_.has_post_ffn_norm) {
        if (!upload_rep_f32(stack_norms(all_layers, [&](int l){ return host_engine.get_layer_tensors(l).post_ffw_norm; }, D), {L, D})) return false;
    }
    if (spec_.has_post_norm) {
        if (!upload_rep_f32(stack_norms(all_layers, [&](int l){ return host_engine.get_layer_tensors(l).post_norm; }, D), {L, D})) return false;
    }
    if (spec_.has_layer_scale) {
        std::vector<float> lscales(static_cast<size_t>(L), 1.0f);
        for (int l = 0; l < L; ++l) {
            auto v = extract_f32_1d(host_engine.get_layer_tensors(l).layer_output_scale, 1, 1.0f);
            lscales[static_cast<size_t>(l)] = v[0];
        }
        if (!upload_rep_f32(lscales, {L})) return false;
    }
    if (spec_.has_q_norm) {
        if (N_swa > 0 && !upload_rep_f32(stack_norms(swa_layers, [&](int l){ return host_engine.get_layer_tensors(l).attn_q_norm; }, HD_swa), {N_swa, HD_swa})) return false;
        if (N_glob > 0 && !upload_rep_f32(stack_norms(glob_layers, [&](int l){ return host_engine.get_layer_tensors(l).attn_q_norm; }, HD_glob), {N_glob, HD_glob})) return false;
    }
    if (spec_.has_k_norm) {
        if (N_swa_kv > 0 && !upload_rep_f32(stack_norms(swa_kv_layers, [&](int l){ return host_engine.get_layer_tensors(l).attn_k_norm; }, HD_swa), {N_swa_kv, HD_swa})) return false;
        if (N_glob_kv > 0 && !upload_rep_f32(stack_norms(glob_kv_layers, [&](int l){ return host_engine.get_layer_tensors(l).attn_k_norm; }, HD_glob), {N_glob_kv, HD_glob})) return false;
    }
    if (!upload_rep_f32(extract_f32_1d(host_engine.get_output_norm_tensor(), D, 1.0f), {D})) return false;

    // 6. Precompute exact RoPE cos/sin tables matching Avx2Math::apply_rope bit-for-bit!
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
        (rf_td->size_bytes == 0 || rf_td->size_bytes >= static_cast<size_t>(HD_glob / 2) * sizeof(float)))
        ? reinterpret_cast<const float*>(rf_td->data) : nullptr;

    if (N_swa > 0) {
        std::vector<float> cos_swa, sin_swa;
        build_rope_tables(HD_swa, model_cfg_.rope_freq_base_swa, nullptr, cos_swa, sin_swa);
        if (!upload_rep_f32(cos_swa, {T, HD_swa / 2})) return false;
        if (!upload_rep_f32(sin_swa, {T, HD_swa / 2})) return false;
    }
    if (N_glob > 0) {
        std::vector<float> cos_glob, sin_glob;
        build_rope_tables(HD_glob, model_cfg_.rope_freq_base, rf_ptr, cos_glob, sin_glob);
        if (!upload_rep_f32(cos_glob, {T, HD_glob / 2})) return false;
        if (!upload_rep_f32(sin_glob, {T, HD_glob / 2})) return false;
    }

    metrics_.total_hbm_weight_bytes = total_bytes;
    return true;
}

bool HavenTpuGemma4Engine::allocate_zero_kv_cache() {
    free_kv_cache();
    const int R = cfg_.num_replicas;
    const int B = cfg_.batch_size;
    const int T = cfg_.max_seq_len;
    const int Hkv_sh = compute_hkv_shard(spec_.num_kv_heads, R);

    const int N_swa_kv = spec_.num_swa_kv_layers;
    const int N_glob_kv = spec_.num_glob_kv_layers;

    const std::vector<int64_t> swa_dims  = {N_swa_kv,  B, Hkv_sh, T, spec_.head_dim_swa};
    const std::vector<int64_t> glob_dims = {N_glob_kv, B, Hkv_sh, T, spec_.head_dim_glob};
    const size_t swa_elems  = (N_swa_kv > 0)  ? (static_cast<size_t>(N_swa_kv)  * B * Hkv_sh * T * static_cast<size_t>(spec_.head_dim_swa))  : 0;
    const size_t glob_elems = (N_glob_kv > 0) ? (static_cast<size_t>(N_glob_kv) * B * Hkv_sh * T * static_cast<size_t>(spec_.head_dim_glob)) : 0;

    std::vector<uint16_t> z_swa(swa_elems, 0);
    std::vector<uint16_t> z_glob(glob_elems, 0);

    replica_kv_bufs_.assign(static_cast<size_t>(R), {});
    size_t kv_bytes = 0;
    for (int rep = 0; rep < R; ++rep) {
        const int dev = replica_dev_indices_[static_cast<size_t>(rep)];
        std::vector<PJRT_Buffer*> bufs;
        if (N_swa_kv > 0) {
            PJRT_Buffer* ks = backend_->upload_bf16_nd(z_swa.data(), swa_dims, dev);
            PJRT_Buffer* vs = backend_->upload_bf16_nd(z_swa.data(), swa_dims, dev);
            if (!ks || !vs) {
                backend_->destroy_buffer(ks); backend_->destroy_buffer(vs);
                free_kv_cache();
                return false;
            }
            bufs.push_back(ks);
            bufs.push_back(vs);
            kv_bytes += 2 * swa_elems * sizeof(uint16_t);
        }
        if (N_glob_kv > 0) {
            PJRT_Buffer* kg = backend_->upload_bf16_nd(z_glob.data(), glob_dims, dev);
            PJRT_Buffer* vg = backend_->upload_bf16_nd(z_glob.data(), glob_dims, dev);
            if (!kg || !vg) {
                for (auto* p : bufs) backend_->destroy_buffer(p);
                backend_->destroy_buffer(kg); backend_->destroy_buffer(vg);
                free_kv_cache();
                return false;
            }
            bufs.push_back(kg);
            bufs.push_back(vg);
            kv_bytes += 2 * glob_elems * sizeof(uint16_t);
        }
        replica_kv_bufs_[static_cast<size_t>(rep)] = std::move(bufs);
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
    std::string support_err;
    if (!is_model_supported(host_engine, support_err)) {
        last_error_ = "Unsupported model for TPU on-chip engine: " + support_err;
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

    const auto& mc = host_engine.get_config();
    TpuModelSpec spec = TpuModelSpec::from_model_config(mc, &host_engine);
    if ((spec.hidden_dim % cfg.num_replicas) != 0 ||
        (spec.num_heads % cfg.num_replicas) != 0 ||
        (spec.vocab_size % cfg.num_replicas) != 0) {
        last_error_ = "Model dims (hidden_dim/num_heads/vocab_size) must be divisible by num_replicas=" +
                      std::to_string(cfg.num_replicas) + ".";
        return false;
    }

    backend_ = backend;
    host_engine_ = &host_engine;
    model_cfg_ = mc;
    spec_ = std::move(spec);
    cfg_ = cfg;
    metrics_ = {};

    // 1. Compile decode graph (S = 1, B = batch_size) FIRST to discover XLA torus replica-to-device mapping
    auto dc0 = std::chrono::high_resolution_clock::now();
    std::string decode_mlir = emit_stablehlo_mlir(spec_, cfg_, 1, false);
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

    if (cfg_.num_replicas == 1) {
        replica_dev_indices_ = {0};
    } else {
        replica_dev_indices_ = backend_->get_executable_device_indices(decode_exec_);
        if (static_cast<int>(replica_dev_indices_.size()) != cfg_.num_replicas) {
            replica_dev_indices_.resize(static_cast<size_t>(cfg_.num_replicas));
            for (int r = 0; r < cfg_.num_replicas; ++r) replica_dev_indices_[static_cast<size_t>(r)] = r;
        }
    }

    // 2. Compile slot-targeted multi-bucket MXU prefill graphs (e.g. S=128, 32, 16, and 1 for B>1)
    auto buckets = resolve_prefill_buckets(cfg_);
    if (!buckets.empty()) {
        auto pc0 = std::chrono::high_resolution_clock::now();
        for (int b_len : buckets) {
            std::string prefill_mlir = emit_stablehlo_mlir(spec_, cfg_, b_len, true);
            PJRT_LoadedExecutable* ex = (cfg_.num_replicas == 1)
                ? backend_->compile_stablehlo(prefill_mlir, -1)
                : backend_->compile_stablehlo_replicated(prefill_mlir, cfg_.num_replicas);
            if (!ex) {
                last_error_ = "Slot-targeted prefill S=" + std::to_string(b_len) + " compile failed: " + backend_->get_last_error();
                shutdown();
                return false;
            }
            slot_prefill_execs_.emplace_back(b_len, ex);
        }
        auto pc1 = std::chrono::high_resolution_clock::now();
        metrics_.prefill_compile_ms = std::chrono::duration<double, std::milli>(pc1 - pc0).count();
    }

    // 3. Upload sharded weights + norms + RoPE tables + zero KV cache onto assigned TPU chips
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

    initialized_ = true;
    return true;
}

bool HavenTpuGemma4Engine::execute_graph(PJRT_LoadedExecutable* exec,
                                         int S,
                                         bool slot_targeted,
                                         int slot_id,
                                         const std::vector<uint32_t>& flat_tokens,
                                         const std::vector<int>& pos_per_slot,
                                         std::vector<float>* out_logits,
                                         std::vector<uint32_t>& out_next_tokens) {
    if (!initialized_ || !exec) return false;
    const int B_cap = cfg_.batch_size;
    const int B = slot_targeted ? 1 : B_cap;
    const int R = cfg_.num_replicas;
    const int L = spec_.num_layers;
    const int D = spec_.embedding_dim;
    const int P_dim = spec_.pleias_dim;
    const int V = spec_.vocab_size;
    const bool per_slot_pos = (!slot_targeted && S == 1 && B_cap > 1);
    const size_t num_kv = static_cast<size_t>(spec_.num_kv_buffers());
    const size_t num_outputs = 1 + num_kv;

    if (static_cast<int>(flat_tokens.size()) != B * S) {
        last_error_ = "flat_tokens size must equal B_exec * S.";
        return false;
    }
    if (per_slot_pos && static_cast<int>(pos_per_slot.size()) != B_cap) {
        last_error_ = "pos_per_slot size must equal batch_size.";
        return false;
    }
    if (!per_slot_pos && pos_per_slot.empty()) {
        last_error_ = "pos_per_slot must not be empty.";
        return false;
    }

    const int max_valid_pos = std::max(0, cfg_.max_seq_len - S);
    int32_t pos_scalar = static_cast<int32_t>(std::clamp(pos_per_slot[0], 0, max_valid_pos));
    int32_t slot_i32 = static_cast<int32_t>(std::clamp(slot_id, 0, std::max(0, B_cap - 1)));
    std::vector<int32_t> pos_vec_i32;
    if (per_slot_pos) {
        pos_vec_i32.resize(static_cast<size_t>(B_cap));
        for (int b = 0; b < B_cap; ++b) {
            pos_vec_i32[static_cast<size_t>(b)] = static_cast<int32_t>(std::clamp(pos_per_slot[static_cast<size_t>(b)], 0, max_valid_pos));
        }
    }

    auto th0 = std::chrono::high_resolution_clock::now();
    std::vector<PJRT_Buffer*> step_toks(static_cast<size_t>(R), nullptr);
    std::vector<PJRT_Buffer*> step_emb(static_cast<size_t>(R), nullptr);
    std::vector<PJRT_Buffer*> step_pl(static_cast<size_t>(R), nullptr);
    std::vector<PJRT_Buffer*> step_pos(static_cast<size_t>(R), nullptr);
    std::vector<PJRT_Buffer*> step_slot(static_cast<size_t>(R), nullptr);
    std::vector<std::vector<PJRT_Buffer*>> args_per_rep(static_cast<size_t>(R));

    auto cleanup_step_bufs = [&]() {
        for (int rep = 0; rep < R; ++rep) {
            backend_->destroy_buffer(step_toks[static_cast<size_t>(rep)]);
            backend_->destroy_buffer(step_emb[static_cast<size_t>(rep)]);
            backend_->destroy_buffer(step_pl[static_cast<size_t>(rep)]);
            backend_->destroy_buffer(step_pos[static_cast<size_t>(rep)]);
            backend_->destroy_buffer(step_slot[static_cast<size_t>(rep)]);
        }
    };

    if (cfg_.use_onchip_embeddings) {
        std::vector<int32_t> tok_ids_i32(static_cast<size_t>(B) * static_cast<size_t>(S));
        for (size_t i = 0; i < tok_ids_i32.size(); ++i) {
            tok_ids_i32[i] = static_cast<int32_t>(std::min<uint32_t>(flat_tokens[i], static_cast<uint32_t>(V - 1)));
        }
        for (int rep = 0; rep < R; ++rep) {
            const int dev = replica_dev_indices_[static_cast<size_t>(rep)];
            step_toks[static_cast<size_t>(rep)] = backend_->upload_i32_nd(tok_ids_i32.data(), {B, S}, dev);
            step_pos[static_cast<size_t>(rep)]  = per_slot_pos
                ? backend_->upload_i32_nd(pos_vec_i32.data(), {B_cap}, dev)
                : backend_->upload_i32_nd(&pos_scalar, {}, dev);
            if (slot_targeted) {
                step_slot[static_cast<size_t>(rep)] = backend_->upload_i32_nd(&slot_i32, {}, dev);
            }
            if (!step_toks[static_cast<size_t>(rep)] || !step_pos[static_cast<size_t>(rep)] ||
                (slot_targeted && !step_slot[static_cast<size_t>(rep)])) {
                cleanup_step_bufs();
                last_error_ = "Per-step token-ID H2D upload failed: " + backend_->get_last_error();
                return false;
            }
            auto& a = args_per_rep[static_cast<size_t>(rep)];
            a.reserve(3 + replica_weight_bufs_[static_cast<size_t>(rep)].size() + num_kv);
            a.push_back(step_toks[static_cast<size_t>(rep)]);
            a.push_back(step_pos[static_cast<size_t>(rep)]);
            if (slot_targeted) a.push_back(step_slot[static_cast<size_t>(rep)]);
            for (auto* wb : replica_weight_bufs_[static_cast<size_t>(rep)]) a.push_back(wb);
            for (auto* kv : replica_kv_bufs_[static_cast<size_t>(rep)]) a.push_back(kv);
        }
    } else {
        const size_t emb_n = static_cast<size_t>(B) * static_cast<size_t>(S) * static_cast<size_t>(D);
        emb_f32_scratch_.resize(emb_n);
        emb_bf16_scratch_.resize(emb_n);

        const size_t pl_n = spec_.has_pleias
            ? (static_cast<size_t>(B) * static_cast<size_t>(S) * static_cast<size_t>(L) * static_cast<size_t>(P_dim))
            : 0;
        if (spec_.has_pleias) {
            pleias_f32_scratch_.resize(pl_n);
            pleias_bf16_scratch_.resize(pl_n);
        }

        const TensorDesc* tok_td = host_engine_->get_token_embd_tensor();
        const TensorDesc* pl_td  = spec_.has_pleias ? host_engine_->get_per_layer_token_embd_tensor() : nullptr;
        const float emb_scale = spec_.emb_scale;
        const float tok_pl_scale = std::sqrt(static_cast<float>(P_dim));

        for (int idx = 0; idx < B * S; ++idx) {
            const int tok = static_cast<int>(std::min<uint32_t>(flat_tokens[static_cast<size_t>(idx)], static_cast<uint32_t>(V - 1)));
            float* edst = emb_f32_scratch_.data() + static_cast<size_t>(idx) * static_cast<size_t>(D);
            if (tok_td) {
                Avx2Math::dequantize_row(edst, *tok_td, tok, D);
                if (std::fabs(emb_scale - 1.0f) > 1e-6f) {
                    for (int c = 0; c < D; ++c) edst[c] *= emb_scale;
                }
            } else {
                std::memset(edst, 0, static_cast<size_t>(D) * sizeof(float));
            }
            if (spec_.has_pleias) {
                float* pdst = pleias_f32_scratch_.data() + static_cast<size_t>(idx) * static_cast<size_t>(L * P_dim);
                if (pl_td) {
                    Avx2Math::dequantize_row(pdst, *pl_td, tok, L * P_dim);
                    for (int c = 0; c < L * P_dim; ++c) pdst[c] *= tok_pl_scale;
                } else {
                    std::memset(pdst, 0, static_cast<size_t>(L * P_dim) * sizeof(float));
                }
            }
        }

        HavenTpuPjrtBackend::convert_f32_to_bf16(emb_f32_scratch_.data(), emb_bf16_scratch_.data(), emb_n);
        if (spec_.has_pleias) {
            HavenTpuPjrtBackend::convert_f32_to_bf16(pleias_f32_scratch_.data(), pleias_bf16_scratch_.data(), pl_n);
        }

        for (int rep = 0; rep < R; ++rep) {
            const int dev = replica_dev_indices_[static_cast<size_t>(rep)];
            step_emb[static_cast<size_t>(rep)] = backend_->upload_bf16_nd(emb_bf16_scratch_.data(), {B, S, D}, dev);
            if (spec_.has_pleias) {
                step_pl[static_cast<size_t>(rep)] = backend_->upload_bf16_nd(pleias_bf16_scratch_.data(), {B, S, L, P_dim}, dev);
            }
            step_pos[static_cast<size_t>(rep)] = per_slot_pos
                ? backend_->upload_i32_nd(pos_vec_i32.data(), {B_cap}, dev)
                : backend_->upload_i32_nd(&pos_scalar, {}, dev);
            if (slot_targeted) {
                step_slot[static_cast<size_t>(rep)] = backend_->upload_i32_nd(&slot_i32, {}, dev);
            }
            if (!step_emb[static_cast<size_t>(rep)] || (spec_.has_pleias && !step_pl[static_cast<size_t>(rep)]) ||
                !step_pos[static_cast<size_t>(rep)] || (slot_targeted && !step_slot[static_cast<size_t>(rep)])) {
                cleanup_step_bufs();
                last_error_ = "Per-step H2D upload failed: " + backend_->get_last_error();
                return false;
            }
            auto& a = args_per_rep[static_cast<size_t>(rep)];
            a.reserve(4 + replica_weight_bufs_[static_cast<size_t>(rep)].size() + num_kv);
            a.push_back(step_emb[static_cast<size_t>(rep)]);
            if (spec_.has_pleias) a.push_back(step_pl[static_cast<size_t>(rep)]);
            a.push_back(step_pos[static_cast<size_t>(rep)]);
            if (slot_targeted) a.push_back(step_slot[static_cast<size_t>(rep)]);
            for (auto* wb : replica_weight_bufs_[static_cast<size_t>(rep)]) a.push_back(wb);
            for (auto* kv : replica_kv_bufs_[static_cast<size_t>(rep)]) a.push_back(kv);
        }
    }

    auto te0 = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<PJRT_Buffer*>> outs_per_rep;
    bool ok = false;
    if (R == 1) {
        outs_per_rep.resize(1);
        ok = backend_->execute_1d(exec, replica_dev_indices_[0], args_per_rep[0], num_outputs, outs_per_rep[0]);
    } else {
        ok = backend_->execute_replicated(exec, R, args_per_rep, num_outputs, outs_per_rep);
    }
    auto td0 = std::chrono::high_resolution_clock::now();

    cleanup_step_bufs();

    if (!ok) {
        last_error_ = "On-chip TPU execute failed: " + backend_->get_last_error();
        return false;
    }

    // Swap mutated/donated KV cache handles in-place
    for (int rep = 0; rep < R; ++rep) {
        for (size_t k = 0; k < num_kv; ++k) {
            backend_->destroy_buffer(replica_kv_bufs_[static_cast<size_t>(rep)][k]);
            replica_kv_bufs_[static_cast<size_t>(rep)][k] = outs_per_rep[static_cast<size_t>(rep)][1 + k];
        }
    }

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
    auto td1 = std::chrono::high_resolution_clock::now();
    metrics_.last_h2d_ms  = std::chrono::duration<double, std::milli>(te0 - th0).count();
    metrics_.last_exec_ms = std::chrono::duration<double, std::milli>(td0 - te0).count();
    metrics_.last_d2h_ms  = std::chrono::duration<double, std::milli>(td1 - td0).count();
    return ok;
}

bool HavenTpuGemma4Engine::prefill_slot(int slot_id,
                                        const std::vector<uint32_t>& tokens,
                                        int start_pos,
                                        std::vector<float>* out_logits,
                                        uint32_t* out_next_token) {
    if (!initialized_ || tokens.empty()) return false;
    if (slot_id < 0 || slot_id >= cfg_.batch_size) {
        last_error_ = "prefill_slot: slot_id out of range.";
        return false;
    }

    int pos = start_pos;
    size_t cursor = 0;
    std::vector<uint32_t> next_toks;

    // Greedy descent through compiled slot-targeted MXU buckets (e.g. 128 -> 32 -> 16 -> 1)
    while (cursor < tokens.size()) {
        size_t rem = tokens.size() - cursor;
        PJRT_LoadedExecutable* chosen_exec = nullptr;
        int chosen_len = 0;
        bool use_slot_targeted = true;

        for (const auto& [blen, bexec] : slot_prefill_execs_) {
            if (bexec && static_cast<size_t>(blen) <= rem) {
                chosen_exec = bexec;
                chosen_len = blen;
                use_slot_targeted = true;
                break;
            }
        }

        if (!chosen_exec) {
            if (cfg_.batch_size == 1) {
                // For B=1, decode_exec_ IS a single-slot S=1 graph!
                chosen_exec = decode_exec_;
                chosen_len = 1;
                use_slot_targeted = false;
            } else {
                last_error_ = "prefill_slot requires prefill_chunk_size > 0 (or S=1 bucket) when batch_size > 1.";
                return false;
            }
        }

        std::vector<uint32_t> chunk(tokens.begin() + static_cast<std::ptrdiff_t>(cursor),
                                    tokens.begin() + static_cast<std::ptrdiff_t>(cursor + static_cast<size_t>(chosen_len)));
        bool is_last = (cursor + static_cast<size_t>(chosen_len) == tokens.size());
        if (!execute_graph(chosen_exec, chosen_len, use_slot_targeted, slot_id, chunk, {pos},
                           is_last ? out_logits : nullptr, next_toks)) {
            return false;
        }
        cursor += static_cast<size_t>(chosen_len);
        pos += chosen_len;
    }

    if (out_next_token && !next_toks.empty()) {
        *out_next_token = next_toks[0];
    }
    return true;
}

bool HavenTpuGemma4Engine::prefill_prompt(const std::vector<uint32_t>& tokens,
                                          int start_pos,
                                          std::vector<float>* out_logits,
                                          uint32_t* out_next_token) {
    if (!initialized_ || tokens.empty()) return false;
    if (cfg_.batch_size == 1) {
        return prefill_slot(0, tokens, start_pos, out_logits, out_next_token);
    }

    // For batch_size > 1: broadcast identical prompt across all B slots (used by synchronous batch throughput benchmarks)
    const int B = cfg_.batch_size;
    int pos = start_pos;
    std::vector<uint32_t> next_toks;
    for (size_t cursor = 0; cursor < tokens.size(); ++cursor) {
        std::vector<uint32_t> step_toks(static_cast<size_t>(B), tokens[cursor]);
        std::vector<int> step_pos(static_cast<size_t>(B), pos);
        bool is_last = (cursor + 1 == tokens.size());
        if (!execute_graph(decode_exec_, 1, false, 0, step_toks, step_pos,
                           is_last ? out_logits : nullptr, next_toks)) {
            return false;
        }
        pos++;
    }
    if (out_next_token && !next_toks.empty()) {
        *out_next_token = next_toks[0];
    }
    return true;
}

bool HavenTpuGemma4Engine::decode_step_batched(const std::vector<uint32_t>& tokens_per_batch,
                                               const std::vector<int>& pos_per_slot,
                                               std::vector<float>* out_logits,
                                               std::vector<uint32_t>& out_next_tokens) {
    return execute_graph(decode_exec_, 1, false, 0, tokens_per_batch, pos_per_slot, out_logits, out_next_tokens);
}

bool HavenTpuGemma4Engine::decode_step(const std::vector<uint32_t>& tokens_per_batch,
                                       int pos,
                                       std::vector<float>* out_logits,
                                       std::vector<uint32_t>& out_next_tokens) {
    std::vector<int> pos_vec(static_cast<size_t>(std::max(1, cfg_.batch_size)), pos);
    return execute_graph(decode_exec_, 1, false, 0, tokens_per_batch, pos_vec, out_logits, out_next_tokens);
}

} // namespace haven
