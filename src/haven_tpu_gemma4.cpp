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
    ss << std::scientific << std::setprecision(9) << v;
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

// Quantize a [rows, cols] float row-major slice into symmetric INT8 + FP64 least-squares refined per-row F32 scales
static void quantize_f32_to_s8_per_row(const float* src, int rows, int cols,
                                       int8_t* dst_q, float* dst_scale) {
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) {
        const float* row = src + static_cast<size_t>(r) * static_cast<size_t>(cols);
        int8_t* qrow = dst_q + static_cast<size_t>(r) * static_cast<size_t>(cols);
        double max_abs = 0.0;
        for (int c = 0; c < cols; ++c) {
            const double a = std::fabs(static_cast<double>(row[c]));
            if (a > max_abs) max_abs = a;
        }
        if (max_abs <= 1e-12) {
            dst_scale[r] = 1.0f;
            std::memset(qrow, 0, static_cast<size_t>(cols));
            continue;
        }
        const double inv_scale = 127.0 / max_abs;
        double sum_wq = 0.0;
        double sum_qq = 0.0;
        for (int c = 0; c < cols; ++c) {
            const double w = static_cast<double>(row[c]);
            int q = static_cast<int>(std::lrint(w * inv_scale));
            q = std::clamp(q, -127, 127);
            qrow[c] = static_cast<int8_t>(q);
            const double qd = static_cast<double>(q);
            sum_wq += w * qd;
            sum_qq += qd * qd;
        }
        // Closed-form FP64 least-squares optimal scale minimizes L2 quantization error across the row
        dst_scale[r] = (sum_qq > 0.0) ? static_cast<float>(sum_wq / sum_qq) : static_cast<float>(max_abs / 127.0);
    }
}

// Quantize a single row [cols] into packed group-G signed INT4 (2 two's-complement [-8,+7] nibbles/byte)
// + MSE-optimal clipped & FP64 least-squares refined [cols/G] F32 scales (30-45% lower RMSE on heavy-tailed LLM weights,
// 100% compatible with shift_left/shift_right_arithmetic StableHLO unpack with ZERO runtime overhead!).
static void quantize_row_f32_to_s4_packed_group(const float* row, int cols, int group_size,
                                                int8_t* dst_packed_row, float* dst_scale_row) {
    const int ng = cols / group_size;
    const int half_g = group_size / 2;
    static constexpr double kClipAlphas[12] = {
        1.00, 0.92, 0.84, 0.76, 0.68, 0.60, 0.52, 0.45, 0.38, 0.32, 0.26, 0.20
    };

    for (int g = 0; g < ng; ++g) {
        const float* grp = row + static_cast<size_t>(g) * static_cast<size_t>(group_size);
        double max_abs = 0.0;
        for (int i = 0; i < group_size; ++i) {
            const double a = std::fabs(static_cast<double>(grp[i]));
            if (a > max_abs) max_abs = a;
        }
        int8_t* pdst = dst_packed_row + static_cast<size_t>(g) * static_cast<size_t>(half_g);
        if (max_abs <= 1e-12) {
            dst_scale_row[g] = 1.0f;
            std::memset(pdst, 0, static_cast<size_t>(half_g));
            continue;
        }

        // Grid-search clip ratio maximizing explained energy (sum_wq^2 / sum_qq) == minimizing exact L2 MSE
        double best_score = -1.0;
        double best_inv_s = 7.0 / max_abs;
        double best_scale = max_abs / 7.0;
        auto eval_candidate = [&](double inv_s) {
            double sum_wq = 0.0;
            double sum_qq = 0.0;
            for (int i = 0; i < group_size; ++i) {
                const double w = static_cast<double>(grp[i]);
                int q = static_cast<int>(std::lrint(w * inv_s));
                q = std::clamp(q, -8, 7);
                const double qd = static_cast<double>(q);
                sum_wq += w * qd;
                sum_qq += qd * qd;
            }
            if (sum_qq > 0.0 && sum_wq > 0.0) {
                const double score = (sum_wq * sum_wq) / sum_qq;
                if (score > best_score) {
                    best_score = score;
                    best_inv_s = inv_s;
                    best_scale = sum_wq / sum_qq;
                }
            }
        };

        for (double alpha : kClipAlphas) {
            eval_candidate(7.0 / (alpha * max_abs));
        }
        // Two alternating Lloyd-Max refinement passes around the winning scale
        for (int iter = 0; iter < 2; ++iter) {
            if (best_scale > 1e-12) {
                eval_candidate(1.0 / best_scale);
            }
        }

        dst_scale_row[g] = static_cast<float>(best_scale);
        for (int p = 0; p < half_g; ++p) {
            int q0 = std::clamp(static_cast<int>(std::lrint(static_cast<double>(grp[2 * p])     * best_inv_s)), -8, 7);
            int q1 = std::clamp(static_cast<int>(std::lrint(static_cast<double>(grp[2 * p + 1]) * best_inv_s)), -8, 7);
            uint8_t byte = (static_cast<uint8_t>(q0) & 0x0Fu) | ((static_cast<uint8_t>(q1) & 0x0Fu) << 4);
            pdst[p] = static_cast<int8_t>(byte);
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

    // Unpack packed group-G symmetric signed INT4 matrix [N, K/2] (i8) + [N, K/G] (f32) -> [N, K] (bf16)
    // Pure standard StableHLO ops (shift_left + shift_right_arithmetic + convert i8->f32 + multiply exact f32 scale -> single bf16 convert),
    // eliminating BF16 scale double-rounding while XLA fuses directly into TPU VMEM / CUDA Tensor Core registers!
    std::string emit_unpack_s4_2d(const std::string& w_p, const std::string& s_2d, int N, int K) {
        const int K2 = K / 2;
        const int G  = HavenTpuGemma4Engine::compute_w4_group_size(K);
        const int NG = K / G;
        const std::string tp_i8 = "tensor<" + std::to_string(N) + "x" + std::to_string(K2) + "xi8>";
        const std::string tp_f  = "tensor<" + std::to_string(N) + "x" + std::to_string(K2) + "xf32>";
        const std::string t3_1  = "tensor<" + std::to_string(N) + "x" + std::to_string(K2) + "x1xf32>";
        const std::string t3_2  = "tensor<" + std::to_string(N) + "x" + std::to_string(K2) + "x2xf32>";
        const std::string t_grp = "tensor<" + std::to_string(N) + "x" + std::to_string(NG) + "x" + std::to_string(G) + "xf32>";
        const std::string ts_f  = "tensor<" + std::to_string(N) + "x" + std::to_string(NG) + "xf32>";
        const std::string wf_t  = "tensor<" + std::to_string(N) + "x" + std::to_string(K) + "xf32>";
        const std::string wb_t  = "tensor<" + std::to_string(N) + "x" + std::to_string(K) + "xbf16>";

        std::string c4 = tmp("c4"), shl = tmp("w4sl"), lo = tmp("w4lo"), hi = tmp("w4hi");
        std::string lof = tmp("w4lf"), hif = tmp("w4hf"), lo3 = tmp("w4l3"), hi3 = tmp("w4h3");
        std::string cat = tmp("w4ct"), grp = tmp("w4gr"), sbc = tmp("w4bc");
        std::string dq  = tmp("w4dq"), dqf = tmp("w4f2"), out = tmp("w4bf");
        ss << "    " << c4  << " = stablehlo.constant dense<4> : " << tp_i8 << "\n"
           << "    " << shl << " = stablehlo.shift_left " << w_p << ", " << c4 << " : " << tp_i8 << "\n"
           << "    " << lo  << " = stablehlo.shift_right_arithmetic " << shl << ", " << c4 << " : " << tp_i8 << "\n"
           << "    " << hi  << " = stablehlo.shift_right_arithmetic " << w_p << ", " << c4 << " : " << tp_i8 << "\n"
           << "    " << lof << " = stablehlo.convert " << lo << " : (" << tp_i8 << ") -> " << tp_f << "\n"
           << "    " << hif << " = stablehlo.convert " << hi << " : (" << tp_i8 << ") -> " << tp_f << "\n"
           << "    " << lo3 << " = stablehlo.reshape " << lof << " : (" << tp_f << ") -> " << t3_1 << "\n"
           << "    " << hi3 << " = stablehlo.reshape " << hif << " : (" << tp_f << ") -> " << t3_1 << "\n"
           << "    " << cat << " = stablehlo.concatenate " << lo3 << ", " << hi3 << ", dim = 2 : (" << t3_1 << ", " << t3_1 << ") -> " << t3_2 << "\n"
           << "    " << grp << " = stablehlo.reshape " << cat << " : (" << t3_2 << ") -> " << t_grp << "\n"
           << "    " << sbc << " = stablehlo.broadcast_in_dim " << s_2d << ", dims = [0, 1] : (" << ts_f << ") -> " << t_grp << "\n"
           << "    " << dq  << " = stablehlo.multiply " << grp << ", " << sbc << " : " << t_grp << "\n"
           << "    " << dqf << " = stablehlo.reshape " << dq << " : (" << t_grp << ") -> " << wf_t << "\n"
           << "    " << out << " = stablehlo.convert " << dqf << " : (" << wf_t << ") -> " << wb_t << "\n";
        return out;
    }

    // Slice 2D layer matrix [N, K] from 3D stack [L, N, K] (or [L, N, K/2] for W4A16) and multiply: [B, S, K] x [N, K]^T -> [B, S, N] (F32)
    std::string emit_linear_3d(const std::string& x_bf16,
                               const std::string& w_stack, const std::string& s_stack,
                               int layer_idx, int L, int B, int S, int N, int K,
                               bool use_int8, bool use_int4 = false) {
        const std::string x_t  = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(K) + "xbf16>";
        const std::string y_t  = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(N) + "xf32>";
        const std::string wb_t = "tensor<" + std::to_string(N) + "x" + std::to_string(K) + "xbf16>";

        if (use_int4) {
            const int K2 = K / 2;
            const int NG = K / HavenTpuGemma4Engine::compute_w4_group_size(K);
            const std::string ws_t = "tensor<" + std::to_string(L) + "x" + std::to_string(N) + "x" + std::to_string(K2) + "xi8>";
            const std::string w1_t = "tensor<1x" + std::to_string(N) + "x" + std::to_string(K2) + "xi8>";
            const std::string w2_t = "tensor<" + std::to_string(N) + "x" + std::to_string(K2) + "xi8>";
            const std::string ss_t = "tensor<" + std::to_string(L) + "x" + std::to_string(N) + "x" + std::to_string(NG) + "xf32>";
            const std::string s1_t = "tensor<1x" + std::to_string(N) + "x" + std::to_string(NG) + "xf32>";
            const std::string s2_t = "tensor<" + std::to_string(N) + "x" + std::to_string(NG) + "xf32>";

            std::string w_sl = tmp("wsl"), w_2d = tmp("w2d"), s_sl = tmp("ssl"), s_2d = tmp("s2d");
            ss << "    " << w_sl << " = stablehlo.slice " << w_stack << " [" << layer_idx << ":" << (layer_idx + 1)
               << ", 0:" << N << ", 0:" << K2 << "] : (" << ws_t << ") -> " << w1_t << "\n"
               << "    " << w_2d << " = stablehlo.reshape " << w_sl << " : (" << w1_t << ") -> " << w2_t << "\n"
               << "    " << s_sl << " = stablehlo.slice " << s_stack << " [" << layer_idx << ":" << (layer_idx + 1)
               << ", 0:" << N << ", 0:" << NG << "] : (" << ss_t << ") -> " << s1_t << "\n"
               << "    " << s_2d << " = stablehlo.reshape " << s_sl << " : (" << s1_t << ") -> " << s2_t << "\n";
            std::string w_mat = emit_unpack_s4_2d(w_2d, s_2d, N, K);
            std::string dot = tmp("dot");
            ss << "    " << dot << " = stablehlo.dot_general " << x_bf16 << ", " << w_mat
               << ", contracting_dims = [2] x [1] : (" << x_t << ", " << wb_t << ") -> " << y_t << "\n";
            return dot;
        }

        const std::string el   = use_int8 ? "i8" : "bf16";
        const std::string ws_t = "tensor<" + std::to_string(L) + "x" + std::to_string(N) + "x" + std::to_string(K) + "x" + el + ">";
        const std::string w1_t = "tensor<1x" + std::to_string(N) + "x" + std::to_string(K) + "x" + el + ">";
        const std::string w2_t = "tensor<" + std::to_string(N) + "x" + std::to_string(K) + "x" + el + ">";

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

    // Dynamic per-row symmetric INT8 quantization [N, K] (F32) -> (x_i8 [N, K] i8, x_scale [N] f32)
    // Enables native hardware TPU v6e i8 x i8 -> i32 MXU GEMMs on MoE Gate+Up (2x BF16 TOPS, zero i8->bf16 spill!).
    std::pair<std::string, std::string> emit_quant_i8_per_row(const std::string& x_f32, int N, int K,
                                                              const std::string& c_zero) {
        const std::string t2_f = "tensor<" + std::to_string(N) + "x" + std::to_string(K) + "xf32>";
        const std::string t2_i = "tensor<" + std::to_string(N) + "x" + std::to_string(K) + "xi8>";
        const std::string t1_f = "tensor<" + std::to_string(N) + "xf32>";
        std::string ax = tmp("qax"), mx = tmp("qmx"), c_i127 = tmp("qi127"), c_eps = tmp("qeps");
        std::string sc0 = tmp("qsc0"), sc = tmp("qsc"), scb = tmp("qscb"), dv = tmp("qdv");
        std::string rnd = tmp("qrnd"), c_hi = tmp("qhi"), c_lo = tmp("qlo"), cl1 = tmp("qcl1"), cl2 = tmp("qcl2"), xi8 = tmp("qi8");
        ss << "    " << ax << " = stablehlo.abs " << x_f32 << " : " << t2_f << "\n"
           << "    " << mx << " = stablehlo.reduce(" << ax << " init: " << c_zero << ") across dimensions = [1] : ("
           << t2_f << ", tensor<f32>) -> " << t1_f << "\n"
           << "      reducer(%ra: tensor<f32>, %rb: tensor<f32>) {\n"
           << "        %rc = stablehlo.maximum %ra, %rb : tensor<f32>\n"
           << "        stablehlo.return %rc : tensor<f32>\n"
           << "      }\n"
           << "    " << c_i127 << " = stablehlo.constant dense<" << fmt_f32(1.0f / 127.0f) << "> : " << t1_f << "\n"
           << "    " << c_eps  << " = stablehlo.constant dense<1.0000000e-12> : " << t1_f << "\n"
           << "    " << sc0    << " = stablehlo.multiply " << mx << ", " << c_i127 << " : " << t1_f << "\n"
           << "    " << sc     << " = stablehlo.maximum " << sc0 << ", " << c_eps << " : " << t1_f << "\n"
           << "    " << scb    << " = stablehlo.broadcast_in_dim " << sc << ", dims = [0] : (" << t1_f << ") -> " << t2_f << "\n"
           << "    " << dv     << " = stablehlo.divide " << x_f32 << ", " << scb << " : " << t2_f << "\n"
           << "    " << rnd    << " = stablehlo.round_nearest_even " << dv << " : " << t2_f << "\n"
           << "    " << c_hi   << " = stablehlo.constant dense<1.2700000e+02> : " << t2_f << "\n"
           << "    " << c_lo   << " = stablehlo.constant dense<-1.2700000e+02> : " << t2_f << "\n"
           << "    " << cl1    << " = stablehlo.minimum " << rnd << ", " << c_hi << " : " << t2_f << "\n"
           << "    " << cl2    << " = stablehlo.maximum " << cl1 << ", " << c_lo << " : " << t2_f << "\n"
           << "    " << xi8    << " = stablehlo.convert " << cl2 << " : (" << t2_f << ") -> " << t2_i << "\n";
        return {xi8, sc};
    }
};

// Exact O(N log N) Normalized Orthogonal Walsh-Hadamard Transform (H^T = H, H^2 = I, entries ±1/sqrt(N))
// For N = 256, 1/sqrt(256) = 1/16 = 0.0625 is 100% EXACT in BF16 (zero mantissa rounding error!).
static void fwht_inplace(float* a, int n) {
    for (int h = 1; h < n; h <<= 1) {
        for (int i = 0; i < n; i += (h << 1)) {
            for (int j = i; j < i + h; ++j) {
                const float x = a[j];
                const float y = a[j + h];
                a[j]     = x + y;
                a[j + h] = x - y;
            }
        }
    }
    const float inv = 1.0f / std::sqrt(static_cast<float>(n));
    for (int i = 0; i < n; ++i) a[i] *= inv;
}

static std::vector<int> resolve_prefill_buckets(const TpuGemma4Config& cfg, bool is_cuda = false) {
    std::vector<int> buckets;
    if (!cfg.prefill_buckets.empty()) {
        for (int b : cfg.prefill_buckets) {
            if (b >= 1 && b <= cfg.max_seq_len) buckets.push_back(b);
        }
    } else if (cfg.prefill_chunk_size > 1) {
        if (cfg.prefill_chunk_size >= 128) {
            if (is_cuda) {
                for (int b : {cfg.prefill_chunk_size, 32}) {
                    if (b <= cfg.max_seq_len) buckets.push_back(b);
                }
            } else {
                for (int b : {cfg.prefill_chunk_size, 64, 32}) {
                    if (b <= cfg.max_seq_len) buckets.push_back(b);
                }
            }
        } else {
            buckets.push_back(std::min(cfg.prefill_chunk_size, cfg.max_seq_len));
            if (cfg.prefill_chunk_size > 32) buckets.push_back(32);
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

// Compute exact TPU (8,128)-aligned SWA KV buffer length:
// Hybrid SWA+Global models (Gemma 4 / Ava 26B) use exact sink-free sliding window (e.g. T_swa=1024 exact power of 2!);
// SWA-only models (Mistral) retain 4 StreamingLLM attention sinks.
static int compute_swa_kv_len(const TpuModelSpec& spec, const TpuGemma4Config& tc) {
    const int T = std::max(1, tc.max_seq_len);
    int max_s = std::max(1, tc.prefill_chunk_size);
    for (int b : tc.prefill_buckets) {
        if (b > max_s && b <= T) max_s = b;
    }
    if (spec.sliding_window <= 0) return T;
    const int num_sinks = (spec.num_glob_layers > 0) ? 0 : 4;
    const int min_ring = num_sinks + spec.sliding_window + std::max(0, max_s - 1);
    const int aligned_ring = ((min_ring + 127) / 128) * 128;
    return (aligned_ring < T) ? aligned_ring : T;
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
    s.head_dim_swa = (mc.head_dim_swa > 0) ? static_cast<int>(mc.head_dim_swa) : default_hd;
    s.head_dim_glob = default_hd;
    s.num_kv_heads_swa = s.num_kv_heads;
    s.num_kv_heads_glob = s.num_kv_heads;
    s.has_swa_v = true;
    s.has_glob_v = true;

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

        // Detect Sparse Mixture-of-Experts (Gemma 4 26B-A4B / Ava 26B Turbo MoE)
        if (lt0.ffn_gate_inp != nullptr && lt0.ffn_down_exps != nullptr &&
            (lt0.ffn_gate_up_exps != nullptr || (lt0.ffn_gate_exps != nullptr && lt0.ffn_up_exps != nullptr))) {
            s.is_moe = true;
            s.num_experts = (mc.expert_count > 0)
                ? static_cast<int>(mc.expert_count)
                : ((lt0.ffn_gate_inp->shape.size() > 1) ? static_cast<int>(lt0.ffn_gate_inp->shape[1]) : 128);
            s.num_experts_used = (mc.expert_used_count > 0) ? static_cast<int>(mc.expert_used_count) : 8;
            s.expert_ffn_dim = (mc.expert_ffn_dim > 0)
                ? static_cast<int>(mc.expert_ffn_dim)
                : ((lt0.ffn_down_exps->shape.size() > 0) ? static_cast<int>(lt0.ffn_down_exps->shape[0]) : s.hidden_dim);
            s.has_moe_router_scale = (lt0.ffn_gate_inp_scale != nullptr && !mc.turbo_folded_scales);
            s.has_post_ffn_norm_1  = (lt0.post_ffw_norm_1 != nullptr);
            s.has_pre_ffn_norm_2   = (lt0.pre_ffw_norm_2 != nullptr);
            s.has_post_ffn_norm_2  = (lt0.post_ffw_norm_2 != nullptr);
        }

        bool found_swa_hd = false, found_glob_hd = false;
        bool found_swa_kv = false, found_glob_kv = false;
        for (int l = 0; l < s.num_layers; ++l) {
            const auto& lt = host_engine->get_layer_tensors(static_cast<uint32_t>(l));
            bool is_swa = (static_cast<size_t>(l) < mc.sliding_window_pattern.size())
                ? mc.sliding_window_pattern[static_cast<size_t>(l)]
                : (has_swa_arch && (mc.architecture == "mistral" || (l % 6 != 5)));
            bool is_shared_kv = (mc.shared_kv_layers > 0 && mc.shared_kv_layers < mc.num_layers &&
                static_cast<uint32_t>(l) >= (mc.num_layers - mc.shared_kv_layers));
            bool has_kv = !is_shared_kv && (lt.attn_k != nullptr);
            s.layer_is_swa[static_cast<size_t>(l)] = is_swa;
            s.layer_has_kv[static_cast<size_t>(l)] = has_kv;

            int layer_hd = default_hd;
            if (lt.attn_q && lt.attn_q->shape.size() > 1 && s.num_heads > 0) {
                int hd = static_cast<int>(lt.attn_q->shape[1]) / s.num_heads;
                if (hd > 0) {
                    layer_hd = hd;
                    if (is_swa && !found_swa_hd) { s.head_dim_swa = hd; found_swa_hd = true; }
                    if (!is_swa && !found_glob_hd) { s.head_dim_glob = hd; found_glob_hd = true; }
                }
            }
            if (has_kv) {
                int layer_hkv = (static_cast<size_t>(l) < mc.head_count_kv_per_layer.size() && mc.head_count_kv_per_layer[static_cast<size_t>(l)] > 0)
                    ? static_cast<int>(mc.head_count_kv_per_layer[static_cast<size_t>(l)])
                    : ((lt.attn_k && lt.attn_k->shape.size() > 1 && layer_hd > 0)
                        ? std::max(1, static_cast<int>(lt.attn_k->shape[1]) / layer_hd)
                        : s.num_kv_heads);
                if (is_swa && !found_swa_kv) {
                    s.num_kv_heads_swa = layer_hkv;
                    s.has_swa_v = (lt.attn_v != nullptr);
                    found_swa_kv = true;
                }
                if (!is_swa && !found_glob_kv) {
                    s.num_kv_heads_glob = layer_hkv;
                    s.has_glob_v = (lt.attn_v != nullptr);
                    found_glob_kv = true;
                }
            }
        }
    } else {
        // Synthetic specification from ModelConfig (for offline CI & IR inspection)
        const bool is_gemma4_moe = (mc.is_gemma && mc.expert_count > 0);
        const bool is_gemma4_42 = (mc.is_gemma && s.num_layers == 42 && !is_gemma4_moe);
        const bool is_gemma3_or_4 = (mc.is_gemma && (mc.architecture.find("gemma3") != std::string::npos ||
                                                     mc.architecture.find("gemma4") != std::string::npos ||
                                                     is_gemma4_42 || is_gemma4_moe));
        s.has_pleias = is_gemma4_42;
        s.has_q_norm = is_gemma3_or_4 || (mc.architecture == "qwen3");
        s.has_k_norm = is_gemma3_or_4 || (mc.architecture == "qwen3");
        s.has_v_norm = is_gemma4_42 || is_gemma4_moe;
        s.has_post_attn_norm = mc.is_gemma;
        s.has_post_ffn_norm = mc.is_gemma;
        s.has_post_norm = is_gemma4_42;
        s.has_layer_scale = is_gemma4_42 || is_gemma4_moe;
        s.has_qkv_bias = (mc.architecture == "qwen2");
        s.has_tied_embeddings = mc.is_gemma;
        if (is_gemma4_42) {
            s.head_dim_swa = 256;
            s.head_dim_glob = 512;
        }
        if (is_gemma4_moe) {
            s.is_moe = true;
            s.num_experts = static_cast<int>(mc.expert_count);
            s.num_experts_used = (mc.expert_used_count > 0) ? static_cast<int>(mc.expert_used_count) : 8;
            s.expert_ffn_dim = (mc.expert_ffn_dim > 0) ? static_cast<int>(mc.expert_ffn_dim) : 704;
            s.has_moe_router_scale = true;
            s.has_post_ffn_norm_1 = true;
            s.has_pre_ffn_norm_2 = true;
            s.has_post_ffn_norm_2 = true;
            s.head_dim_swa = (mc.head_dim_swa > 0) ? static_cast<int>(mc.head_dim_swa) : 256;
            s.head_dim_glob = (mc.head_dim > 0) ? static_cast<int>(mc.head_dim) : 512;
            s.num_kv_heads_swa = s.num_kv_heads;
            s.num_kv_heads_glob = 2;
            s.has_swa_v = true;
            s.has_glob_v = false;
        }
        const int kv_owner_limit = (is_gemma4_42 && s.num_layers > 20)
            ? (s.num_layers - static_cast<int>(mc.shared_kv_layers > 0 ? mc.shared_kv_layers : 18))
            : s.num_layers;
        for (int l = 0; l < s.num_layers; ++l) {
            bool is_swa = (static_cast<size_t>(l) < mc.sliding_window_pattern.size())
                ? mc.sliding_window_pattern[static_cast<size_t>(l)]
                : (has_swa_arch && (mc.architecture == "mistral" || (l % 6 != 5)));
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
    if (!lt0.attn_q || !lt0.attn_k || !lt0.attn_output ||
        !lt0.ffn_gate || !lt0.ffn_up || !lt0.ffn_down) {
        reason_out = "Layer 0 is missing required attention/FFN tensors.";
        return false;
    }
    TpuModelSpec spec = TpuModelSpec::from_model_config(mc, &host_engine);
    if (spec.num_kv_buffers() == 0) {
        reason_out = "Model has no KV-owning layers.";
        return false;
    }
    if ((spec.num_swa_layers > 0 && (spec.head_dim_swa <= 0 || (spec.head_dim_swa % 2) != 0 || (spec.num_heads % std::max(1, spec.num_kv_heads_swa)) != 0)) ||
        (spec.num_glob_layers > 0 && (spec.head_dim_glob <= 0 || (spec.head_dim_glob % 2) != 0 || (spec.num_heads % std::max(1, spec.num_kv_heads_glob)) != 0))) {
        reason_out = "Head dimension must be positive/even and num_heads divisible by per-type num_kv_heads.";
        return false;
    }
    if (spec.is_moe && (spec.num_experts <= 0 || spec.num_experts_used <= 0 || spec.num_experts_used > spec.num_experts || spec.expert_ffn_dim <= 0)) {
        reason_out = "Invalid MoE configuration (experts/top-k/expert_ffn_dim).";
        return false;
    }
    reason_out.clear();
    return true;
}

int HavenTpuGemma4Engine::compute_w4_group_size(int cols) {
    if (cols <= 0) return 1;
    for (int g : {128, 64, 32, 16, 8, 4, 2}) {
        if ((cols % g) == 0) return g;
    }
    return cols;
}

void HavenTpuGemma4Engine::quantize_f32_to_s4_packed_group(const float* src, int rows, int cols, int group_size,
                                                           int8_t* dst_packed, float* dst_scale) {
    if (!src || !dst_packed || !dst_scale || rows <= 0 || cols <= 0 || (cols % 2) != 0 || group_size <= 0 || (cols % group_size) != 0) {
        return;
    }
    const int ng = cols / group_size;
    const int cols_half = cols / 2;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) {
        quantize_row_f32_to_s4_packed_group(
            src + static_cast<size_t>(r) * static_cast<size_t>(cols),
            cols, group_size,
            dst_packed + static_cast<size_t>(r) * static_cast<size_t>(cols_half),
            dst_scale + static_cast<size_t>(r) * static_cast<size_t>(ng));
    }
}

void HavenTpuGemma4Engine::dequantize_s4_packed_group_to_f32(const int8_t* src_packed, const float* src_scale,
                                                             int rows, int cols, int group_size, float* dst) {
    if (!src_packed || !src_scale || !dst || rows <= 0 || cols <= 0 || (cols % 2) != 0 || group_size <= 0 || (cols % group_size) != 0) {
        return;
    }
    const int ng = cols / group_size;
    const int half_g = group_size / 2;
    const int cols_half = cols / 2;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) {
        const int8_t* prow = src_packed + static_cast<size_t>(r) * static_cast<size_t>(cols_half);
        const float* srow  = src_scale  + static_cast<size_t>(r) * static_cast<size_t>(ng);
        float* drow        = dst        + static_cast<size_t>(r) * static_cast<size_t>(cols);
        for (int g = 0; g < ng; ++g) {
            const float sc = srow[g];
            const int8_t* pgrp = prow + static_cast<size_t>(g) * static_cast<size_t>(half_g);
            float* dgrp = drow + static_cast<size_t>(g) * static_cast<size_t>(group_size);
            for (int p = 0; p < half_g; ++p) {
                const int8_t b = pgrp[p];
                const int8_t lo = static_cast<int8_t>(static_cast<int8_t>(static_cast<uint8_t>(b) << 4) >> 4);
                const int8_t hi = static_cast<int8_t>(b >> 4);
                dgrp[2 * p]     = static_cast<float>(lo) * sc;
                dgrp[2 * p + 1] = static_cast<float>(hi) * sc;
            }
        }
    }
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
    const bool i4 = tc.use_int4_weights;
    const bool i8 = (!i4 && tc.use_int8_weights);
    const bool has_q = (i8 || i4);
    const bool kv_fp8 = tc.use_fp8_kv;
    const bool kv_f32 = (!kv_fp8 && tc.use_f32_kv);
    const bool onchip_emb = tc.use_onchip_embeddings;
    const bool per_slot_pos = (!slot_targeted && S == 1 && B_cap > 1);
    const std::string wel = has_q ? "i8" : "bf16";
    const std::string kv_el = kv_fp8 ? "f8E4M3FN" : (kv_f32 ? "f32" : "bf16");
    auto proj_k = [&](int K) -> int { return i4 ? (K / 2) : K; };
    auto w4_ng  = [&](int K) -> int { return K / compute_w4_group_size(K); };

    const int F_sh = F / R;
    const int V_sh = V / R;
    const int H_sh = H / R;
    const int Hkv_swa = (spec.num_kv_heads_swa > 0) ? spec.num_kv_heads_swa : Hkv;
    const int Hkv_glob = (spec.num_kv_heads_glob > 0) ? spec.num_kv_heads_glob : Hkv;
    const int Hkv_swa_sh = compute_hkv_shard(Hkv_swa, R);
    const int Hkv_glob_sh = compute_hkv_shard(Hkv_glob, R);
    const int G_swa_sh = std::max(1, H_sh / Hkv_swa_sh);
    const int G_glob_sh = std::max(1, H_sh / Hkv_glob_sh);

    const int HD_swa = spec.head_dim_swa;
    const int HD_glob = spec.head_dim_glob;
    const int Q_swa_sh = H_sh * HD_swa;
    const int KV_swa_sh = Hkv_swa_sh * HD_swa;
    const int Q_glob_sh = H_sh * HD_glob;
    const int KV_glob_sh = Hkv_glob_sh * HD_glob;
    const int P_dim = spec.pleias_dim;

    const bool is_moe = spec.is_moe;
    const bool use_moe_hadamard = (is_moe && has_q && (D % 256) == 0);
    const int E = is_moe ? spec.num_experts : 0;
    const int K_exp = is_moe ? spec.num_experts_used : 0;
    const int F_exp_sh = is_moe ? (spec.expert_ffn_dim / R) : 0;
    const int GU_exp_sh = 2 * F_exp_sh;

    const int N_swa = spec.num_swa_layers;
    const int N_glob = spec.num_glob_layers;
    const int N_swa_kv = spec.num_swa_kv_layers;
    const int N_glob_kv = spec.num_glob_kv_layers;
    const bool has_swa = (N_swa > 0);
    const bool has_glob = (N_glob > 0);
    const bool has_swa_kv = (N_swa_kv > 0);
    const bool has_glob_kv = (N_glob_kv > 0);
    const int T_swa = has_swa ? std::max(S, compute_swa_kv_len(spec, tc)) : T;
    const bool swa_ring_mode = has_swa && (T_swa < T);
    const int swa_sinks = has_glob ? 0 : 4;
    const int swa_ring_cap = swa_ring_mode ? std::max(1, T_swa - swa_sinks) : T;

    MlirEmitter em;
    // Per-Layer Coalesced TPU (8,128)-aligned 3D KV Cache buffers: [B_cap, T_kv, KV_sh] (bf16 or scale-free f8E4M3FN)
    const std::string kv_ks_t = "tensor<" + std::to_string(B_cap) + "x" + std::to_string(T_swa) + "x" + std::to_string(KV_swa_sh)  + "x" + kv_el + ">";
    const std::string kv_kg_t = "tensor<" + std::to_string(B_cap) + "x" + std::to_string(T)     + "x" + std::to_string(KV_glob_sh) + "x" + kv_el + ">";
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
    if (S > 1) {
        em.ss << "    %valid_last: tensor<i32>,\n";
    }

    // 1. Stacked projection weights (fused FFN Gate+Up halves dispatch count & activation reads!)
    em.ss << "    %w_ffn_gu: tensor<" << L << "x" << (2 * F_sh) << "x" << proj_k(D) << "x" << wel << ">,\n"
          << "    %w_ffn_down: tensor<" << L << "x" << D << "x" << proj_k(F_sh) << "x" << wel << ">,\n";
    if (is_moe) {
        em.ss << "    %w_moe_router: tensor<" << L << "x" << E << "x" << D << "xbf16>,\n";
        for (int l = 0; l < L; ++l) {
            em.ss << "    %w_moe_gu_" << l << ": tensor<" << E << "x" << GU_exp_sh << "x" << proj_k(D) << "x" << wel << ">,\n";
        }
        for (int l = 0; l < L; ++l) {
            em.ss << "    %w_moe_dn_" << l << ": tensor<" << E << "x" << F_exp_sh << "x" << proj_k(D) << "x" << wel << ">,\n";
        }
        if (use_moe_hadamard) {
            em.ss << "    %w_had_moe: tensor<256x256xbf16>,\n";
        }
    }
    if (has_swa) {
        em.ss << "    %w_swa_q: tensor<" << N_swa << "x" << Q_swa_sh << "x" << proj_k(D) << "x" << wel << ">,\n"
              << "    %w_swa_o: tensor<" << N_swa << "x" << D << "x" << proj_k(Q_swa_sh) << "x" << wel << ">,\n";
        if (has_swa_kv) {
            em.ss << "    %w_swa_k: tensor<" << N_swa_kv << "x" << KV_swa_sh << "x" << proj_k(D) << "x" << wel << ">,\n";
            if (spec.has_swa_v) {
                em.ss << "    %w_swa_v: tensor<" << N_swa_kv << "x" << KV_swa_sh << "x" << proj_k(D) << "x" << wel << ">,\n";
            }
        }
    }
    if (has_glob) {
        em.ss << "    %w_glob_q: tensor<" << N_glob << "x" << Q_glob_sh << "x" << proj_k(D) << "x" << wel << ">,\n"
              << "    %w_glob_o: tensor<" << N_glob << "x" << D << "x" << proj_k(Q_glob_sh) << "x" << wel << ">,\n";
        if (has_glob_kv) {
            em.ss << "    %w_glob_k: tensor<" << N_glob_kv << "x" << KV_glob_sh << "x" << proj_k(D) << "x" << wel << ">,\n";
            if (spec.has_glob_v) {
                em.ss << "    %w_glob_v: tensor<" << N_glob_kv << "x" << KV_glob_sh << "x" << proj_k(D) << "x" << wel << ">,\n";
            }
        }
    }
    // Vocab tables stay W8 INT8 even in W4A16 mode to protect 262k-vocab logit fidelity!
    em.ss << "    %w_lm_head: tensor<" << V_sh << "x" << D << "x" << wel << ">,\n";
    if (onchip_emb && !spec.has_tied_embeddings) {
        em.ss << "    %w_tok_emb: tensor<" << V_sh << "x" << D << "x" << wel << ">,\n";
    }
    if (onchip_emb && spec.has_pleias) {
        em.ss << "    %w_pl_tok_emb: tensor<" << V_sh << "x" << (L * P_dim) << "x" << wel << ">,\n";
    }

    // 2. Optional INT8 per-row or W4A16 per-group F32 scales (exact 1:1 mirror of projection weights)
    if (has_q) {
        auto sc_suffix = [&](int K) -> std::string {
            return i4 ? ("x" + std::to_string(w4_ng(K)) + "xf32>") : "xf32>";
        };
        em.ss << "    %s_ffn_gu: tensor<" << L << "x" << (2 * F_sh) << sc_suffix(D) << ",\n"
              << "    %s_ffn_down: tensor<" << L << "x" << D << sc_suffix(F_sh) << ",\n";
        if (is_moe) {
            for (int l = 0; l < L; ++l) {
                em.ss << "    %s_moe_gu_" << l << ": tensor<" << E << "x" << GU_exp_sh << sc_suffix(D) << ",\n";
            }
            for (int l = 0; l < L; ++l) {
                em.ss << "    %s_moe_dn_" << l << ": tensor<" << E << "x" << F_exp_sh << sc_suffix(D) << ",\n";
            }
        }
        if (has_swa) {
            em.ss << "    %s_swa_q: tensor<" << N_swa << "x" << Q_swa_sh << sc_suffix(D) << ",\n"
                  << "    %s_swa_o: tensor<" << N_swa << "x" << D << sc_suffix(Q_swa_sh) << ",\n";
            if (has_swa_kv) {
                em.ss << "    %s_swa_k: tensor<" << N_swa_kv << "x" << KV_swa_sh << sc_suffix(D) << ",\n";
                if (spec.has_swa_v) {
                    em.ss << "    %s_swa_v: tensor<" << N_swa_kv << "x" << KV_swa_sh << sc_suffix(D) << ",\n";
                }
            }
        }
        if (has_glob) {
            em.ss << "    %s_glob_q: tensor<" << N_glob << "x" << Q_glob_sh << sc_suffix(D) << ",\n"
                  << "    %s_glob_o: tensor<" << N_glob << "x" << D << sc_suffix(Q_glob_sh) << ",\n";
            if (has_glob_kv) {
                em.ss << "    %s_glob_k: tensor<" << N_glob_kv << "x" << KV_glob_sh << sc_suffix(D) << ",\n";
                if (spec.has_glob_v) {
                    em.ss << "    %s_glob_v: tensor<" << N_glob_kv << "x" << KV_glob_sh << sc_suffix(D) << ",\n";
                }
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
                em.ss << "    %b_swa_k: tensor<" << N_swa_kv << "x" << KV_swa_sh << "xf32>,\n";
                if (spec.has_swa_v) {
                    em.ss << "    %b_swa_v: tensor<" << N_swa_kv << "x" << KV_swa_sh << "xf32>,\n";
                }
            }
        }
        if (has_glob) {
            em.ss << "    %b_glob_q: tensor<" << N_glob << "x" << Q_glob_sh << "xf32>,\n";
            if (has_glob_kv) {
                em.ss << "    %b_glob_k: tensor<" << N_glob_kv << "x" << KV_glob_sh << "xf32>,\n";
                if (spec.has_glob_v) {
                    em.ss << "    %b_glob_v: tensor<" << N_glob_kv << "x" << KV_glob_sh << "xf32>,\n";
                }
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
    if (is_moe && spec.has_moe_router_scale) {
        em.ss << "    %w_moe_rscale: tensor<" << L << "x" << D << "xf32>,\n";
    }
    if (is_moe && spec.has_post_ffn_norm_1) {
        em.ss << "    %w_post_ffn_norm_1: tensor<" << L << "x" << D << "xf32>,\n";
    }
    if (is_moe && spec.has_pre_ffn_norm_2) {
        em.ss << "    %w_pre_ffn_norm_2: tensor<" << L << "x" << D << "xf32>,\n";
    }
    if (is_moe && spec.has_post_ffn_norm_2) {
        em.ss << "    %w_post_ffn_norm_2: tensor<" << L << "x" << D << "xf32>,\n";
    }
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

    // 7. Mutable per-layer donated 3D HBM3 KV Cache buffers (coalesced [B_cap, T_kv, KV_sh] in bf16 or scale-free f8E4M3FN)
    int alias_slot = 1;
    std::vector<std::string> kv_ret_types;
    const int total_kv_pairs = N_swa_kv + N_glob_kv;
    int kv_pair_emitted = 0;
    for (int i = 0; i < N_swa_kv; ++i) {
        const int a_ks = alias_slot++;
        const int a_vs = alias_slot++;
        kv_pair_emitted++;
        const bool more = (kv_pair_emitted < total_kv_pairs);
        em.ss << "    %kv_ks_" << i << ": " << kv_ks_t << donate_attr(a_ks) << ",\n"
              << "    %kv_vs_" << i << ": " << kv_ks_t << donate_attr(a_vs) << (more ? ",\n" : "\n");
        kv_ret_types.push_back(kv_ks_t);
        kv_ret_types.push_back(kv_ks_t);
    }
    for (int i = 0; i < N_glob_kv; ++i) {
        const int a_kg = alias_slot++;
        const int a_vg = alias_slot++;
        kv_pair_emitted++;
        const bool more = (kv_pair_emitted < total_kv_pairs);
        em.ss << "    %kv_kg_" << i << ": " << kv_kg_t << donate_attr(a_kg) << ",\n"
              << "    %kv_vg_" << i << ": " << kv_kg_t << donate_attr(a_vg) << (more ? ",\n" : "\n");
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

    // Pre-slice/gather RoPE cos/sin AND build Causal + Sink-Preserving SWA masks + 4D KV scatter indices once!
    std::string cos_swa_var, sin_swa_var, cos_glob_var, sin_glob_var;
    std::string mask_glob_t_i1, mask_swa_t_i1;
    std::string sc_idx_swa_var = "%sc_idx_glob";
    if (per_slot_pos) {
        // Independent per-slot positions: %pos_vec: tensor<B_cap x i32>
        const std::string t_b_i32  = "tensor<" + std::to_string(B_cap) + "xi32>";
        const std::string t_b_i1   = "tensor<" + std::to_string(B_cap) + "xi1>";
        const std::string t_b1_i32 = "tensor<" + std::to_string(B_cap) + "x1xi32>";
        const std::string t_b2_i32 = "tensor<" + std::to_string(B_cap) + "x2xi32>";
        const std::string t_bt_i32 = "tensor<" + std::to_string(B_cap) + "x" + std::to_string(T) + "xi32>";
        mask_glob_t_i1 = "tensor<" + std::to_string(B_cap) + "x" + std::to_string(T) + "xi1>";
        mask_swa_t_i1  = "tensor<" + std::to_string(B_cap) + "x" + std::to_string(T_swa) + "xi1>";

        em.ss << "    %pos_col = stablehlo.reshape %pos_vec : (" << t_b_i32 << ") -> " << t_b1_i32 << "\n"
              << "    %b_iota_1d = stablehlo.iota dim = 0 : " << t_b_i32 << "\n"
              << "    %b_iota_col = stablehlo.reshape %b_iota_1d : (" << t_b_i32 << ") -> " << t_b1_i32 << "\n"
              << "    %sc_idx_glob = stablehlo.concatenate %b_iota_col, %pos_col, dim = 1 : (" << t_b1_i32 << ", " << t_b1_i32 << ") -> " << t_b2_i32 << "\n";

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
              << "    %q_bt = stablehlo.broadcast_in_dim %pos_vec, dims = [0] : (" << t_b_i32 << ") -> " << t_bt_i32 << "\n"
              << "    %mask_glob = stablehlo.compare LE, %k_bt, %q_bt, SIGNED : (" << t_bt_i32 << ", " << t_bt_i32 << ") -> " << mask_glob_t_i1 << "\n";
        if (has_swa) {
            if (!swa_ring_mode) {
                em.ss << "    %c_win = stablehlo.constant dense<" << std::max(0, spec.sliding_window - 1) << "> : " << t_bt_i32 << "\n"
                      << "    %win_lo = stablehlo.subtract %q_bt, %c_win : " << t_bt_i32 << "\n"
                      << "    %in_win = stablehlo.compare GE, %k_bt, %win_lo, SIGNED : (" << t_bt_i32 << ", " << t_bt_i32 << ") -> " << mask_swa_t_i1 << "\n";
                if (swa_sinks > 0) {
                    em.ss << "    %c_sink = stablehlo.constant dense<" << swa_sinks << "> : " << t_bt_i32 << "\n"
                          << "    %is_sink = stablehlo.compare LT, %k_bt, %c_sink, SIGNED : (" << t_bt_i32 << ", " << t_bt_i32 << ") -> " << mask_swa_t_i1 << "\n"
                          << "    %win_or_sink = stablehlo.or %in_win, %is_sink : " << mask_swa_t_i1 << "\n"
                          << "    %mask_swa = stablehlo.and %mask_glob, %win_or_sink : " << mask_swa_t_i1 << "\n";
                } else {
                    em.ss << "    %mask_swa = stablehlo.and %mask_glob, %in_win : " << mask_swa_t_i1 << "\n";
                }
            } else if (swa_sinks == 0) {
                // Exact Sink-Free Circular SWA Ring Buffer: slot(p) = p % T_swa (exact power-of-2 bitmask when T_swa=1024!)
                const std::string t_bts_i32 = "tensor<" + std::to_string(B_cap) + "x" + std::to_string(T_swa) + "xi32>";
                em.ss << "    %crc_b = stablehlo.constant dense<" << swa_ring_cap << "> : " << t_b_i32 << "\n"
                      << "    %p_rem = stablehlo.remainder %pos_vec, %crc_b : " << t_b_i32 << "\n"
                      << "    %swa_pos_col = stablehlo.reshape %p_rem : (" << t_b_i32 << ") -> " << t_b1_i32 << "\n"
                      << "    %sc_idx_swa = stablehlo.concatenate %b_iota_col, %swa_pos_col, dim = 1 : (" << t_b1_i32 << ", " << t_b1_i32 << ") -> " << t_b2_i32 << "\n"
                      << "    %ks_iota = stablehlo.iota dim = 0 : tensor<" << T_swa << "xi32>\n"
                      << "    %ks_bt = stablehlo.broadcast_in_dim %ks_iota, dims = [1] : (tensor<" << T_swa << "xi32>) -> " << t_bts_i32 << "\n"
                      << "    %qs_bt = stablehlo.broadcast_in_dim %pos_vec, dims = [0] : (" << t_b_i32 << ") -> " << t_bts_i32 << "\n"
                      << "    %qr_bt = stablehlo.broadcast_in_dim %p_rem, dims = [0] : (" << t_b_i32 << ") -> " << t_bts_i32 << "\n"
                      << "    %c0_bts = stablehlo.constant dense<0> : " << t_bts_i32 << "\n"
                      << "    %crc_bts = stablehlo.constant dense<" << swa_ring_cap << "> : " << t_bts_i32 << "\n"
                      << "    %cwin_bts = stablehlo.constant dense<" << std::max(0, spec.sliding_window - 1) << "> : " << t_bts_i32 << "\n"
                      << "    %r_diff = stablehlo.subtract %qr_bt, %ks_bt : " << t_bts_i32 << "\n"
                      << "    %r_neg = stablehlo.compare LT, %r_diff, %c0_bts, SIGNED : (" << t_bts_i32 << ", " << t_bts_i32 << ") -> " << mask_swa_t_i1 << "\n"
                      << "    %r_wrap = stablehlo.add %r_diff, %crc_bts : " << t_bts_i32 << "\n"
                      << "    %r_dist = stablehlo.select %r_neg, %r_wrap, %r_diff : " << mask_swa_t_i1 << ", " << t_bts_i32 << "\n"
                      << "    %p_stored = stablehlo.subtract %qs_bt, %r_dist : " << t_bts_i32 << "\n"
                      << "    %ring_ge0 = stablehlo.compare GE, %p_stored, %c0_bts, SIGNED : (" << t_bts_i32 << ", " << t_bts_i32 << ") -> " << mask_swa_t_i1 << "\n"
                      << "    %ring_in_win = stablehlo.compare LE, %r_dist, %cwin_bts, SIGNED : (" << t_bts_i32 << ", " << t_bts_i32 << ") -> " << mask_swa_t_i1 << "\n"
                      << "    %mask_swa = stablehlo.and %ring_ge0, %ring_in_win : " << mask_swa_t_i1 << "\n";
                sc_idx_swa_var = "%sc_idx_swa";
            } else {
                // Exact 4-Sink-Preserving SWA Ring Buffer: slot(p) = (p < 4) ? p : 4 + ((p - 4) % swa_ring_cap)
                const std::string t_bts_i32 = "tensor<" + std::to_string(B_cap) + "x" + std::to_string(T_swa) + "xi32>";
                em.ss << "    %c4_b = stablehlo.constant dense<4> : " << t_b_i32 << "\n"
                      << "    %c0_b = stablehlo.constant dense<0> : " << t_b_i32 << "\n"
                      << "    %crc_b = stablehlo.constant dense<" << swa_ring_cap << "> : " << t_b_i32 << "\n"
                      << "    %p_lt4 = stablehlo.compare LT, %pos_vec, %c4_b, SIGNED : (" << t_b_i32 << ", " << t_b_i32 << ") -> " << t_b_i1 << "\n"
                      << "    %p_m4 = stablehlo.subtract %pos_vec, %c4_b : " << t_b_i32 << "\n"
                      << "    %p_m4p = stablehlo.maximum %p_m4, %c0_b : " << t_b_i32 << "\n"
                      << "    %p_rem = stablehlo.remainder %p_m4p, %crc_b : " << t_b_i32 << "\n"
                      << "    %p_ring = stablehlo.add %p_rem, %c4_b : " << t_b_i32 << "\n"
                      << "    %swa_pos_vec = stablehlo.select %p_lt4, %pos_vec, %p_ring : " << t_b_i1 << ", " << t_b_i32 << "\n"
                      << "    %swa_pos_col = stablehlo.reshape %swa_pos_vec : (" << t_b_i32 << ") -> " << t_b1_i32 << "\n"
                      << "    %sc_idx_swa = stablehlo.concatenate %b_iota_col, %swa_pos_col, dim = 1 : (" << t_b1_i32 << ", " << t_b1_i32 << ") -> " << t_b2_i32 << "\n"
                      << "    %ks_iota = stablehlo.iota dim = 0 : tensor<" << T_swa << "xi32>\n"
                      << "    %ks_bt = stablehlo.broadcast_in_dim %ks_iota, dims = [1] : (tensor<" << T_swa << "xi32>) -> " << t_bts_i32 << "\n"
                      << "    %qs_bt = stablehlo.broadcast_in_dim %pos_vec, dims = [0] : (" << t_b_i32 << ") -> " << t_bts_i32 << "\n"
                      << "    %qr_bt = stablehlo.broadcast_in_dim %p_rem, dims = [0] : (" << t_b_i32 << ") -> " << t_bts_i32 << "\n"
                      << "    %c4_bts = stablehlo.constant dense<4> : " << t_bts_i32 << "\n"
                      << "    %c0_bts = stablehlo.constant dense<0> : " << t_bts_i32 << "\n"
                      << "    %crc_bts = stablehlo.constant dense<" << swa_ring_cap << "> : " << t_bts_i32 << "\n"
                      << "    %cwin_bts = stablehlo.constant dense<" << std::max(0, spec.sliding_window - 1) << "> : " << t_bts_i32 << "\n"
                      << "    %is_sink = stablehlo.compare LT, %ks_bt, %c4_bts, SIGNED : (" << t_bts_i32 << ", " << t_bts_i32 << ") -> " << mask_swa_t_i1 << "\n"
                      << "    %sink_ok = stablehlo.compare LE, %ks_bt, %qs_bt, SIGNED : (" << t_bts_i32 << ", " << t_bts_i32 << ") -> " << mask_swa_t_i1 << "\n"
                      << "    %sink_valid = stablehlo.and %is_sink, %sink_ok : " << mask_swa_t_i1 << "\n"
                      << "    %j_ring = stablehlo.subtract %ks_bt, %c4_bts : " << t_bts_i32 << "\n"
                      << "    %r_diff = stablehlo.subtract %qr_bt, %j_ring : " << t_bts_i32 << "\n"
                      << "    %r_neg = stablehlo.compare LT, %r_diff, %c0_bts, SIGNED : (" << t_bts_i32 << ", " << t_bts_i32 << ") -> " << mask_swa_t_i1 << "\n"
                      << "    %r_wrap = stablehlo.add %r_diff, %crc_bts : " << t_bts_i32 << "\n"
                      << "    %r_dist = stablehlo.select %r_neg, %r_wrap, %r_diff : " << mask_swa_t_i1 << ", " << t_bts_i32 << "\n"
                      << "    %p_stored = stablehlo.subtract %qs_bt, %r_dist : " << t_bts_i32 << "\n"
                      << "    %ring_ge4 = stablehlo.compare GE, %p_stored, %c4_bts, SIGNED : (" << t_bts_i32 << ", " << t_bts_i32 << ") -> " << mask_swa_t_i1 << "\n"
                      << "    %ring_in_win = stablehlo.compare LE, %r_dist, %cwin_bts, SIGNED : (" << t_bts_i32 << ", " << t_bts_i32 << ") -> " << mask_swa_t_i1 << "\n"
                      << "    %not_sink = stablehlo.compare GE, %ks_bt, %c4_bts, SIGNED : (" << t_bts_i32 << ", " << t_bts_i32 << ") -> " << mask_swa_t_i1 << "\n"
                      << "    %ring_v1 = stablehlo.and %not_sink, %ring_ge4 : " << mask_swa_t_i1 << "\n"
                      << "    %ring_valid = stablehlo.and %ring_v1, %ring_in_win : " << mask_swa_t_i1 << "\n"
                      << "    %mask_swa = stablehlo.or %sink_valid, %ring_valid : " << mask_swa_t_i1 << "\n";
                sc_idx_swa_var = "%sc_idx_swa";
            }
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
        mask_glob_t_i1 = "tensor<" + std::to_string(S) + "x" + std::to_string(T) + "xi1>";
        mask_swa_t_i1  = "tensor<" + std::to_string(S) + "x" + std::to_string(T_swa) + "xi1>";
        em.ss << "    %k_iota = stablehlo.iota dim = 0 : tensor<" << T << "xi32>\n"
              << "    %q_iota = stablehlo.iota dim = 0 : tensor<" << S << "xi32>\n"
              << "    %pos_s = stablehlo.broadcast_in_dim %pos, dims = [] : (tensor<i32>) -> tensor<" << S << "xi32>\n"
              << "    %q_pos = stablehlo.add %q_iota, %pos_s : tensor<" << S << "xi32>\n"
              << "    %k_st = stablehlo.broadcast_in_dim %k_iota, dims = [1] : (tensor<" << T << "xi32>) -> " << t_st_i32 << "\n"
              << "    %q_st = stablehlo.broadcast_in_dim %q_pos, dims = [0] : (tensor<" << S << "xi32>) -> " << t_st_i32 << "\n"
              << "    %mask_glob = stablehlo.compare LE, %k_st, %q_st, SIGNED : (" << t_st_i32 << ", " << t_st_i32 << ") -> " << mask_glob_t_i1 << "\n";
        if (has_swa) {
            if (!swa_ring_mode) {
                em.ss << "    %c_win = stablehlo.constant dense<" << std::max(0, spec.sliding_window - 1) << "> : " << t_st_i32 << "\n"
                      << "    %win_lo = stablehlo.subtract %q_st, %c_win : " << t_st_i32 << "\n"
                      << "    %in_win = stablehlo.compare GE, %k_st, %win_lo, SIGNED : (" << t_st_i32 << ", " << t_st_i32 << ") -> " << mask_swa_t_i1 << "\n";
                if (swa_sinks > 0) {
                    em.ss << "    %c_sink = stablehlo.constant dense<" << swa_sinks << "> : " << t_st_i32 << "\n"
                          << "    %is_sink = stablehlo.compare LT, %k_st, %c_sink, SIGNED : (" << t_st_i32 << ", " << t_st_i32 << ") -> " << mask_swa_t_i1 << "\n"
                          << "    %win_or_sink = stablehlo.or %in_win, %is_sink : " << mask_swa_t_i1 << "\n"
                          << "    %mask_swa = stablehlo.and %mask_glob, %win_or_sink : " << mask_swa_t_i1 << "\n";
                } else {
                    em.ss << "    %mask_swa = stablehlo.and %mask_glob, %in_win : " << mask_swa_t_i1 << "\n";
                }
            } else if (swa_sinks == 0) {
                // Exact Sink-Free Circular SWA Ring Buffer for S=1 decode and S-chunk prefill
                const std::string t_s_i32   = "tensor<" + std::to_string(S) + "xi32>";
                const std::string t_s1_i32  = "tensor<" + std::to_string(S) + "x1xi32>";
                const std::string t_s2_i32  = "tensor<" + std::to_string(S) + "x2xi32>";
                const std::string t_ts_i32  = "tensor<" + std::to_string(T_swa) + "xi32>";
                const std::string t_ts_i1   = "tensor<" + std::to_string(T_swa) + "xi1>";
                const std::string t_sts_i32 = "tensor<" + std::to_string(S) + "x" + std::to_string(T_swa) + "xi32>";

                em.ss << "    %crc_s = stablehlo.constant dense<" << swa_ring_cap << "> : " << t_s_i32 << "\n"
                      << "    %swa_write_pos = stablehlo.remainder %q_pos, %crc_s : " << t_s_i32 << "\n";
                if (S == 1) {
                    em.ss << "    %swa_write_pos_0 = stablehlo.reshape %swa_write_pos : (tensor<1xi32>) -> tensor<i32>\n";
                } else {
                    em.ss << "    %b_idx_s1 = stablehlo.broadcast_in_dim " << c0_i32 << ", dims = [] : (tensor<i32>) -> " << t_s1_i32 << "\n"
                          << "    %swa_wp_col = stablehlo.reshape %swa_write_pos : (" << t_s_i32 << ") -> " << t_s1_i32 << "\n"
                          << "    %sc_idx_swa_pre = stablehlo.concatenate %b_idx_s1, %swa_wp_col, dim = 1 : (" << t_s1_i32 << ", " << t_s1_i32 << ") -> " << t_s2_i32 << "\n";
                }
                em.ss << "    %c_sm1 = stablehlo.constant dense<" << (S - 1) << "> : tensor<i32>\n"
                      << "    %max_q = stablehlo.add %pos, %c_sm1 : tensor<i32>\n"
                      << "    %crc_0 = stablehlo.constant dense<" << swa_ring_cap << "> : tensor<i32>\n"
                      << "    %mq_rem = stablehlo.remainder %max_q, %crc_0 : tensor<i32>\n"
                      << "    %ks_iota = stablehlo.iota dim = 0 : " << t_ts_i32 << "\n"
                      << "    %c0_ts = stablehlo.constant dense<0> : " << t_ts_i32 << "\n"
                      << "    %crc_ts = stablehlo.constant dense<" << swa_ring_cap << "> : " << t_ts_i32 << "\n"
                      << "    %mq_ts = stablehlo.broadcast_in_dim %max_q, dims = [] : (tensor<i32>) -> " << t_ts_i32 << "\n"
                      << "    %mqr_ts = stablehlo.broadcast_in_dim %mq_rem, dims = [] : (tensor<i32>) -> " << t_ts_i32 << "\n"
                      << "    %r_diff = stablehlo.subtract %mqr_ts, %ks_iota : " << t_ts_i32 << "\n"
                      << "    %r_neg = stablehlo.compare LT, %r_diff, %c0_ts, SIGNED : (" << t_ts_i32 << ", " << t_ts_i32 << ") -> " << t_ts_i1 << "\n"
                      << "    %r_wrap = stablehlo.add %r_diff, %crc_ts : " << t_ts_i32 << "\n"
                      << "    %r_dist = stablehlo.select %r_neg, %r_wrap, %r_diff : " << t_ts_i1 << ", " << t_ts_i32 << "\n"
                      << "    %p_st_1d = stablehlo.subtract %mq_ts, %r_dist : " << t_ts_i32 << "\n"
                      << "    %p_st = stablehlo.broadcast_in_dim %p_st_1d, dims = [1] : (" << t_ts_i32 << ") -> " << t_sts_i32 << "\n"
                      << "    %q_st_s = stablehlo.broadcast_in_dim %q_pos, dims = [0] : (" << t_s_i32 << ") -> " << t_sts_i32 << "\n"
                      << "    %c0_sts = stablehlo.constant dense<0> : " << t_sts_i32 << "\n"
                      << "    %cwin_sts = stablehlo.constant dense<" << std::max(0, spec.sliding_window - 1) << "> : " << t_sts_i32 << "\n"
                      << "    %swa_causal = stablehlo.compare LE, %p_st, %q_st_s, SIGNED : (" << t_sts_i32 << ", " << t_sts_i32 << ") -> " << mask_swa_t_i1 << "\n"
                      << "    %p_ge0 = stablehlo.compare GE, %p_st, %c0_sts, SIGNED : (" << t_sts_i32 << ", " << t_sts_i32 << ") -> " << mask_swa_t_i1 << "\n"
                      << "    %win_lo = stablehlo.subtract %q_st_s, %cwin_sts : " << t_sts_i32 << "\n"
                      << "    %in_win = stablehlo.compare GE, %p_st, %win_lo, SIGNED : (" << t_sts_i32 << ", " << t_sts_i32 << ") -> " << mask_swa_t_i1 << "\n"
                      << "    %r1_ok = stablehlo.and %p_ge0, %swa_causal : " << mask_swa_t_i1 << "\n"
                      << "    %mask_swa = stablehlo.and %r1_ok, %in_win : " << mask_swa_t_i1 << "\n";
            } else {
                // Exact 4-Sink-Preserving SWA Ring Buffer for S=1 decode and S-chunk prefill!
                const std::string t_s_i32   = "tensor<" + std::to_string(S) + "xi32>";
                const std::string t_s_i1    = "tensor<" + std::to_string(S) + "xi1>";
                const std::string t_s1_i32  = "tensor<" + std::to_string(S) + "x1xi32>";
                const std::string t_s2_i32  = "tensor<" + std::to_string(S) + "x2xi32>";
                const std::string t_ts_i32  = "tensor<" + std::to_string(T_swa) + "xi32>";
                const std::string t_ts_i1   = "tensor<" + std::to_string(T_swa) + "xi1>";
                const std::string t_sts_i32 = "tensor<" + std::to_string(S) + "x" + std::to_string(T_swa) + "xi32>";

                em.ss << "    %c4_s = stablehlo.constant dense<4> : " << t_s_i32 << "\n"
                      << "    %c0_s = stablehlo.constant dense<0> : " << t_s_i32 << "\n"
                      << "    %crc_s = stablehlo.constant dense<" << swa_ring_cap << "> : " << t_s_i32 << "\n"
                      << "    %qp_lt4 = stablehlo.compare LT, %q_pos, %c4_s, SIGNED : (" << t_s_i32 << ", " << t_s_i32 << ") -> " << t_s_i1 << "\n"
                      << "    %qp_m4 = stablehlo.subtract %q_pos, %c4_s : " << t_s_i32 << "\n"
                      << "    %qp_m4p = stablehlo.maximum %qp_m4, %c0_s : " << t_s_i32 << "\n"
                      << "    %qp_rem = stablehlo.remainder %qp_m4p, %crc_s : " << t_s_i32 << "\n"
                      << "    %qp_ring = stablehlo.add %qp_rem, %c4_s : " << t_s_i32 << "\n"
                      << "    %swa_write_pos = stablehlo.select %qp_lt4, %q_pos, %qp_ring : " << t_s_i1 << ", " << t_s_i32 << "\n";
                if (S == 1) {
                    em.ss << "    %swa_write_pos_0 = stablehlo.reshape %swa_write_pos : (tensor<1xi32>) -> tensor<i32>\n";
                } else {
                    // Always scatter into 1-slot [1, T_swa, KV_sh] buffer (batch index 0) for zero-copy HBM3 updates!
                    em.ss << "    %b_idx_s1 = stablehlo.broadcast_in_dim " << c0_i32 << ", dims = [] : (tensor<i32>) -> " << t_s1_i32 << "\n"
                          << "    %swa_wp_col = stablehlo.reshape %swa_write_pos : (" << t_s_i32 << ") -> " << t_s1_i32 << "\n"
                          << "    %sc_idx_swa_pre = stablehlo.concatenate %b_idx_s1, %swa_wp_col, dim = 1 : (" << t_s1_i32 << ", " << t_s1_i32 << ") -> " << t_s2_i32 << "\n";
                }
                em.ss << "    %c_sm1 = stablehlo.constant dense<" << (S - 1) << "> : tensor<i32>\n"
                      << "    %max_q = stablehlo.add %pos, %c_sm1 : tensor<i32>\n"
                      << "    %c4_0 = stablehlo.constant dense<4> : tensor<i32>\n"
                      << "    %crc_0 = stablehlo.constant dense<" << swa_ring_cap << "> : tensor<i32>\n"
                      << "    %mq_m4 = stablehlo.subtract %max_q, %c4_0 : tensor<i32>\n"
                      << "    %mq_m4p = stablehlo.maximum %mq_m4, " << c0_i32 << " : tensor<i32>\n"
                      << "    %mq_rem = stablehlo.remainder %mq_m4p, %crc_0 : tensor<i32>\n"
                      << "    %ks_iota = stablehlo.iota dim = 0 : " << t_ts_i32 << "\n"
                      << "    %c4_ts = stablehlo.constant dense<4> : " << t_ts_i32 << "\n"
                      << "    %c0_ts = stablehlo.constant dense<0> : " << t_ts_i32 << "\n"
                      << "    %crc_ts = stablehlo.constant dense<" << swa_ring_cap << "> : " << t_ts_i32 << "\n"
                      << "    %mq_ts = stablehlo.broadcast_in_dim %max_q, dims = [] : (tensor<i32>) -> " << t_ts_i32 << "\n"
                      << "    %mqr_ts = stablehlo.broadcast_in_dim %mq_rem, dims = [] : (tensor<i32>) -> " << t_ts_i32 << "\n"
                      << "    %j_ring = stablehlo.subtract %ks_iota, %c4_ts : " << t_ts_i32 << "\n"
                      << "    %r_diff = stablehlo.subtract %mqr_ts, %j_ring : " << t_ts_i32 << "\n"
                      << "    %r_neg = stablehlo.compare LT, %r_diff, %c0_ts, SIGNED : (" << t_ts_i32 << ", " << t_ts_i32 << ") -> " << t_ts_i1 << "\n"
                      << "    %r_wrap = stablehlo.add %r_diff, %crc_ts : " << t_ts_i32 << "\n"
                      << "    %r_dist = stablehlo.select %r_neg, %r_wrap, %r_diff : " << t_ts_i1 << ", " << t_ts_i32 << "\n"
                      << "    %p_ring_st = stablehlo.subtract %mq_ts, %r_dist : " << t_ts_i32 << "\n"
                      << "    %j_lt4 = stablehlo.compare LT, %ks_iota, %c4_ts, SIGNED : (" << t_ts_i32 << ", " << t_ts_i32 << ") -> " << t_ts_i1 << "\n"
                      << "    %p_st_1d = stablehlo.select %j_lt4, %ks_iota, %p_ring_st : " << t_ts_i1 << ", " << t_ts_i32 << "\n"
                      << "    %p_st = stablehlo.broadcast_in_dim %p_st_1d, dims = [1] : (" << t_ts_i32 << ") -> " << t_sts_i32 << "\n"
                      << "    %q_st_s = stablehlo.broadcast_in_dim %q_pos, dims = [0] : (" << t_s_i32 << ") -> " << t_sts_i32 << "\n"
                      << "    %j_lt4_st = stablehlo.broadcast_in_dim %j_lt4, dims = [1] : (" << t_ts_i1 << ") -> " << mask_swa_t_i1 << "\n"
                      << "    %c4_sts = stablehlo.constant dense<4> : " << t_sts_i32 << "\n"
                      << "    %cwin_sts = stablehlo.constant dense<" << std::max(0, spec.sliding_window - 1) << "> : " << t_sts_i32 << "\n"
                      << "    %swa_causal = stablehlo.compare LE, %p_st, %q_st_s, SIGNED : (" << t_sts_i32 << ", " << t_sts_i32 << ") -> " << mask_swa_t_i1 << "\n"
                      << "    %sink_ok = stablehlo.and %j_lt4_st, %swa_causal : " << mask_swa_t_i1 << "\n"
                      << "    %p_ge4 = stablehlo.compare GE, %p_st, %c4_sts, SIGNED : (" << t_sts_i32 << ", " << t_sts_i32 << ") -> " << mask_swa_t_i1 << "\n"
                      << "    %win_lo = stablehlo.subtract %q_st_s, %cwin_sts : " << t_sts_i32 << "\n"
                      << "    %in_win = stablehlo.compare GE, %p_st, %win_lo, SIGNED : (" << t_sts_i32 << ", " << t_sts_i32 << ") -> " << mask_swa_t_i1 << "\n"
                      << "    %not_sink = stablehlo.not %j_lt4_st : " << mask_swa_t_i1 << "\n"
                      << "    %r1_ok = stablehlo.and %not_sink, %p_ge4 : " << mask_swa_t_i1 << "\n"
                      << "    %r2_ok = stablehlo.and %r1_ok, %swa_causal : " << mask_swa_t_i1 << "\n"
                      << "    %ring_ok = stablehlo.and %r2_ok, %in_win : " << mask_swa_t_i1 << "\n"
                      << "    %mask_swa = stablehlo.or %sink_ok, %ring_ok : " << mask_swa_t_i1 << "\n";
            }
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
        std::string emb_loc = em.emit_table_gather_2d(emb_tbl, emb_scl, "%idx_3d", V_sh, D, B, S, has_q);
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
            std::string pl_loc = em.emit_table_gather_2d("%w_pl_tok_emb", "%s_pl_tok_emb", "%idx_3d", V_sh, L * P_dim, B, S, has_q);
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
              << "    %c_pl_pscale = stablehlo.constant dense<" << fmt_f32(static_cast<float>(1.0 / std::sqrt(static_cast<double>(D)))) << "> : " << t_pl_flat << "\n"
              << "    %pl_proj_sc = stablehlo.multiply %pl_proj_raw, %c_pl_pscale : " << t_pl_flat << "\n"
              << "    %pl_proj_4d = stablehlo.reshape %pl_proj_sc : (" << t_pl_flat << ") -> " << t_pl_f32 << "\n";
        std::string pl_proj_norm = em.emit_rms_norm_4d("%pl_proj_4d", "%w_pl_pnorm", B, S, L, P_dim, spec.rms_norm_eps, c0_f32);
        em.ss << "    %pl_sum = stablehlo.add " << pl_proj_norm << ", " << pl_tok_f32_var << " : " << t_pl_f32 << "\n"
              << "    %c_inv_s2 = stablehlo.constant dense<" << fmt_f32(static_cast<float>(1.0 / std::sqrt(2.0))) << "> : " << t_pl_f32 << "\n"
              << "    %pleias_all = stablehlo.multiply %pl_sum, %c_inv_s2 : " << t_pl_f32 << "\n";
    }

    std::string x_cur = x0_f32_var;
    std::vector<std::string> kv_ks(static_cast<size_t>(N_swa_kv));
    std::vector<std::string> kv_vs(static_cast<size_t>(N_swa_kv));
    for (int i = 0; i < N_swa_kv; ++i) {
        kv_ks[static_cast<size_t>(i)] = "%kv_ks_" + std::to_string(i);
        kv_vs[static_cast<size_t>(i)] = "%kv_vs_" + std::to_string(i);
    }
    std::vector<std::string> kv_kg(static_cast<size_t>(N_glob_kv));
    std::vector<std::string> kv_vg(static_cast<size_t>(N_glob_kv));
    for (int i = 0; i < N_glob_kv; ++i) {
        kv_kg[static_cast<size_t>(i)] = "%kv_kg_" + std::to_string(i);
        kv_vg[static_cast<size_t>(i)] = "%kv_vg_" + std::to_string(i);
    }
    // Cached isolated 1-slot [1, T_kv, KV_sh] views when slot_targeted == true
    std::vector<std::string> slot_ks(static_cast<size_t>(N_swa_kv));
    std::vector<std::string> slot_vs(static_cast<size_t>(N_swa_kv));
    std::vector<std::string> slot_kg(static_cast<size_t>(N_glob_kv));
    std::vector<std::string> slot_vg(static_cast<size_t>(N_glob_kv));

    // Step 2: Unroll all L transformer layers with exact static shapes
    for (int l = 0; l < L; ++l) {
        const bool is_swa = spec.layer_is_swa[static_cast<size_t>(l)];
        const bool has_kv = spec.layer_has_kv[static_cast<size_t>(l)];
        const int ti = spec.layer_type_idx[static_cast<size_t>(l)];
        const int kvi = has_kv ? spec.layer_kv_idx[static_cast<size_t>(l)]
                               : (is_swa ? spec.last_swa_kv_idx : spec.last_glob_kv_idx);
        const int HD = is_swa ? HD_swa : HD_glob;
        const int T_kv = is_swa ? T_swa : T;
        const int Q_sh = is_swa ? Q_swa_sh : Q_glob_sh;
        const int Hkv_sh = is_swa ? Hkv_swa_sh : Hkv_glob_sh;
        const int G_sh = is_swa ? G_swa_sh : G_glob_sh;
        const int KV_sh = is_swa ? KV_swa_sh : KV_glob_sh;
        const bool layer_has_v = is_swa ? spec.has_swa_v : spec.has_glob_v;
        const int N_type = is_swa ? N_swa : N_glob;
        const int N_kv_type = is_swa ? N_swa_kv : N_glob_kv;
        const std::string& cos_s = is_swa ? cos_swa_var : cos_glob_var;
        const std::string& sin_s = is_swa ? sin_swa_var : sin_glob_var;
        const std::string& mask_t_layer = is_swa ? mask_swa_t_i1 : mask_glob_t_i1;
        const float attn_scale = spec.is_gemma ? 1.0f : static_cast<float>(1.0 / std::sqrt(static_cast<double>(HD)));

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
            ti, N_type, B, S, Q_sh, D, i8, i4);
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

        // 2c. K, V Projections (+ optional bias) + optional K/V-Norm + RoPE + Zero-Copy Per-Layer 3D KV Update (BF16 or Scale-Free FP8 E4M3FN)
        const std::string& kv_buf_t = is_swa ? kv_ks_t : kv_kg_t;
        const std::string t_kv3_t   = "tensor<" + std::to_string(B) + "x" + std::to_string(T_kv) + "x" + std::to_string(KV_sh) + "x" + kv_el + ">";

        if (has_kv) {
            const std::string& kv_k_in = is_swa ? kv_ks[static_cast<size_t>(kvi)] : kv_kg[static_cast<size_t>(kvi)];
            const std::string& kv_v_in = is_swa ? kv_vs[static_cast<size_t>(kvi)] : kv_vg[static_cast<size_t>(kvi)];

            std::string k_flat = em.emit_linear_3d(x_an_bf,
                is_swa ? "%w_swa_k" : "%w_glob_k",
                is_swa ? "%s_swa_k" : "%s_glob_k",
                kvi, N_kv_type, B, S, KV_sh, D, i8, i4);
            if (spec.has_qkv_bias) {
                k_flat = em.emit_add_bias_3d(k_flat, is_swa ? "%b_swa_k" : "%b_glob_k", kvi, N_kv_type, B, S, KV_sh);
            }
            std::string v_flat = k_flat; // Gemma 4 k_eq_v when layer_has_v == false
            if (layer_has_v) {
                v_flat = em.emit_linear_3d(x_an_bf,
                    is_swa ? "%w_swa_v" : "%w_glob_v",
                    is_swa ? "%s_swa_v" : "%s_glob_v",
                    kvi, N_kv_type, B, S, KV_sh, D, i8, i4);
                if (spec.has_qkv_bias) {
                    v_flat = em.emit_add_bias_3d(v_flat, is_swa ? "%b_swa_v" : "%b_glob_v", kvi, N_kv_type, B, S, KV_sh);
                }
            }
            const std::string t_kv3    = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(KV_sh) + "xf32>";
            const std::string t_kv4    = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(Hkv_sh) + "x" + std::to_string(HD) + "xf32>";
            const std::string t_kv4_el = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(Hkv_sh) + "x" + std::to_string(HD) + "x" + kv_el + ">";

            std::string k_4d = em.tmp("k4");
            em.ss << "    " << k_4d << " = stablehlo.reshape " << k_flat << " : (" << t_kv3 << ") -> " << t_kv4 << "\n";
            std::string v_4d = k_4d;
            if (layer_has_v) {
                v_4d = em.tmp("v4");
                em.ss << "    " << v_4d << " = stablehlo.reshape " << v_flat << " : (" << t_kv3 << ") -> " << t_kv4 << "\n";
            }
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

            std::string k_src_f = k_rot, v_src_f = v_final;
            if (kv_fp8) {
                // Clamp strictly inside finite f8E4M3FN dynamic range [-448, +448] to guarantee zero NaN overflow
                std::string c_fhi = em.tmp("f8hi"), c_flo = em.tmp("f8lo");
                std::string kc1 = em.tmp("kc1"), kc2 = em.tmp("kc2"), vc1 = em.tmp("vc1"), vc2 = em.tmp("vc2");
                em.ss << "    " << c_fhi << " = stablehlo.constant dense<4.4800000e+02> : " << t_kv4 << "\n"
                      << "    " << c_flo << " = stablehlo.constant dense<-4.4800000e+02> : " << t_kv4 << "\n"
                      << "    " << kc1   << " = stablehlo.minimum " << k_rot   << ", " << c_fhi << " : " << t_kv4 << "\n"
                      << "    " << kc2   << " = stablehlo.maximum " << kc1     << ", " << c_flo << " : " << t_kv4 << "\n"
                      << "    " << vc1   << " = stablehlo.minimum " << v_final << ", " << c_fhi << " : " << t_kv4 << "\n"
                      << "    " << vc2   << " = stablehlo.maximum " << vc1     << ", " << c_flo << " : " << t_kv4 << "\n";
                k_src_f = kc2;
                v_src_f = vc2;
            }
            std::string k_el_v = k_src_f, v_el_v = v_src_f;
            if (!kv_f32) {
                k_el_v = em.tmp("kel");
                v_el_v = em.tmp("vel");
                em.ss << "    " << k_el_v << " = stablehlo.convert " << k_src_f << " : (" << t_kv4 << ") -> " << t_kv4_el << "\n"
                      << "    " << v_el_v << " = stablehlo.convert " << v_src_f << " : (" << t_kv4 << ") -> " << t_kv4_el << "\n";
            }

            std::string nk = em.tmp("kvk"), nv = em.tmp("kvv");
            if (per_slot_pos) {
                // Coalesced in-place 3D per-layer scatter into [B_cap, T_kv, KV_sh] (major dims [0, 1], contiguous minor dim KV_sh!)
                const std::string t_upd   = "tensor<" + std::to_string(B_cap) + "x" + std::to_string(KV_sh) + "x" + kv_el + ">";
                const std::string t_scidx = "tensor<" + std::to_string(B_cap) + "x2xi32>";
                const std::string sc_idx = is_swa ? sc_idx_swa_var : std::string("%sc_idx_glob");
                std::string k_upd = em.tmp("kscu"), v_upd = em.tmp("vscu");
                em.ss << "    " << k_upd << " = stablehlo.reshape " << k_el_v << " : (" << t_kv4_el << ") -> " << t_upd << "\n"
                      << "    " << v_upd << " = stablehlo.reshape " << v_el_v << " : (" << t_kv4_el << ") -> " << t_upd << "\n"
                      << "    " << nk << " = \"stablehlo.scatter\"(" << kv_k_in << ", " << sc_idx << ", " << k_upd << ") ({\n"
                      << "    ^bb0(%old: tensor<" << kv_el << ">, %new: tensor<" << kv_el << ">):\n"
                      << "      stablehlo.return %new : tensor<" << kv_el << ">\n"
                      << "    }) {\n"
                      << "      scatter_dimension_numbers = #stablehlo.scatter<\n"
                      << "        update_window_dims = [1],\n"
                      << "        inserted_window_dims = [0, 1],\n"
                      << "        scatter_dims_to_operand_dims = [0, 1],\n"
                      << "        index_vector_dim = 1\n"
                      << "      >,\n"
                      << "      indices_are_sorted = false,\n"
                      << "      unique_indices = true\n"
                      << "    } : (" << kv_buf_t << ", " << t_scidx << ", " << t_upd << ") -> " << kv_buf_t << "\n"
                      << "    " << nv << " = \"stablehlo.scatter\"(" << kv_v_in << ", " << sc_idx << ", " << v_upd << ") ({\n"
                      << "    ^bb0(%old: tensor<" << kv_el << ">, %new: tensor<" << kv_el << ">):\n"
                      << "      stablehlo.return %new : tensor<" << kv_el << ">\n"
                      << "    }) {\n"
                      << "      scatter_dimension_numbers = #stablehlo.scatter<\n"
                      << "        update_window_dims = [1],\n"
                      << "        inserted_window_dims = [0, 1],\n"
                      << "        scatter_dims_to_operand_dims = [0, 1],\n"
                      << "        index_vector_dim = 1\n"
                      << "      >,\n"
                      << "      indices_are_sorted = false,\n"
                      << "      unique_indices = true\n"
                      << "    } : (" << kv_buf_t << ", " << t_scidx << ", " << t_upd << ") -> " << kv_buf_t << "\n";
            } else if (slot_targeted) {
                // Zero-Copy Slot-Targeted Prefill:
                // 1) Slice ONLY the single target slot [1, T_kv, KV_sh] BEFORE mutating [B_cap, T_kv, KV_sh].
                // 2) Apply ring scatter or linear update entirely on the tiny 1-slot buffer (VMEM speed!).
                // 3) Write back ONLY the 1 updated slot via pure dynamic_update_slice with ZERO downstream readers
                //    on %nk/%nv (guarantees 100% in-place HBM3 update with ZERO 64/256-slot copy!).
                std::string k_sl0 = em.tmp("ksl0"), v_sl0 = em.tmp("vsl0");
                std::string k_sl1 = em.tmp("ksl1"), v_sl1 = em.tmp("vsl1");
                em.ss << "    " << k_sl0 << " = stablehlo.dynamic_slice " << kv_k_in << ", %slot_id, "
                      << c0_i32 << ", " << c0_i32 << ", sizes = [1, " << T_kv << ", " << KV_sh
                      << "] : (" << kv_buf_t << ", tensor<i32>, tensor<i32>, tensor<i32>) -> " << t_kv3_t << "\n"
                      << "    " << v_sl0 << " = stablehlo.dynamic_slice " << kv_v_in << ", %slot_id, "
                      << c0_i32 << ", " << c0_i32 << ", sizes = [1, " << T_kv << ", " << KV_sh
                      << "] : (" << kv_buf_t << ", tensor<i32>, tensor<i32>, tensor<i32>) -> " << t_kv3_t << "\n";
                if (is_swa && swa_ring_mode && S > 1) {
                    const std::string t_upd   = "tensor<" + std::to_string(S) + "x" + std::to_string(KV_sh) + "x" + kv_el + ">";
                    const std::string t_scidx = "tensor<" + std::to_string(S) + "x2xi32>";
                    std::string k_upd = em.tmp("kscu"), v_upd = em.tmp("vscu");
                    em.ss << "    " << k_upd << " = stablehlo.reshape " << k_el_v << " : (" << t_kv4_el << ") -> " << t_upd << "\n"
                          << "    " << v_upd << " = stablehlo.reshape " << v_el_v << " : (" << t_kv4_el << ") -> " << t_upd << "\n"
                          << "    " << k_sl1 << " = \"stablehlo.scatter\"(" << k_sl0 << ", %sc_idx_swa_pre, " << k_upd << ") ({\n"
                          << "    ^bb0(%old: tensor<" << kv_el << ">, %new: tensor<" << kv_el << ">):\n"
                          << "      stablehlo.return %new : tensor<" << kv_el << ">\n"
                          << "    }) {\n"
                          << "      scatter_dimension_numbers = #stablehlo.scatter<\n"
                          << "        update_window_dims = [1],\n"
                          << "        inserted_window_dims = [0, 1],\n"
                          << "        scatter_dims_to_operand_dims = [0, 1],\n"
                          << "        index_vector_dim = 1\n"
                          << "      >,\n"
                          << "      indices_are_sorted = false,\n"
                          << "      unique_indices = true\n"
                          << "    } : (" << t_kv3_t << ", " << t_scidx << ", " << t_upd << ") -> " << t_kv3_t << "\n"
                          << "    " << v_sl1 << " = \"stablehlo.scatter\"(" << v_sl0 << ", %sc_idx_swa_pre, " << v_upd << ") ({\n"
                          << "    ^bb0(%old: tensor<" << kv_el << ">, %new: tensor<" << kv_el << ">):\n"
                          << "      stablehlo.return %new : tensor<" << kv_el << ">\n"
                          << "    }) {\n"
                          << "      scatter_dimension_numbers = #stablehlo.scatter<\n"
                          << "        update_window_dims = [1],\n"
                          << "        inserted_window_dims = [0, 1],\n"
                          << "        scatter_dims_to_operand_dims = [0, 1],\n"
                          << "        index_vector_dim = 1\n"
                          << "      >,\n"
                          << "      indices_are_sorted = false,\n"
                          << "      unique_indices = true\n"
                          << "    } : (" << t_kv3_t << ", " << t_scidx << ", " << t_upd << ") -> " << t_kv3_t << "\n"
                          << "    " << nk << " = stablehlo.dynamic_update_slice " << kv_k_in << ", " << k_sl1 << ", "
                          << "%slot_id, " << c0_i32 << ", " << c0_i32
                          << " : (" << kv_buf_t << ", " << t_kv3_t << ", tensor<i32>, tensor<i32>, tensor<i32>) -> " << kv_buf_t << "\n"
                          << "    " << nv << " = stablehlo.dynamic_update_slice " << kv_v_in << ", " << v_sl1 << ", "
                          << "%slot_id, " << c0_i32 << ", " << c0_i32
                          << " : (" << kv_buf_t << ", " << t_kv3_t << ", tensor<i32>, tensor<i32>, tensor<i32>) -> " << kv_buf_t << "\n";
                } else {
                    const std::string t_kv3_el = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(KV_sh) + "x" + kv_el + ">";
                    std::string k_3 = em.tmp("k3"), v_3 = em.tmp("v3");
                    const std::string wpos_var = (is_swa && swa_ring_mode) ? "%swa_write_pos_0" : "%pos";
                    em.ss << "    " << k_3 << " = stablehlo.reshape " << k_el_v << " : (" << t_kv4_el << ") -> " << t_kv3_el << "\n"
                          << "    " << v_3 << " = stablehlo.reshape " << v_el_v << " : (" << t_kv4_el << ") -> " << t_kv3_el << "\n"
                          << "    " << k_sl1 << " = stablehlo.dynamic_update_slice " << k_sl0 << ", " << k_3 << ", "
                          << c0_i32 << ", " << wpos_var << ", " << c0_i32
                          << " : (" << t_kv3_t << ", " << t_kv3_el << ", tensor<i32>, tensor<i32>, tensor<i32>) -> " << t_kv3_t << "\n"
                          << "    " << v_sl1 << " = stablehlo.dynamic_update_slice " << v_sl0 << ", " << v_3 << ", "
                          << c0_i32 << ", " << wpos_var << ", " << c0_i32
                          << " : (" << t_kv3_t << ", " << t_kv3_el << ", tensor<i32>, tensor<i32>, tensor<i32>) -> " << t_kv3_t << "\n"
                          << "    " << nk << " = stablehlo.dynamic_update_slice " << kv_k_in << ", " << k_3 << ", "
                          << "%slot_id, " << wpos_var << ", " << c0_i32
                          << " : (" << kv_buf_t << ", " << t_kv3_el << ", tensor<i32>, tensor<i32>, tensor<i32>) -> " << kv_buf_t << "\n"
                          << "    " << nv << " = stablehlo.dynamic_update_slice " << kv_v_in << ", " << v_3 << ", "
                          << "%slot_id, " << wpos_var << ", " << c0_i32
                          << " : (" << kv_buf_t << ", " << t_kv3_el << ", tensor<i32>, tensor<i32>, tensor<i32>) -> " << kv_buf_t << "\n";
                }
                if (is_swa) {
                    slot_ks[static_cast<size_t>(kvi)] = k_sl1;
                    slot_vs[static_cast<size_t>(kvi)] = v_sl1;
                } else {
                    slot_kg[static_cast<size_t>(kvi)] = k_sl1;
                    slot_vg[static_cast<size_t>(kvi)] = v_sl1;
                }
            } else if (is_swa && swa_ring_mode && S > 1) {
                // Exact wrap-safe coalesced ring scatter for B_cap == 1 S-chunk prefill
                const std::string t_upd   = "tensor<" + std::to_string(S) + "x" + std::to_string(KV_sh) + "x" + kv_el + ">";
                const std::string t_scidx = "tensor<" + std::to_string(S) + "x2xi32>";
                std::string k_upd = em.tmp("kscu"), v_upd = em.tmp("vscu");
                em.ss << "    " << k_upd << " = stablehlo.reshape " << k_el_v << " : (" << t_kv4_el << ") -> " << t_upd << "\n"
                      << "    " << v_upd << " = stablehlo.reshape " << v_el_v << " : (" << t_kv4_el << ") -> " << t_upd << "\n"
                      << "    " << nk << " = \"stablehlo.scatter\"(" << kv_k_in << ", %sc_idx_swa_pre, " << k_upd << ") ({\n"
                      << "    ^bb0(%old: tensor<" << kv_el << ">, %new: tensor<" << kv_el << ">):\n"
                      << "      stablehlo.return %new : tensor<" << kv_el << ">\n"
                      << "    }) {\n"
                      << "      scatter_dimension_numbers = #stablehlo.scatter<\n"
                      << "        update_window_dims = [1],\n"
                      << "        inserted_window_dims = [0, 1],\n"
                      << "        scatter_dims_to_operand_dims = [0, 1],\n"
                      << "        index_vector_dim = 1\n"
                      << "      >,\n"
                      << "      indices_are_sorted = false,\n"
                      << "      unique_indices = true\n"
                      << "    } : (" << kv_buf_t << ", " << t_scidx << ", " << t_upd << ") -> " << kv_buf_t << "\n"
                      << "    " << nv << " = \"stablehlo.scatter\"(" << kv_v_in << ", %sc_idx_swa_pre, " << v_upd << ") ({\n"
                      << "    ^bb0(%old: tensor<" << kv_el << ">, %new: tensor<" << kv_el << ">):\n"
                      << "      stablehlo.return %new : tensor<" << kv_el << ">\n"
                      << "    }) {\n"
                      << "      scatter_dimension_numbers = #stablehlo.scatter<\n"
                      << "        update_window_dims = [1],\n"
                      << "        inserted_window_dims = [0, 1],\n"
                      << "        scatter_dims_to_operand_dims = [0, 1],\n"
                      << "        index_vector_dim = 1\n"
                      << "      >,\n"
                      << "      indices_are_sorted = false,\n"
                      << "      unique_indices = true\n"
                      << "    } : (" << kv_buf_t << ", " << t_scidx << ", " << t_upd << ") -> " << kv_buf_t << "\n";
            } else {
                // Single-stream (B=1): zero-transpose 3D dynamic_update_slice!
                const std::string t_kv3_el = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(KV_sh) + "x" + kv_el + ">";
                std::string k_3 = em.tmp("k3"), v_3 = em.tmp("v3");
                const std::string wpos_var  = (is_swa && swa_ring_mode) ? "%swa_write_pos_0" : "%pos";
                em.ss << "    " << k_3 << " = stablehlo.reshape " << k_el_v << " : (" << t_kv4_el << ") -> " << t_kv3_el << "\n"
                      << "    " << v_3 << " = stablehlo.reshape " << v_el_v << " : (" << t_kv4_el << ") -> " << t_kv3_el << "\n"
                      << "    " << nk << " = stablehlo.dynamic_update_slice " << kv_k_in << ", " << k_3 << ", "
                      << c0_i32 << ", " << wpos_var << ", " << c0_i32
                      << " : (" << kv_buf_t << ", " << t_kv3_el << ", tensor<i32>, tensor<i32>, tensor<i32>) -> " << kv_buf_t << "\n"
                      << "    " << nv << " = stablehlo.dynamic_update_slice " << kv_v_in << ", " << v_3 << ", "
                      << c0_i32 << ", " << wpos_var << ", " << c0_i32
                      << " : (" << kv_buf_t << ", " << t_kv3_el << ", tensor<i32>, tensor<i32>, tensor<i32>) -> " << kv_buf_t << "\n";
            }
            if (is_swa) {
                kv_ks[static_cast<size_t>(kvi)] = nk;
                kv_vs[static_cast<size_t>(kvi)] = nv;
            } else {
                kv_kg[static_cast<size_t>(kvi)] = nk;
                kv_vg[static_cast<size_t>(kvi)] = nv;
            }
        }

        // 2e. GQA Attention directly on per-layer coalesced [B, T_kv, Hkv_sh, HD] HBM3 KV Cache (BF16 or Fused FP8 E4M3FN -> BF16!)
        const std::string& cur_k_buf = is_swa ? kv_ks[static_cast<size_t>(kvi)] : kv_kg[static_cast<size_t>(kvi)];
        const std::string& cur_v_buf = is_swa ? kv_vs[static_cast<size_t>(kvi)] : kv_vg[static_cast<size_t>(kvi)];
        const std::string t_kv4_t   = "tensor<" + std::to_string(B) + "x" + std::to_string(T_kv) + "x" + std::to_string(Hkv_sh) + "x" + std::to_string(HD) + "x" + kv_el + ">";
        const std::string t_kv4_bf  = "tensor<" + std::to_string(B) + "x" + std::to_string(T_kv) + "x" + std::to_string(Hkv_sh) + "x" + std::to_string(HD) + "xbf16>";
        const std::string t_q5_bf   = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(Hkv_sh) + "x" + std::to_string(G_sh) + "x" + std::to_string(HD) + "xbf16>";
        const std::string t_sc5     = "tensor<" + std::to_string(B) + "x" + std::to_string(Hkv_sh) + "x" + std::to_string(S) + "x" + std::to_string(G_sh) + "x" + std::to_string(T_kv) + "xf32>";
        const std::string t_sc5_i1  = "tensor<" + std::to_string(B) + "x" + std::to_string(Hkv_sh) + "x" + std::to_string(S) + "x" + std::to_string(G_sh) + "x" + std::to_string(T_kv) + "xi1>";
        const std::string t_sc5_bf  = "tensor<" + std::to_string(B) + "x" + std::to_string(Hkv_sh) + "x" + std::to_string(S) + "x" + std::to_string(G_sh) + "x" + std::to_string(T_kv) + "xbf16>";
        const std::string t_sc4     = "tensor<" + std::to_string(B) + "x" + std::to_string(Hkv_sh) + "x" + std::to_string(S) + "x" + std::to_string(G_sh) + "xf32>";
        const std::string t_ctx5    = "tensor<" + std::to_string(B) + "x" + std::to_string(Hkv_sh) + "x" + std::to_string(S) + "x" + std::to_string(G_sh) + "x" + std::to_string(HD) + "xf32>";
        const std::string t_ctx5t   = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(Hkv_sh) + "x" + std::to_string(G_sh) + "x" + std::to_string(HD) + "xf32>";
        const std::string t_ao_bf   = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(Q_sh) + "xbf16>";

        std::string k3_src = cur_k_buf;
        std::string v3_src = cur_v_buf;
        if (slot_targeted) {
            const std::string& cached_k = is_swa ? slot_ks[static_cast<size_t>(kvi)] : slot_kg[static_cast<size_t>(kvi)];
            const std::string& cached_v = is_swa ? slot_vs[static_cast<size_t>(kvi)] : slot_vg[static_cast<size_t>(kvi)];
            if (!cached_k.empty() && !cached_v.empty()) {
                k3_src = cached_k;
                v3_src = cached_v;
            } else {
                k3_src = em.tmp("k3s");
                v3_src = em.tmp("v3s");
                em.ss << "    " << k3_src << " = stablehlo.dynamic_slice " << cur_k_buf << ", %slot_id, "
                      << c0_i32 << ", " << c0_i32 << ", sizes = [1, " << T_kv << ", " << KV_sh
                      << "] : (" << kv_buf_t << ", tensor<i32>, tensor<i32>, tensor<i32>) -> " << t_kv3_t << "\n"
                      << "    " << v3_src << " = stablehlo.dynamic_slice " << cur_v_buf << ", %slot_id, "
                      << c0_i32 << ", " << c0_i32 << ", sizes = [1, " << T_kv << ", " << KV_sh
                      << "] : (" << kv_buf_t << ", tensor<i32>, tensor<i32>, tensor<i32>) -> " << t_kv3_t << "\n";
            }
        }
        std::string klay = em.tmp("klay"), vlay = em.tmp("vlay");
        em.ss << "    " << klay << " = stablehlo.reshape " << k3_src << " : (" << t_kv3_t << ") -> " << t_kv4_t << "\n"
              << "    " << vlay << " = stablehlo.reshape " << v3_src << " : (" << t_kv3_t << ") -> " << t_kv4_t << "\n";

        std::string klay_bf = klay;
        std::string vlay_bf = vlay;
        if (kv_fp8 || kv_f32) {
            klay_bf = em.tmp("klayb");
            vlay_bf = em.tmp("vlayb");
            em.ss << "    " << klay_bf << " = stablehlo.convert " << klay << " : (" << t_kv4_t << ") -> " << t_kv4_bf << "\n"
                  << "    " << vlay_bf << " = stablehlo.convert " << vlay << " : (" << t_kv4_t << ") -> " << t_kv4_bf << "\n";
        }

        std::string qbf4 = em.tmp("qbf"), q5 = em.tmp("q5"), sc = em.tmp("sc");
        em.ss << "    " << qbf4 << " = stablehlo.convert " << q_rot << " : (" << t_q4 << ") -> tensor<" << B << "x" << S << "x" << H_sh << "x" << HD << "xbf16>\n"
              << "    " << q5   << " = stablehlo.reshape " << qbf4 << " : (tensor<" << B << "x" << S << "x" << H_sh << "x" << HD << "xbf16>) -> " << t_q5_bf << "\n"
              << "    " << sc   << " = stablehlo.dot_general " << q5 << ", " << klay_bf
              << ", batching_dims = [0, 2] x [0, 2], contracting_dims = [4] x [3] : (" << t_q5_bf << ", " << t_kv4_bf << ") -> " << t_sc5 << "\n";

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
              << ", dims = " << mask_bdims << " : (" << mask_t_layer << ") -> " << t_sc5_i1 << "\n"
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
              << "    " << ctx  << " = stablehlo.dot_general " << prbf << ", " << vlay_bf
              << ", batching_dims = [0, 1] x [0, 2], contracting_dims = [4] x [1] : (" << t_sc5_bf << ", " << t_kv4_bf << ") -> " << t_ctx5 << "\n"
              << "    " << ctxt << " = stablehlo.transpose " << ctx << ", dims = [0, 2, 1, 3, 4] : (" << t_ctx5 << ") -> " << t_ctx5t << "\n"
              << "    " << ao_f32 << " = stablehlo.reshape " << ctxt << " : (" << t_ctx5t << ") -> " << t_q3 << "\n"
              << "    " << ao_bf  << " = stablehlo.convert " << ao_f32 << " : (" << t_q3 << ") -> " << t_ao_bf << "\n";

        // 2f. Attention Output Projection + Full-F32 Optical ICI AllReduce + Optional Post-Attn Norm + Residual 1
        std::string attn_out_loc = em.emit_linear_3d(ao_bf,
            is_swa ? "%w_swa_o" : "%w_glob_o",
            is_swa ? "%s_swa_o" : "%s_glob_o",
            ti, N_type, B, S, D, Q_sh, i8, i4);
        std::string attn_out = em.emit_all_reduce_sum_3d(attn_out_loc, B, S, D, R);
        std::string attn_final = attn_out;
        if (spec.has_post_attn_norm) {
            std::string pa_nw = em.slice_1d_from_2d("%w_post_attn_norm", l, L, D);
            attn_final = em.emit_rms_norm_3d(attn_out, pa_nw, B, S, D, spec.rms_norm_eps, c0_f32);
        }
        std::string x_res1 = em.tmp("res1");
        em.ss << "    " << x_res1 << " = stablehlo.add " << x_cur << ", " << attn_final << " : " << t_bsd_f32 << "\n";

        // 2g-2i. Pre-FFN RMSNorm + Fused Sharded Gate+Up GeGLU/SwiGLU FFN + Full-F32 Optical ICI AllReduce + Residual 2
        std::string fn_nw = em.slice_1d_from_2d("%w_ffn_norm", l, L, D);
        std::string x_fn_f32 = em.emit_rms_norm_3d(x_res1, fn_nw, B, S, D, spec.rms_norm_eps, c0_f32);
        std::string x_fn_bf = em.tmp("fnbf");
        em.ss << "    " << x_fn_bf << " = stablehlo.convert " << x_fn_f32 << " : (" << t_bsd_f32 << ") -> " << t_bsd_bf << "\n";

        const std::string t_fgu_f32 = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(2 * F_sh) + "xf32>";
        const std::string t_ff_f32  = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(F_sh) + "xf32>";
        const std::string t_ff_bf   = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(F_sh) + "xbf16>";
        std::string f_gu = em.emit_linear_3d(x_fn_bf, "%w_ffn_gu", "%s_ffn_gu", l, L, B, S, 2 * F_sh, D, i8, i4);
        std::string f_gate = em.tmp("fgat"), f_up = em.tmp("fup");
        em.ss << "    " << f_gate << " = stablehlo.slice " << f_gu << " [0:" << B << ", 0:" << S << ", 0:" << F_sh << "] : (" << t_fgu_f32 << ") -> " << t_ff_f32 << "\n"
              << "    " << f_up   << " = stablehlo.slice " << f_gu << " [0:" << B << ", 0:" << S << ", " << F_sh << ":" << (2 * F_sh) << "] : (" << t_fgu_f32 << ") -> " << t_ff_f32 << "\n";
        std::string f_gate_act = (spec.ffn_activation == FfnActivation::GELU)
            ? em.emit_gelu_3d(f_gate, B, S, F_sh)
            : em.emit_silu_3d(f_gate, B, S, F_sh);
        std::string f_act = em.tmp("fact"), f_act_bf = em.tmp("fabf");
        em.ss << "    " << f_act    << " = stablehlo.multiply " << f_gate_act << ", " << f_up << " : " << t_ff_f32 << "\n"
              << "    " << f_act_bf << " = stablehlo.convert " << f_act << " : (" << t_ff_f32 << ") -> " << t_ff_bf << "\n";
        std::string f_down_loc = em.emit_linear_3d(f_act_bf, "%w_ffn_down", "%s_ffn_down", l, L, B, S, D, F_sh, i8, i4);
        std::string f_final;
        if (!is_moe) {
            std::string f_down = em.emit_all_reduce_sum_3d(f_down_loc, B, S, D, R);
            f_final = f_down;
            if (spec.has_post_ffn_norm) {
                std::string pf_nw = em.slice_1d_from_2d("%w_post_ffn_norm", l, L, D);
                f_final = em.emit_rms_norm_3d(f_down, pf_nw, B, S, D, spec.rms_norm_eps, c0_f32);
            }
        } else {
            // Parallel Sparse Mixture-of-Experts Branch (Gemma 4 26B-A4B / Ava 26B Turbo MoE)
            const int N_tok = B * S;
            const std::string t_nd_bf  = "tensor<" + std::to_string(N_tok) + "x" + std::to_string(D) + "xbf16>";
            const std::string t_nd_f32 = "tensor<" + std::to_string(N_tok) + "x" + std::to_string(D) + "xf32>";
            const std::string t_ne_f32 = "tensor<" + std::to_string(N_tok) + "x" + std::to_string(E) + "xf32>";
            const std::string t_ne_i32 = "tensor<" + std::to_string(N_tok) + "x" + std::to_string(E) + "xi32>";
            const std::string t_n1_f32 = "tensor<" + std::to_string(N_tok) + "xf32>";
            const std::string t_nk_f32 = "tensor<" + std::to_string(N_tok) + "x" + std::to_string(K_exp) + "xf32>";
            const std::string t_nk_i32 = "tensor<" + std::to_string(N_tok) + "x" + std::to_string(K_exp) + "xi32>";
            const std::string t_bse_f32 = "tensor<" + std::to_string(B) + "x" + std::to_string(S) + "x" + std::to_string(E) + "xf32>";

            // Router: RMSNorm(x_res1) * (1/sqrt(D)) * ffn_gate_inp.scale -> Linear -> Softmax -> Top-K + Renorm
            std::string r_rn = em.emit_rms_norm_3d(x_res1, "", B, S, D, spec.rms_norm_eps, c0_f32);
            std::string c_inv_sd = em.tmp("risd"), r_sc = em.tmp("rsc");
            em.ss << "    " << c_inv_sd << " = stablehlo.constant dense<" << fmt_f32(1.0f / std::sqrt(static_cast<float>(D))) << "> : " << t_bsd_f32 << "\n"
                  << "    " << r_sc << " = stablehlo.multiply " << r_rn << ", " << c_inv_sd << " : " << t_bsd_f32 << "\n";
            std::string r_in = r_sc;
            if (spec.has_moe_router_scale) {
                std::string rs_1d = em.slice_1d_from_2d("%w_moe_rscale", l, L, D);
                std::string rs_bc = em.tmp("rsbc"), r_mul = em.tmp("rmul");
                em.ss << "    " << rs_bc << " = stablehlo.broadcast_in_dim " << rs_1d << ", dims = [2] : (tensor<" << D << "xf32>) -> " << t_bsd_f32 << "\n"
                      << "    " << r_mul << " = stablehlo.multiply " << r_sc << ", " << rs_bc << " : " << t_bsd_f32 << "\n";
                r_in = r_mul;
            }
            std::string r_bf = em.tmp("rbf");
            em.ss << "    " << r_bf << " = stablehlo.convert " << r_in << " : (" << t_bsd_f32 << ") -> " << t_bsd_bf << "\n";
            std::string r_lg3 = em.emit_linear_3d(r_bf, "%w_moe_router", "", l, L, B, S, E, D, false);
            std::string r_lg2 = em.tmp("rlg2"), r_mx = em.tmp("rmx"), r_mxb = em.tmp("rmxb");
            std::string r_sh = em.tmp("rsh"), r_ex = em.tmp("rex"), r_se = em.tmp("rse"), r_seb = em.tmp("rseb"), r_pr = em.tmp("rpr");
            std::string e_io = em.tmp("eio"), sp = em.tmp("sp"), si = em.tmp("si");
            std::string tp = em.tmp("tp"), ti_v = em.tmp("ti"), tsum = em.tmp("tsum"), c_epsw = em.tmp("cepsw"), tcl = em.tmp("tcl"), tcb = em.tmp("tcb"), tw = em.tmp("tw");

            em.ss << "    " << r_lg2 << " = stablehlo.reshape " << r_lg3 << " : (" << t_bse_f32 << ") -> " << t_ne_f32 << "\n"
                  << "    " << r_mx  << " = stablehlo.reduce(" << r_lg2 << " init: " << c_ninf << ") across dimensions = [1] : (" << t_ne_f32 << ", tensor<f32>) -> " << t_n1_f32 << "\n"
                  << "      reducer(%ra: tensor<f32>, %rb: tensor<f32>) {\n"
                  << "        %rc = stablehlo.maximum %ra, %rb : tensor<f32>\n"
                  << "        stablehlo.return %rc : tensor<f32>\n"
                  << "      }\n"
                  << "    " << r_mxb << " = stablehlo.broadcast_in_dim " << r_mx << ", dims = [0] : (" << t_n1_f32 << ") -> " << t_ne_f32 << "\n"
                  << "    " << r_sh  << " = stablehlo.subtract " << r_lg2 << ", " << r_mxb << " : " << t_ne_f32 << "\n"
                  << "    " << r_ex  << " = stablehlo.exponential " << r_sh << " : " << t_ne_f32 << "\n"
                  << "    " << r_se  << " = stablehlo.reduce(" << r_ex << " init: " << c0_f32 << ") across dimensions = [1] : (" << t_ne_f32 << ", tensor<f32>) -> " << t_n1_f32 << "\n"
                  << "      reducer(%ra: tensor<f32>, %rb: tensor<f32>) {\n"
                  << "        %rc = stablehlo.add %ra, %rb : tensor<f32>\n"
                  << "        stablehlo.return %rc : tensor<f32>\n"
                  << "      }\n"
                  << "    " << r_seb << " = stablehlo.broadcast_in_dim " << r_se << ", dims = [0] : (" << t_n1_f32 << ") -> " << t_ne_f32 << "\n"
                  << "    " << r_pr  << " = stablehlo.divide " << r_ex << ", " << r_seb << " : " << t_ne_f32 << "\n"
                  << "    " << e_io  << " = stablehlo.iota dim = 1 : " << t_ne_i32 << "\n"
                  << "    " << sp << ", " << si << " = \"stablehlo.sort\"(" << r_pr << ", " << e_io << ") ({\n"
                  << "    ^bb0(%sa: tensor<f32>, %sb: tensor<f32>, %ia: tensor<i32>, %ib: tensor<i32>):\n"
                  << "      %sgt = stablehlo.compare GT, %sa, %sb, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>\n"
                  << "      stablehlo.return %sgt : tensor<i1>\n"
                  << "    }) {\n"
                  << "      dimension = 1 : i64,\n"
                  << "      is_stable = true\n"
                  << "    } : (" << t_ne_f32 << ", " << t_ne_i32 << ") -> (" << t_ne_f32 << ", " << t_ne_i32 << ")\n"
                  << "    " << tp    << " = stablehlo.slice " << sp << " [0:" << N_tok << ", 0:" << K_exp << "] : (" << t_ne_f32 << ") -> " << t_nk_f32 << "\n"
                  << "    " << ti_v  << " = stablehlo.slice " << si << " [0:" << N_tok << ", 0:" << K_exp << "] : (" << t_ne_i32 << ") -> " << t_nk_i32 << "\n"
                  << "    " << tsum  << " = stablehlo.reduce(" << tp << " init: " << c0_f32 << ") across dimensions = [1] : (" << t_nk_f32 << ", tensor<f32>) -> " << t_n1_f32 << "\n"
                  << "      reducer(%ra: tensor<f32>, %rb: tensor<f32>) {\n"
                  << "        %rc = stablehlo.add %ra, %rb : tensor<f32>\n"
                  << "        stablehlo.return %rc : tensor<f32>\n"
                  << "      }\n"
                  << "    " << c_epsw << " = stablehlo.constant dense<6.1035156e-05> : " << t_n1_f32 << "\n"
                  << "    " << tcl    << " = stablehlo.maximum " << tsum << ", " << c_epsw << " : " << t_n1_f32 << "\n"
                  << "    " << tcb    << " = stablehlo.broadcast_in_dim " << tcl << ", dims = [0] : (" << t_n1_f32 << ") -> " << t_nk_f32 << "\n"
                  << "    " << tw     << " = stablehlo.divide " << tp << ", " << tcb << " : " << t_nk_f32 << "\n";

            // Pre-FFN Norm 2 on x_res1 -> optional exact Block-256 Orthogonal Hadamard Rotation (x' = xH)
            // Flattens 662x Gemma 4 blk.29 activation outliers down to 5.8x Gaussian BEFORE INT8 quantization!
            std::string x_m2 = x_res1;
            if (spec.has_pre_ffn_norm_2) {
                std::string p2_nw = em.slice_1d_from_2d("%w_pre_ffn_norm_2", l, L, D);
                x_m2 = em.emit_rms_norm_3d(x_res1, p2_nw, B, S, D, spec.rms_norm_eps, c0_f32);
            }
            std::string xm2_bf, xm2_f2_src;
            if (use_moe_hadamard) {
                const int N_hblk = D / 256;
                const std::string t_nhb_f32 = "tensor<" + std::to_string(N_tok) + "x" + std::to_string(N_hblk) + "x256xf32>";
                const std::string t_nhb_bf  = "tensor<" + std::to_string(N_tok) + "x" + std::to_string(N_hblk) + "x256xbf16>";
                std::string xh_in = em.tmp("xhin"), xh_bf = em.tmp("xhbf"), xh_rot = em.tmp("xhrot");
                std::string xh_f2 = em.tmp("xhf2"), xh_b2 = em.tmp("xhb2");
                em.ss << "    " << xh_in  << " = stablehlo.reshape " << x_m2 << " : (" << t_bsd_f32 << ") -> " << t_nhb_f32 << "\n"
                      << "    " << xh_bf  << " = stablehlo.convert " << xh_in << " : (" << t_nhb_f32 << ") -> " << t_nhb_bf << "\n"
                      << "    " << xh_rot << " = stablehlo.dot_general " << xh_bf << ", %w_had_moe"
                      << ", contracting_dims = [2] x [0] : (" << t_nhb_bf << ", tensor<256x256xbf16>) -> " << t_nhb_f32 << "\n"
                      << "    " << xh_f2  << " = stablehlo.reshape " << xh_rot << " : (" << t_nhb_f32 << ") -> " << t_nd_f32 << "\n"
                      << "    " << xh_b2  << " = stablehlo.convert " << xh_f2  << " : (" << t_nd_f32  << ") -> " << t_nd_bf  << "\n";
                xm2_f2_src = xh_f2;
                xm2_bf = xh_b2;
            } else {
                std::string xm2_bf3 = em.tmp("xm2b");
                xm2_bf = em.tmp("xm2");
                em.ss << "    " << xm2_bf3 << " = stablehlo.convert " << x_m2 << " : (" << t_bsd_f32 << ") -> " << t_bsd_bf << "\n"
                      << "    " << xm2_bf  << " = stablehlo.reshape " << xm2_bf3 << " : (" << t_bsd_bf << ") -> " << t_nd_bf << "\n";
            }

            // Per-layer 3D expert weights in TPU (8,128)-aligned zero-padding layout:
            //   %w_moe_gu_l : tensor<E x GU_exp_sh x proj_k(D) x wel>  (zero-cost reshape -> 2D [E*GU_exp_sh, proj_k(D)])
            //   %w_moe_dn_l : tensor<E x F_exp_sh  x proj_k(D) x wel>  (zero-cost reshape -> 2D [E*F_exp_sh, proj_k(D)])
            const int D_pk = proj_k(D);
            const int NG_D = w4_ng(D);
            const std::string t_wgu3  = "tensor<" + std::to_string(E) + "x" + std::to_string(GU_exp_sh) + "x" + std::to_string(D_pk) + "x" + wel + ">";
            const std::string t_wdn3  = "tensor<" + std::to_string(E) + "x" + std::to_string(F_exp_sh)  + "x" + std::to_string(D_pk) + "x" + wel + ">";
            const std::string wgu_l = "%w_moe_gu_" + std::to_string(l);
            const std::string wdn_l = "%w_moe_dn_" + std::to_string(l);
            const std::string sgu_l = "%s_moe_gu_" + std::to_string(l);
            const std::string sdn_l = "%s_moe_dn_" + std::to_string(l);
            const std::string t_sgu2 = "tensor<" + std::to_string(E) + "x" + std::to_string(GU_exp_sh) + "xf32>";
            const std::string t_sdn2 = "tensor<" + std::to_string(E) + "x" + std::to_string(F_exp_sh) + "xf32>";
            const std::string t_sgu3 = "tensor<" + std::to_string(E) + "x" + std::to_string(GU_exp_sh) + "x" + std::to_string(NG_D) + "xf32>";
            const std::string t_sdn3 = "tensor<" + std::to_string(E) + "x" + std::to_string(F_exp_sh)  + "x" + std::to_string(NG_D) + "xf32>";

            std::string moe_2d;
            if (N_tok == 1) {
                // Regime 1A (Interactive B=1 S=1): Direct HBM3 dynamic_slice of ONLY K_exp=8 experts
                // (compiles to 8 direct contiguous VMEM DMA loads vs slow full-table gather!).
                const std::string t_1gu    = "tensor<1x" + std::to_string(GU_exp_sh) + "x" + std::to_string(D_pk) + "x" + wel + ">";
                const std::string t_kgu    = "tensor<" + std::to_string(K_exp) + "x" + std::to_string(GU_exp_sh) + "x" + std::to_string(D_pk) + "x" + wel + ">";
                const std::string t_kgu2   = "tensor<" + std::to_string(K_exp * GU_exp_sh) + "x" + std::to_string(D_pk) + "x" + wel + ">";
                const std::string t_kgu2b  = "tensor<" + std::to_string(K_exp * GU_exp_sh) + "x" + std::to_string(D) + "xbf16>";
                const std::string t_gu2_f  = "tensor<1x" + std::to_string(K_exp * GU_exp_sh) + "xf32>";
                const std::string t_1dn    = "tensor<1x" + std::to_string(F_exp_sh) + "x" + std::to_string(D_pk) + "x" + wel + ">";
                const std::string t_kdn    = "tensor<" + std::to_string(K_exp) + "x" + std::to_string(F_exp_sh) + "x" + std::to_string(D_pk) + "x" + wel + ">";
                const std::string t_kdn2   = "tensor<" + std::to_string(K_exp * F_exp_sh) + "x" + std::to_string(D_pk) + "x" + wel + ">";
                const std::string t_kdn2b  = "tensor<" + std::to_string(K_exp * F_exp_sh) + "x" + std::to_string(D) + "xbf16>";
                const std::string t_gu_f32 = "tensor<1x" + std::to_string(K_exp) + "x" + std::to_string(GU_exp_sh) + "xf32>";
                const std::string t_ef_f32 = "tensor<1x" + std::to_string(K_exp) + "x" + std::to_string(F_exp_sh) + "xf32>";
                const std::string t_ef_bf  = "tensor<1x" + std::to_string(K_exp) + "x" + std::to_string(F_exp_sh) + "xbf16>";
                const std::string t_ef2_bf = "tensor<1x" + std::to_string(K_exp * F_exp_sh) + "xbf16>";
                const std::string t_1sgu4  = "tensor<1x" + std::to_string(GU_exp_sh) + "x" + std::to_string(NG_D) + "xf32>";
                const std::string t_ksgu4  = "tensor<" + std::to_string(K_exp) + "x" + std::to_string(GU_exp_sh) + "x" + std::to_string(NG_D) + "xf32>";
                const std::string t_ksgu42 = "tensor<" + std::to_string(K_exp * GU_exp_sh) + "x" + std::to_string(NG_D) + "xf32>";
                const std::string t_1sdn4  = "tensor<1x" + std::to_string(F_exp_sh) + "x" + std::to_string(NG_D) + "xf32>";
                const std::string t_ksdn4  = "tensor<" + std::to_string(K_exp) + "x" + std::to_string(F_exp_sh) + "x" + std::to_string(NG_D) + "xf32>";
                const std::string t_ksdn42 = "tensor<" + std::to_string(K_exp * F_exp_sh) + "x" + std::to_string(NG_D) + "xf32>";

                std::vector<std::string> ek_vars, gu_parts, dn_parts, sgu_parts, sdn_parts;
                for (int k = 0; k < K_exp; ++k) {
                    std::string esl = em.tmp("eksl"), ek = em.tmp("ek");
                    em.ss << "    " << esl << " = stablehlo.slice " << ti_v << " [0:1, " << k << ":" << (k + 1) << "] : (" << t_nk_i32 << ") -> tensor<1x1xi32>\n"
                          << "    " << ek  << " = stablehlo.reshape " << esl << " : (tensor<1x1xi32>) -> tensor<i32>\n";
                    ek_vars.push_back(ek);

                    std::string guk = em.tmp("guk"), dnk = em.tmp("dnk");
                    em.ss << "    " << guk << " = stablehlo.dynamic_slice " << wgu_l << ", " << ek << ", " << c0_i32 << ", " << c0_i32
                          << ", sizes = [1, " << GU_exp_sh << ", " << D_pk << "] : (" << t_wgu3 << ", tensor<i32>, tensor<i32>, tensor<i32>) -> " << t_1gu << "\n"
                          << "    " << dnk << " = stablehlo.dynamic_slice " << wdn_l << ", " << ek << ", " << c0_i32 << ", " << c0_i32
                          << ", sizes = [1, " << F_exp_sh << ", " << D_pk << "] : (" << t_wdn3 << ", tensor<i32>, tensor<i32>, tensor<i32>) -> " << t_1dn << "\n";
                    gu_parts.push_back(guk);
                    dn_parts.push_back(dnk);

                    if (i8) {
                        std::string sguk = em.tmp("sguk"), sdnk = em.tmp("sdnk");
                        em.ss << "    " << sguk << " = stablehlo.dynamic_slice " << sgu_l << ", " << ek << ", " << c0_i32
                              << ", sizes = [1, " << GU_exp_sh << "] : (" << t_sgu2 << ", tensor<i32>, tensor<i32>) -> tensor<1x" << GU_exp_sh << "xf32>\n"
                              << "    " << sdnk << " = stablehlo.dynamic_slice " << sdn_l << ", " << ek << ", " << c0_i32
                              << ", sizes = [1, " << F_exp_sh << "] : (" << t_sdn2 << ", tensor<i32>, tensor<i32>) -> tensor<1x" << F_exp_sh << "xf32>\n";
                        sgu_parts.push_back(sguk);
                        sdn_parts.push_back(sdnk);
                    } else if (i4) {
                        std::string sguk = em.tmp("sguk"), sdnk = em.tmp("sdnk");
                        em.ss << "    " << sguk << " = stablehlo.dynamic_slice " << sgu_l << ", " << ek << ", " << c0_i32 << ", " << c0_i32
                              << ", sizes = [1, " << GU_exp_sh << ", " << NG_D << "] : (" << t_sgu3 << ", tensor<i32>, tensor<i32>, tensor<i32>) -> " << t_1sgu4 << "\n"
                              << "    " << sdnk << " = stablehlo.dynamic_slice " << sdn_l << ", " << ek << ", " << c0_i32 << ", " << c0_i32
                              << ", sizes = [1, " << F_exp_sh << ", " << NG_D << "] : (" << t_sdn3 << ", tensor<i32>, tensor<i32>, tensor<i32>) -> " << t_1sdn4 << "\n";
                        sgu_parts.push_back(sguk);
                        sdn_parts.push_back(sdnk);
                    }
                }

                auto emit_concat_k = [&](const std::vector<std::string>& parts, const std::string& elem_t, const std::string& out_t) -> std::string {
                    std::string out = em.tmp("kcat");
                    em.ss << "    " << out << " = stablehlo.concatenate ";
                    for (size_t i = 0; i < parts.size(); ++i) em.ss << (i ? ", " : "") << parts[i];
                    em.ss << ", dim = 0 : (";
                    for (size_t i = 0; i < parts.size(); ++i) em.ss << (i ? ", " : "") << elem_t;
                    em.ss << ") -> " << out_t << "\n";
                    return out;
                };

                std::string kgu = emit_concat_k(gu_parts, t_1gu, t_kgu);
                std::string kgu_2d = em.tmp("kgu2");
                em.ss << "    " << kgu_2d << " = stablehlo.reshape " << kgu << " : (" << t_kgu << ") -> " << t_kgu2 << "\n";
                std::string kgu_bf = kgu_2d;
                if (i4) {
                    std::string ksgu = emit_concat_k(sgu_parts, t_1sgu4, t_ksgu4);
                    std::string ksgu_2d = em.tmp("ksg42");
                    em.ss << "    " << ksgu_2d << " = stablehlo.reshape " << ksgu << " : (" << t_ksgu4 << ") -> " << t_ksgu42 << "\n";
                    kgu_bf = em.emit_unpack_s4_2d(kgu_2d, ksgu_2d, K_exp * GU_exp_sh, D);
                } else if (i8) {
                    kgu_bf = em.tmp("kgub");
                    em.ss << "    " << kgu_bf << " = stablehlo.convert " << kgu_2d << " : (" << t_kgu2 << ") -> " << t_kgu2b << "\n";
                }
                std::string gu_dot2 = em.tmp("gud2"), gu_dot = em.tmp("gudot");
                em.ss << "    " << gu_dot2 << " = stablehlo.dot_general " << xm2_bf << ", " << kgu_bf
                      << ", contracting_dims = [1] x [1] : (" << t_nd_bf << ", " << t_kgu2b << ") -> " << t_gu2_f << "\n"
                      << "    " << gu_dot  << " = stablehlo.reshape " << gu_dot2 << " : (" << t_gu2_f << ") -> " << t_gu_f32 << "\n";
                std::string gu_cur = gu_dot;
                if (i8) {
                    std::string ksgu = emit_concat_k(sgu_parts, "tensor<1x" + std::to_string(GU_exp_sh) + "xf32>",
                                                     "tensor<" + std::to_string(K_exp) + "x" + std::to_string(GU_exp_sh) + "xf32>");
                    std::string ksgu3 = em.tmp("ksg3"), gu_sc = em.tmp("gusc");
                    em.ss << "    " << ksgu3 << " = stablehlo.reshape " << ksgu << " : (tensor<" << K_exp << "x" << GU_exp_sh << "xf32>) -> " << t_gu_f32 << "\n"
                          << "    " << gu_sc << " = stablehlo.multiply " << gu_dot << ", " << ksgu3 << " : " << t_gu_f32 << "\n";
                    gu_cur = gu_sc;
                }
                std::string eg = em.tmp("eg"), eu = em.tmp("eu");
                em.ss << "    " << eg << " = stablehlo.slice " << gu_cur << " [0:1, 0:" << K_exp << ", 0:" << F_exp_sh << "] : (" << t_gu_f32 << ") -> " << t_ef_f32 << "\n"
                      << "    " << eu << " = stablehlo.slice " << gu_cur << " [0:1, 0:" << K_exp << ", " << F_exp_sh << ":" << GU_exp_sh << "] : (" << t_gu_f32 << ") -> " << t_ef_f32 << "\n";
                std::string eg_act = (spec.ffn_activation == FfnActivation::GELU)
                    ? em.emit_gelu_3d(eg, 1, K_exp, F_exp_sh)
                    : em.emit_silu_3d(eg, 1, K_exp, F_exp_sh);
                std::string eact = em.tmp("eact"), twb = em.tmp("twb"), eact_w = em.tmp("eactw");
                em.ss << "    " << eact   << " = stablehlo.multiply " << eg_act << ", " << eu << " : " << t_ef_f32 << "\n"
                      << "    " << twb    << " = stablehlo.broadcast_in_dim " << tw << ", dims = [0, 1] : (" << t_nk_f32 << ") -> " << t_ef_f32 << "\n"
                      << "    " << eact_w << " = stablehlo.multiply " << eact << ", " << twb << " : " << t_ef_f32 << "\n";
                std::string eact_sc = eact_w;
                if (i8) {
                    std::string ksdn = emit_concat_k(sdn_parts, "tensor<1x" + std::to_string(F_exp_sh) + "xf32>",
                                                     "tensor<" + std::to_string(K_exp) + "x" + std::to_string(F_exp_sh) + "xf32>");
                    std::string ksdn3 = em.tmp("ksd3"), eact_s = em.tmp("eas");
                    em.ss << "    " << ksdn3  << " = stablehlo.reshape " << ksdn << " : (tensor<" << K_exp << "x" << F_exp_sh << "xf32>) -> " << t_ef_f32 << "\n"
                          << "    " << eact_s << " = stablehlo.multiply " << eact_w << ", " << ksdn3 << " : " << t_ef_f32 << "\n";
                    eact_sc = eact_s;
                }
                std::string eact_bf = em.tmp("eab"), eact_2d = em.tmp("ea2");
                em.ss << "    " << eact_bf << " = stablehlo.convert " << eact_sc << " : (" << t_ef_f32 << ") -> " << t_ef_bf << "\n"
                      << "    " << eact_2d << " = stablehlo.reshape " << eact_bf << " : (" << t_ef_bf << ") -> " << t_ef2_bf << "\n";
                std::string kdn = emit_concat_k(dn_parts, t_1dn, t_kdn);
                std::string kdn_2d = em.tmp("kdn2");
                em.ss << "    " << kdn_2d << " = stablehlo.reshape " << kdn << " : (" << t_kdn << ") -> " << t_kdn2 << "\n";
                std::string kdn_bf = kdn_2d;
                if (i4) {
                    std::string ksdn = emit_concat_k(sdn_parts, t_1sdn4, t_ksdn4);
                    std::string ksdn_2d = em.tmp("ksd42");
                    em.ss << "    " << ksdn_2d << " = stablehlo.reshape " << ksdn << " : (" << t_ksdn4 << ") -> " << t_ksdn42 << "\n";
                    kdn_bf = em.emit_unpack_s4_2d(kdn_2d, ksdn_2d, K_exp * F_exp_sh, D);
                } else if (i8) {
                    kdn_bf = em.tmp("kdnb");
                    em.ss << "    " << kdn_bf << " = stablehlo.convert " << kdn_2d << " : (" << t_kdn2 << ") -> " << t_kdn2b << "\n";
                }
                moe_2d = em.tmp("moe2");
                em.ss << "    " << moe_2d << " = stablehlo.dot_general " << eact_2d << ", " << kdn_bf
                      << ", contracting_dims = [1] x [0] : (" << t_ef2_bf << ", " << t_kdn2b << ") -> " << t_nd_f32 << "\n";
            } else {
                // Regime 2 (All Multi-Token Batches & Prefill N_tok >= 2):
                // Zero-Gather Fused Block-256 Hadamard W8A8 (INT8), W4A16 (INT4), or BF16!
                const std::string t_nk1_i32 = "tensor<" + std::to_string(N_tok) + "x" + std::to_string(K_exp) + "x1xi32>";
                const std::string t_nk2_i32 = "tensor<" + std::to_string(N_tok) + "x" + std::to_string(K_exp) + "x2xi32>";
                const std::string t_nd_i8   = "tensor<" + std::to_string(N_tok) + "x" + std::to_string(D) + "xi8>";
                const std::string t_wgu2    = "tensor<" + std::to_string(E * GU_exp_sh) + "x" + std::to_string(D_pk) + "x" + wel + ">";
                const std::string t_wgu2b   = "tensor<" + std::to_string(E * GU_exp_sh) + "x" + std::to_string(D) + "xbf16>";
                const std::string t_sgu42   = "tensor<" + std::to_string(E * GU_exp_sh) + "x" + std::to_string(NG_D) + "xf32>";
                const std::string t_gu2_f   = "tensor<" + std::to_string(N_tok) + "x" + std::to_string(E * GU_exp_sh) + "xf32>";
                const std::string t_gu2_i32 = "tensor<" + std::to_string(N_tok) + "x" + std::to_string(E * GU_exp_sh) + "xi32>";
                const std::string t_gu_f32  = "tensor<" + std::to_string(N_tok) + "x" + std::to_string(E) + "x" + std::to_string(GU_exp_sh) + "xf32>";
                const std::string t_ef_f32  = "tensor<" + std::to_string(N_tok) + "x" + std::to_string(E) + "x" + std::to_string(F_exp_sh) + "xf32>";
                const std::string t_ef_bf   = "tensor<" + std::to_string(N_tok) + "x" + std::to_string(E) + "x" + std::to_string(F_exp_sh) + "xbf16>";
                const std::string t_ef2_bf  = "tensor<" + std::to_string(N_tok) + "x" + std::to_string(E * F_exp_sh) + "xbf16>";
                const std::string t_wdn2    = "tensor<" + std::to_string(E * F_exp_sh) + "x" + std::to_string(D_pk) + "x" + wel + ">";
                const std::string t_wdn2b   = "tensor<" + std::to_string(E * F_exp_sh) + "x" + std::to_string(D) + "xbf16>";
                const std::string t_sdn42   = "tensor<" + std::to_string(E * F_exp_sh) + "x" + std::to_string(NG_D) + "xf32>";

                std::string n_io = em.tmp("nio"), n_io3 = em.tmp("nio3"), ti3 = em.tmp("ti3"), sc_idx = em.tmp("msc");
                std::string z_ne = em.tmp("zne"), w_rt = em.tmp("wrt");
                em.ss << "    " << n_io   << " = stablehlo.iota dim = 0 : " << t_nk_i32 << "\n"
                      << "    " << n_io3  << " = stablehlo.reshape " << n_io << " : (" << t_nk_i32 << ") -> " << t_nk1_i32 << "\n"
                      << "    " << ti3    << " = stablehlo.reshape " << ti_v << " : (" << t_nk_i32 << ") -> " << t_nk1_i32 << "\n"
                      << "    " << sc_idx << " = stablehlo.concatenate " << n_io3 << ", " << ti3 << ", dim = 2 : (" << t_nk1_i32 << ", " << t_nk1_i32 << ") -> " << t_nk2_i32 << "\n"
                      << "    " << z_ne   << " = stablehlo.constant dense<0.0000000e+00> : " << t_ne_f32 << "\n"
                      << "    " << w_rt   << " = \"stablehlo.scatter\"(" << z_ne << ", " << sc_idx << ", " << tw << ") ({\n"
                      << "    ^bb0(%old: tensor<f32>, %new: tensor<f32>):\n"
                      << "      stablehlo.return %new : tensor<f32>\n"
                      << "    }) {\n"
                      << "      scatter_dimension_numbers = #stablehlo.scatter<\n"
                      << "        update_window_dims = [],\n"
                      << "        inserted_window_dims = [0, 1],\n"
                      << "        scatter_dims_to_operand_dims = [0, 1],\n"
                      << "        index_vector_dim = 2\n"
                      << "      >,\n"
                      << "      indices_are_sorted = false,\n"
                      << "      unique_indices = true\n"
                      << "    } : (" << t_ne_f32 << ", " << t_nk2_i32 << ", " << t_nk_f32 << ") -> " << t_ne_f32 << "\n";

                std::string wgu_2d = em.tmp("wgu2");
                em.ss << "    " << wgu_2d << " = stablehlo.reshape " << wgu_l << " : (" << t_wgu3 << ") -> " << t_wgu2 << "\n";
                std::string gu_cur;
                if (i4) {
                    std::string sgu_2d = em.tmp("sgu42");
                    em.ss << "    " << sgu_2d << " = stablehlo.reshape " << sgu_l << " : (" << t_sgu3 << ") -> " << t_sgu42 << "\n";
                    std::string wgu_bf = em.emit_unpack_s4_2d(wgu_2d, sgu_2d, E * GU_exp_sh, D);
                    std::string gu_dot2 = em.tmp("gud2"), gu_dot = em.tmp("gudot");
                    em.ss << "    " << gu_dot2 << " = stablehlo.dot_general " << xm2_bf << ", " << wgu_bf
                          << ", contracting_dims = [1] x [1] : (" << t_nd_bf << ", " << t_wgu2b << ") -> " << t_gu2_f << "\n"
                          << "    " << gu_dot  << " = stablehlo.reshape " << gu_dot2 << " : (" << t_gu2_f << ") -> " << t_gu_f32 << "\n";
                    gu_cur = gu_dot;
                } else if (i8) {
                    std::string xm2_f2 = xm2_f2_src;
                    if (xm2_f2.empty()) {
                        xm2_f2 = em.tmp("xm2f");
                        em.ss << "    " << xm2_f2 << " = stablehlo.reshape " << x_m2 << " : (" << t_bsd_f32 << ") -> " << t_nd_f32 << "\n";
                    }
                    auto [xm2_i8, xm2_sc] = em.emit_quant_i8_per_row(xm2_f2, N_tok, D, c0_f32);
                    std::string gu_i32 = em.tmp("gui32"), gu_f2 = em.tmp("guf2"), gu_dot = em.tmp("gudot");
                    std::string xscb = em.tmp("xscb"), sgub = em.tmp("sgub"), gu_s1 = em.tmp("gus1"), gu_sc = em.tmp("gusc");
                    em.ss << "    " << gu_i32 << " = stablehlo.dot_general " << xm2_i8 << ", " << wgu_2d
                          << ", contracting_dims = [1] x [1] : (" << t_nd_i8 << ", " << t_wgu2 << ") -> " << t_gu2_i32 << "\n"
                          << "    " << gu_f2  << " = stablehlo.convert " << gu_i32 << " : (" << t_gu2_i32 << ") -> " << t_gu2_f << "\n"
                          << "    " << gu_dot << " = stablehlo.reshape " << gu_f2 << " : (" << t_gu2_f << ") -> " << t_gu_f32 << "\n"
                          << "    " << xscb   << " = stablehlo.broadcast_in_dim " << xm2_sc << ", dims = [0] : (" << t_n1_f32 << ") -> " << t_gu_f32 << "\n"
                          << "    " << gu_s1  << " = stablehlo.multiply " << gu_dot << ", " << xscb << " : " << t_gu_f32 << "\n"
                          << "    " << sgub   << " = stablehlo.broadcast_in_dim " << sgu_l << ", dims = [1, 2] : (" << t_sgu2 << ") -> " << t_gu_f32 << "\n"
                          << "    " << gu_sc  << " = stablehlo.multiply " << gu_s1 << ", " << sgub << " : " << t_gu_f32 << "\n";
                    gu_cur = gu_sc;
                } else {
                    std::string gu_dot2 = em.tmp("gud2"), gu_dot = em.tmp("gudot");
                    em.ss << "    " << gu_dot2 << " = stablehlo.dot_general " << xm2_bf << ", " << wgu_2d
                          << ", contracting_dims = [1] x [1] : (" << t_nd_bf << ", " << t_wgu2 << ") -> " << t_gu2_f << "\n"
                          << "    " << gu_dot  << " = stablehlo.reshape " << gu_dot2 << " : (" << t_gu2_f << ") -> " << t_gu_f32 << "\n";
                    gu_cur = gu_dot;
                }
                std::string eg = em.tmp("eg"), eu = em.tmp("eu");
                em.ss << "    " << eg << " = stablehlo.slice " << gu_cur << " [0:" << N_tok << ", 0:" << E << ", 0:" << F_exp_sh << "] : (" << t_gu_f32 << ") -> " << t_ef_f32 << "\n"
                      << "    " << eu << " = stablehlo.slice " << gu_cur << " [0:" << N_tok << ", 0:" << E << ", " << F_exp_sh << ":" << GU_exp_sh << "] : (" << t_gu_f32 << ") -> " << t_ef_f32 << "\n";
                std::string eg_act = (spec.ffn_activation == FfnActivation::GELU)
                    ? em.emit_gelu_3d(eg, N_tok, E, F_exp_sh)
                    : em.emit_silu_3d(eg, N_tok, E, F_exp_sh);
                std::string eact = em.tmp("eact"), wrtb = em.tmp("wrtb"), eact_w = em.tmp("eactw");
                em.ss << "    " << eact   << " = stablehlo.multiply " << eg_act << ", " << eu << " : " << t_ef_f32 << "\n"
                      << "    " << wrtb   << " = stablehlo.broadcast_in_dim " << w_rt << ", dims = [0, 1] : (" << t_ne_f32 << ") -> " << t_ef_f32 << "\n"
                      << "    " << eact_w << " = stablehlo.multiply " << eact << ", " << wrtb << " : " << t_ef_f32 << "\n";
                std::string eact_sc = eact_w;
                if (i8) {
                    std::string sdnb = em.tmp("sdnb"), eact_s = em.tmp("eas");
                    em.ss << "    " << sdnb   << " = stablehlo.broadcast_in_dim " << sdn_l << ", dims = [1, 2] : (" << t_sdn2 << ") -> " << t_ef_f32 << "\n"
                          << "    " << eact_s << " = stablehlo.multiply " << eact_w << ", " << sdnb << " : " << t_ef_f32 << "\n";
                    eact_sc = eact_s;
                }
                std::string eact_bf = em.tmp("eab"), eact_2d = em.tmp("ea2"), wdn_2d = em.tmp("wdn2");
                em.ss << "    " << eact_bf << " = stablehlo.convert " << eact_sc << " : (" << t_ef_f32 << ") -> " << t_ef_bf << "\n"
                      << "    " << eact_2d << " = stablehlo.reshape " << eact_bf << " : (" << t_ef_bf << ") -> " << t_ef2_bf << "\n"
                      << "    " << wdn_2d  << " = stablehlo.reshape " << wdn_l   << " : (" << t_wdn3  << ") -> " << t_wdn2 << "\n";
                std::string wdn_bf = wdn_2d;
                if (i4) {
                    std::string sdn_2d = em.tmp("sdn42");
                    em.ss << "    " << sdn_2d << " = stablehlo.reshape " << sdn_l << " : (" << t_sdn3 << ") -> " << t_sdn42 << "\n";
                    wdn_bf = em.emit_unpack_s4_2d(wdn_2d, sdn_2d, E * F_exp_sh, D);
                } else if (i8) {
                    wdn_bf = em.tmp("wdnb");
                    em.ss << "    " << wdn_bf << " = stablehlo.convert " << wdn_2d << " : (" << t_wdn2 << ") -> " << t_wdn2b << "\n";
                }
                moe_2d = em.tmp("moe2");
                em.ss << "    " << moe_2d << " = stablehlo.dot_general " << eact_2d << ", " << wdn_bf
                      << ", contracting_dims = [1] x [0] : (" << t_ef2_bf << ", " << t_wdn2b << ") -> " << t_nd_f32 << "\n";
            }

            std::string moe_loc = em.tmp("moel");
            em.ss << "    " << moe_loc << " = stablehlo.reshape " << moe_2d << " : (" << t_nd_f32 << ") -> " << t_bsd_f32 << "\n";

            // Fused Single Full-F32 Optical ICI AllReduce for BOTH Shared-MLP and Sparse-MoE branches!
            std::string mlp_ar = f_down_loc;
            std::string moe_ar = moe_loc;
            if (R > 1) {
                const std::string t_2bsd = "tensor<" + std::to_string(2 * B) + "x" + std::to_string(S) + "x" + std::to_string(D) + "xf32>";
                std::string both_loc = em.tmp("bothl");
                em.ss << "    " << both_loc << " = stablehlo.concatenate " << f_down_loc << ", " << moe_loc << ", dim = 0 : ("
                      << t_bsd_f32 << ", " << t_bsd_f32 << ") -> " << t_2bsd << "\n";
                std::string both_ar = em.emit_all_reduce_sum_3d(both_loc, 2 * B, S, D, R);
                mlp_ar = em.tmp("mlpar");
                moe_ar = em.tmp("moear");
                em.ss << "    " << mlp_ar << " = stablehlo.slice " << both_ar << " [0:" << B << ", 0:" << S << ", 0:" << D << "] : (" << t_2bsd << ") -> " << t_bsd_f32 << "\n"
                      << "    " << moe_ar << " = stablehlo.slice " << both_ar << " [" << B << ":" << (2 * B) << ", 0:" << S << ", 0:" << D << "] : (" << t_2bsd << ") -> " << t_bsd_f32 << "\n";
            }

            if (spec.has_post_ffn_norm_1) {
                std::string p1_nw = em.slice_1d_from_2d("%w_post_ffn_norm_1", l, L, D);
                mlp_ar = em.emit_rms_norm_3d(mlp_ar, p1_nw, B, S, D, spec.rms_norm_eps, c0_f32);
            }
            if (spec.has_post_ffn_norm_2) {
                std::string p2_nw = em.slice_1d_from_2d("%w_post_ffn_norm_2", l, L, D);
                moe_ar = em.emit_rms_norm_3d(moe_ar, p2_nw, B, S, D, spec.rms_norm_eps, c0_f32);
            }
            std::string f_comb = em.tmp("fcomb");
            em.ss << "    " << f_comb << " = stablehlo.add " << mlp_ar << ", " << moe_ar << " : " << t_bsd_f32 << "\n";
            f_final = f_comb;
            if (spec.has_post_ffn_norm) {
                std::string pf_nw = em.slice_1d_from_2d("%w_post_ffn_norm", l, L, D);
                f_final = em.emit_rms_norm_3d(f_comb, pf_nw, B, S, D, spec.rms_norm_eps, c0_f32);
            }
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
        em.ss << "    " << x_last << " = stablehlo.dynamic_slice " << x_cur << ", " << c0_i32 << ", %valid_last, " << c0_i32
              << ", sizes = [" << B << ", 1, " << D << "] : (" << t_bsd_f32
              << ", tensor<i32>, tensor<i32>, tensor<i32>) -> " << t_b1d_f32 << "\n";
    }
    std::string x_final_n = em.emit_rms_norm_3d(x_last, "%w_out_norm", B, 1, D, spec.rms_norm_eps, c0_f32);
    em.ss << "    %xf_2d = stablehlo.reshape " << x_final_n << " : (" << t_b1d_f32 << ") -> " << t_bd_f32 << "\n"
          << "    %xf_bf = stablehlo.convert %xf_2d : (" << t_bd_f32 << ") -> " << t_bd_bf << "\n";

    std::string lm_w = "%w_lm_head";
    if (has_q) {
        em.ss << "    %lm_wbf = stablehlo.convert %w_lm_head : (tensor<" << V_sh << "x" << D << "xi8>) -> tensor<" << V_sh << "x" << D << "xbf16>\n";
        lm_w = "%lm_wbf";
    }
    em.ss << "    %logits_raw = stablehlo.dot_general %xf_bf, " << lm_w << ", contracting_dims = [1] x [1] : ("
          << t_bd_bf << ", tensor<" << V_sh << "x" << D << "xbf16>) -> " << t_bvsh_f32 << "\n";

    std::string logits_sc = "%logits_raw";
    if (has_q) {
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
    for (int i = 0; i < N_swa_kv; ++i) {
        em.ss << ", " << kv_ks[static_cast<size_t>(i)] << ", " << kv_vs[static_cast<size_t>(i)];
    }
    for (int i = 0; i < N_glob_kv; ++i) {
        em.ss << ", " << kv_kg[static_cast<size_t>(i)] << ", " << kv_vg[static_cast<size_t>(i)];
    }
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
    const bool i4 = cfg_.use_int4_weights;
    const bool i8 = (!i4 && cfg_.use_int8_weights);
    const bool has_q = (i8 || i4);
    const int T = cfg_.max_seq_len;

    const int V_sh = V / R;
    const int H_sh = H / R;
    const int Hkv_swa = (spec_.num_kv_heads_swa > 0) ? spec_.num_kv_heads_swa : Hkv;
    const int Hkv_glob = (spec_.num_kv_heads_glob > 0) ? spec_.num_kv_heads_glob : Hkv;
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

    const int h_qkv    = (model_cfg_.turbo_hadamard_qkv > 0 && (D % static_cast<int>(model_cfg_.turbo_hadamard_qkv)) == 0) ? static_cast<int>(model_cfg_.turbo_hadamard_qkv) : 0;
    const int h_ffn    = (model_cfg_.turbo_hadamard_ffn > 0 && (D % static_cast<int>(model_cfg_.turbo_hadamard_ffn)) == 0) ? static_cast<int>(model_cfg_.turbo_hadamard_ffn) : 0;
    const int h_moe_in = static_cast<int>(model_cfg_.turbo_hadamard_moe_in);
    const int h_moe_dn = static_cast<int>(model_cfg_.turbo_hadamard_moe_down);
    const bool t_folded_norms  = model_cfg_.turbo_folded_norms;
    const bool t_folded_scales = model_cfg_.turbo_folded_scales;

    auto unrotate_matrix_rows = [](std::vector<float>& mat, int rows, int cols, int blk) {
        if (blk <= 0 || cols <= 0 || (cols % blk) != 0) return;
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < rows; ++r) {
            Avx2Math::fwht_block_inplace(mat.data() + static_cast<size_t>(r) * static_cast<size_t>(cols), cols, blk);
        }
    };

    auto upload_sharded_stack = [&](const std::vector<int>& layer_list,
                                    const std::function<const TensorDesc*(int)>& getter,
                                    int full_rows, int full_cols,
                                    bool shard_rows, bool shard_cols,
                                    bool is_kv_head_shard,
                                    int layer_hkv = 0,
                                    int unrot_fwht_blk = 0) -> bool {
        const int cnt = static_cast<int>(layer_list.size());
        if (cnt <= 0) return true;
        const int eff_hkv = (layer_hkv > 0) ? layer_hkv : Hkv;
        const int eff_hkv_sh = compute_hkv_shard(eff_hkv, R);
        const int eff_gqa = std::max(1, H / eff_hkv);
        const int r_sh = shard_rows ? (is_kv_head_shard ? (full_rows / eff_hkv * eff_hkv_sh) : (full_rows / R)) : full_rows;
        const int c_sh = shard_cols ? (full_cols / R) : full_cols;
        const int c_pk = i4 ? (c_sh / 2) : c_sh;
        const int g_w4 = i4 ? compute_w4_group_size(c_sh) : 1;
        const int ng_w4 = i4 ? (c_sh / g_w4) : 1;

        const size_t shard_elems = static_cast<size_t>(cnt) * static_cast<size_t>(r_sh) * static_cast<size_t>(c_pk);
        const size_t sc_elems    = static_cast<size_t>(cnt) * static_cast<size_t>(r_sh) * static_cast<size_t>(ng_w4);
        std::vector<std::vector<uint16_t>> bf16_shards(has_q ? 0 : static_cast<size_t>(R));
        std::vector<std::vector<int8_t>>   s8_shards(has_q ? static_cast<size_t>(R) : 0);
        std::vector<std::vector<float>>    sc_shards(has_q ? static_cast<size_t>(R) : 0);
        for (int r = 0; r < R; ++r) {
            if (!has_q) bf16_shards[static_cast<size_t>(r)].resize(shard_elems);
            else {
                s8_shards[static_cast<size_t>(r)].resize(shard_elems);
                sc_shards[static_cast<size_t>(r)].resize(sc_elems);
            }
        }

        std::vector<float> full_f32;
        std::vector<float> slice_f32(static_cast<size_t>(r_sh) * static_cast<size_t>(c_sh));
        for (int idx = 0; idx < cnt; ++idx) {
            const TensorDesc* td = getter(layer_list[static_cast<size_t>(idx)]);
            dequantize_full_matrix_f32(td, full_rows, full_cols, full_f32);
            if (unrot_fwht_blk > 0) unrotate_matrix_rows(full_f32, full_rows, full_cols, unrot_fwht_blk);

            for (int rep = 0; rep < R; ++rep) {
                int r_off = 0;
                if (shard_rows) {
                    if (is_kv_head_shard) {
                        int kv_head_start = (rep * H_sh) / eff_gqa;
                        r_off = kv_head_start * (full_rows / eff_hkv);
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

                const size_t dst_off = static_cast<size_t>(idx) * static_cast<size_t>(r_sh) * static_cast<size_t>(c_pk);
                const size_t sc_off  = static_cast<size_t>(idx) * static_cast<size_t>(r_sh) * static_cast<size_t>(ng_w4);
                if (!has_q) {
                    HavenTpuPjrtBackend::convert_f32_to_bf16(slice_f32.data(), bf16_shards[static_cast<size_t>(rep)].data() + dst_off, slice_f32.size());
                } else if (i4) {
                    quantize_f32_to_s4_packed_group(slice_f32.data(), r_sh, c_sh, g_w4,
                        s8_shards[static_cast<size_t>(rep)].data() + dst_off,
                        sc_shards[static_cast<size_t>(rep)].data() + sc_off);
                } else {
                    quantize_f32_to_s8_per_row(slice_f32.data(), r_sh, c_sh,
                        s8_shards[static_cast<size_t>(rep)].data() + dst_off,
                        sc_shards[static_cast<size_t>(rep)].data() + sc_off);
                }
            }
        }

        for (int rep = 0; rep < R; ++rep) {
            const int dev = replica_dev_indices_[static_cast<size_t>(rep)];
            PJRT_Buffer* wb = nullptr;
            if (!has_q) {
                wb = backend_->upload_bf16_nd(bf16_shards[static_cast<size_t>(rep)].data(), {cnt, r_sh, c_sh}, dev);
                total_bytes += shard_elems * sizeof(uint16_t);
            } else {
                wb = backend_->upload_s8_nd(s8_shards[static_cast<size_t>(rep)].data(), {cnt, r_sh, c_pk}, dev);
                PJRT_Buffer* sb = i4
                    ? backend_->upload_f32_nd(sc_shards[static_cast<size_t>(rep)].data(), {cnt, r_sh, ng_w4}, dev)
                    : backend_->upload_f32_nd(sc_shards[static_cast<size_t>(rep)].data(), {cnt, r_sh}, dev);
                if (!sb) { backend_->destroy_buffer(wb); return false; }
                replica_scale_bufs[static_cast<size_t>(rep)].push_back(sb);
                total_bytes += shard_elems + sc_elems * sizeof(float);
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
            if (!has_q) {
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
                            double max_abs = 0.0;
                            for (int c = 0; c < cols; ++c) {
                                const double a = std::fabs(static_cast<double>(row_f32[static_cast<size_t>(c)]));
                                if (a > max_abs) max_abs = a;
                            }
                            int8_t* dst_r = sq.data() + static_cast<size_t>(r) * static_cast<size_t>(cols);
                            if (max_abs <= 1e-12) {
                                sc[static_cast<size_t>(r)] = 1.0f;
                                std::memset(dst_r, 0, static_cast<size_t>(cols));
                                continue;
                            }
                            const double inv_scale = 127.0 / max_abs;
                            double sum_wq = 0.0;
                            double sum_qq = 0.0;
                            for (int c = 0; c < cols; ++c) {
                                const double w = static_cast<double>(row_f32[static_cast<size_t>(c)]);
                                int q = static_cast<int>(std::lrint(w * inv_scale));
                                q = std::clamp(q, -127, 127);
                                dst_r[c] = static_cast<int8_t>(q);
                                const double qd = static_cast<double>(q);
                                sum_wq += w * qd;
                                sum_qq += qd * qd;
                            }
                            // Closed-form FP64 least-squares optimal scale minimizes L2 quantization error on vocab/LM head
                            sc[static_cast<size_t>(r)] = (sum_qq > 0.0)
                                ? static_cast<float>(sum_wq / sum_qq)
                                : static_cast<float>(max_abs / 127.0);
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

    // 1. Stacked layer projection matrices (fused FFN Gate+Up [L, 2*F_sh, D(_pk)] + Down [L, D, F_sh(_pk)])
    // Supports per-layer variable FFN width (e.g. Gemma 4/4.5 E2B double-wide shared-KV MLP: 6144 -> 12288)
    // by zero-padding shorter layers per-replica so GeGLU(0, 0) * 0 == 0 with exact mathematical parity!
    {
        const int F_sh = F / R;
        const int GU_sh = 2 * F_sh;
        const int D_pk = i4 ? (D / 2) : D;
        const int F_pk = i4 ? (F_sh / 2) : F_sh;
        const int G_D  = i4 ? compute_w4_group_size(D) : 1;
        const int NG_D = i4 ? (D / G_D) : 1;
        const int G_F  = i4 ? compute_w4_group_size(F_sh) : 1;
        const int NG_F = i4 ? (F_sh / G_F) : 1;
        const size_t gu_shard_elems = static_cast<size_t>(L) * static_cast<size_t>(GU_sh) * static_cast<size_t>(D_pk);
        const size_t dn_shard_elems = static_cast<size_t>(L) * static_cast<size_t>(D) * static_cast<size_t>(F_pk);
        const size_t gu_sc_elems    = static_cast<size_t>(L) * static_cast<size_t>(GU_sh) * static_cast<size_t>(NG_D);
        const size_t dn_sc_elems    = static_cast<size_t>(L) * static_cast<size_t>(D) * static_cast<size_t>(NG_F);
        std::vector<std::vector<uint16_t>> gu_bf16(static_cast<size_t>(R)), dn_bf16(static_cast<size_t>(R));
        std::vector<std::vector<int8_t>>   gu_s8(static_cast<size_t>(R)),   dn_s8(static_cast<size_t>(R));
        std::vector<std::vector<float>>    gu_sc(static_cast<size_t>(R)),   dn_sc(static_cast<size_t>(R));
        for (int rep = 0; rep < R; ++rep) {
            if (!has_q) {
                gu_bf16[static_cast<size_t>(rep)].assign(gu_shard_elems, 0);
                dn_bf16[static_cast<size_t>(rep)].assign(dn_shard_elems, 0);
            } else {
                gu_s8[static_cast<size_t>(rep)].assign(gu_shard_elems, 0);
                gu_sc[static_cast<size_t>(rep)].assign(gu_sc_elems, 1.0f);
                dn_s8[static_cast<size_t>(rep)].assign(dn_shard_elems, 0);
                dn_sc[static_cast<size_t>(rep)].assign(dn_sc_elems, 1.0f);
            }
        }
        std::vector<float> gate_f32, up_f32, down_f32;
        std::vector<float> dn_slice(static_cast<size_t>(D) * static_cast<size_t>(F_sh));
        for (int l = 0; l < L; ++l) {
            const auto& lt = host_engine.get_layer_tensors(static_cast<uint32_t>(l));
            const int layer_F = (lt.ffn_gate && lt.ffn_gate->shape.size() > 1)
                ? std::min<int>(F, static_cast<int>(lt.ffn_gate->shape[1]))
                : ((static_cast<size_t>(l) < model_cfg_.hidden_dim_per_layer.size() && model_cfg_.hidden_dim_per_layer[static_cast<size_t>(l)] > 0)
                    ? std::min<int>(F, static_cast<int>(model_cfg_.hidden_dim_per_layer[static_cast<size_t>(l)]))
                    : F);
            const int lF_sh = layer_F / R;
            const int l_h_ffn_dn = (model_cfg_.turbo_hadamard_ffn_down > 0 && (layer_F % static_cast<int>(model_cfg_.turbo_hadamard_ffn_down)) == 0)
                ? static_cast<int>(model_cfg_.turbo_hadamard_ffn_down) : 0;

            dequantize_full_matrix_f32(lt.ffn_gate, layer_F, D, gate_f32);
            dequantize_full_matrix_f32(lt.ffn_up,   layer_F, D, up_f32);
            dequantize_full_matrix_f32(lt.ffn_down, D, layer_F, down_f32);
            if (h_ffn > 0) {
                unrotate_matrix_rows(gate_f32, layer_F, D, h_ffn);
                unrotate_matrix_rows(up_f32,   layer_F, D, h_ffn);
            }
            if (l_h_ffn_dn > 0) {
                unrotate_matrix_rows(down_f32, D, layer_F, l_h_ffn_dn);
            }

            for (int rep = 0; rep < R; ++rep) {
                const size_t base_el = static_cast<size_t>(l) * static_cast<size_t>(GU_sh) * static_cast<size_t>(D_pk);
                const size_t base_sc = static_cast<size_t>(l) * static_cast<size_t>(GU_sh) * static_cast<size_t>(NG_D);
                const float* g_src = gate_f32.data() + static_cast<size_t>(rep * lF_sh) * static_cast<size_t>(D);
                const float* u_src = up_f32.data()   + static_cast<size_t>(rep * lF_sh) * static_cast<size_t>(D);
                if (!has_q) {
                    HavenTpuPjrtBackend::convert_f32_to_bf16(g_src, gu_bf16[static_cast<size_t>(rep)].data() + base_el, static_cast<size_t>(lF_sh) * static_cast<size_t>(D));
                    HavenTpuPjrtBackend::convert_f32_to_bf16(u_src, gu_bf16[static_cast<size_t>(rep)].data() + base_el + static_cast<size_t>(F_sh) * static_cast<size_t>(D), static_cast<size_t>(lF_sh) * static_cast<size_t>(D));
                } else if (i4) {
                    quantize_f32_to_s4_packed_group(g_src, lF_sh, D, G_D,
                        gu_s8[static_cast<size_t>(rep)].data() + base_el,
                        gu_sc[static_cast<size_t>(rep)].data() + base_sc);
                    quantize_f32_to_s4_packed_group(u_src, lF_sh, D, G_D,
                        gu_s8[static_cast<size_t>(rep)].data() + base_el + static_cast<size_t>(F_sh) * static_cast<size_t>(D_pk),
                        gu_sc[static_cast<size_t>(rep)].data() + base_sc + static_cast<size_t>(F_sh) * static_cast<size_t>(NG_D));
                } else {
                    quantize_f32_to_s8_per_row(g_src, lF_sh, D,
                        gu_s8[static_cast<size_t>(rep)].data() + base_el,
                        gu_sc[static_cast<size_t>(rep)].data() + base_sc);
                    quantize_f32_to_s8_per_row(u_src, lF_sh, D,
                        gu_s8[static_cast<size_t>(rep)].data() + base_el + static_cast<size_t>(F_sh) * static_cast<size_t>(D),
                        gu_sc[static_cast<size_t>(rep)].data() + base_sc + static_cast<size_t>(F_sh));
                }

                std::fill(dn_slice.begin(), dn_slice.end(), 0.0f);
                const int c_off = rep * lF_sh;
                for (int rr = 0; rr < D; ++rr) {
                    const float* src_row = down_f32.data() + static_cast<size_t>(rr) * static_cast<size_t>(layer_F) + static_cast<size_t>(c_off);
                    std::memcpy(dn_slice.data() + static_cast<size_t>(rr) * static_cast<size_t>(F_sh),
                                src_row, static_cast<size_t>(lF_sh) * sizeof(float));
                }
                const size_t dn_off    = static_cast<size_t>(l) * static_cast<size_t>(D) * static_cast<size_t>(F_pk);
                const size_t dn_sc_off = static_cast<size_t>(l) * static_cast<size_t>(D) * static_cast<size_t>(NG_F);
                if (!has_q) {
                    HavenTpuPjrtBackend::convert_f32_to_bf16(dn_slice.data(), dn_bf16[static_cast<size_t>(rep)].data() + dn_off, dn_slice.size());
                } else if (i4) {
                    quantize_f32_to_s4_packed_group(dn_slice.data(), D, F_sh, G_F,
                        dn_s8[static_cast<size_t>(rep)].data() + dn_off,
                        dn_sc[static_cast<size_t>(rep)].data() + dn_sc_off);
                } else {
                    quantize_f32_to_s8_per_row(dn_slice.data(), D, F_sh,
                        dn_s8[static_cast<size_t>(rep)].data() + dn_off,
                        dn_sc[static_cast<size_t>(rep)].data() + dn_sc_off);
                }
            }
        }
        for (int rep = 0; rep < R; ++rep) {
            const int dev = replica_dev_indices_[static_cast<size_t>(rep)];
            PJRT_Buffer* wb_gu = nullptr;
            PJRT_Buffer* wb_dn = nullptr;
            if (!has_q) {
                wb_gu = backend_->upload_bf16_nd(gu_bf16[static_cast<size_t>(rep)].data(), {L, GU_sh, D}, dev);
                wb_dn = backend_->upload_bf16_nd(dn_bf16[static_cast<size_t>(rep)].data(), {L, D, F_sh}, dev);
                total_bytes += (gu_shard_elems + dn_shard_elems) * sizeof(uint16_t);
            } else {
                wb_gu = backend_->upload_s8_nd(gu_s8[static_cast<size_t>(rep)].data(), {L, GU_sh, D_pk}, dev);
                PJRT_Buffer* sb_gu = i4
                    ? backend_->upload_f32_nd(gu_sc[static_cast<size_t>(rep)].data(), {L, GU_sh, NG_D}, dev)
                    : backend_->upload_f32_nd(gu_sc[static_cast<size_t>(rep)].data(), {L, GU_sh}, dev);
                wb_dn = backend_->upload_s8_nd(dn_s8[static_cast<size_t>(rep)].data(), {L, D, F_pk}, dev);
                PJRT_Buffer* sb_dn = i4
                    ? backend_->upload_f32_nd(dn_sc[static_cast<size_t>(rep)].data(), {L, D, NG_F}, dev)
                    : backend_->upload_f32_nd(dn_sc[static_cast<size_t>(rep)].data(), {L, D}, dev);
                if (!sb_gu || !sb_dn) {
                    backend_->destroy_buffer(wb_gu); backend_->destroy_buffer(sb_gu);
                    backend_->destroy_buffer(wb_dn); backend_->destroy_buffer(sb_dn);
                    return false;
                }
                replica_scale_bufs[static_cast<size_t>(rep)].push_back(sb_gu);
                replica_scale_bufs[static_cast<size_t>(rep)].push_back(sb_dn);
                total_bytes += gu_shard_elems + dn_shard_elems + (gu_sc_elems + dn_sc_elems) * sizeof(float);
            }
            if (!wb_gu || !wb_dn) {
                backend_->destroy_buffer(wb_gu);
                backend_->destroy_buffer(wb_dn);
                return false;
            }
            replica_weight_bufs_[static_cast<size_t>(rep)].push_back(wb_gu);
            replica_weight_bufs_[static_cast<size_t>(rep)].push_back(wb_dn);
        }
    }

    // 1b. Sparse Mixture-of-Experts 4D weight stacks (streamed replica-by-replica to keep peak host RAM minimal!)
    if (spec_.is_moe) {
        const int E = spec_.num_experts;
        const int F_exp = spec_.expert_ffn_dim;
        const int F_exp_sh = F_exp / R;
        const int GU_exp_sh = 2 * F_exp_sh;
        const bool use_moe_hadamard = (has_q && (D % 256) == 0);
        const int D_pk = i4 ? (D / 2) : D;
        const int G_D  = i4 ? compute_w4_group_size(D) : 1;
        const int NG_D = i4 ? (D / G_D) : 1;

        // %w_moe_router: [L, E, D] BF16 (replicated across R chips)
        {
            const size_t r_elems = static_cast<size_t>(L) * static_cast<size_t>(E) * static_cast<size_t>(D);
            std::vector<uint16_t> r_bf(r_elems, 0);
            std::vector<float> mat;
            for (int l = 0; l < L; ++l) {
                dequantize_full_matrix_f32(host_engine.get_layer_tensors(l).ffn_gate_inp, E, D, mat);
                HavenTpuPjrtBackend::convert_f32_to_bf16(mat.data(), r_bf.data() + static_cast<size_t>(l) * mat.size(), mat.size());
            }
            for (int rep = 0; rep < R; ++rep) {
                const int dev = replica_dev_indices_[static_cast<size_t>(rep)];
                PJRT_Buffer* b = backend_->upload_bf16_nd(r_bf.data(), {L, E, D}, dev);
                if (!b) return false;
                replica_weight_bufs_[static_cast<size_t>(rep)].push_back(b);
                total_bytes += r_elems * sizeof(uint16_t);
            }
        }

        // %w_moe_gu_0..L-1: L separate 3D buffers [E, GU_exp_sh, D(_pk)] in TPU (8,128)-aligned layout!
        {
            const size_t layer_elems = static_cast<size_t>(E) * static_cast<size_t>(GU_exp_sh) * static_cast<size_t>(D_pk);
            const size_t layer_scales = static_cast<size_t>(E) * static_cast<size_t>(GU_exp_sh) * static_cast<size_t>(NG_D);
            std::vector<uint16_t> bf_buf(has_q ? 0 : layer_elems, 0);
            std::vector<int8_t>   s8_buf(has_q ? layer_elems : 0, 0);
            std::vector<float>    sc_buf(has_q ? layer_scales : 0, 1.0f);

            for (int l = 0; l < L; ++l) {
                const auto& lt = host_engine.get_layer_tensors(static_cast<uint32_t>(l));
                const bool fused_gu = (lt.ffn_gate_up_exps != nullptr);
                for (int rep = 0; rep < R; ++rep) {
                    const int dev = replica_dev_indices_[static_cast<size_t>(rep)];
                    #pragma omp parallel
                    {
                        std::vector<float> row_f32(static_cast<size_t>(D), 0.0f);
                        #pragma omp for schedule(static)
                        for (int e = 0; e < E; ++e) {
                            const size_t exp_base = static_cast<size_t>(e) * static_cast<size_t>(GU_exp_sh) * static_cast<size_t>(D_pk);
                            const size_t sc_base  = static_cast<size_t>(e) * static_cast<size_t>(GU_exp_sh) * static_cast<size_t>(NG_D);
                            for (int half = 0; half < 2; ++half) {
                                const TensorDesc* src_td = fused_gu ? lt.ffn_gate_up_exps : (half == 0 ? lt.ffn_gate_exps : lt.ffn_up_exps);
                                for (int rr = 0; rr < F_exp_sh; ++rr) {
                                    const int col_out = half * F_exp_sh + rr;
                                    const int src_row = fused_gu
                                        ? (e * (2 * F_exp) + half * F_exp + rep * F_exp_sh + rr)
                                        : (e * F_exp + rep * F_exp_sh + rr);
                                    if (src_td && src_td->data) {
                                        Avx2Math::dequantize_row(row_f32.data(), *src_td, src_row, D);
                                    } else {
                                        std::fill(row_f32.begin(), row_f32.end(), 0.0f);
                                    }
                                    // Offline Block-256 Orthogonal Walsh-Hadamard pre-rotation (W' = W H)
                                    // Pairs with runtime x' = x H to crush 662x blk.29 spikes -> 5.8x Gaussian!
                                    // If GGUF is already Gemma 4.5 Turbo rotated (h_moe_in == 256), skip double-rotation!
                                    if (h_moe_in > 0 && h_moe_in != 256 && (D % h_moe_in) == 0) {
                                        Avx2Math::fwht_block_inplace(row_f32.data(), D, h_moe_in);
                                    }
                                    const bool already_rot_256 = (h_moe_in == 256);
                                    if (use_moe_hadamard != already_rot_256) {
                                        for (int blk = 0; blk < D; blk += 256) {
                                            fwht_inplace(row_f32.data() + blk, 256);
                                        }
                                    }
                                    const size_t dst_off = exp_base + static_cast<size_t>(col_out) * static_cast<size_t>(D_pk);
                                    if (!has_q) {
                                        HavenTpuPjrtBackend::convert_f32_to_bf16(row_f32.data(), bf_buf.data() + dst_off, static_cast<size_t>(D));
                                    } else if (i4) {
                                        quantize_row_f32_to_s4_packed_group(row_f32.data(), D, G_D,
                                            s8_buf.data() + dst_off,
                                            sc_buf.data() + sc_base + static_cast<size_t>(col_out) * static_cast<size_t>(NG_D));
                                    } else {
                                        float max_abs = 0.0f;
                                        for (int d = 0; d < D; ++d) {
                                            float a = std::fabs(row_f32[static_cast<size_t>(d)]);
                                            if (a > max_abs) max_abs = a;
                                        }
                                        const float scale = (max_abs > 1e-12f) ? (max_abs / 127.0f) : 1.0f;
                                        const float inv_scale = (max_abs > 1e-12f) ? (127.0f / max_abs) : 0.0f;
                                        sc_buf[sc_base + static_cast<size_t>(col_out)] = scale;
                                        int8_t* dst = s8_buf.data() + dst_off;
                                        for (int d = 0; d < D; ++d) {
                                            int q = static_cast<int>(std::lrintf(row_f32[static_cast<size_t>(d)] * inv_scale));
                                            dst[d] = static_cast<int8_t>(std::clamp(q, -127, 127));
                                        }
                                    }
                                }
                            }
                        }
                    }

                    PJRT_Buffer* wb = nullptr;
                    if (!has_q) {
                        wb = backend_->upload_bf16_nd(bf_buf.data(), {E, GU_exp_sh, D}, dev);
                        total_bytes += layer_elems * sizeof(uint16_t);
                    } else {
                        wb = backend_->upload_s8_nd(s8_buf.data(), {E, GU_exp_sh, D_pk}, dev);
                        PJRT_Buffer* sb = i4
                            ? backend_->upload_f32_nd(sc_buf.data(), {E, GU_exp_sh, NG_D}, dev)
                            : backend_->upload_f32_nd(sc_buf.data(), {E, GU_exp_sh}, dev);
                        if (!sb) { backend_->destroy_buffer(wb); return false; }
                        replica_scale_bufs[static_cast<size_t>(rep)].push_back(sb);
                        total_bytes += layer_elems + layer_scales * sizeof(float);
                    }
                    if (!wb) return false;
                    replica_weight_bufs_[static_cast<size_t>(rep)].push_back(wb);
                }
            }
        }

        // %w_moe_dn_0..L-1: L separate 3D buffers [E, F_exp_sh, D(_pk)] in XLA-native MXU layout!
        {
            const size_t layer_elems = static_cast<size_t>(E) * static_cast<size_t>(F_exp_sh) * static_cast<size_t>(D_pk);
            const size_t layer_scales = static_cast<size_t>(E) * static_cast<size_t>(F_exp_sh) * static_cast<size_t>(NG_D);
            std::vector<uint16_t> bf_buf(has_q ? 0 : layer_elems, 0);
            std::vector<int8_t>   s8_buf(has_q ? layer_elems : 0, 0);
            std::vector<float>    sc_buf(has_q ? layer_scales : 0, 1.0f);

            for (int l = 0; l < L; ++l) {
                const auto& lt = host_engine.get_layer_tensors(static_cast<uint32_t>(l));
                const auto down_s = !t_folded_scales
                    ? extract_f32_1d(lt.ffn_down_exps_scale, E, 1.0f)
                    : std::vector<float>(static_cast<size_t>(E), 1.0f);
                for (int rep = 0; rep < R; ++rep) {
                    const int dev = replica_dev_indices_[static_cast<size_t>(rep)];
                    const int col_off = rep * F_exp_sh;
                    #pragma omp parallel
                    {
                        std::vector<float> full_row(static_cast<size_t>(F_exp), 0.0f);
                        std::vector<float> exp_tr(static_cast<size_t>(F_exp_sh) * static_cast<size_t>(D), 0.0f);
                        #pragma omp for schedule(static)
                        for (int e = 0; e < E; ++e) {
                            const float es = down_s[static_cast<size_t>(e)];
                            // Dequantize [D, F_exp_sh] and transpose into [F_exp_sh, D] inside L2 cache
                            for (int d = 0; d < D; ++d) {
                                if (lt.ffn_down_exps && lt.ffn_down_exps->data) {
                                    Avx2Math::dequantize_row(full_row.data(), *lt.ffn_down_exps, e * D + d, F_exp);
                                    if (h_moe_dn > 0 && (F_exp % h_moe_dn) == 0) {
                                        Avx2Math::fwht_block_inplace(full_row.data(), F_exp, h_moe_dn);
                                    }
                                    for (int f = 0; f < F_exp_sh; ++f) {
                                        exp_tr[static_cast<size_t>(f) * static_cast<size_t>(D) + static_cast<size_t>(d)] =
                                            full_row[static_cast<size_t>(col_off + f)] * es;
                                    }
                                } else {
                                    for (int f = 0; f < F_exp_sh; ++f) {
                                        exp_tr[static_cast<size_t>(f) * static_cast<size_t>(D) + static_cast<size_t>(d)] = 0.0f;
                                    }
                                }
                            }
                            const size_t exp_base = static_cast<size_t>(e) * static_cast<size_t>(F_exp_sh) * static_cast<size_t>(D_pk);
                            const size_t sc_base  = static_cast<size_t>(e) * static_cast<size_t>(F_exp_sh) * static_cast<size_t>(NG_D);
                            if (!has_q) {
                                HavenTpuPjrtBackend::convert_f32_to_bf16(exp_tr.data(), bf_buf.data() + exp_base, exp_tr.size());
                            } else if (i4) {
                                for (int f = 0; f < F_exp_sh; ++f) {
                                    const float* frow = exp_tr.data() + static_cast<size_t>(f) * static_cast<size_t>(D);
                                    quantize_row_f32_to_s4_packed_group(frow, D, G_D,
                                        s8_buf.data() + exp_base + static_cast<size_t>(f) * static_cast<size_t>(D_pk),
                                        sc_buf.data() + sc_base + static_cast<size_t>(f) * static_cast<size_t>(NG_D));
                                }
                            } else {
                                for (int f = 0; f < F_exp_sh; ++f) {
                                    const float* frow = exp_tr.data() + static_cast<size_t>(f) * static_cast<size_t>(D);
                                    float max_abs = 0.0f;
                                    for (int d = 0; d < D; ++d) {
                                        float a = std::fabs(frow[d]);
                                        if (a > max_abs) max_abs = a;
                                    }
                                    const float scale = (max_abs > 1e-12f) ? (max_abs / 127.0f) : 1.0f;
                                    const float inv_scale = (max_abs > 1e-12f) ? (127.0f / max_abs) : 0.0f;
                                    sc_buf[sc_base + static_cast<size_t>(f)] = scale;
                                    int8_t* dst = s8_buf.data() + exp_base + static_cast<size_t>(f) * static_cast<size_t>(D);
                                    for (int d = 0; d < D; ++d) {
                                        int q = static_cast<int>(std::lrintf(frow[d] * inv_scale));
                                        dst[d] = static_cast<int8_t>(std::clamp(q, -127, 127));
                                    }
                                }
                            }
                        }
                    }

                    PJRT_Buffer* wb = nullptr;
                    if (!has_q) {
                        wb = backend_->upload_bf16_nd(bf_buf.data(), {E, F_exp_sh, D}, dev);
                        total_bytes += layer_elems * sizeof(uint16_t);
                    } else {
                        wb = backend_->upload_s8_nd(s8_buf.data(), {E, F_exp_sh, D_pk}, dev);
                        PJRT_Buffer* sb = i4
                            ? backend_->upload_f32_nd(sc_buf.data(), {E, F_exp_sh, NG_D}, dev)
                            : backend_->upload_f32_nd(sc_buf.data(), {E, F_exp_sh}, dev);
                        if (!sb) { backend_->destroy_buffer(wb); return false; }
                        replica_scale_bufs[static_cast<size_t>(rep)].push_back(sb);
                        total_bytes += layer_elems + layer_scales * sizeof(float);
                    }
                    if (!wb) return false;
                    replica_weight_bufs_[static_cast<size_t>(rep)].push_back(wb);
                }
            }
        }

        // %w_had_moe: single replicated [256, 256] BF16 exact power-of-two (±1/16 = ±0.0625) Sylvester Hadamard matrix (128 KB)
        if (use_moe_hadamard) {
            std::vector<float> h_f32(static_cast<size_t>(256) * 256);
            constexpr float inv16 = 0.0625f; // Exact 2^-4 in BF16 (zero mantissa rounding error!)
            for (int i = 0; i < 256; ++i) {
                for (int j = 0; j < 256; ++j) {
                    unsigned x = static_cast<unsigned>(i & j);
                    int parity = 0;
                    while (x) { parity ^= 1; x &= (x - 1); }
                    h_f32[static_cast<size_t>(i) * 256 + static_cast<size_t>(j)] = parity ? -inv16 : inv16;
                }
            }
            std::vector<uint16_t> h_bf(h_f32.size());
            HavenTpuPjrtBackend::convert_f32_to_bf16(h_f32.data(), h_bf.data(), h_f32.size());
            for (int rep = 0; rep < R; ++rep) {
                const int dev = replica_dev_indices_[static_cast<size_t>(rep)];
                PJRT_Buffer* hb = backend_->upload_bf16_nd(h_bf.data(), {256, 256}, dev);
                if (!hb) return false;
                replica_weight_bufs_[static_cast<size_t>(rep)].push_back(hb);
                total_bytes += h_bf.size() * sizeof(uint16_t);
            }
        }
    }

    if (N_swa > 0) {
        if (!upload_sharded_stack(swa_layers,    [&](int l){ return host_engine.get_layer_tensors(l).attn_q; },      H * HD_swa, D, true, false, false, 0, h_qkv)) return false;
        if (!upload_sharded_stack(swa_layers,    [&](int l){ return host_engine.get_layer_tensors(l).attn_output; }, D, H * HD_swa, false, true, false)) return false;
        if (N_swa_kv > 0) {
            if (!upload_sharded_stack(swa_kv_layers, [&](int l){ return host_engine.get_layer_tensors(l).attn_k; }, Hkv_swa * HD_swa, D, (R > 1), false, true, Hkv_swa, h_qkv)) return false;
            if (spec_.has_swa_v) {
                if (!upload_sharded_stack(swa_kv_layers, [&](int l){ return host_engine.get_layer_tensors(l).attn_v; }, Hkv_swa * HD_swa, D, (R > 1), false, true, Hkv_swa, h_qkv)) return false;
            }
        }
    }
    if (N_glob > 0) {
        if (!upload_sharded_stack(glob_layers,    [&](int l){ return host_engine.get_layer_tensors(l).attn_q; },      H * HD_glob, D, true, false, false, 0, h_qkv)) return false;
        if (!upload_sharded_stack(glob_layers,    [&](int l){ return host_engine.get_layer_tensors(l).attn_output; }, D, H * HD_glob, false, true, false)) return false;
        if (N_glob_kv > 0) {
            if (!upload_sharded_stack(glob_kv_layers, [&](int l){ return host_engine.get_layer_tensors(l).attn_k; }, Hkv_glob * HD_glob, D, (R > 1), false, true, Hkv_glob, h_qkv)) return false;
            if (spec_.has_glob_v) {
                if (!upload_sharded_stack(glob_kv_layers, [&](int l){ return host_engine.get_layer_tensors(l).attn_v; }, Hkv_glob * HD_glob, D, (R > 1), false, true, Hkv_glob, h_qkv)) return false;
            }
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

    // Append INT8 per-row or W4A16 per-group scales immediately after projection weights (matching MLIR signature)
    if (has_q) {
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
                                       int full_dim, bool is_kv_head_shard,
                                       int layer_hkv = 0) -> bool {
            const int cnt = static_cast<int>(layer_list.size());
            if (cnt <= 0) return true;
            const int eff_hkv = (layer_hkv > 0) ? layer_hkv : Hkv;
            const int eff_hkv_sh = compute_hkv_shard(eff_hkv, R);
            const int eff_gqa = std::max(1, H / eff_hkv);
            const int d_sh = is_kv_head_shard ? (full_dim / eff_hkv * eff_hkv_sh) : (full_dim / R);
            std::vector<std::vector<float>> rep_data(static_cast<size_t>(R),
                std::vector<float>(static_cast<size_t>(cnt) * static_cast<size_t>(d_sh), 0.0f));
            for (int idx = 0; idx < cnt; ++idx) {
                auto full_b = extract_f32_1d(getter(layer_list[static_cast<size_t>(idx)]), full_dim, 0.0f);
                for (int rep = 0; rep < R; ++rep) {
                    int off = is_kv_head_shard ? (((rep * H_sh) / eff_gqa) * (full_dim / eff_hkv)) : (rep * d_sh);
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
                if (!upload_sharded_bias(swa_kv_layers, [&](int l){ return host_engine.get_layer_tensors(l).attn_k_bias; }, Hkv_swa * HD_swa, true, Hkv_swa)) return false;
                if (spec_.has_swa_v) {
                    if (!upload_sharded_bias(swa_kv_layers, [&](int l){ return host_engine.get_layer_tensors(l).attn_v_bias; }, Hkv_swa * HD_swa, true, Hkv_swa)) return false;
                }
            }
        }
        if (N_glob > 0) {
            if (!upload_sharded_bias(glob_layers, [&](int l){ return host_engine.get_layer_tensors(l).attn_q_bias; }, H * HD_glob, false)) return false;
            if (N_glob_kv > 0) {
                if (!upload_sharded_bias(glob_kv_layers, [&](int l){ return host_engine.get_layer_tensors(l).attn_k_bias; }, Hkv_glob * HD_glob, true, Hkv_glob)) return false;
                if (spec_.has_glob_v) {
                    if (!upload_sharded_bias(glob_kv_layers, [&](int l){ return host_engine.get_layer_tensors(l).attn_v_bias; }, Hkv_glob * HD_glob, true, Hkv_glob)) return false;
                }
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

    const std::vector<float> ones_ld(static_cast<size_t>(L) * static_cast<size_t>(D), 1.0f);
    if (!upload_rep_f32(t_folded_norms ? ones_ld : stack_norms(all_layers, [&](int l){ return host_engine.get_layer_tensors(l).attn_norm; }, D), {L, D})) return false;
    if (spec_.has_post_attn_norm) {
        if (!upload_rep_f32(stack_norms(all_layers, [&](int l){ return host_engine.get_layer_tensors(l).post_attention_norm; }, D), {L, D})) return false;
    }
    if (!upload_rep_f32(t_folded_norms ? ones_ld : stack_norms(all_layers, [&](int l){ return host_engine.get_layer_tensors(l).ffn_norm; }, D), {L, D})) return false;
    if (spec_.is_moe && spec_.has_moe_router_scale) {
        if (!upload_rep_f32(stack_norms(all_layers, [&](int l){ return host_engine.get_layer_tensors(l).ffn_gate_inp_scale; }, D), {L, D})) return false;
    }
    if (spec_.is_moe && spec_.has_post_ffn_norm_1) {
        if (!upload_rep_f32(stack_norms(all_layers, [&](int l){ return host_engine.get_layer_tensors(l).post_ffw_norm_1; }, D), {L, D})) return false;
    }
    if (spec_.is_moe && spec_.has_pre_ffn_norm_2) {
        if (!upload_rep_f32(t_folded_norms ? ones_ld : stack_norms(all_layers, [&](int l){ return host_engine.get_layer_tensors(l).pre_ffw_norm_2; }, D), {L, D})) return false;
    }
    if (spec_.is_moe && spec_.has_post_ffn_norm_2) {
        if (!upload_rep_f32(stack_norms(all_layers, [&](int l){ return host_engine.get_layer_tensors(l).post_ffw_norm_2; }, D), {L, D})) return false;
    }
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

    // 6. Precompute exact FP64 RoPE cos/sin tables matching Avx2Math::apply_rope & CUDA bit-for-bit!
    auto build_rope_tables = [&](int head_dim, float freq_base, const float* freq_factors,
                                 std::vector<float>& cos_tbl, std::vector<float>& sin_tbl) {
        const int pairs = head_dim / 2;
        cos_tbl.resize(static_cast<size_t>(T) * static_cast<size_t>(pairs));
        sin_tbl.resize(static_cast<size_t>(T) * static_cast<size_t>(pairs));
        const double base_d = static_cast<double>(freq_base);
        const double fscale_d = static_cast<double>(model_cfg_.rope_freq_scale);
        const double inv_hd = -2.0 / static_cast<double>(head_dim);
        for (int p = 0; p < T; ++p) {
            const double pos_scale = static_cast<double>(p) * fscale_d;
            for (int i = 0; i < pairs; ++i) {
                double theta = pos_scale * std::pow(base_d, static_cast<double>(i) * inv_hd);
                if (freq_factors && freq_factors[i] > 0.0f) {
                    theta /= static_cast<double>(freq_factors[i]);
                }
                cos_tbl[static_cast<size_t>(p) * static_cast<size_t>(pairs) + static_cast<size_t>(i)] = static_cast<float>(std::cos(theta));
                sin_tbl[static_cast<size_t>(p) * static_cast<size_t>(pairs) + static_cast<size_t>(i)] = static_cast<float>(std::sin(theta));
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
    const bool kv_fp8 = cfg_.use_fp8_kv;
    const bool kv_f32 = (!kv_fp8 && cfg_.use_f32_kv);
    const int T_swa = (spec_.num_swa_kv_layers > 0) ? compute_swa_kv_len(spec_, cfg_) : T;
    const int Hkv_swa_sh  = compute_hkv_shard(spec_.num_kv_heads_swa  > 0 ? spec_.num_kv_heads_swa  : spec_.num_kv_heads, R);
    const int Hkv_glob_sh = compute_hkv_shard(spec_.num_kv_heads_glob > 0 ? spec_.num_kv_heads_glob : spec_.num_kv_heads, R);

    const int N_swa_kv = spec_.num_swa_kv_layers;
    const int N_glob_kv = spec_.num_glob_kv_layers;

    const std::vector<int64_t> swa_dims  = {B, T_swa, Hkv_swa_sh  * spec_.head_dim_swa};
    const std::vector<int64_t> glob_dims = {B, T,     Hkv_glob_sh * spec_.head_dim_glob};
    const size_t swa_layer_elems  = (N_swa_kv > 0)  ? (static_cast<size_t>(B) * Hkv_swa_sh  * T_swa * static_cast<size_t>(spec_.head_dim_swa))  : 0;
    const size_t glob_layer_elems = (N_glob_kv > 0) ? (static_cast<size_t>(B) * Hkv_glob_sh * T     * static_cast<size_t>(spec_.head_dim_glob)) : 0;

    std::vector<uint16_t> z_swa_bf((kv_fp8 || kv_f32) ? 0 : swa_layer_elems, 0);
    std::vector<uint16_t> z_glob_bf((kv_fp8 || kv_f32) ? 0 : glob_layer_elems, 0);
    std::vector<uint8_t>  z_swa_f8(kv_fp8 ? swa_layer_elems : 0, 0);  // 0x00 is exact +0.0 in f8E4M3FN
    std::vector<uint8_t>  z_glob_f8(kv_fp8 ? glob_layer_elems : 0, 0);
    std::vector<float>    z_swa_f32(kv_f32 ? swa_layer_elems : 0, 0.0f);
    std::vector<float>    z_glob_f32(kv_f32 ? glob_layer_elems : 0, 0.0f);
    const size_t kv_elem_bytes = kv_fp8 ? sizeof(uint8_t) : (kv_f32 ? sizeof(float) : sizeof(uint16_t));

    replica_kv_bufs_.assign(static_cast<size_t>(R), {});
    size_t kv_bytes = 0;
    for (int rep = 0; rep < R; ++rep) {
        const int dev = replica_dev_indices_[static_cast<size_t>(rep)];
        auto& bufs = replica_kv_bufs_[static_cast<size_t>(rep)];
        bufs.reserve(static_cast<size_t>(spec_.num_kv_buffers()));
        for (int i = 0; i < N_swa_kv; ++i) {
            PJRT_Buffer* ks = kv_fp8
                ? backend_->upload_f8e4m3fn_nd(z_swa_f8.data(), swa_dims, dev)
                : (kv_f32 ? backend_->upload_f32_nd(z_swa_f32.data(), swa_dims, dev)
                          : backend_->upload_bf16_nd(z_swa_bf.data(), swa_dims, dev));
            if (!ks) { free_kv_cache(); return false; }
            bufs.push_back(ks);
            PJRT_Buffer* vs = kv_fp8
                ? backend_->upload_f8e4m3fn_nd(z_swa_f8.data(), swa_dims, dev)
                : (kv_f32 ? backend_->upload_f32_nd(z_swa_f32.data(), swa_dims, dev)
                          : backend_->upload_bf16_nd(z_swa_bf.data(), swa_dims, dev));
            if (!vs) { free_kv_cache(); return false; }
            bufs.push_back(vs);
            kv_bytes += 2 * swa_layer_elems * kv_elem_bytes;
        }
        for (int i = 0; i < N_glob_kv; ++i) {
            PJRT_Buffer* kg = kv_fp8
                ? backend_->upload_f8e4m3fn_nd(z_glob_f8.data(), glob_dims, dev)
                : (kv_f32 ? backend_->upload_f32_nd(z_glob_f32.data(), glob_dims, dev)
                          : backend_->upload_bf16_nd(z_glob_bf.data(), glob_dims, dev));
            if (!kg) { free_kv_cache(); return false; }
            bufs.push_back(kg);
            PJRT_Buffer* vg = kv_fp8
                ? backend_->upload_f8e4m3fn_nd(z_glob_f8.data(), glob_dims, dev)
                : (kv_f32 ? backend_->upload_f32_nd(z_glob_f32.data(), glob_dims, dev)
                          : backend_->upload_bf16_nd(z_glob_bf.data(), glob_dims, dev));
            if (!vg) { free_kv_cache(); return false; }
            bufs.push_back(vg);
            kv_bytes += 2 * glob_layer_elems * kv_elem_bytes;
        }
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
    if (cfg.num_replicas != 1 && cfg.num_replicas != 2 && cfg.num_replicas != 4 && cfg.num_replicas != 8) {
        last_error_ = "num_replicas must be 1, 2, 4, or 8.";
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
        (spec.vocab_size % cfg.num_replicas) != 0 ||
        (spec.is_moe && (spec.expert_ffn_dim % cfg.num_replicas) != 0)) {
        last_error_ = "Model dims (hidden_dim/expert_ffn_dim/num_heads/vocab_size) must be divisible by num_replicas=" +
                      std::to_string(cfg.num_replicas) + ".";
        return false;
    }
    if (cfg.num_replicas > 1) {
        auto hkv_valid = [&](int hkv) {
            return hkv <= 0 || (hkv % cfg.num_replicas == 0) || (cfg.num_replicas % hkv == 0);
        };
        if (!hkv_valid(spec.num_kv_heads_swa) || !hkv_valid(spec.num_kv_heads_glob)) {
            last_error_ = "Multi-chip TP requires KV head count divisible by or dividing num_replicas=" +
                          std::to_string(cfg.num_replicas) + ".";
            return false;
        }
    }
    if (cfg.use_int4_weights &&
        ((spec.embedding_dim % 2) != 0 || ((spec.hidden_dim / cfg.num_replicas) % 2) != 0)) {
        last_error_ = "W4A16 INT4 requires even embedding_dim and sharded hidden_dim.";
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

    // 2. Compile slot-targeted multi-bucket MXU prefill graphs (TPU: {128, 64, 32}; CUDA: {128, 32}; + 1 for B>1)
    auto buckets = resolve_prefill_buckets(cfg_, backend_->is_cuda_platform());
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
                                         std::vector<uint32_t>& out_next_tokens,
                                         int valid_last) {
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
    int32_t valid_last_i32 = static_cast<int32_t>((valid_last >= 0) ? std::clamp(valid_last, 0, S - 1) : (S - 1));
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
    std::vector<PJRT_Buffer*> step_vlast(static_cast<size_t>(R), nullptr);
    std::vector<std::vector<PJRT_Buffer*>> args_per_rep(static_cast<size_t>(R));

    auto cleanup_step_bufs = [&]() {
        for (int rep = 0; rep < R; ++rep) {
            backend_->destroy_buffer(step_toks[static_cast<size_t>(rep)]);
            backend_->destroy_buffer(step_emb[static_cast<size_t>(rep)]);
            backend_->destroy_buffer(step_pl[static_cast<size_t>(rep)]);
            backend_->destroy_buffer(step_pos[static_cast<size_t>(rep)]);
            backend_->destroy_buffer(step_slot[static_cast<size_t>(rep)]);
            backend_->destroy_buffer(step_vlast[static_cast<size_t>(rep)]);
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
            if (S > 1) {
                step_vlast[static_cast<size_t>(rep)] = backend_->upload_i32_nd(&valid_last_i32, {}, dev);
            }
            if (!step_toks[static_cast<size_t>(rep)] || !step_pos[static_cast<size_t>(rep)] ||
                (slot_targeted && !step_slot[static_cast<size_t>(rep)]) ||
                (S > 1 && !step_vlast[static_cast<size_t>(rep)])) {
                cleanup_step_bufs();
                last_error_ = "Per-step token-ID H2D upload failed: " + backend_->get_last_error();
                return false;
            }
            auto& a = args_per_rep[static_cast<size_t>(rep)];
            a.reserve(4 + replica_weight_bufs_[static_cast<size_t>(rep)].size() + num_kv);
            a.push_back(step_toks[static_cast<size_t>(rep)]);
            a.push_back(step_pos[static_cast<size_t>(rep)]);
            if (slot_targeted) a.push_back(step_slot[static_cast<size_t>(rep)]);
            if (S > 1) a.push_back(step_vlast[static_cast<size_t>(rep)]);
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
            if (S > 1) {
                step_vlast[static_cast<size_t>(rep)] = backend_->upload_i32_nd(&valid_last_i32, {}, dev);
            }
            if (!step_emb[static_cast<size_t>(rep)] || (spec_.has_pleias && !step_pl[static_cast<size_t>(rep)]) ||
                !step_pos[static_cast<size_t>(rep)] || (slot_targeted && !step_slot[static_cast<size_t>(rep)]) ||
                (S > 1 && !step_vlast[static_cast<size_t>(rep)])) {
                cleanup_step_bufs();
                last_error_ = "Per-step H2D upload failed: " + backend_->get_last_error();
                return false;
            }
            auto& a = args_per_rep[static_cast<size_t>(rep)];
            a.reserve(5 + replica_weight_bufs_[static_cast<size_t>(rep)].size() + num_kv);
            a.push_back(step_emb[static_cast<size_t>(rep)]);
            if (spec_.has_pleias) a.push_back(step_pl[static_cast<size_t>(rep)]);
            a.push_back(step_pos[static_cast<size_t>(rep)]);
            if (slot_targeted) a.push_back(step_slot[static_cast<size_t>(rep)]);
            if (S > 1) a.push_back(step_vlast[static_cast<size_t>(rep)]);
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
        if (cfg_.donate_kv_cache) {
            // Donated KV buffers were consumed by PJRT on launch; clear handles and reallocate clean KV slabs
            for (auto& rkv : replica_kv_bufs_) {
                for (auto*& b : rkv) b = nullptr;
            }
            allocate_zero_kv_cache();
        }
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

    const int T_swa = (spec_.num_swa_layers > 0) ? compute_swa_kv_len(spec_, cfg_) : cfg_.max_seq_len;
    const bool swa_ring_mode = (spec_.num_swa_layers > 0) && (T_swa < cfg_.max_seq_len);

    // Greedy descent with Single-Shot Padded Final Bucket:
    // Any final tail (2 <= rem <= max_bucket) pads into the smallest compiled bucket >= rem in ONE TPU call!
    // Causal attention guarantees valid tokens 0..rem-1 never attend to pad tokens; %valid_last extracts exact
    // logits at rem-1, and subsequent decode starts at pos+rem immediately overwriting pad KV positions.
    while (cursor < tokens.size()) {
        size_t rem = tokens.size() - cursor;
        PJRT_LoadedExecutable* chosen_exec = nullptr;
        int chosen_len = 0;
        size_t consume_len = 0;
        bool use_slot_targeted = true;

        // 1. Check if the entire remaining tail fits in ONE padded bucket (eliminates 6-call tail cascades!)
        if (rem >= 2) {
            PJRT_LoadedExecutable* pad_exec = nullptr;
            int pad_len = 0;
            for (auto it = slot_prefill_execs_.rbegin(); it != slot_prefill_execs_.rend(); ++it) {
                const int blen = it->first;
                if (it->second && blen > 1 && static_cast<size_t>(blen) >= rem &&
                    pos + blen <= cfg_.max_seq_len &&
                    (!swa_ring_mode || pos + blen <= T_swa)) {
                    pad_exec = it->second;
                    pad_len = blen;
                    break;
                }
            }
            if (pad_exec) {
                chosen_exec = pad_exec;
                chosen_len = pad_len;
                consume_len = rem;
                use_slot_targeted = true;
            }
        }

        // 2. Otherwise take the largest unpadded bucket <= rem (e.g. full S=128 chunk or S=1 tail)
        if (!chosen_exec) {
            for (const auto& [blen, bexec] : slot_prefill_execs_) {
                if (bexec && static_cast<size_t>(blen) <= rem) {
                    chosen_exec = bexec;
                    chosen_len = blen;
                    consume_len = static_cast<size_t>(blen);
                    use_slot_targeted = true;
                    break;
                }
            }
        }

        if (!chosen_exec) {
            if (cfg_.batch_size == 1) {
                // For B=1, decode_exec_ IS a single-slot S=1 graph!
                chosen_exec = decode_exec_;
                chosen_len = 1;
                consume_len = 1;
                use_slot_targeted = false;
            } else {
                last_error_ = "prefill_slot requires prefill_chunk_size > 0 (or S=1 bucket) when batch_size > 1.";
                return false;
            }
        }

        std::vector<uint32_t> chunk(static_cast<size_t>(chosen_len), 0u);
        for (size_t i = 0; i < consume_len; ++i) {
            chunk[i] = tokens[cursor + i];
        }
        bool is_last = (cursor + consume_len == tokens.size());
        int valid_last = static_cast<int>(consume_len) - 1;
        if (!execute_graph(chosen_exec, chosen_len, use_slot_targeted, slot_id, chunk, {pos},
                           is_last ? out_logits : nullptr, next_toks, valid_last)) {
            return false;
        }
        cursor += consume_len;
        pos += static_cast<int>(consume_len);
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
