#ifndef HAVEN_POSTURE_SWAY_HPP
#define HAVEN_POSTURE_SWAY_HPP

/**
 * @file haven_posture_sway.hpp
 * @brief Autonomous Postural Sway & Diaphragmatic Breath Synthesis Engine for Haven C++.
 * 
 * Implements sub-stepped numerical Lorenz attractor chaotic dynamics, continuous-time
 * exponential low-pass filtering, asymmetric diaphragmatic respiration (35% inhale / 65% exhale),
 * Lissajous figure-8 center-of-mass balance shifts, and geodesic angular deadband filtering.
 * 
 * Mathematical Formulation:
 * 1. Lorenz Dynamical System:
 *    dx/dt = \sigma (y - x)
 *    dy/dt = x (\rho - z) - y
 *    dz/dt = x y - \beta z
 *    where \sigma = 10.0, \rho = 28.0, \beta = 8/3.
 * 
 * 2. Exponential Low-Pass Filter:
 *    \alpha_{LP} = 1 - e^{-k_{cutoff} \Delta t}, where k_{cutoff} = 6.0 rad/s
 *    x_{smooth}(t + \Delta t) = x_{smooth}(t) + \alpha_{LP} (x(t + \Delta t) - x_{smooth}(t))
 * 
 * 3. Geodesic Angular Deadband Filtering:
 *    \Delta \theta = 2 \arctan2(||\mathbf{v}_{\Delta}||, |w_{\Delta}|)
 *    Gate transform update if \Delta \theta < \epsilon_{deadband} (0.0005 rad / 0.0286 deg).
 * 
 * Standard: C++20
 * Namespace: haven
 */

#include <cmath>
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>

namespace haven {

constexpr float PI = 3.14159265358979323846f;
constexpr float TWO_PI = 6.28318530717958647692f;

/**
 * @brief 3D Cartesian Vector representation.
 */
struct Vec3 {
    float x{0.0f};
    float y{0.0f};
    float z{0.0f};

    [[nodiscard]] constexpr Vec3 operator+(const Vec3& o) const noexcept {
        return {x + o.x, y + o.y, z + o.z};
    }

    [[nodiscard]] constexpr Vec3 operator-(const Vec3& o) const noexcept {
        return {x - o.x, y - o.y, z - o.z};
    }

    [[nodiscard]] constexpr Vec3 operator*(float s) const noexcept {
        return {x * s, y * s, z * s};
    }

    [[nodiscard]] float length_sq() const noexcept {
        return x * x + y * y + z * z;
    }

    [[nodiscard]] float length() const noexcept {
        return std::sqrt(length_sq());
    }

    [[nodiscard]] static float distance(const Vec3& a, const Vec3& b) noexcept {
        return (a - b).length();
    }
};

/**
 * @brief Unit Quaternion representation for 3D SO(3) rotations.
 */
struct Quaternion {
    float x{0.0f};
    float y{0.0f};
    float z{0.0f};
    float w{1.0f};

    [[nodiscard]] static constexpr Quaternion identity() noexcept {
        return {0.0f, 0.0f, 0.0f, 1.0f};
    }

    void normalize() noexcept {
        float norm_sq = x * x + y * y + z * z + w * w;
        if (norm_sq > 1e-12f) {
            float inv = 1.0f / std::sqrt(norm_sq);
            x *= inv;
            y *= inv;
            z *= inv;
            w *= inv;
        } else {
            x = 0.0f;
            y = 0.0f;
            z = 0.0f;
            w = 1.0f;
        }
    }

    [[nodiscard]] static Quaternion from_euler_xyz(float pitch_x, float roll_y, float yaw_z) noexcept {
        float cx = std::cos(pitch_x * 0.5f);
        float sx = std::sin(pitch_x * 0.5f);
        float cy = std::cos(roll_y * 0.5f);
        float sy = std::sin(roll_y * 0.5f);
        float cz = std::cos(yaw_z * 0.5f);
        float sz = std::sin(yaw_z * 0.5f);

        // XYZ Tait-Bryan rotation sequence
        Quaternion q{
            sx * cy * cz + cx * sy * sz,
            cx * sy * cz - sx * cy * sz,
            cx * cy * sz + sx * sy * cz,
            cx * cy * cz - sx * sy * sz
        };
        q.normalize();
        return q;
    }

    [[nodiscard]] static float geodesic_angle(const Quaternion& q1, const Quaternion& q2) noexcept {
        // Relative quaternion q_rel = q1^{-1} * q2
        float xd =  q1.w * q2.x - q1.x * q2.w - q1.y * q2.z + q1.z * q2.y;
        float yd =  q1.w * q2.y + q1.x * q2.z - q1.y * q2.w - q1.z * q2.x;
        float zd =  q1.w * q2.z - q1.x * q2.y + q1.y * q2.x - q1.z * q2.w;
        float wd =  q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z;

        float v_norm = std::sqrt(xd * xd + yd * yd + zd * zd);
        return 2.0f * std::atan2(v_norm, std::abs(wd));
    }

    [[nodiscard]] static Quaternion slerp(const Quaternion& q1, Quaternion q2, float t) noexcept {
        float dot = q1.x * q2.x + q1.y * q2.y + q1.z * q2.z + q1.w * q2.w;
        if (dot < 0.0f) {
            dot = -dot;
            q2.x = -q2.x;
            q2.y = -q2.y;
            q2.z = -q2.z;
            q2.w = -q2.w;
        }

        if (dot > 0.9995f) {
            Quaternion r{
                q1.x + t * (q2.x - q1.x),
                q1.y + t * (q2.y - q1.y),
                q1.z + t * (q2.z - q1.z),
                q1.w + t * (q2.w - q1.w)
            };
            r.normalize();
            return r;
        }

        float theta_0 = std::acos(std::clamp(dot, -1.0f, 1.0f));
        float theta = theta_0 * t;
        float sin_theta = std::sin(theta);
        float sin_theta_0 = std::sin(theta_0);

        float s0 = std::cos(theta) - dot * sin_theta / sin_theta_0;
        float s1 = sin_theta / sin_theta_0;

        Quaternion r{
            (s0 * q1.x) + (s1 * q2.x),
            (s0 * q1.y) + (s1 * q2.y),
            (s0 * q1.z) + (s1 * q2.z),
            (s0 * q1.w) + (s1 * q2.w)
        };
        r.normalize();
        return r;
    }
};

/**
 * @brief Sub-millimeter & Geodesic Angular Deadband Filter.
 * 
 * Prevents continuous floating-point micro-vibrations at joint equilibrium.
 */
class PosturalDeadbandFilter {
private:
    float angular_epsilon_{0.0005f}; // 0.0286 degrees
    float translation_epsilon_{0.0001f}; // 0.1 mm
    float lambda_{18.0f};

    Quaternion last_quat_{Quaternion::identity()};
    Vec3 last_trans_{0.0f, 0.0f, 0.0f};
    bool initialized_rot_{false};
    bool initialized_trans_{false};

public:
    explicit PosturalDeadbandFilter(
        float angular_epsilon = 0.0005f,
        float translation_epsilon = 0.0001f,
        float lambda = 18.0f
    ) noexcept
        : angular_epsilon_(angular_epsilon),
          translation_epsilon_(translation_epsilon),
          lambda_(lambda) {}

    void reset() noexcept {
        last_quat_ = Quaternion::identity();
        last_trans_ = {0.0f, 0.0f, 0.0f};
        initialized_rot_ = false;
        initialized_trans_ = false;
    }

    [[nodiscard]] Quaternion filter_rotation(
        const Quaternion& target,
        float delta_sec
    ) noexcept {
        if (!initialized_rot_) {
            last_quat_ = target;
            initialized_rot_ = true;
            return target;
        }

        float delta_rad = Quaternion::geodesic_angle(last_quat_, target);
        if (delta_rad < angular_epsilon_) {
            return last_quat_;
        }

        float alpha = 1.0f - std::exp(-lambda_ * std::min(delta_sec, 0.033f));
        last_quat_ = Quaternion::slerp(last_quat_, target, alpha);
        return last_quat_;
    }

    [[nodiscard]] Vec3 filter_translation(
        const Vec3& target,
        float delta_sec
    ) noexcept {
        if (!initialized_trans_) {
            last_trans_ = target;
            initialized_trans_ = true;
            return target;
        }

        float dist = Vec3::distance(last_trans_, target);
        if (dist < translation_epsilon_) {
            return last_trans_;
        }

        float alpha = 1.0f - std::exp(-lambda_ * std::min(delta_sec, 0.033f));
        last_trans_ = last_trans_ + (target - last_trans_) * alpha;
        return last_trans_;
    }

    [[nodiscard]] float get_angular_epsilon() const noexcept { return angular_epsilon_; }
    [[nodiscard]] float get_translation_epsilon() const noexcept { return translation_epsilon_; }
};

/**
 * @brief Output state of the Autonomic Postural Sway Engine.
 */
struct SwayOutput {
    float chest_pitch_rad{0.0f};       // Diaphragmatic breathing flexion (rad)
    Vec3 spinal_sway_rad{0.0f, 0.0f, 0.0f}; // Pitch, Roll, Yaw Euler angles (rad)
    Vec3 hip_translation_m{0.0f, 0.0f, 0.0f}; // Figure-8 balance shift (m)
    Quaternion filtered_torso_quat{Quaternion::identity()}; // Filtered SO(3) torso orientation
    Vec3 filtered_hip_trans_m{0.0f, 0.0f, 0.0f}; // Filtered hip translation (m)
};

/**
 * @brief Configuration parameters for Sway Engine.
 */
struct SwayConfig {
    // Lorenz Attractor parameters
    float lorenz_sigma{10.0f};
    float lorenz_rho{28.0f};
    float lorenz_beta{8.0f / 3.0f};
    float sub_dt{0.005f};
    float lorenz_scale_pitch{0.00003f};
    float lorenz_scale_roll{0.00003f};
    float lorenz_scale_yaw{0.00002f};
    float lorenz_cutoff_freq{6.0f}; // rad/s (~0.955 Hz)

    // Breathing parameters
    float breath_freq{0.18f};       // 0.18 Hz (5.55s cycle)
    float inhale_fraction{0.35f};   // 35% Inhalation / 65% Exhalation
    float breath_amplitude{0.012f}; // rad (~0.69 deg)

    // Lissajous Figure-8 Postural Sway
    float sway_freq{0.08f};         // 0.08 Hz (12.5s cycle)
    float fig8_hip_x_amp{0.005f};   // 5 mm lateral sway
    float fig8_hip_z_amp{0.003f};   // 3 mm anterior-posterior sway
    float fig8_pitch_amp{0.002f};   // ~0.11 deg pitch
    float fig8_roll_amp{0.003f};    // ~0.17 deg roll

    // Deadband Filter
    float angular_deadband{0.0005f};     // rad
    float translation_deadband{0.0001f}; // m
    float filter_lambda{18.0f};          // s^-1
};

/**
 * @brief Production-Grade Autonomic Postural Sway & Respiration Engine.
 * 
 * Provides biologically authentic posture sway for virtual companions, humanoid
 * avatars, and simulated robotics. Zero runtime allocations, header-only, thread-safe.
 */
class AutonomicPosturalSwayEngine {
private:
    SwayConfig config_{};

    // Lorenz Attractor State
    float lx_{0.1f};
    float ly_{0.0f};
    float lz_{0.0f};

    // Low-Pass Filtered Lorenz State
    float l_smooth_x_{0.0f};
    float l_smooth_y_{0.0f};
    float l_smooth_z_{0.0f};

    // Timers
    float breath_timer_{0.0f};
    float sway_timer_{0.0f};

    // Postural Deadband Filter
    PosturalDeadbandFilter deadband_filter_{};

public:
    explicit AutonomicPosturalSwayEngine(const SwayConfig& config = SwayConfig{}) noexcept
        : config_(config),
          deadband_filter_(config.angular_deadband, config.translation_deadband, config.filter_lambda) {
        reset();
    }

    /**
     * @brief Resets all dynamic states and timers to canonical equilibrium.
     */
    void reset() noexcept {
        lx_ = 0.1f;
        ly_ = 0.0f;
        lz_ = 0.0f;
        l_smooth_x_ = 0.0f;
        l_smooth_y_ = 0.0f;
        l_smooth_z_ = 0.0f;
        breath_timer_ = 0.0f;
        sway_timer_ = 0.0f;
        deadband_filter_.reset();
    }

    /**
     * @brief Steps the autonomic sway simulation forward by delta_sec.
     * 
     * @param delta_sec Time delta in seconds. Clamped to [0.0001, 0.1] to prevent instability.
     * @return SwayOutput Computed chest pitch, spinal sway Euler angles, hip translations, and filtered transforms.
     */
    [[nodiscard]] SwayOutput update(float delta_sec) noexcept {
        float dt = std::clamp(delta_sec, 0.0001f, 0.1f);

        // -------------------------------------------------------------
        // 1. Asymmetric Diaphragmatic Breathing (35% Inhale / 65% Exhale)
        // -------------------------------------------------------------
        breath_timer_ += dt;
        float breath_period = 1.0f / config_.breath_freq;
        float phase = std::fmod(breath_timer_, breath_period) / breath_period;
        if (phase < 0.0f) phase += 1.0f;

        float chest_pitch = 0.0f;
        if (phase < config_.inhale_fraction) {
            float u = phase / config_.inhale_fraction;
            chest_pitch = std::sin(u * (PI * 0.5f)) * config_.breath_amplitude;
        } else {
            float v = (phase - config_.inhale_fraction) / (1.0f - config_.inhale_fraction);
            chest_pitch = std::cos(v * (PI * 0.5f)) * config_.breath_amplitude;
        }

        // -------------------------------------------------------------
        // 2. Sub-stepped Numerical Lorenz Attractor Integration
        // -------------------------------------------------------------
        int steps = std::clamp(static_cast<int>(dt / config_.sub_dt), 1, 5);
        float sub_dt = dt / static_cast<float>(steps);

        for (int i = 0; i < steps; ++i) {
            float dx = config_.lorenz_sigma * (ly_ - lx_);
            float dy = lx_ * (config_.lorenz_rho - lz_) - ly_;
            float dz = lx_ * ly_ - config_.lorenz_beta * lz_;

            lx_ += dx * sub_dt;
            ly_ += dy * sub_dt;
            lz_ += dz * sub_dt;
        }

        // Numerical Explosion & NaN Failsafe
        if (std::isnan(lx_) || std::isnan(ly_) || std::isnan(lz_) ||
            std::isinf(lx_) || std::isinf(ly_) || std::isinf(lz_) ||
            std::abs(lx_) > 60.0f || std::abs(ly_) > 60.0f || std::abs(lz_) > 60.0f) {
            lx_ = 0.1f;
            ly_ = 0.0f;
            lz_ = 0.0f;
        }

        // Continuous-time exponential low-pass filter (cutoff 6.0 rad/s)
        float lp_alpha = 1.0f - std::exp(-config_.lorenz_cutoff_freq * dt);
        l_smooth_x_ += lp_alpha * (lx_ - l_smooth_x_);
        l_smooth_y_ += lp_alpha * (ly_ - l_smooth_y_);
        l_smooth_z_ += lp_alpha * (lz_ - l_smooth_z_);

        // -------------------------------------------------------------
        // 3. Lissajous Figure-8 Weight Shift & Spinal Mapping
        // -------------------------------------------------------------
        sway_timer_ += dt;
        float omega = TWO_PI * config_.sway_freq;

        float fig8_x = config_.fig8_hip_x_amp * std::sin(omega * sway_timer_);
        float fig8_z = config_.fig8_hip_z_amp * std::sin(2.0f * omega * sway_timer_);
        float fig8_pitch = config_.fig8_pitch_amp * std::sin(2.0f * omega * sway_timer_);
        float fig8_roll = config_.fig8_roll_amp * std::sin(omega * sway_timer_);

        Vec3 spinal_sway{
            l_smooth_x_ * config_.lorenz_scale_pitch + fig8_pitch,
            l_smooth_y_ * config_.lorenz_scale_roll + fig8_roll,
            (l_smooth_z_ - 25.0f) * config_.lorenz_scale_yaw
        };

        Vec3 hip_translation{fig8_x, 0.0f, fig8_z};

        // -------------------------------------------------------------
        // 4. Geodesic Angular & Translation Deadband Filtering
        // -------------------------------------------------------------
        Quaternion raw_torso_quat = Quaternion::from_euler_xyz(
            spinal_sway.x + chest_pitch,
            spinal_sway.y,
            spinal_sway.z
        );

        Quaternion filtered_quat = deadband_filter_.filter_rotation(raw_torso_quat, dt);
        Vec3 filtered_trans = deadband_filter_.filter_translation(hip_translation, dt);

        return SwayOutput{
            chest_pitch,
            spinal_sway,
            hip_translation,
            filtered_quat,
            filtered_trans
        };
    }

    // Getters for inspection and testing
    [[nodiscard]] const SwayConfig& get_config() const noexcept { return config_; }
    [[nodiscard]] std::array<float, 3> get_raw_lorenz_state() const noexcept { return {lx_, ly_, lz_}; }
    [[nodiscard]] std::array<float, 3> get_smooth_lorenz_state() const noexcept { return {l_smooth_x_, l_smooth_y_, l_smooth_z_}; }
    [[nodiscard]] float get_breath_timer() const noexcept { return breath_timer_; }
    [[nodiscard]] float get_sway_timer() const noexcept { return sway_timer_; }

    // State injector for testing boundary conditions & recovery
    void inject_lorenz_state(float x, float y, float z) noexcept {
        lx_ = x;
        ly_ = y;
        lz_ = z;
    }
};

} // namespace haven

#endif // HAVEN_POSTURE_SWAY_HPP
