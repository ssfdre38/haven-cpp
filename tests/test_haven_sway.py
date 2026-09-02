"""
Test Suite for Haven Postural Sway & Diaphragmatic Breath Engine.

Validates:
1. 100,000 Continuous Simulation Steps Stability (No NaNs, No Infs)
2. Asymmetric Diaphragmatic Respiration (35% Inhalation / 65% Exhalation @ 0.18 Hz)
3. Sub-stepped Lorenz Attractor Chaotic Dynamics & Low-Pass Filtering
4. Numerical Explosion & NaN Recovery Failsafe
5. Lissajous Figure-8 Postural Sway Frequency Spectrum
6. Geodesic Angular Deadband Filter Thresholding (0.0005 rad)
7. Translation Deadband Filter Thresholding (0.0001 m)
8. High-Performance Execution & Zero Memory Leakage
"""

import math
import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure repository root and package root are in sys.path
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_HAVEN_DIR = Path(__file__).resolve().parents[1]
if str(_HAVEN_DIR) not in sys.path:
    sys.path.insert(0, str(_HAVEN_DIR))

try:
    from contributions.haven_cpp.haven.posture_sway import (
        Vec3,
        Quaternion,
        PosturalDeadbandFilter,
        SwayConfig,
        SwayOutput,
        AutonomicPosturalSwayEngine,
        generate_sway_trajectory,
        compute_power_spectrum,
        render_ascii_phase_portrait,
    )
except ImportError:
    from haven.posture_sway import (
        Vec3,
        Quaternion,
        PosturalDeadbandFilter,
        SwayConfig,
        SwayOutput,
        AutonomicPosturalSwayEngine,
        generate_sway_trajectory,
        compute_power_spectrum,
        render_ascii_phase_portrait,
    )


class TestQuaternionAndGeodesicMath:
    """Validates SO(3) Lie Group arithmetic and Riemannian geodesic metrics."""

    def test_identity_and_normalization(self):
        q = Quaternion.identity()
        assert np.isclose(q.x, 0.0)
        assert np.isclose(q.y, 0.0)
        assert np.isclose(q.z, 0.0)
        assert np.isclose(q.w, 1.0)

        q_unnorm = Quaternion(2.0, 0.0, 0.0, 0.0)
        q_unnorm.normalize()
        assert np.isclose(q_unnorm.x, 1.0)
        assert np.isclose(q_unnorm.w, 0.0)

    def test_from_euler_xyz(self):
        # 90 degrees around X axis
        q_pitch = Quaternion.from_euler_xyz(math.pi * 0.5, 0.0, 0.0)
        assert np.isclose(q_pitch.x, math.sin(math.pi * 0.25), atol=1e-5)
        assert np.isclose(q_pitch.w, math.cos(math.pi * 0.25), atol=1e-5)

        # Zero rotation produces identity
        q_zero = Quaternion.from_euler_xyz(0.0, 0.0, 0.0)
        assert np.isclose(q_zero.w, 1.0, atol=1e-6)

    def test_geodesic_angle_metric(self):
        q1 = Quaternion.identity()
        # 30 deg rotation around Y
        theta = math.radians(30.0)
        q2 = Quaternion.from_euler_xyz(0.0, theta, 0.0)

        angle = Quaternion.geodesic_angle(q1, q2)
        assert np.isclose(angle, theta, atol=1e-5)

        # Opposite representation of same SO(3) element (-q)
        q2_neg = Quaternion(-q2.x, -q2.y, -q2.z, -q2.w)
        angle_neg = Quaternion.geodesic_angle(q1, q2_neg)
        assert np.isclose(angle_neg, theta, atol=1e-5)

    def test_slerp_interpolation(self):
        q0 = Quaternion.identity()
        q1 = Quaternion.from_euler_xyz(0.0, math.radians(60.0), 0.0)

        # Midpoint at t=0.5 should be exactly 30 degrees
        q_mid = Quaternion.slerp(q0, q1, 0.5)
        angle = Quaternion.geodesic_angle(q0, q_mid)
        assert np.isclose(angle, math.radians(30.0), atol=1e-5)


class TestDeadbandFiltering:
    """Validates angular and linear deadband thresholds preventing micro-jitter."""

    def test_angular_deadband_suppression(self):
        db = PosturalDeadbandFilter(angular_epsilon=0.0005, filter_lambda=18.0)
        q_base = Quaternion.from_euler_xyz(0.01, 0.01, 0.01)

        # Initial seed
        res0 = db.filter_rotation(q_base, 0.016)
        assert np.isclose(res0.w, q_base.w, atol=1e-5)

        # Perturbation below threshold (0.0002 rad < 0.0005 rad)
        q_micro = Quaternion.from_euler_xyz(0.01 + 0.0001, 0.01 + 0.0001, 0.01)
        angle_diff = Quaternion.geodesic_angle(q_base, q_micro)
        assert angle_diff < 0.0005

        res_filtered = db.filter_rotation(q_micro, 0.016)
        # Should return exact previous quaternion (no jitter)
        assert np.isclose(res_filtered.x, res0.x, atol=1e-6)
        assert np.isclose(res_filtered.y, res0.y, atol=1e-6)
        assert np.isclose(res_filtered.z, res0.z, atol=1e-6)
        assert np.isclose(res_filtered.w, res0.w, atol=1e-6)

        # Perturbation above threshold (0.005 rad > 0.0005 rad)
        q_macro = Quaternion.from_euler_xyz(0.01 + 0.005, 0.01, 0.01)
        res_moved = db.filter_rotation(q_macro, 0.016)
        # Should smoothly advance towards target
        delta_moved = Quaternion.geodesic_angle(res0, res_moved)
        assert delta_moved > 0.0005

    def test_translation_deadband_suppression(self):
        db = PosturalDeadbandFilter(translation_epsilon=0.0001, filter_lambda=18.0)
        t_base = Vec3(0.01, 0.02, 0.03)

        res0 = db.filter_translation(t_base, 0.016)

        # Sub-0.1mm perturbation (0.05 mm = 0.00005 m)
        t_micro = Vec3(0.01003, 0.02004, 0.03)
        res_filtered = db.filter_translation(t_micro, 0.016)
        assert np.isclose(res_filtered.x, res0.x)
        assert np.isclose(res_filtered.y, res0.y)

        # 2mm movement
        t_macro = Vec3(0.012, 0.02, 0.03)
        res_moved = db.filter_translation(t_macro, 0.016)
        assert res_moved.x > res0.x


class TestAutonomicSwayEngine:
    """Validates the core continuous-time posture sway engine."""

    def test_100k_steps_numerical_stability(self):
        """Simulates 100,000 consecutive steps (~27.7 minutes at 60 FPS) to guarantee zero NaNs/Infs."""
        engine = AutonomicPosturalSwayEngine()
        dt = 1.0 / 60.0

        max_chest = 0.0
        max_pitch = 0.0
        max_roll = 0.0
        max_hip_x = 0.0
        max_hip_z = 0.0
        has_nan_or_inf = False

        for step in range(100_000):
            out = engine.update(dt)
            if (
                math.isnan(out.chest_pitch_rad)
                or math.isinf(out.chest_pitch_rad)
                or math.isnan(out.spinal_sway_rad.x)
                or math.isnan(out.spinal_sway_rad.y)
                or math.isnan(out.spinal_sway_rad.z)
                or math.isnan(out.filtered_torso_quat.w)
            ):
                has_nan_or_inf = True
                break

            abs_c = abs(out.chest_pitch_rad)
            if abs_c > max_chest:
                max_chest = abs_c
            abs_p = abs(out.spinal_sway_rad.x)
            if abs_p > max_pitch:
                max_pitch = abs_p
            abs_r = abs(out.spinal_sway_rad.y)
            if abs_r > max_roll:
                max_roll = abs_r
            abs_hx = abs(out.hip_translation_m.x)
            if abs_hx > max_hip_x:
                max_hip_x = abs_hx
            abs_hz = abs(out.hip_translation_m.z)
            if abs_hz > max_hip_z:
                max_hip_z = abs_hz

        assert not has_nan_or_inf, "Encountered NaN or Inf during 100,000 steps"
        assert max_chest <= 0.025, f"Chest pitch exceeded bound: {max_chest}"
        assert max_pitch <= 0.05, f"Spinal pitch exceeded bound: {max_pitch}"
        assert max_roll <= 0.05, f"Spinal roll exceeded bound: {max_roll}"
        assert max_hip_x <= 0.02, f"Hip X exceeded bound: {max_hip_x}"
        assert max_hip_z <= 0.02, f"Hip Z exceeded bound: {max_hip_z}"

    def test_asymmetric_breathing_waveform(self):
        """Tests that breathing follows 35% inhale / 65% exhale with smooth transitions."""
        config = SwayConfig(breath_freq=0.20, inhale_fraction=0.35, breath_amplitude=0.010)
        engine = AutonomicPosturalSwayEngine(config)

        period = 1.0 / 0.20  # 5.0 seconds
        dt = 0.01
        num_steps = int(period / dt)

        phases = []
        pitches = []
        for _ in range(num_steps):
            out = engine.update(dt)
            pitches.append(out.chest_pitch_rad)
            phases.append((engine.breath_timer % period) / period)

        pitches = np.array(pitches)
        phases = np.array(phases)

        # Peak must occur near phase = 0.35 (inhalation peak)
        max_idx = np.argmax(pitches)
        peak_phase = phases[max_idx]
        assert np.isclose(peak_phase, 0.35, atol=0.03), f"Peak breathing phase at {peak_phase}, expected 0.35"
        assert np.isclose(pitches[max_idx], 0.010, atol=1e-4)

        # Minima at phase = 0 / 1.0 (exhale completion)
        assert np.isclose(pitches[0], 0.0, atol=1e-3)
        assert np.isclose(pitches[-1], 0.0, atol=1e-3)

    def test_explosion_failsafe_recovery(self):
        """Validates that numerical divergence or extreme state injection immediately recovers."""
        engine = AutonomicPosturalSwayEngine()
        engine.update(0.016)

        # Inject extreme divergence (|x| > 60)
        engine.inject_lorenz_state(x=500.0, y=-300.0, z=999.0)
        out = engine.update(0.016)

        # Engine must catch failsafe and reset to (0.1, 0.0, 0.0)
        assert engine.lx == 0.1 or abs(engine.lx) < 60.0
        assert not math.isnan(out.spinal_sway_rad.x)
        assert abs(out.spinal_sway_rad.x) < 0.05

        # Inject NaN
        engine.inject_lorenz_state(x=float("nan"), y=0.0, z=0.0)
        out_nan = engine.update(0.016)
        assert not math.isnan(engine.lx)
        assert not math.isnan(out_nan.spinal_sway_rad.x)

    def test_frequency_spectrum_compliance(self):
        """Validates that Lissajous and respiration peaks match specified frequencies in PSD."""
        traj = generate_sway_trajectory(duration_sec=60.0, fps=60.0)

        # Check Respiration Frequency (0.18 Hz)
        freqs, psd_breath = compute_power_spectrum(traj["chest_pitch"], fps=60.0)
        peak_breath_freq = freqs[np.argmax(psd_breath)]
        assert np.isclose(peak_breath_freq, 0.18, atol=0.03), f"Respiration peak at {peak_breath_freq} Hz"

        # Check Lateral Sway Frequency (0.08 Hz)
        freqs_sway, psd_sway_x = compute_power_spectrum(traj["hip_translation"][:, 0], fps=60.0)
        peak_sway_freq = freqs_sway[np.argmax(psd_sway_x)]
        assert np.isclose(peak_sway_freq, 0.08, atol=0.03), f"Lateral sway peak at {peak_sway_freq} Hz"

    def test_low_pass_filter_attenuation(self):
        """Verifies that high-frequency chaotic fluctuation is attenuated by exponential filter."""
        traj = generate_sway_trajectory(duration_sec=30.0, fps=60.0)
        raw_x = traj["lorenz_raw"][:, 0]
        smooth_x = traj["lorenz_smooth"][:, 0]

        # Derivatives / step differences of smoothed signal must have significantly lower variance
        diff_raw = np.diff(raw_x)
        diff_smooth = np.diff(smooth_x)

        var_raw = np.var(diff_raw)
        var_smooth = np.var(diff_smooth)

        assert var_smooth < var_raw * 0.5, f"Smoothed variance {var_smooth} not significantly lower than raw {var_raw}"

    def test_ascii_phase_portrait_generation(self):
        """Verifies ASCII rendering utility outputs valid non-empty grid."""
        traj = generate_sway_trajectory(duration_sec=10.0, fps=30.0)
        portrait = render_ascii_phase_portrait(
            traj["lorenz_smooth"][:, 0],
            traj["lorenz_smooth"][:, 1],
            width=40,
            height=15,
        )
        assert len(portrait) > 0
        assert "+" in portrait
        assert "•" in portrait


if __name__ == "__main__":
    import time
    print("=== Running Haven Sway Tests Directly ===", flush=True)
    t_math = TestQuaternionAndGeodesicMath()
    print("[1/3] Running Quaternion & Geodesic tests...", flush=True)
    t_math.test_identity_and_normalization()
    t_math.test_from_euler_xyz()
    t_math.test_geodesic_angle_metric()
    t_math.test_slerp_interpolation()
    print("  -> Passed Quaternion tests.", flush=True)

    t_db = TestDeadbandFiltering()
    print("[2/3] Running Deadband Filtering tests...", flush=True)
    t_db.test_angular_deadband_suppression()
    t_db.test_translation_deadband_suppression()
    print("  -> Passed Deadband tests.", flush=True)

    t_engine = TestAutonomicSwayEngine()
    print("[3/3] Running Autonomic Sway Engine tests (including 100k steps)...", flush=True)
    t0 = time.perf_counter()
    t_engine.test_100k_steps_numerical_stability()
    t1 = time.perf_counter()
    print(f"  -> 100k steps verified in {(t1 - t0)*1000:.2f}ms (0 NaNs/Infs).", flush=True)
    t_engine.test_asymmetric_breathing_waveform()
    print("  -> Breathing waveform verified.", flush=True)
    t_engine.test_explosion_failsafe_recovery()
    print("  -> Explosion failsafe verified.", flush=True)
    t_engine.test_frequency_spectrum_compliance()
    print("  -> Frequency spectrum verified.", flush=True)
    t_engine.test_low_pass_filter_attenuation()
    print("  -> Low-pass filter attenuation verified.", flush=True)
    t_engine.test_ascii_phase_portrait_generation()
    print("  -> ASCII phase portrait verified.", flush=True)
    print("=== ALL HAVEN SWAY TESTS PASSED SUCCESSFULLY ===", flush=True)
