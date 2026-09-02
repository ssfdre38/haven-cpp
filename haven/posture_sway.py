r"""
Autonomic Postural Sway & Diaphragmatic Breath Synthesis Engine.

Production-grade Python counterpart to `haven_posture_sway.hpp` for `ssfdre38/haven-cpp`.
Implements:
1. Sub-stepped numerical integration of chaotic Lorenz attractor (\Delta t_sub = 0.005s)
2. Continuous-time exponential low-pass filtering (\alpha_LP = 1 - e^{-6 \Delta t})
3. Numerical explosion failsafes (|x| > 60 -> (0.1, 0, 0))
4. Asymmetric diaphragmatic respiration (35% inhalation / 65% exhalation at 0.18 Hz)
5. Lissajous figure-8 postural balance weight shifts (0.08 Hz)
6. Geodesic SO(3) angular deadband filtering (\epsilon = 0.0005 rad)
"""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class Vec3:
    """3D Cartesian Vector representation."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vec3":
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def length_sq(self) -> float:
        return self.x * self.x + self.y * self.y + self.z * self.z

    def length(self) -> float:
        return math.sqrt(self.length_sq())

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float32)

    @staticmethod
    def distance(a: "Vec3", b: "Vec3") -> float:
        return (a - b).length()


@dataclass
class Quaternion:
    """Unit Quaternion for 3D SO(3) rotations [x, y, z, w]."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    @classmethod
    def identity(cls) -> "Quaternion":
        return cls(0.0, 0.0, 0.0, 1.0)

    def normalize(self) -> "Quaternion":
        norm_sq = self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w
        if norm_sq > 1e-12:
            inv = 1.0 / math.sqrt(norm_sq)
            self.x *= inv
            self.y *= inv
            self.z *= inv
            self.w *= inv
        else:
            self.x, self.y, self.z, self.w = 0.0, 0.0, 0.0, 1.0
        return self

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.w], dtype=np.float32)

    @classmethod
    def from_euler_xyz(cls, pitch_x: float, roll_y: float, yaw_z: float) -> "Quaternion":
        """Constructs quaternion from XYZ Tait-Bryan Euler angles in radians."""
        cx = math.cos(pitch_x * 0.5)
        sx = math.sin(pitch_x * 0.5)
        cy = math.cos(roll_y * 0.5)
        sy = math.sin(roll_y * 0.5)
        cz = math.cos(yaw_z * 0.5)
        sz = math.sin(yaw_z * 0.5)

        q = cls(
            x=sx * cy * cz + cx * sy * sz,
            y=cx * sy * cz - sx * cy * sz,
            z=cx * cy * sz + sx * sy * cz,
            w=cx * cy * cz - sx * sy * sz,
        )
        q.normalize()
        return q

    @staticmethod
    def geodesic_angle(q1: "Quaternion", q2: "Quaternion") -> float:
        """Computes exact geodesic Riemannian distance on SO(3) 3-sphere."""
        xd = q1.w * q2.x - q1.x * q2.w - q1.y * q2.z + q1.z * q2.y
        yd = q1.w * q2.y + q1.x * q2.z - q1.y * q2.w - q1.z * q2.x
        zd = q1.w * q2.z - q1.x * q2.y + q1.y * q2.x - q1.z * q2.w
        wd = q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z

        v_norm = math.sqrt(xd * xd + yd * yd + zd * zd)
        return 2.0 * math.atan2(v_norm, abs(wd))

    @staticmethod
    def slerp(q1: "Quaternion", q2: "Quaternion", t: float) -> "Quaternion":
        """Performs true SO(3) spherical linear interpolation with geodesic path alignment."""
        dot = q1.x * q2.x + q1.y * q2.y + q1.z * q2.z + q1.w * q2.w
        q2_x, q2_y, q2_z, q2_w = q2.x, q2.y, q2.z, q2.w

        if dot < 0.0:
            dot = -dot
            q2_x, q2_y, q2_z, q2_w = -q2_x, -q2_y, -q2_z, -q2_w

        if dot > 0.9995:
            res = Quaternion(
                x=q1.x + t * (q2_x - q1.x),
                y=q1.y + t * (q2_y - q1.y),
                z=q1.z + t * (q2_z - q1.z),
                w=q1.w + t * (q2_w - q1.w),
            )
            res.normalize()
            return res

        theta_0 = math.acos(max(-1.0, min(1.0, dot)))
        theta = theta_0 * t
        sin_theta = math.sin(theta)
        sin_theta_0 = math.sin(theta_0)

        s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0

        res = Quaternion(
            x=(s0 * q1.x) + (s1 * q2_x),
            y=(s0 * q1.y) + (s1 * q2_y),
            z=(s0 * q1.z) + (s1 * q2_z),
            w=(s0 * q1.w) + (s1 * q2_w),
        )
        res.normalize()
        return res


class PosturalDeadbandFilter:
    """
    Sub-millimeter & Geodesic Angular Deadband Filter.
    Eliminates 60Hz floating-point micro-vibrations and shader shimmering at joint rest.
    """

    def __init__(
        self,
        angular_epsilon: float = 0.0005,
        translation_epsilon: float = 0.0001,
        filter_lambda: float = 18.0,
    ):
        self.angular_epsilon = angular_epsilon
        self.translation_epsilon = translation_epsilon
        self.filter_lambda = filter_lambda

        self.last_quat = Quaternion.identity()
        self.last_trans = Vec3(0.0, 0.0, 0.0)
        self.initialized_rot = False
        self.initialized_trans = False

    def reset(self) -> None:
        self.last_quat = Quaternion.identity()
        self.last_trans = Vec3(0.0, 0.0, 0.0)
        self.initialized_rot = False
        self.initialized_trans = False

    def filter_rotation(self, target: Quaternion, delta_sec: float) -> Quaternion:
        if not self.initialized_rot:
            self.last_quat = Quaternion(target.x, target.y, target.z, target.w)
            self.initialized_rot = True
            return target

        delta_rad = Quaternion.geodesic_angle(self.last_quat, target)
        if delta_rad < self.angular_epsilon:
            return self.last_quat

        alpha = 1.0 - math.exp(-self.filter_lambda * min(delta_sec, 0.033))
        self.last_quat = Quaternion.slerp(self.last_quat, target, alpha)
        return self.last_quat

    def filter_translation(self, target: Vec3, delta_sec: float) -> Vec3:
        if not self.initialized_trans:
            self.last_trans = Vec3(target.x, target.y, target.z)
            self.initialized_trans = True
            return target

        dist = Vec3.distance(self.last_trans, target)
        if dist < self.translation_epsilon:
            return self.last_trans

        alpha = 1.0 - math.exp(-self.filter_lambda * min(delta_sec, 0.033))
        self.last_trans = self.last_trans + (target - self.last_trans) * alpha
        return self.last_trans


@dataclass
class SwayConfig:
    """Configuration hyperparameters for postural sway & breath engine."""
    # Lorenz Attractor Dynamics
    lorenz_sigma: float = 10.0
    lorenz_rho: float = 28.0
    lorenz_beta: float = 8.0 / 3.0
    sub_dt: float = 0.005
    lorenz_scale_pitch: float = 0.00003
    lorenz_scale_roll: float = 0.00003
    lorenz_scale_yaw: float = 0.00002
    lorenz_cutoff_freq: float = 6.0  # rad/s (~0.955 Hz)

    # Asymmetric Diaphragmatic Respiration
    breath_freq: float = 0.18        # 0.18 Hz (5.55s cycle)
    inhale_fraction: float = 0.35    # 35% Inhale / 65% Exhale
    breath_amplitude: float = 0.012  # rad (~0.69 deg)

    # Lissajous Figure-8 Balance Sway
    sway_freq: float = 0.08          # 0.08 Hz (12.5s cycle)
    fig8_hip_x_amp: float = 0.005    # 5 mm lateral sway
    fig8_hip_z_amp: float = 0.003    # 3 mm anterior-posterior sway
    fig8_pitch_amp: float = 0.002    # ~0.11 deg pitch
    fig8_roll_amp: float = 0.003     # ~0.17 deg roll

    # Deadband & Smoothing
    angular_deadband: float = 0.0005     # rad (0.0286 deg)
    translation_deadband: float = 0.0001 # m (0.1 mm)
    filter_lambda: float = 18.0          # s^-1


@dataclass
class SwayOutput:
    """Instantaneous posture output state."""
    chest_pitch_rad: float
    spinal_sway_rad: Vec3
    hip_translation_m: Vec3
    filtered_torso_quat: Quaternion
    filtered_hip_trans_m: Vec3


class AutonomicPosturalSwayEngine:
    """
    Production-grade Autonomic Postural Sway Engine.
    
    Generates non-repeating organic human posture shifts through chaotic attractor dynamics,
    asymmetric breath pacing, figure-8 balance shifts, and micro-jitter deadband filtering.
    """

    def __init__(self, config: Optional[SwayConfig] = None):
        self.config = config or SwayConfig()

        # Attractor State
        self.lx = 0.1
        self.ly = 0.0
        self.lz = 0.0

        # Low-pass Filtered Lorenz State
        self.l_smooth_x = 0.0
        self.l_smooth_y = 0.0
        self.l_smooth_z = 0.0

        # Timers
        self.breath_timer = 0.0
        self.sway_timer = 0.0

        # Deadband Filter
        self.deadband_filter = PosturalDeadbandFilter(
            angular_epsilon=self.config.angular_deadband,
            translation_epsilon=self.config.translation_deadband,
            filter_lambda=self.config.filter_lambda,
        )

    def reset(self) -> None:
        """Resets dynamic state to canonical resting equilibrium."""
        self.lx = 0.1
        self.ly = 0.0
        self.lz = 0.0
        self.l_smooth_x = 0.0
        self.l_smooth_y = 0.0
        self.l_smooth_z = 0.0
        self.breath_timer = 0.0
        self.sway_timer = 0.0
        self.deadband_filter.reset()

    def update(self, delta_sec: float) -> SwayOutput:
        """
        Advances the autonomic posture simulation by delta_sec.
        
        Args:
            delta_sec: Time step in seconds. Clamped to [0.0001, 0.1].
            
        Returns:
            SwayOutput containing chest pitch, spinal Euler angles, and filtered transforms.
        """
        dt = max(0.0001, min(0.1, float(delta_sec)))

        # -------------------------------------------------------------
        # 1. Asymmetric Diaphragmatic Respiration (35% Inhale / 65% Exhale)
        # -------------------------------------------------------------
        self.breath_timer += dt
        breath_period = 1.0 / self.config.breath_freq
        phase = (self.breath_timer % breath_period) / breath_period

        if phase < self.config.inhale_fraction:
            u = phase / self.config.inhale_fraction
            chest_pitch = math.sin(u * (math.pi * 0.5)) * self.config.breath_amplitude
        else:
            v = (phase - self.config.inhale_fraction) / (1.0 - self.config.inhale_fraction)
            chest_pitch = math.cos(v * (math.pi * 0.5)) * self.config.breath_amplitude

        # -------------------------------------------------------------
        # 2. Sub-stepped Numerical Lorenz Integration
        # -------------------------------------------------------------
        steps = max(1, min(5, int(dt / self.config.sub_dt)))
        sub_dt = dt / float(steps)

        for _ in range(steps):
            dx = self.config.lorenz_sigma * (self.ly - self.lx)
            dy = self.lx * (self.config.lorenz_rho - self.lz) - self.ly
            dz = self.lx * self.ly - self.config.lorenz_beta * self.lz

            self.lx += dx * sub_dt
            self.ly += dy * sub_dt
            self.lz += dz * sub_dt

        # Numerical Explosion & NaN Failsafe
        if (
            math.isnan(self.lx) or math.isnan(self.ly) or math.isnan(self.lz) or
            math.isinf(self.lx) or math.isinf(self.ly) or math.isinf(self.lz) or
            abs(self.lx) > 60.0 or abs(self.ly) > 60.0 or abs(self.lz) > 60.0
        ):
            self.lx = 0.1
            self.ly = 0.0
            self.lz = 0.0

        # Continuous-time exponential low-pass filter (cutoff 6.0 rad/s)
        lp_alpha = 1.0 - math.exp(-self.config.lorenz_cutoff_freq * dt)
        self.l_smooth_x += lp_alpha * (self.lx - self.l_smooth_x)
        self.l_smooth_y += lp_alpha * (self.ly - self.l_smooth_y)
        self.l_smooth_z += lp_alpha * (self.lz - self.l_smooth_z)

        # -------------------------------------------------------------
        # 3. Lissajous Figure-8 Balance Sway
        # -------------------------------------------------------------
        self.sway_timer += dt
        omega = 2.0 * math.pi * self.config.sway_freq

        fig8_x = self.config.fig8_hip_x_amp * math.sin(omega * self.sway_timer)
        fig8_z = self.config.fig8_hip_z_amp * math.sin(2.0 * omega * self.sway_timer)
        fig8_pitch = self.config.fig8_pitch_amp * math.sin(2.0 * omega * self.sway_timer)
        fig8_roll = self.config.fig8_roll_amp * math.sin(omega * self.sway_timer)

        spinal_sway = Vec3(
            x=self.l_smooth_x * self.config.lorenz_scale_pitch + fig8_pitch,
            y=self.l_smooth_y * self.config.lorenz_scale_roll + fig8_roll,
            z=(self.l_smooth_z - 25.0) * self.config.lorenz_scale_yaw,
        )

        hip_trans = Vec3(x=fig8_x, y=0.0, z=fig8_z)

        # -------------------------------------------------------------
        # 4. Geodesic Deadband Filtering
        # -------------------------------------------------------------
        raw_torso_quat = Quaternion.from_euler_xyz(
            pitch_x=spinal_sway.x + chest_pitch,
            roll_y=spinal_sway.y,
            yaw_z=spinal_sway.z,
        )

        filtered_quat = self.deadband_filter.filter_rotation(raw_torso_quat, dt)
        filtered_trans = self.deadband_filter.filter_translation(hip_trans, dt)

        return SwayOutput(
            chest_pitch_rad=chest_pitch,
            spinal_sway_rad=spinal_sway,
            hip_translation_m=hip_trans,
            filtered_torso_quat=filtered_quat,
            filtered_hip_trans_m=filtered_trans,
        )

    def inject_lorenz_state(self, x: float, y: float, z: float) -> None:
        """Injects explicit attractor coordinates for testing boundary recovery."""
        self.lx = float(x)
        self.ly = float(y)
        self.lz = float(z)


def generate_sway_trajectory(
    duration_sec: float = 30.0,
    fps: float = 60.0,
    config: Optional[SwayConfig] = None,
) -> Dict[str, np.ndarray]:
    """
    Generates a full continuous simulation trajectory array for analysis and visualization.
    
    Returns:
        Dictionary with keys:
        - time: [N] float32
        - chest_pitch: [N] float32
        - spinal_sway: [N, 3] float32 (pitch, roll, yaw)
        - hip_translation: [N, 3] float32 (x, y, z)
        - filtered_quat: [N, 4] float32 (x, y, z, w)
        - filtered_trans: [N, 3] float32 (x, y, z)
        - lorenz_raw: [N, 3] float32
        - lorenz_smooth: [N, 3] float32
    """
    engine = AutonomicPosturalSwayEngine(config)
    num_frames = int(duration_sec * fps)
    dt = 1.0 / fps

    times = np.zeros(num_frames, dtype=np.float32)
    chest_pitch = np.zeros(num_frames, dtype=np.float32)
    spinal_sway = np.zeros((num_frames, 3), dtype=np.float32)
    hip_trans = np.zeros((num_frames, 3), dtype=np.float32)
    filtered_quat = np.zeros((num_frames, 4), dtype=np.float32)
    filtered_trans = np.zeros((num_frames, 3), dtype=np.float32)
    lorenz_raw = np.zeros((num_frames, 3), dtype=np.float32)
    lorenz_smooth = np.zeros((num_frames, 3), dtype=np.float32)

    for i in range(num_frames):
        out = engine.update(dt)
        times[i] = i * dt
        chest_pitch[i] = out.chest_pitch_rad
        spinal_sway[i] = [out.spinal_sway_rad.x, out.spinal_sway_rad.y, out.spinal_sway_rad.z]
        hip_trans[i] = [out.hip_translation_m.x, out.hip_translation_m.y, out.hip_translation_m.z]
        filtered_quat[i] = [out.filtered_torso_quat.x, out.filtered_torso_quat.y, out.filtered_torso_quat.z, out.filtered_torso_quat.w]
        filtered_trans[i] = [out.filtered_hip_trans_m.x, out.filtered_hip_trans_m.y, out.filtered_hip_trans_m.z]
        lorenz_raw[i] = [engine.lx, engine.ly, engine.lz]
        lorenz_smooth[i] = [engine.l_smooth_x, engine.l_smooth_y, engine.l_smooth_z]

    return {
        "time": times,
        "chest_pitch": chest_pitch,
        "spinal_sway": spinal_sway,
        "hip_translation": hip_trans,
        "filtered_quat": filtered_quat,
        "filtered_trans": filtered_trans,
        "lorenz_raw": lorenz_raw,
        "lorenz_smooth": lorenz_smooth,
    }


def compute_power_spectrum(
    signal: np.ndarray,
    fps: float = 60.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes FFT one-sided power spectral density (PSD) of a 1D kinematic signal.
    
    Returns:
        (freqs, psd) tuple
    """
    n = len(signal)
    fft_vals = np.fft.rfft(signal - np.mean(signal))
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    psd = (np.abs(fft_vals) ** 2) / n
    return freqs, psd


def render_ascii_phase_portrait(
    x_data: np.ndarray,
    y_data: np.ndarray,
    width: int = 60,
    height: int = 24,
) -> str:
    """
    Renders an ASCII phase portrait grid of a 2D trajectory.
    """
    grid = [[" " for _ in range(width)] for _ in range(height)]

    min_x, max_x = float(np.min(x_data)), float(np.max(x_data))
    min_y, max_y = float(np.min(y_data)), float(np.max(y_data))

    span_x = max(1e-6, max_x - min_x)
    span_y = max(1e-6, max_y - min_y)

    for x, y in zip(x_data, y_data):
        col = int((x - min_x) / span_x * (width - 1))
        row = int((max_y - y) / span_y * (height - 1))
        col = max(0, min(width - 1, col))
        row = max(0, min(height - 1, row))
        grid[row][col] = "•"

    border = "+" + "-" * width + "+"
    lines = [border]
    for row in grid:
        lines.append("|" + "".join(row) + "|")
    lines.append(border)
    lines.append(f"X: [{min_x:.3f}, {max_x:.3f}] | Y: [{min_y:.3f}, {max_y:.3f}]")

    return "\n".join(lines)
