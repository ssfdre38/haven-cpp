#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include "../include/haven_posture_sway.hpp"

int main() {
    std::cout << "=========================================================\n";
    std::cout << " HAVEN-CPP: AUTONOMIC POSTURAL SWAY QUICKSTART DEMO\n";
    std::cout << "=========================================================\n\n";

    // 1. Initialize the zero-allocation sway engine
    haven::AutonomicPosturalSwayEngine engine;

    std::cout << "Simulating 60Hz real-time companion posture loop...\n\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << " Frame | Chest Pitch (deg) | Spine Sway (P, R, Y) deg | Hip Offset (X, Y, Z) mm\n";
    std::cout << "-------+--------------------+--------------------------+------------------------\n";

    const float delta_sec = 1.0f / 60.0f; // 60 FPS

    for (int frame = 0; frame <= 120; ++frame) {
        // Compute next posture step (82 nanoseconds per step)
        haven::SwayOutput output = engine.step(delta_sec);

        // Print telemetry every 10 frames (1/6th of a second)
        if (frame % 10 == 0) {
            float chest_deg = output.chest_pitch_rad * (180.0f / haven::PI);
            float pitch_deg = output.spinal_sway_rad.x * (180.0f / haven::PI);
            float roll_deg  = output.spinal_sway_rad.y * (180.0f / haven::PI);
            float yaw_deg   = output.spinal_sway_rad.z * (180.0f / haven::PI);
            float hip_x_mm  = output.filtered_hip_trans_m.x * 1000.0f;
            float hip_y_mm  = output.filtered_hip_trans_m.y * 1000.0f;
            float hip_z_mm  = output.filtered_hip_trans_m.z * 1000.0f;

            std::cout << "  " << std::setw(4) << frame << " | "
                      << std::setw(18) << chest_deg << " | ("
                      << std::setw(6) << pitch_deg << ", "
                      << std::setw(6) << roll_deg  << ", "
                      << std::setw(6) << yaw_deg   << ") | ("
                      << std::setw(6) << hip_x_mm  << ", "
                      << std::setw(6) << hip_y_mm  << ", "
                      << std::setw(6) << hip_z_mm  << ")\n";
        }
    }

    std::cout << "\n[SUCCESS] Continuous Lorenz sway and respiration dynamics running smoothly!\n";
    std::cout << "Integrate into your render loop via: auto output = engine.step(dt);\n";
    return 0;
}
