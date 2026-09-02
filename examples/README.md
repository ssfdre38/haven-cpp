# 🚀 Quickstart: Autonomic Postural Sway Engine

This example demonstrates how to integrate the **Autonomous Postural Sway & Diaphragmatic Breath Synthesis Engine** into your companion runtime in **less than 10 lines of C++20**.

---

## ⚡ 1-Line Build & Run

From the repository root:

```bash
# Standard C++20 compilation (Zero dependencies)
g++ -std=c++20 examples/quickstart_sway.cpp -I include -o quickstart
./quickstart
```

---

## 💻 30-Second Integration Pattern

```cpp
#include "haven_posture_sway.hpp"

class CompanionAvatar {
private:
    haven::AutonomicPosturalSwayEngine swayEngine_;

public:
    void onRenderFrame(float deltaSeconds) {
        // 1. Step the engine (82ns execution time, zero heap allocations)
        haven::SwayOutput sway = swayEngine_.step(deltaSeconds);

        // 2. Apply SO(3) quaternion directly to character torso/spine
        torsoBone.setRotation(sway.filtered_torso_quat.x,
                              sway.filtered_torso_quat.y,
                              sway.filtered_torso_quat.z,
                              sway.filtered_torso_quat.w);

        // 3. Apply figure-8 weight shift to hips
        hipBone.setPosition(sway.filtered_hip_trans_m.x,
                            sway.filtered_hip_trans_m.y,
                            sway.filtered_hip_trans_m.z);
    }
};
```
