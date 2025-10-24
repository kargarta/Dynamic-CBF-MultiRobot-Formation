# Dynamic Control Barrier Function–Based Multi-Robot Formation Control

This repository implements a **Safety-Critical Control Framework** for **multi-robot formation navigation** in environments containing **dynamic and static polygonal obstacles**.  
The controller is based on **Control Barrier Functions (CBFs)** and **Quadratic Programming (QP)** to guarantee safety while maintaining formation and convergence to a goal position.

---

## 🚀 Key Features

- **Dynamic Obstacle Avoidance:** Real-time adaptation to moving obstacles using time-varying CBF constraints.
- **Formation Preservation:** Maintains square or arbitrary robot formations using pairwise CBFs.
- **Mode Switching Logic:** Automatically transitions between:
  - **Base Mode:** Normal formation tracking
  - **Slack Mode:** Temporarily relaxes formation constraints to navigate narrow passages
  - **Recovery Mode:** Re-forms the structure and stabilizes after obstacle avoidance
- **Safety and Goal Convergence:** Ensures both obstacle clearance and convergence to a designated target.
- **Visualization and Data Logging:** Generates trajectory and distance plots, and exports simulation data to CSV.

---

## 🧩 System Overview

The proposed approach combines:
1. **Nominal Control Law:** Drives each robot toward its goal and maintains formation geometry.
2. **CBF-QP Safety Filter:** Enforces safety constraints for obstacle avoidance, formation maintenance, and goal convergence.
3. **Dynamic Environment Model:** Includes both static and linearly moving obstacles with known velocities.

---

## ⚙️ Dependencies

Make sure Python ≥ 3.8 is installed and then run:

```bash
pip install numpy scipy cvxpy matplotlib shapely pandas robotarium
