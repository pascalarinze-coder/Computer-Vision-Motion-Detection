# Computer-Vision-Motion-Detection

# 👁️ Logic-Driven Motion Detection & AR System

## 📌 Project Overview
A real-time Computer Vision system that processes a 5x5 grid to detect motion. It demonstrates "Systems Thinking" by applying specific rules to how automation spreads across a grid.

### ⚙️ Core Logic (State Machine)
- **Grid Partitioning:** Divides the camera feed into 25 independent monitoring zones.
- **Neighbor-Constrained Activation:** Motion is only registered if it is the first event OR if it occurs adjacent to an already active cell.
- **Automatic Reset:** Features a 10-second watchdog timer that flushes the system state during inactivity.

## 🛠️ Tech Stack
- **OpenCV:** Image processing and AR projection.
- **NumPy:** Matrix manipulation for grid logic.
- **Python:** Core system architecture.
