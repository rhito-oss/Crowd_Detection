# Real-Time Multi-Person Crowd & Density Detection

A real-time crowd and multi-person detection system using OpenCV and YOLOv8, optimized for CPU inference. The system detects multiple people simultaneously, evaluates physical closeness and spatial density per frame, and displays an on-screen HUD with bounding boxes, proximity alert lines, and a live crowd indicator.

---

## Table of Contents

- [Features](#features)
- [Prerequisites: Installing `uv`](#prerequisites-installing-uv)
  - [Linux](#linux)
  - [macOS](#macos)
  - [Windows](#windows)
- [Installation from Scratch](#installation-from-scratch)
- [Configuration (`config.yaml`)](#configuration-configyaml)
  - [Tuning for Your OS & Hardware](#tuning-for-your-os--hardware)
- [Running the Application](#running-the-application)
- [Controls & Shortcuts](#controls--shortcuts)
- [License](#license)

---

## Features

- **Multi-Person Detection**: Detects and tracks multiple people in real-time using YOLOv8 nano (`yolov8n.pt`).
- **Live Density & Closeness Evaluation**:
  - Detects when people are standing shoulder-to-shoulder, overlapping, or within a close proximity threshold.
  - Draws visual proximity lines connecting close individuals and highlights their bounding boxes in red (`[CLOSE]`).
  - Displays a clean real-time HUD status badge: `crowded: true` or `crowded: false`.
- **Hardware Agnostic**: Fully optimized for standard laptop/desktop CPUs (Intel/AMD) with CPU-only PyTorch, but can also run on NVIDIA GPUs if available.
- **Cross-Platform**: Supports Linux (V4L2), Windows (DirectShow / MSMF), and macOS (AVFoundation).

---

## Prerequisites: Installing `uv`

[`uv`](https://github.com/astral-sh/uv) is an extremely fast Python package and project manager written in Rust. Choose the installation command for your operating system:

### Linux

Run in your terminal (bash/zsh):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
*On Arch Linux, you can also install via pacman:*
```bash
sudo pacman -S uv
```

### macOS

Using Homebrew:
```bash
brew install uv
```
Or via curl:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows

Run in PowerShell:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
Or using `winget`:
```powershell
winget install --id=astral-sh.uv -e
```

> **Note:** After installing `uv`, restart your terminal or ensure `uv` is in your system `PATH`.

---

## Installation from Scratch

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rhito-oss/Crowd_Detection.git
   cd Crowd_Detection
   uv venv
   overlay use .venv/bin/activate.nu
   ```

2. **Create virtual environment and install CPU dependencies:**
   
   - **Linux / macOS:**
     ```bash
     uv pip install --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision
     uv pip install -e .
     ```
   
   - **Windows (PowerShell):**
     ```powershell
     uv pip install --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision
     uv pip install -e .
     ```

*(The YOLOv8 nano model weights `yolov8n.pt` (~6.2 MB) will automatically download on the first run).*

---

## Configuration (`config.yaml`)

The application is completely configured through `config.yaml`.

```yaml
camera:
  device_index: 0              # Camera index (0 is typically default/integrated webcam)
  width: 1280                  # Video stream width
  height: 720                  # Video stream height
  fps: 30                      # Camera target frame rate
  api_preference: "v4l2"       # Camera backend (see tuning guide below)

detection:
  model: "yolov8n.pt"          # Model weights (yolov8n.pt, yolov8s.pt, etc.)
  conf_threshold: 0.35         # Detection confidence threshold (0.0 to 1.0)
  iou_threshold: 0.70          # NMS IoU threshold (higher keeps heavily overlapping people)
  device: "cpu"                # Compute device ("cpu" or "cuda" / "0")

density:
  enabled: true
  min_people_for_crowd: 2      # Minimum people to consider frame crowded
  closeness_threshold: 0.25    # Proximity threshold relative to frame diagonal
  min_close_pairs: 1           # Number of close pairs required to trigger crowded: true

display:
  window_name: "Crowd Detection - Person Tracking"
  show_fps: true               # Display live FPS counter
  show_count: true             # Display detected people counter
  show_conf: true              # Display confidence percentage on boxes
  show_density: true           # Display crowded: true/false status
  box_color: [0, 255, 0]       # Normal bounding box color (BGR: Green)
  crowd_box_color: [0, 0, 255] # Close/Crowded bounding box color (BGR: Red)
  line_color: [0, 0, 255]      # Proximity line color (BGR: Red)
  text_color: [255, 255, 255]  # Label text color (BGR: White)
  badge_color: [40, 40, 40]    # HUD overlay background (BGR: Dark Gray)
```

### Tuning for Your OS & Hardware

#### 1. Operating System Camera Backend (`api_preference`)
- **Arch Linux / Ubuntu / Debian**: Keep `"v4l2"` (Video4Linux).
- **Windows (Laptop or Desktop with USB Webcam)**: Change to `"any"` or `"dshow"` (DirectShow).
- **macOS**: Change to `"any"` (uses AVFoundation automatically).

#### 2. External / USB Webcams
- If you have an external webcam plugged in (especially on a desktop PC), change `device_index: 0` to `device_index: 1` (or test `2` if index 0 is an integrated camera or virtual device).

#### 3. Hardware & GPU Acceleration (`device`)
- **Integrated GPU / CPU Only (Default)**: Leave `device: "cpu"`. YOLOv8 nano runs smoothly at 25-30+ FPS on modern Intel/AMD processors.
- **Dedicated NVIDIA GPU**: 
  - Install CUDA PyTorch: `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`
  - In `config.yaml`, change `device: "cpu"` to `device: "cuda"` (or `"0"`).

#### 4. Closeness & Crowd Sensitivity
- **`closeness_threshold`**: 
  - Increase (e.g. `0.30` - `0.35`) to make closeness detection **more sensitive** (triggers even when people are standing further apart).
  - Decrease (e.g. `0.15` - `0.20`) to require people to be very close / touching.
- **`min_people_for_crowd`**: Set the minimum crowd headcount required (e.g. `2`, `3`, `5`).

---

## Running the Application

To start the detection window:

```bash
uv run --no-sync crowd-detection
```

Or run directly with Python:

```bash
# Linux / macOS
.venv/bin/python -m crowd_detection.main

# Windows (PowerShell)
.venv\Scripts\python -m crowd_detection.main
```

---

## Controls & Shortcuts

- **`q`** or **`ESC`**: Quit and safely close the camera stream and window.
- **Window `X` Button**: Closes the application cleanly.

---

## License

This project is open-source under the [MIT License](LICENSE).
