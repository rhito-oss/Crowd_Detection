# Two-Tier Real-Time Face Recognition Engine

This is a standalone, decoupled facial recognition module. It does not touch or modify any code in the original crowd detection pipeline.

---

## 1. Architecture Overview

- **Face Vector Database (`face_rec_module/data/faces.faiss`)**:
  - Powered by **FAISS `IndexFlatIP`** (512-D ArcFace normalized vectors).
  - Sub-millisecond Cosine Similarity matching (< 0.2 ms).
- **Metadata Database (`face_rec_module/data/identities.db`)**:
  - Powered by **SQLite in WAL mode**.
  - Microsecond identity and profile lookup (< 0.02 ms) matched 1:1 by row ID.
- **Inference Engine**:
  - Uses InsightFace (`buffalo_sc`) running on CPU with ONNX Runtime.

---

## 2. Modules & Scripts

| File | Purpose |
| :--- | :--- |
| **`enroll_face.py`** | Interactive 5-burst webcam face scanner to enroll a new face. |
| **`live_recognition.py`** | Standalone live real-time camera face recognition stream with tactical HUD. |
| **`face_database.py`** | Low-latency two-tier FAISS + SQLite database abstraction. |
| **`face_recognizer.py`** | Face alignment, 512-D embedding extraction, and matching wrapper. |

---

## 3. How to Use

### A. Enroll a New Face (Webcam Scanner)
```bash
.venv/bin/python -m face_rec_module.enroll_face --name "YourName"
```
* Align face in the green guide circle and press **SPACE**.
* Follow the 5-angle prompts (Center, Left, Right, Up, Down).

### B. Run Live Real-Time Face Recognition
```bash
.venv/bin/python -m face_rec_module.live_recognition
```
* Opens your camera and starts live recognition against your database.
* Renders tactical corner brackets, 2D/3D landmarks, and identity pill: `RHITO // 92%` (or `UNKNOWN SUBJECT` if not recognized).
* Press `Q` or `ESC` to quit.
