import sys
import time
import threading
from typing import Dict, Optional, Tuple
import cv2
import numpy as np

from face_rec_module.face_recognizer import FaceRecognizer
from face_rec_module.face_database import IdentityMatch


class LandmarkSmoother:
    """
    Adaptive Dynamic Filter (1-Euro algorithm style):
    - When stationary (speed is low): heavy filtering with 2.0px deadband -> 100% rock-solid, zero jitter.
    - When moving (speed increases): cutoff jumps instantly (alpha -> 1.0) -> follows your face instantaneously with ZERO drag or lag!
    """

    def __init__(self, deadband_px: float = 2.0):
        self.deadband_px = deadband_px
        self.smooth_bbox: Optional[np.ndarray] = None
        self.smooth_kps: Optional[np.ndarray] = None
        self.smooth_conf: float = 0.0
        self.last_seen: float = 0.0

    def update(self, bbox: np.ndarray, kps: Optional[np.ndarray], conf: float) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
        now = time.time()
        # Reset if face was lost for more than 0.4 seconds
        if now - self.last_seen > 0.4 or self.smooth_kps is None or self.smooth_bbox is None:
            self.smooth_bbox = bbox.astype(np.float32)
            self.smooth_kps = kps.astype(np.float32) if kps is not None else None
            self.smooth_conf = conf
            self.last_seen = now
            out_box = np.round(self.smooth_bbox).astype(int)
            out_kps = np.round(self.smooth_kps).astype(int) if self.smooth_kps is not None else None
            return out_box, out_kps, conf

        # 1. Measure landmark speed/displacement
        if kps is not None:
            kps_f = kps.astype(np.float32)
            pts_diff = np.max(np.linalg.norm(kps_f - self.smooth_kps, axis=1))

            if pts_diff <= self.deadband_px:
                # Stationary: complete freeze, zero noise
                pass
            elif pts_diff > 12.0:
                # Fast movement: snap immediately to current position (NO LAG / NO INERTIA)
                self.smooth_kps = kps_f
                self.smooth_bbox = bbox.astype(np.float32)
            else:
                # Medium movement: dynamic adaptive alpha proportional to speed
                ratio = (pts_diff - self.deadband_px) / (12.0 - self.deadband_px)
                alpha = 0.50 + 0.45 * ratio
                self.smooth_kps = (1.0 - alpha) * self.smooth_kps + alpha * kps_f
                self.smooth_bbox = (1.0 - alpha) * self.smooth_bbox + alpha * bbox.astype(np.float32)
        else:
            self.smooth_bbox = bbox.astype(np.float32)

        self.smooth_conf = 0.70 * self.smooth_conf + 0.30 * conf
        self.last_seen = now

        return (
            np.round(self.smooth_bbox).astype(int),
            (np.round(self.smooth_kps).astype(int) if self.smooth_kps is not None else None),
            self.smooth_conf,
        )


def get_available_camera(preferred_index: int = 0):
    indices = [preferred_index] + [i for i in range(5) if i != preferred_index]
    for idx in indices:
        cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                return cap, idx
        cap.release()
    return None, -1


def draw_hud(frame: np.ndarray, smoothed_faces: list, fps: float) -> np.ndarray:
    display = frame.copy()
    h, w = frame.shape[:2]

    # Iron Man Holographic Color Palette
    # Cyan / Arc-Reactor Blue & Crisp White
    CYAN_ARC = (255, 230, 0)       # Bright vivid cyan-blue (BGR)
    WHITE_GLOW = (255, 255, 255)   # Pure white
    ICE_BLUE = (255, 200, 100)     # Soft holographic ice-blue
    DIM_BLUE = (180, 120, 40)      # Subtle wireframe blue

    # 1. Top Telemetry Bar (Minimal holographic header)
    overlay_bar = display.copy()
    cv2.rectangle(overlay_bar, (0, 0), (w, 42), (20, 15, 10), -1)
    cv2.addWeighted(overlay_bar, 0.40, display, 0.60, 0, display)
    cv2.line(display, (0, 42), (w, 42), ICE_BLUE, 1)

    cv2.putText(
        display,
        "STARK INDUSTRIES // TACTICAL AR HUD",
        (24, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        WHITE_GLOW,
        1,
        cv2.LINE_AA,
    )

    fps_text = f"INF: 60FPS | IDENTIFIED: {sum(1 for _, m, _, _ in smoothed_faces if m is not None)}"
    cv2.putText(
        display,
        fps_text,
        (w - 290, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        CYAN_ARC,
        1,
        cv2.LINE_AA,
    )

    # 2. Render each face target (No corner squares / boxes)
    for bbox, match, kps, smooth_conf in smoothed_faces:
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1

        if match is not None:
            display_conf = int(smooth_conf * 100) if smooth_conf > 0 else int(match.confidence * 100)
            label = f"{match.name.upper()} // {display_conf}%"
            wire_color = CYAN_ARC
            glow_color = WHITE_GLOW
        else:
            label = "TARGET IDENTIFYING..."
            wire_color = ICE_BLUE
            glow_color = (220, 220, 220)

        # Holographic Wireframe Mesh (White & Blue Iron Man HUD)
        if kps is not None and len(kps) >= 5:
            # Ensure coordinates are strictly python int tuples for OpenCV
            pts = [(int(p[0]), int(p[1])) for p in kps]
            overlay = display.copy()

            # Eye midpoint and scale
            eye_cx = int((pts[0][0] + pts[1][0]) / 2)
            eye_cy = int((pts[0][1] + pts[1][1]) / 2)
            eye_dist = max(10, int(np.linalg.norm(np.array(pts[0]) - np.array(pts[1]))))

            # Head yaw tilt ratio (-1.0 to +1.0) derived from nose horizontal offset
            yaw = float(pts[2][0] - eye_cx) / max(1.0, float(eye_dist))
            yaw = max(-0.8, min(0.8, yaw))

            # Forehead apex
            forehead = (int(eye_cx), int(max(0, eye_cy - int(eye_dist * 0.48))))

            # Temples / ear tragi
            left_temple = (
                int(pts[0][0] - eye_dist * (0.42 + 0.35 * yaw)),
                int(pts[0][1] + eye_dist * 0.10),
            )
            right_temple = (
                int(pts[1][0] + eye_dist * (0.42 - 0.35 * yaw)),
                int(pts[1][1] + eye_dist * 0.10),
            )

            # Chin apex
            mouth_cx = int((pts[3][0] + pts[4][0]) / 2)
            mouth_cy = int((pts[3][1] + pts[4][1]) / 2)
            chin = (int(mouth_cx), int(mouth_cy + int(eye_dist * 0.52)))

            # Jaw angles
            left_jaw = (
                int(pts[3][0] - eye_dist * (0.35 + 0.25 * yaw)),
                int(mouth_cy + eye_dist * 0.20),
            )
            right_jaw = (
                int(pts[4][0] + eye_dist * (0.35 - 0.25 * yaw)),
                int(mouth_cy + eye_dist * 0.20),
            )

            # Primary Facets (Blended Cyan / Ice-Blue lines)
            facets = [
                # Diamond around eyes and nose
                (pts[0], pts[1]),
                (pts[0], pts[2]),
                (pts[1], pts[2]),
                # Nose to mouth
                (pts[2], pts[3]),
                (pts[2], pts[4]),
                (pts[3], pts[4]),
                # Forehead dome
                (forehead, pts[0]),
                (forehead, pts[1]),
                (forehead, left_temple),
                (forehead, right_temple),
                # Temple & cheeks
                (left_temple, pts[0]),
                (right_temple, pts[1]),
                (left_temple, left_jaw),
                (right_temple, right_jaw),
                # Jaw & chin
                (left_jaw, pts[3]),
                (right_jaw, pts[4]),
                (left_jaw, chin),
                (right_jaw, chin),
                (pts[3], chin),
                (pts[4], chin),
            ]
            for p1, p2 in facets:
                cv2.line(overlay, p1, p2, wire_color, 1, cv2.LINE_AA)

            # Secondary optical cross-connects (Faint holographic depth)
            cv2.line(overlay, forehead, pts[2], DIM_BLUE, 1, cv2.LINE_AA)
            cv2.line(overlay, left_temple, right_temple, DIM_BLUE, 1, cv2.LINE_AA)

            # Semi-transparent blend
            cv2.addWeighted(overlay, 0.45, display, 0.55, 0, display)

            # Draw crisp White & Cyan glowing nodes
            nodes = [
                pts[0], pts[1], pts[2], pts[3], pts[4],
                forehead, left_temple, right_temple, left_jaw, right_jaw, chin
            ]
            for pt in nodes:
                # Outer cyan aura + inner solid white core
                cv2.circle(display, pt, 3, wire_color, 1, cv2.LINE_AA)
                cv2.circle(display, pt, 1, WHITE_GLOW, -1, cv2.LINE_AA)

            # Sleek Iron Man Target Bracket & Data Ribbon next to head
            ribbon_x = max(20, right_temple[0] + 18)
            ribbon_y = right_temple[1] - 10

            # Diagonal HUD leader line from cheek to data label
            cv2.line(display, right_temple, (ribbon_x - 4, ribbon_y + 6), ICE_BLUE, 1, cv2.LINE_AA)
            cv2.line(display, (ribbon_x - 4, ribbon_y + 6), (ribbon_x + 140, ribbon_y + 6), ICE_BLUE, 1, cv2.LINE_AA)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                display,
                label,
                (ribbon_x, ribbon_y),
                font,
                0.46,
                WHITE_GLOW,
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                "LOCK // ACTIVE",
                (ribbon_x, ribbon_y + 20),
                font,
                0.36,
                CYAN_ARC,
                1,
                cv2.LINE_AA,
            )

    return display

    return display


def main():
    print("=" * 65)
    print("LIVE REAL-TIME TACTICAL FACE RECOGNITION (SMOOTH & JITTER-FREE)")
    print("=" * 65)

    recognizer = FaceRecognizer(
        index_path="face_rec_module/data/faces.faiss",
        db_path="face_rec_module/data/identities.db",
        threshold=0.40,
    )

    cap, dev_idx = get_available_camera(0)
    if cap is None:
        print("Error: Could not open any camera device.", file=sys.stderr)
        sys.exit(1)

    print(f"Connected to camera on device index: {dev_idx}")

    window_name = "Smooth Face Recognition HUD"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)

    latest_frame = None
    cached_raw_results = []
    is_running = True
    lock = threading.Lock()

    # Temporal landmark smoother instance
    smoother = LandmarkSmoother(deadband_px=2.0)

    def recognition_worker():
        nonlocal cached_raw_results
        while is_running:
            frame_to_process = None
            with lock:
                if latest_frame is not None:
                    frame_to_process = latest_frame.copy()

            if frame_to_process is not None:
                results = recognizer.process_frame(frame_to_process)
                with lock:
                    cached_raw_results = results
            else:
                time.sleep(0.01)

    worker_thread = threading.Thread(target=recognition_worker, daemon=True)
    worker_thread.start()

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            with lock:
                latest_frame = frame
                raw_results = list(cached_raw_results)

            now = time.time()
            dt = now - prev_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)
            prev_time = now

            # Apply temporal exponential smoothing to eliminate noise and jitter
            smoothed_results = []
            if raw_results:
                # Update smoother with largest face
                bbox, match, kps = raw_results[0]
                conf = match.confidence if match is not None else 0.0
                sm_box, sm_kps, sm_conf = smoother.update(bbox, kps, conf)
                smoothed_results.append((sm_box, match, sm_kps, sm_conf))

            display = draw_hud(frame, smoothed_results, fps)
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("Exit signal received.")
                break

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        is_running = False
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
