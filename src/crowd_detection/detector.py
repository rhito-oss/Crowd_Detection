import math
import time
from dataclasses import dataclass
from typing import List, Tuple
from ultralytics import YOLO
import numpy as np


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    track_id: int = -1
    speed: float = 0.0
    class_id: int = 0
    label: str = "person"


class PersonDetector:
    def __init__(self, model_name: str = "yolov8n.pt", conf_threshold: float = 0.35, iou_threshold: float = 0.70, device: str = "cpu"):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = YOLO(model_name)
        # Store previous positions and timestamps: {track_id: (center_x, center_y, timestamp)}
        self._track_history = {}

    def detect(self, frame: np.ndarray, timestamp: float = None) -> Tuple[List[Detection], float]:
        if timestamp is None:
            timestamp = time.time()

        # Class 0 in COCO is person. persist=True maintains track IDs across frames
        results = self.model.track(
            source=frame,
            classes=[0],
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            persist=True,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results:
            return detections, 0.0

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections, 0.0

        current_track_ids = set()
        total_speed = 0.0

        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())

            track_id = int(box.id[0].cpu().numpy()) if box.id is not None else -1

            cx = (xyxy[0] + xyxy[2]) / 2.0
            cy = (xyxy[1] + xyxy[3]) / 2.0

            speed_kmh = 0.0
            if track_id != -1:
                current_track_ids.add(track_id)
                box_h = max(1, xyxy[3] - xyxy[1])
                # Estimate scale: an average person is ~1.7m tall
                # meters_per_pixel = 1.7 / box_h
                m_per_px = 1.7 / box_h

                if track_id in self._track_history:
                    prev_cx, prev_cy, prev_t = self._track_history[track_id]
                    dt = timestamp - prev_t
                    if dt > 0:
                        dist_px = math.hypot(cx - prev_cx, cy - prev_cy)
                        speed_px_s = dist_px / dt
                        # Convert px/s -> m/s -> km/h: (m/s) * 3.6
                        speed_m_s = speed_px_s * m_per_px
                        speed_kmh = speed_m_s * 3.6
                self._track_history[track_id] = (cx, cy, timestamp)

            total_speed += speed_kmh

            detections.append(
                Detection(
                    x1=int(xyxy[0]),
                    y1=int(xyxy[1]),
                    x2=int(xyxy[2]),
                    y2=int(xyxy[3]),
                    confidence=conf,
                    track_id=track_id,
                    speed=speed_kmh,
                    class_id=cls_id,
                    label="person",
                )
            )

        # Cleanup stale tracks not seen in over 5 seconds
        stale_ids = [tid for tid, (_, _, t) in self._track_history.items() if timestamp - t > 5.0]
        for tid in stale_ids:
            del self._track_history[tid]

        # Formula: (speed of person 1 + person 2 + ... + person n) / total persons
        n = len(detections)
        avg_speed = (total_speed / n) if n > 0 else 0.0

        return detections, avg_speed
