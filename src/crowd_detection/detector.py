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
    class_id: int = 0
    label: str = "person"


class PersonDetector:
    def __init__(self, model_name: str = "yolov8n.pt", conf_threshold: float = 0.35, iou_threshold: float = 0.70, device: str = "cpu"):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        # Load YOLO model (will automatically download weights if not cached)
        self.model = YOLO(model_name)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        # Class 0 in COCO is person
        results = self.model.predict(
            source=frame,
            classes=[0],
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())

            detections.append(
                Detection(
                    x1=int(xyxy[0]),
                    y1=int(xyxy[1]),
                    x2=int(xyxy[2]),
                    y2=int(xyxy[3]),
                    confidence=conf,
                    class_id=cls_id,
                    label="person",
                )
            )

        return detections
