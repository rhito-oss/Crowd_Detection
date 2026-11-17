from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import yaml

@dataclass
class CameraConfig:
    device_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    api_preference: str = "v4l2"

@dataclass
class DetectionConfig:
    model: str = "yolov8n.pt"
    conf_threshold: float = 0.45
    iou_threshold: float = 0.45
    device: str = "cpu"


@dataclass
class DensityConfig:
    enabled: bool = True
    min_people_for_crowd: int = 2
    closeness_threshold: float = 0.15
    min_close_pairs: int = 2


@dataclass
class DisplayConfig:
    window_name: str = "Crowd Detection - Person Tracking"
    show_fps: bool = True
    show_count: bool = True
    show_conf: bool = True
    show_density: bool = True
    box_color: List[int] = field(default_factory=lambda: [0, 255, 0])
    crowd_box_color: List[int] = field(default_factory=lambda: [0, 0, 255])
    line_color: List[int] = field(default_factory=lambda: [0, 0, 255])
    text_color: List[int] = field(default_factory=lambda: [255, 255, 255])
    badge_color: List[int] = field(default_factory=lambda: [40, 40, 40])


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    density: DensityConfig = field(default_factory=DensityConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)


def load_config(config_path: str = "config.yaml") -> AppConfig:
    path = Path(config_path)
    if not path.is_file():
        return AppConfig()

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    cam_data = data.get("camera", {})
    det_data = data.get("detection", {})
    density_data = data.get("density", {})
    disp_data = data.get("display", {})

    return AppConfig(
        camera=CameraConfig(**cam_data),
        detection=DetectionConfig(**det_data),
        density=DensityConfig(**density_data),
        display=DisplayConfig(**disp_data),
    )
