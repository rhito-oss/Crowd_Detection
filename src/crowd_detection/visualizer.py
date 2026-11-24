from typing import List, Tuple
import cv2
import numpy as np
from crowd_detection.config import DisplayConfig
from crowd_detection.detector import Detection
from crowd_detection.density import DensityResult
from crowd_detection.stampede import StampedeRiskResult


def draw_hud(
    frame: np.ndarray,
    detections: List[Detection],
    density_result: DensityResult,
    fps: float,
    config: DisplayConfig,
    avg_speed: float = 0.0,
    stampede_result: StampedeRiskResult = None,
) -> np.ndarray:
    annotated = frame.copy()
    normal_box_color = tuple(config.box_color)
    crowd_box_color = tuple(config.crowd_box_color)
    line_color = tuple(config.line_color)
    text_color = tuple(config.text_color)
    badge_bg = tuple(config.badge_color)

    for i, j in density_result.close_pairs:
        if i < len(detections) and j < len(detections):
            c1 = (int((detections[i].x1 + detections[i].x2) / 2), int((detections[i].y1 + detections[i].y2) / 2))
            c2 = (int((detections[j].x1 + detections[j].x2) / 2), int((detections[j].y1 + detections[j].y2) / 2))
            cv2.line(annotated, c1, c2, color=line_color, thickness=2, lineType=cv2.LINE_AA)

    for idx, det in enumerate(detections):
        box_color = crowd_box_color if idx in density_result.crowded_indices else normal_box_color

        cv2.rectangle(
            annotated,
            (det.x1, det.y1),
            (det.x2, det.y2),
            color=box_color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        status_tag = " [CLOSE]" if idx in density_result.crowded_indices else ""
        speed_tag = f" {det.speed:.1f}km/h" if det.speed is not None else ""
        label = f"Person {det.confidence:.0%}{speed_tag}{status_tag}" if config.show_conf else f"Person{speed_tag}{status_tag}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        label_top = max(0, det.y1 - label_h - 6)
        cv2.rectangle(
            annotated,
            (det.x1, label_top),
            (det.x1 + label_w + 8, label_top + label_h + 6),
            color=box_color,
            thickness=-1,
        )

        cv2.putText(
            annotated,
            label,
            (det.x1 + 4, label_top + label_h + 2),
            fontFace=font,
            fontScale=font_scale,
            color=(0, 0, 0),
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

    hud_lines: List[Tuple[str, Tuple[int, int, int]]] = []

    if config.show_count:
        count_text = f"People Count: {len(detections)}"
        hud_lines.append((count_text, (0, 255, 128)))

    if config.show_density:
        crowd_text = f"crowded: {str(density_result.is_crowded).lower()}"
        crowd_color = (0, 0, 255) if density_result.is_crowded else (0, 255, 0)
        hud_lines.append((crowd_text, crowd_color))

    if config.show_fps:
        fps_text = f"FPS: {fps:.1f}"
        hud_lines.append((fps_text, text_color))

    avg_speed_text = f"Avg Speed: {avg_speed:.1f} km/h"
    hud_lines.append((avg_speed_text, (255, 200, 0)))

    if stampede_result is not None:
        prob_pct = stampede_result.probability * 100.0
        level = stampede_result.risk_level
        # Color coding: Green (LOW), Amber/Yellow (MODERATE), Red (HIGH/CRITICAL)
        if level == "LOW":
            risk_color = (0, 255, 0)
        elif level == "MODERATE":
            risk_color = (0, 255, 255)
        else:
            risk_color = (0, 0, 255)

        stampede_text = f"Stampede Risk: {prob_pct:.1f}% ({level})"
        hud_lines.append((stampede_text, risk_color))

    if hud_lines:
        margin = 15
        badge_w = 330
        line_height = 28
        badge_h = 15 + len(hud_lines) * line_height

        overlay = annotated.copy()
        cv2.rectangle(
            overlay,
            (margin, margin),
            (margin + badge_w, margin + badge_h),
            color=badge_bg,
            thickness=-1,
        )
        border_col = (0, 0, 255) if density_result.is_crowded else (100, 100, 100)
        cv2.rectangle(
            overlay,
            (margin, margin),
            (margin + badge_w, margin + badge_h),
            color=border_col,
            thickness=2 if density_result.is_crowded else 1,
            lineType=cv2.LINE_AA,
        )

        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

        for i, (line, col) in enumerate(hud_lines):
            y_pos = margin + 22 + (i * line_height)
            cv2.putText(
                annotated,
                line,
                (margin + 12, y_pos),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=col,
                thickness=2,
                lineType=cv2.LINE_AA,
            )

    return annotated
