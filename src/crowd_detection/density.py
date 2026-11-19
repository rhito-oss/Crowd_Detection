import math
from dataclasses import dataclass
from typing import List, Set, Tuple
from crowd_detection.config import DensityConfig
from crowd_detection.detector import Detection


@dataclass
class DensityResult:
    is_crowded: bool
    close_pairs: List[Tuple[int, int]] 
    crowded_indices: Set[int]          
    density_score: float              

def evaluate_density(
    detections: List[Detection],
    frame_width: int,
    frame_height: int,
    config: DensityConfig,
) -> DensityResult:
    n = len(detections)
    if not config.enabled or n < 2:
        return DensityResult(
            is_crowded=False,
            close_pairs=[],
            crowded_indices=set(),
            density_score=0.0,
        )

    frame_diag = math.hypot(frame_width, frame_height)
    if frame_diag == 0:
        frame_diag = 1.0

    centers = [
        ((d.x1 + d.x2) / 2.0, (d.y1 + d.y2) / 2.0)
        for d in detections
    ]

    close_pairs: List[Tuple[int, int]] = []
    crowded_indices: Set[int] = set()

    for i in range(n):
        d1 = detections[i]
        w1 = max(1, d1.x2 - d1.x1)
        h1 = max(1, d1.y2 - d1.y1)

        for j in range(i + 1, n):
            d2 = detections[j]
            w2 = max(1, d2.x2 - d2.x1)
            h2 = max(1, d2.y2 - d2.y1)

            overlap_x = max(0, min(d1.x2, d2.x2) - max(d1.x1, d2.x1))
            overlap_y = max(0, min(d1.y2, d2.y2) - max(d1.y1, d2.y1))
            boxes_overlap = (overlap_x > 0) and (overlap_y > 0)

            dx_gap = max(0, max(d1.x1, d2.x1) - min(d1.x2, d2.x2))
            dy_gap = max(0, max(d1.y1, d2.y1) - min(d1.y2, d2.y2))

            avg_width = (w1 + w2) / 2.0
            avg_height = (h1 + h2) / 2.0

            dist = math.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1])
            norm_dist = dist / frame_diag

            touching_or_near = (
                boxes_overlap
                or (dx_gap <= 0.60 * avg_width and dy_gap <= 0.70 * avg_height)
                or (norm_dist <= max(config.closeness_threshold, 0.25))
            )

            if touching_or_near:
                close_pairs.append((i, j))
                crowded_indices.add(i)
                crowded_indices.add(j)

    is_crowded = len(close_pairs) > 0 or len(crowded_indices) >= 2
    density_score = len(close_pairs) / (n * (n - 1) / 2.0) if n > 1 else 0.0

    return DensityResult(
        is_crowded=is_crowded,
        close_pairs=close_pairs,
        crowded_indices=crowded_indices,
        density_score=density_score,
    )
