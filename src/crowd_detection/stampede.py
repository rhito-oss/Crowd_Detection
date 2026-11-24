"""
Stampede Risk & Probability Estimation Module

Accurate kinetic-gated formulation:
1. Crowd Volume Gate: Requires a critical minimum crowd (default >= 3). Pairs (2 people) evaluate to 0.0%.
2. Spatial Density & Clustering: Evaluates close pairs and proportion of people packed close together.
3. Kinetic Gating (Crucial): Normal walking or stationary posture (<= 4 km/h) forces probability to ~0.0%.
4. Multiplicative Model:
   P = Volume_Factor * Spatial_Factor * Kinetic_Factor
   If kinetic movement or crowd volume is zero, total stampede risk is zero.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from crowd_detection.config import StampedeConfig
from crowd_detection.detector import Detection
from crowd_detection.density import DensityResult


@dataclass
class StampedeRiskResult:
    probability: float               # Value in [0.0, 1.0] (0% to 100%)
    risk_level: str                  # "LOW", "MODERATE", "HIGH", "CRITICAL"
    factors: dict = field(default_factory=dict)


def calculate_stampede_probability(
    detections: List[Detection],
    density_result: DensityResult,
    avg_speed_kmh: float,
    config: Optional[StampedeConfig] = None,
) -> StampedeRiskResult:
    """
    Computes a kinetic-gated stampede risk probability score.
    """
    if config is None:
        config = StampedeConfig()

    n = len(detections)

    # Gate 1: If stampede detection disabled or crowd size is below minimum threshold
    if not config.enabled or n < config.min_people:
        return StampedeRiskResult(
            probability=0.0,
            risk_level="LOW",
            factors={
                "volume_factor": 0.0,
                "spatial_factor": 0.0,
                "kinetic_factor": 0.0,
                "speed_factor": 0.0,
                "variance_factor": 0.0,
            },
        )

    # 1. Crowd Volume Factor (Normalized between min_people and critical_people)
    if config.critical_people > config.min_people:
        vol_range = config.critical_people - config.min_people
        volume_factor = min(1.0, max(0.0, (n - config.min_people + 1) / (vol_range + 1)))
    else:
        volume_factor = 1.0

    # 2. Spatial Density & Clustering Factor
    density_score = min(1.0, max(0.0, density_result.density_score))
    clustering_ratio = len(density_result.crowded_indices) / float(n) if n > 0 else 0.0
    spatial_factor = min(1.0, 0.5 * density_score + 0.5 * clustering_ratio)

    # 3. Kinetic & Turbulence Factor
    # Subtract walking baseline: speeds below walking threshold produce 0 speed factor
    if config.panic_speed_kmh > config.walking_speed_kmh:
        speed_delta = max(0.0, avg_speed_kmh - config.walking_speed_kmh)
        speed_range = config.panic_speed_kmh - config.walking_speed_kmh
        speed_factor = min(1.0, speed_delta / speed_range)
    else:
        speed_factor = 1.0 if avg_speed_kmh >= config.panic_speed_kmh else 0.0

    # Speed Variance across individuals (turbulent, chaotic panic motion)
    speeds = [d.speed for d in detections if d.speed > 0]
    if len(speeds) >= 2 and config.panic_variance_kmh > 0:
        speed_std = float(np.std(speeds))
        variance_factor = min(1.0, speed_std / config.panic_variance_kmh)
    else:
        variance_factor = 0.0

    # Kinetic factor triggers on high uniform surge OR high chaotic turbulence
    kinetic_factor = max(speed_factor, 0.7 * speed_factor + 0.3 * variance_factor)

    # 4. Multiplicative Risk Formulation
    # If stationary or normal walking (kinetic_factor == 0), probability is 0.0
    base_probability = volume_factor * spatial_factor * kinetic_factor

    # Non-linear surge boost when severe clustering meets fast movement
    if spatial_factor > 0.5 and kinetic_factor > 0.4:
        boost = (spatial_factor - 0.5) * (kinetic_factor - 0.4) * 0.8
        combined_probability = min(1.0, base_probability + boost)
    else:
        combined_probability = base_probability

    # Categorize Risk Level
    if combined_probability < 0.25:
        risk_level = "LOW"
    elif combined_probability < 0.55:
        risk_level = "MODERATE"
    elif combined_probability < 0.80:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"

    return StampedeRiskResult(
        probability=round(combined_probability, 3),
        risk_level=risk_level,
        factors={
            "volume_factor": round(volume_factor, 3),
            "spatial_factor": round(spatial_factor, 3),
            "kinetic_factor": round(kinetic_factor, 3),
            "speed_factor": round(speed_factor, 3),
            "variance_factor": round(variance_factor, 3),
        },
    )
