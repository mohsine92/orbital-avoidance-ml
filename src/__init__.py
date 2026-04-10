

# Supervised learning system for orbital avoidance maneuvers
from .orbital_mechanics import (
    OrbitalState,
    create_circular_orbit,
    create_elliptical_orbit,
    propagate_orbit,
    compute_orbital_period,
    apply_maneuver,
    MU_EARTH,
    R_EARTH
)

from .sensor_model import (
    OpticalSensor,
    SensorParameters
)

from .collision_risk import (
    CollisionRisk,
    R_COMBINED
)

from .optimizer import (
    ManeuverOptimizer,
    ManeuverOptimizationResult
)

from .dataset_generator import (
    DatasetGenerator
)

from .ml_model import (
    ManeuverPredictor,
    compare_models,
    analyze_predictions
)

__version__ = "1.0.0"
__author__ = "Orbital Avoidance ML Team"

__all__ = [
    # Orbital mechanics
    'OrbitalState',
    'create_circular_orbit',
    'create_elliptical_orbit',
    'propagate_orbit',
    'compute_orbital_period',
    'apply_maneuver',
    'MU_EARTH',
    'R_EARTH',
    
    # Sensor
    'OpticalSensor',
    'SensorParameters',
    
    # Collision risk
    'CollisionRisk',
    'R_COMBINED',
    
    # Optimizer
    'ManeuverOptimizer',
    'ManeuverOptimizationResult',
    
    # Dataset
    'DatasetGenerator',
    
    # ML
    'ManeuverPredictor',
    'compare_models',
    'analyze_predictions',
]
