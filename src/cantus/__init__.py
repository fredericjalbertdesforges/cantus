"""
cantus — real-time score-following for voice and continuous-pitch instruments.
"""
from cantus.algorithms.oltw_dixon import OLTWResult, oltw_align
from cantus.features.base import FeatureExtractor, FeatureSpec
from cantus.features.piano_roll import PianoRollExtractor
from cantus.features.pitch_trajectory import (
    PitchTrajectoryExtractor,
    f0_to_one_hot,
    hz_to_midi,
)
from cantus.metrics import (
    alignment_error_frames,
    alignment_rate,
    summarise,
)

__version__ = "0.0.1"

__all__ = [
    "OLTWResult",
    "oltw_align",
    "FeatureExtractor",
    "FeatureSpec",
    "PianoRollExtractor",
    "PitchTrajectoryExtractor",
    "f0_to_one_hot",
    "hz_to_midi",
    "alignment_error_frames",
    "alignment_rate",
    "summarise",
    "__version__",
]
