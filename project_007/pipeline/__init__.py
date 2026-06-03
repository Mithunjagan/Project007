"""PROJECT 007 — Pipeline modules."""

from pipeline.models import FrameMeta, DetectionResult, PoseResult, MotionResult
from pipeline.events import RuleEvent, BehaviorScore, EventCandidate
from pipeline.detector import PersonDetector
from pipeline.pose import PoseExtractor
from pipeline.buffer import TrackBuffer
from pipeline.motion import MotionEngine
from pipeline.rules import ProxyRuleEngine
from pipeline.persistence import PersistenceFilter
from pipeline.scoring import BehaviorScoringEngine
from pipeline.recorder import ClipRecorder
from pipeline.opticalflow import OpticalFlowWorker
from pipeline.scene import SceneDynamicsEngine
from pipeline.tamper import CameraTamperEngine
from pipeline.intrusion import IntrusionEngine

__all__ = [
    "FrameMeta", "DetectionResult", "PoseResult", "MotionResult",
    "RuleEvent", "BehaviorScore", "EventCandidate",
    "PersonDetector", "PoseExtractor", "TrackBuffer", "MotionEngine",
    "ProxyRuleEngine", "PersistenceFilter", "BehaviorScoringEngine", "ClipRecorder",
    "OpticalFlowWorker", "SceneDynamicsEngine", "CameraTamperEngine", "IntrusionEngine"
]
