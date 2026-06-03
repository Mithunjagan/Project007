import json
from pathlib import Path
from dataclasses import dataclass
from typing import List

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class GroundTruthEvent:
    start_frame: int
    end_frame: int
    label: str


class AnnotationLoader:
    """
    Loads manual ground-truth labels for dataset evaluation.
    """
    def __init__(self, annotations_dir: str = "dataset/annotations"):
        self.annotations_dir = Path(annotations_dir)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)

    def get_events_for_video(self, video_filename: str) -> List[GroundTruthEvent]:
        """
        Look up the ground-truth events for a given video.
        Uses the stem of the video (e.g. sample_01) to find sample_01.json.
        """
        stem = Path(video_filename).stem
        json_path = self.annotations_dir / f"{stem}.json"

        if not json_path.exists():
            # If no annotation exists, we assume it's entirely NORMAL
            return []

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            events = []
            for ev in data.get("events", []):
                events.append(GroundTruthEvent(
                    start_frame=ev["start_frame"],
                    end_frame=ev["end_frame"],
                    label=ev["label"]
                ))
            return events
        except Exception as e:
            logger.error(f"Failed to load annotations from {json_path}: {e}")
            return []

    def save_events_for_video(self, video_filename: str, events: List[GroundTruthEvent]):
        """
        Save ground-truth events to a JSON file.
        """
        stem = Path(video_filename).stem
        json_path = self.annotations_dir / f"{stem}.json"
        
        data = {
            "video": video_filename,
            "events": [
                {
                    "start_frame": e.start_frame,
                    "end_frame": e.end_frame,
                    "label": e.label
                } for e in events
            ]
        }
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
