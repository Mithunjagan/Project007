"""
PROJECT 007 — P3.0 Session Manager
Manages recording sessions and their metadata.
"""

import json
from pathlib import Path
from typing import List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

SCENARIO_CATEGORIES = [
    "normal",
    "camera_tamper",
    "intrusion",
    "interaction",
    "crowded",
    "occlusion",
    "lighting_change",
]


class SessionManager:
    """Manages dataset sessions: list, filter, and retrieve metadata."""

    def __init__(self, dataset_dir: str = "dataset"):
        self.dataset_dir = Path(dataset_dir)

    def list_sessions(self, category: Optional[str] = None) -> List[dict]:
        """List all sessions, optionally filtered by category."""
        sessions = []
        categories = [category] if category else SCENARIO_CATEGORIES

        for cat in categories:
            cat_dir = self.dataset_dir / cat
            if not cat_dir.exists():
                continue
            for meta_file in sorted(cat_dir.glob("session_*_meta.json")):
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    # Verify video exists
                    video_path = cat_dir / f"{meta['session_id']}.mp4"
                    if video_path.exists():
                        meta["video_path"] = str(video_path)
                        sessions.append(meta)
                except Exception as e:
                    logger.warning(f"Error reading {meta_file}: {e}")

        return sessions

    def get_session(self, session_id: str) -> Optional[dict]:
        """Look up a single session by its ID."""
        for cat in SCENARIO_CATEGORIES:
            meta_path = self.dataset_dir / cat / f"{session_id}_meta.json"
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                video_path = self.dataset_dir / cat / f"{session_id}.mp4"
                if video_path.exists():
                    meta["video_path"] = str(video_path)
                return meta
        return None

    def get_total_duration(self, category: Optional[str] = None) -> float:
        """Sum of all session durations in seconds."""
        sessions = self.list_sessions(category)
        return sum(s.get("duration", 0) for s in sessions)

    def get_category_counts(self) -> dict:
        """Count sessions per category."""
        counts = {}
        for cat in SCENARIO_CATEGORIES:
            sessions = self.list_sessions(cat)
            counts[cat] = len(sessions)
        return counts
