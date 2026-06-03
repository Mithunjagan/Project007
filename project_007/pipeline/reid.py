"""
PROJECT 007 — Soft ReID Tracker (P5.1)
Lightweight, real-time identity persistence engine that associates
new tracker IDs with recently lost target "ghosts" using color histograms
and scale-invariant bounding box dynamics.
"""

import time
import cv2
import numpy as np

from config import (
    ENABLE_SOFT_REID,
    REID_GHOST_TIMEOUT_S,
    REID_SIMILARITY_THRESHOLD,
    REID_COLOR_HIST_WEIGHT,
    REID_GEOMETRY_WEIGHT,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class SoftReIDTracker:
    def __init__(self):
        # persistent_id -> track_data dict
        self._active_tracks: dict[int, dict] = {}
        self._ghost_tracks: dict[int, dict] = {}
        
        # raw_id (from YOLO) -> persistent_id (our unified ID)
        self._raw_to_persistent: dict[int, int] = {}
        
        self._next_persistent_id = 1

    def _compute_hsv_histogram(self, crop: np.ndarray) -> np.ndarray:
        """Compute concatenated 1D HSV color histogram for a crop."""
        try:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            # H channel (hue): 16 bins, S channel (saturation): 8 bins
            hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256])
            
            # Normalize to sum = 1
            sum_h = np.sum(hist_h)
            sum_s = np.sum(hist_s)
            
            if sum_h > 0:
                hist_h /= sum_h
            if sum_s > 0:
                hist_s /= sum_s
                
            return np.concatenate([hist_h.ravel(), hist_s.ravel()]).astype(np.float32)
        except Exception as e:
            logger.warning(f"Failed to compute HSV histogram: {e}")
            return np.zeros(24, dtype=np.float32)

    def update(self, raw_detections: list, frame: np.ndarray, timestamp: float) -> list[int]:
        """
        Map a list of DetectionResults with raw tracker IDs to persistent IDs.
        
        Returns:
            list[int]: Persistent IDs matching the input detections in order.
        """
        if not ENABLE_SOFT_REID:
            return [det.track_id for det in raw_detections]

        # ── 1. Clean expired ghost tracks ──────────────────────────────────────
        expired_ghosts = [
            pid for pid, data in self._ghost_tracks.items()
            if (timestamp - data["last_seen_ts"]) > REID_GHOST_TIMEOUT_S
        ]
        for pid in expired_ghosts:
            del self._ghost_tracks[pid]

        current_persistent_ids = []
        new_active_tracks = {}

        # ── 2. Match current detections to active tracks or ghosts ─────────────
        for det in raw_detections:
            raw_id = det.track_id
            bbox = det.bbox
            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]
            
            # Safe crop
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            crop = frame[y1:y2, x1:x2]
            hist = self._compute_hsv_histogram(crop)
            
            centroid = (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)
            bbox_h = float(max(1, y2 - y1))
            bbox_w = float(max(1, x2 - x1))

            matched_pid = None

            # A. If raw_id was recently mapped to an active persistent ID, keep it
            if raw_id in self._raw_to_persistent:
                pid = self._raw_to_persistent[raw_id]
                # Double-check it was actually in active_tracks
                if pid in self._active_tracks:
                    matched_pid = pid

            # B. If not matched to active, search ghosts
            if matched_pid is None:
                best_ghost_pid = None
                best_sim = -1.0

                for g_pid, g_data in self._ghost_tracks.items():
                    # Calculate dt since last seen
                    dt = timestamp - g_data["last_seen_ts"]
                    if dt <= 0:
                        dt = 0.04

                    # 1. Color histogram similarity (correlation metric: 1.0 = perfect, -1.0 = opposite)
                    sim_color = float(cv2.compareHist(g_data["hist"], hist, cv2.HISTCMP_CORREL))
                    sim_color = max(0.0, sim_color) # Clamp to [0, 1]

                    # 2. Geometric similarity
                    # Project centroid using last known velocity
                    pred_cx = g_data["centroid"][0] + g_data["velocity"][0] * dt
                    pred_cy = g_data["centroid"][1] + g_data["velocity"][1] * dt
                    
                    dist = np.linalg.norm(np.array(centroid) - np.array([pred_cx, pred_cy]))
                    # Scale-invariant distance score: smaller boxes shrink the search radius
                    sim_dist = float(np.exp(-dist / (g_data["bbox_h"] * 1.8)))

                    # Aspect ratio and height similarity
                    sim_size = float(min(bbox_h, g_data["bbox_h"]) / max(bbox_h, g_data["bbox_h"]))
                    sim_geom = 0.70 * sim_dist + 0.30 * sim_size

                    # Weighted total similarity
                    sim_total = REID_COLOR_HIST_WEIGHT * sim_color + REID_GEOMETRY_WEIGHT * sim_geom

                    if sim_total > REID_SIMILARITY_THRESHOLD and sim_total > best_sim:
                        best_sim = sim_total
                        best_ghost_pid = g_pid

                if best_ghost_pid is not None:
                    # Identity Restored!
                    matched_pid = best_ghost_pid
                    logger.info(
                        f"Identity Restored: Raw ID {raw_id} matched to Ghost Persistent ID {matched_pid} "
                        f"(Similarity={best_sim:.3f})"
                    )
                    # Remove from ghosts
                    del self._ghost_tracks[matched_pid]

            # C. If still not matched, assign a new persistent ID
            if matched_pid is None:
                matched_pid = self._next_persistent_id
                self._next_persistent_id += 1
                logger.info(f"New Identity Registered: Raw ID {raw_id} -> Persistent ID {matched_pid}")

            # ── 3. Calculate velocity and update active tracks ──────────────────
            velocity = (0.0, 0.0)
            if matched_pid in self._active_tracks:
                prev_data = self._active_tracks[matched_pid]
                dt = timestamp - prev_data["timestamp"]
                if dt > 0:
                    vx = (centroid[0] - prev_data["centroid"][0]) / dt
                    vy = (centroid[1] - prev_data["centroid"][1]) / dt
                    velocity = (vx, vy)
            elif matched_pid in self._ghost_tracks:
                # If matched to a ghost, inherit its last velocity
                velocity = self._ghost_tracks[matched_pid]["velocity"]

            # Store current track data
            new_active_tracks[matched_pid] = {
                "raw_id": raw_id,
                "persistent_id": matched_pid,
                "bbox": bbox,
                "centroid": centroid,
                "bbox_h": bbox_h,
                "bbox_w": bbox_w,
                "velocity": velocity,
                "hist": hist,
                "timestamp": timestamp,
            }

            self._raw_to_persistent[raw_id] = matched_pid
            current_persistent_ids.append(matched_pid)

        # ── 4. Move missing active tracks to ghosts ───────────────────────────
        for pid, data in self._active_tracks.items():
            if pid not in new_active_tracks:
                # Move to ghost cache
                data["last_seen_ts"] = timestamp
                self._ghost_tracks[pid] = data

        self._active_tracks = new_active_tracks
        return current_persistent_ids
