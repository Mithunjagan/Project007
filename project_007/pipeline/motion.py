"""
PROJECT 007 — Motion Feature Engine
Computes normalized arm velocity, body displacement, fall score, approach
velocity, and motion vectors from rolling keypoint history.
"""

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


class MotionEngine:
    """
    Stateless feature calculator for P1 proxy rules.

    Outputs are normalized by bounding box height or time to ensure
    camera-distance invariance.
    """

    def compute(self, track_id: int, buffer) -> dict:
        """
        Compute per-person motion features for *track_id*.
        """
        history = buffer.get_history(track_id)

        if len(history) < 2:
            return {
                "arm_velocity": 0.0,
                "arm_motion_vector": (0.0, 0.0),
                "body_displacement": 0.0,
                "fall_score": 0.0,
                "fall_score_delta": 0.0,
                "uncertainty": 1.0,
            }

        current = history[-1]
        previous = history[-2]

        dt = current["timestamp"] - previous["timestamp"]
        if dt <= 0:
            dt = 0.04  # Fallback to ~25 FPS

        bbox = current["bbox"]
        bbox_h = max(1.0, bbox[3] - bbox[1])

        arm_vel, arm_vec = self._arm_motion(current["keypoints"], previous["keypoints"], dt, bbox_h)
        body_disp = self._body_displacement(current["center"], previous["center"], dt, bbox_h)
        
        curr_fall_score = self._fall_score(current["keypoints"])
        prev_fall_score = self._fall_score(previous["keypoints"])
        fall_score_delta = curr_fall_score - prev_fall_score

        uncertainty = self._estimate_uncertainty(current["keypoints"])

        return {
            "arm_velocity": arm_vel,
            "arm_motion_vector": arm_vec,
            "body_displacement": body_disp,
            "fall_score": curr_fall_score,
            "fall_score_delta": fall_score_delta,
            "uncertainty": uncertainty,
        }

    def compute_pairwise(self, buffer) -> dict:
        """
        Compute normalized distance and approach velocity for all pairs.

        Returns
        -------
        dict
            ``{(id_a, id_b): {"distance": float, "approach_velocity": float, "target_vec_a2b": tuple, "target_vec_b2a": tuple}}``
        """
        active_ids = sorted(buffer.get_active_ids())
        pairwise: dict = {}

        for i in range(len(active_ids)):
            for j in range(i + 1, len(active_ids)):
                id_a, id_b = active_ids[i], active_ids[j]
                hist_a = buffer.get_history(id_a)
                hist_b = buffer.get_history(id_b)

                if len(hist_a) < 2 or len(hist_b) < 2:
                    pairwise[(id_a, id_b)] = {
                        "distance": 999.0,
                        "approach_velocity": 0.0,
                        "target_vec_a2b": (0.0, 0.0),
                        "target_vec_b2a": (0.0, 0.0),
                    }
                    continue

                curr_a = hist_a[-1]
                curr_b = hist_b[-1]
                prev_a = hist_a[-2]
                prev_b = hist_b[-2]

                dt = curr_a["timestamp"] - prev_a["timestamp"]
                if dt <= 0:
                    dt = 0.04

                bbox_h_a = max(1.0, curr_a["bbox"][3] - curr_a["bbox"][1])
                bbox_h_b = max(1.0, curr_b["bbox"][3] - curr_b["bbox"][1])
                avg_bbox_h = (bbox_h_a + bbox_h_b) / 2.0

                ca_now = np.array(curr_a["center"])
                cb_now = np.array(curr_b["center"])
                ca_prev = np.array(prev_a["center"])
                cb_prev = np.array(prev_b["center"])

                dist_now_px = float(np.linalg.norm(ca_now - cb_now))
                dist_prev_px = float(np.linalg.norm(ca_prev - cb_prev))

                # Normalized distance (in units of average bbox height)
                norm_distance = dist_now_px / avg_bbox_h

                # Positive velocity means approaching
                approach_vel_px = (dist_prev_px - dist_now_px) / dt
                norm_approach_vel = approach_vel_px / avg_bbox_h

                # Target direction vectors
                vec_a2b = cb_now - ca_now
                vec_b2a = ca_now - cb_now

                pairwise[(id_a, id_b)] = {
                    "distance": norm_distance,
                    "approach_velocity": norm_approach_vel,
                    "target_vec_a2b": (float(vec_a2b[0]), float(vec_a2b[1])),
                    "target_vec_b2a": (float(vec_b2a[0]), float(vec_b2a[1])),
                }

        return pairwise

    # ── internal helpers ──────────────────────────────

    @staticmethod
    def _arm_motion(current_kps: dict, previous_kps: dict, dt: float, bbox_h: float) -> tuple[float, tuple[float, float]]:
        """
        Calculates normalized arm velocity and the average 2D motion vector.
        """
        arm_keys = ["left_elbow", "right_elbow", "left_wrist", "right_wrist"]
        dx_list = []
        dy_list = []

        for key in arm_keys:
            curr = current_kps.get(key)
            prev = previous_kps.get(key)

            if curr and prev and curr.get("visibility", 0) > 0.3 and prev.get("visibility", 0) > 0.3:
                dx_list.append(curr["x"] - prev["x"])
                dy_list.append(curr["y"] - prev["y"])

        if not dx_list:
            return 0.0, (0.0, 0.0)

        mean_dx = float(np.mean(dx_list))
        mean_dy = float(np.mean(dy_list))

        speed_px_per_sec = np.sqrt(mean_dx**2 + mean_dy**2) / dt
        normalized_speed = float(speed_px_per_sec / bbox_h)

        return normalized_speed, (mean_dx, mean_dy)

    @staticmethod
    def _body_displacement(current_center: tuple, previous_center: tuple, dt: float, bbox_h: float) -> float:
        """Normalized body displacement speed (heights / sec)."""
        if not current_center or not previous_center:
            return 0.0

        dx = current_center[0] - previous_center[0]
        dy = current_center[1] - previous_center[1]
        speed_px_per_sec = np.sqrt(dx ** 2 + dy ** 2) / dt
        return float(speed_px_per_sec / bbox_h)

    @staticmethod
    def _fall_score(keypoints: dict) -> float:
        """Estimate fall likelihood (0.0 to 1.0)."""
        left_hip = keypoints.get("left_hip")
        right_hip = keypoints.get("right_hip")
        left_ankle = keypoints.get("left_ankle")
        right_ankle = keypoints.get("right_ankle")
        left_shoulder = keypoints.get("left_shoulder")
        right_shoulder = keypoints.get("right_shoulder")

        required = [left_hip, right_hip, left_ankle, right_ankle]
        if not all(required) or any(kp.get("visibility", 0) < 0.3 for kp in required):
            return 0.0

        hip_y = (left_hip["y"] + right_hip["y"]) / 2.0
        ankle_y = (left_ankle["y"] + right_ankle["y"]) / 2.0
        vertical_diff = ankle_y - hip_y

        if vertical_diff <= 0:
            return 1.0

        if left_shoulder and right_shoulder:
            shoulder_y = (left_shoulder["y"] + right_shoulder["y"]) / 2.0
            torso_len = abs(hip_y - shoulder_y)
            if torso_len > 0:
                ratio = vertical_diff / torso_len
                return float(max(0.0, min(1.0, 1.0 - ratio / 2.0)))

        return float(max(0.0, min(1.0, 1.0 - vertical_diff / 200.0)))

    @staticmethod
    def _estimate_uncertainty(keypoints: dict) -> float:
        """Estimate uncertainty based on keypoint visibility [0.0 - 1.0]."""
        vis_scores = [kp.get("visibility", 0) for kp in keypoints.values() if isinstance(kp, dict)]
        if not vis_scores:
            return 1.0
        
        avg_vis = sum(vis_scores) / len(vis_scores)
        # high visibility -> low uncertainty
        return max(0.0, min(1.0, 1.0 - avg_vis))
