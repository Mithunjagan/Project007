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

    def compute(self, track_id: int, buffer, egomotion_affine=None, pose_composed_affine=None) -> dict:
        """
        Compute per-person motion features for *track_id*.

        Parameters
        ----------
        egomotion_affine : np.ndarray | None
            2×3 affine matrix mapping previous-frame coords to current-frame coords (1-frame).
        pose_composed_affine : np.ndarray | None
            2×3 affine matrix composed over the pose interval.
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

        # Check if pose was updated on this frame
        curr_pose_fid = current.get("pose_frame_id")
        prev_pose_fid = previous.get("pose_frame_id")

        if curr_pose_fid is not None and prev_pose_fid is not None and curr_pose_fid == prev_pose_fid:
            # No new pose on this frame, velocity is 0
            arm_vel = 0.0
            arm_vec = (0.0, 0.0)
        else:
            # Pose updated. Compute arm velocity.
            # Compensate keypoints using pose_composed_affine (which covers the pose interval)
            arm_vel, arm_vec = self._arm_motion(current["keypoints"], previous["keypoints"], dt, bbox_h, pose_composed_affine)

        body_disp = self._body_displacement(current["center"], previous["center"], dt, bbox_h, egomotion_affine)
        
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

    def compute_pairwise(self, buffer, egomotion_affine=None) -> dict:
        """
        Compute normalized distance and approach velocity for all pairs.

        Parameters
        ----------
        egomotion_affine : np.ndarray | None
            2×3 affine matrix mapping previous-frame coords to current-frame coords.

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

                # Compensate previous centroids with egomotion if available
                if egomotion_affine is not None:
                    ca_prev = self._apply_affine(egomotion_affine, ca_prev)
                    cb_prev = self._apply_affine(egomotion_affine, cb_prev)

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
    def _apply_affine(affine_mat, point) -> np.ndarray:
        """Apply 2×3 affine transform to a 2D point."""
        px, py = float(point[0]), float(point[1])
        ex = affine_mat[0, 0] * px + affine_mat[0, 1] * py + affine_mat[0, 2]
        ey = affine_mat[1, 0] * px + affine_mat[1, 1] * py + affine_mat[1, 2]
        return np.array([ex, ey])

    @staticmethod
    def _arm_motion(current_kps: dict, previous_kps: dict, dt: float, bbox_h: float, egomotion_affine=None) -> tuple[float, tuple[float, float]]:
        """
        Calculates normalized arm velocity and the average 2D motion vector.
        If egomotion_affine is provided, previous keypoint positions are
        projected through the affine transform before computing deltas.
        """
        arm_keys = ["left_elbow", "right_elbow", "left_wrist", "right_wrist"]
        dx_list = []
        dy_list = []

        for key in arm_keys:
            curr = current_kps.get(key)
            prev = previous_kps.get(key)

            if curr and prev and curr.get("visibility", 0) > 0.3 and prev.get("visibility", 0) > 0.3:
                px, py = prev["x"], prev["y"]
                if egomotion_affine is not None:
                    # Project previous position to current frame's coordinate system
                    ex = egomotion_affine[0, 0] * px + egomotion_affine[0, 1] * py + egomotion_affine[0, 2]
                    ey = egomotion_affine[1, 0] * px + egomotion_affine[1, 1] * py + egomotion_affine[1, 2]
                else:
                    ex, ey = px, py
                dx_list.append(curr["x"] - ex)
                dy_list.append(curr["y"] - ey)

        if not dx_list:
            return 0.0, (0.0, 0.0)

        mean_dx = float(np.mean(dx_list))
        mean_dy = float(np.mean(dy_list))

        speed_px_per_sec = np.sqrt(mean_dx**2 + mean_dy**2) / dt
        normalized_speed = float(speed_px_per_sec / bbox_h)

        return normalized_speed, (mean_dx, mean_dy)

    @staticmethod
    def _body_displacement(current_center: tuple, previous_center: tuple, dt: float, bbox_h: float, egomotion_affine=None) -> float:
        """Normalized body displacement speed (heights / sec)."""
        if not current_center or not previous_center:
            return 0.0

        px, py = previous_center[0], previous_center[1]
        if egomotion_affine is not None:
            ex = egomotion_affine[0, 0] * px + egomotion_affine[0, 1] * py + egomotion_affine[0, 2]
            ey = egomotion_affine[1, 0] * px + egomotion_affine[1, 1] * py + egomotion_affine[1, 2]
        else:
            ex, ey = px, py

        dx = current_center[0] - ex
        dy = current_center[1] - ey
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
