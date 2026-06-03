"""
PROJECT 007 — Egomotion Estimation Layer (P6.0-A)
Estimates global camera motion (translation, rotation, scale) between frames
using Lucas-Kanade optical flow and RANSAC-based 2D affine transform.
Masks out person detection bounding boxes to ignore foreground movement.

Key improvements over naive implementation:
1. EMA temporal smoothing on affine parameters (tx, ty, rotation) to suppress
   frame-to-frame estimation noise.
2. Noise-floor gating: only applies compensation when estimated motion exceeds
   a minimum threshold (prevents adding noise on near-static cameras).
3. Confidence-weighted output: the returned affine matrix blends toward identity
   proportionally to stability_score, preventing noisy frames from corrupting
   downstream velocity calculations.
"""

import cv2
import numpy as np
from utils.logger import get_logger
from config import (
    EGOMOTION_MAX_FEATURES,
    EGOMOTION_MIN_FEATURES,
    EGOMOTION_RANSAC_THRESHOLD,
    EGOMOTION_STABILITY_PENALTY_CAP,
)

logger = get_logger(__name__)


class EgomotionEstimator:
    """
    Computes frame-to-frame camera motion (ego-motion) for drone deployment.

    Returns a smoothed, confidence-weighted affine matrix that maps previous-frame
    coordinates to current-frame coordinates. When no significant camera motion is
    detected, returns an identity transform — ensuring zero impact on static cameras.
    """

    def __init__(self, ema_alpha: float = 0.4, noise_floor_px: float = 0.5):
        """
        Parameters
        ----------
        ema_alpha : float
            Smoothing factor for EMA on affine parameters.
            Higher = more responsive, lower = smoother.
        noise_floor_px : float
            Minimum translation magnitude (pixels) below which motion is considered
            noise and the affine is clamped to identity.
        """
        self._prev_frame = None
        self._prev_features = None  # numpy array of shape (N, 1, 2)

        # Lucas-Kanade optical flow parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        # Good Features To Track parameters for background points selection
        self.feature_params = dict(
            maxCorners=EGOMOTION_MAX_FEATURES,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7
        )

        self.min_features_to_track = EGOMOTION_MIN_FEATURES
        self.ransac_threshold = EGOMOTION_RANSAC_THRESHOLD
        self.stability_penalty_cap = EGOMOTION_STABILITY_PENALTY_CAP

        # Temporal smoothing state
        self._ema_alpha = ema_alpha
        self._noise_floor = noise_floor_px
        self._smooth_tx = 0.0
        self._smooth_ty = 0.0
        self._smooth_da = 0.0  # rotation (radians)
        self._smooth_stability = 1.0
        self._initialized = False

        # The actual 2x3 affine matrix (smoothed, confidence-weighted)
        self._affine_matrix = np.eye(2, 3, dtype=np.float32)

    def _create_foreground_mask(self, frame_shape: tuple[int, int], person_bboxes: list) -> np.ndarray:
        """
        Creates a binary mask where person bounding boxes are set to 0 (ignored)
        and static background areas are set to 255.
        Also excludes a small border boundary to prevent feature loss near edges.
        """
        mask = np.ones(frame_shape, dtype=np.uint8) * 255

        h, w = frame_shape
        border = 15
        mask[0:border, :] = 0
        mask[h - border:h, :] = 0
        mask[:, 0:border] = 0
        mask[:, w - border:w] = 0

        # Exclude person boxes with padding
        pad = 10
        for bbox in person_bboxes:
            x1, y1, x2, y2 = bbox
            x1_idx = max(0, int(x1) - pad)
            y1_idx = max(0, int(y1) - pad)
            x2_idx = min(w, int(x2) + pad)
            y2_idx = min(h, int(y2) + pad)
            mask[y1_idx:y2_idx, x1_idx:x2_idx] = 0

        return mask

    def update(self, frame: np.ndarray, person_bboxes: list) -> dict:
        """
        Processes the current frame and estimates camera motion relative to the previous frame.

        Parameters
        ----------
        frame : np.ndarray
            Color BGR frame.
        person_bboxes : list
            List of bounding boxes [x1, y1, x2, y2] to mask out from feature tracking.

        Returns
        -------
        dict
            {
                "translation_x": float,  # Smoothed horizontal camera translation (pixels)
                "translation_y": float,  # Smoothed vertical camera translation (pixels)
                "rotation": float,       # Smoothed camera rotation (radians)
                "stability_score": float, # 0.0 (unstable/noisy) to 1.0 (stable estimate)
                "affine_matrix": np.ndarray,  # 2x3 smoothed, confidence-weighted affine
                "is_compensating": bool,  # True if motion exceeds noise floor
            }
        """
        identity = np.eye(2, 3, dtype=np.float32)
        motion_result = {
            "translation_x": 0.0,
            "translation_y": 0.0,
            "rotation": 0.0,
            "stability_score": 1.0,
            "affine_matrix": identity.copy(),
            "is_compensating": False,
        }

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. First frame initialization
        if self._prev_frame is None:
            self._prev_frame = gray
            mask = self._create_foreground_mask(gray.shape, person_bboxes)
            self._prev_features = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
            if self._prev_features is None:
                self._prev_features = np.empty((0, 1, 2), dtype=np.float32)
            return motion_result

        # Raw estimates for this frame
        raw_tx, raw_ty, raw_da = 0.0, 0.0, 0.0
        raw_stability = 1.0

        # 2. Track existing background features via Lucas-Kanade
        if self._prev_features is not None and len(self._prev_features) >= 4:
            p2, st, err = cv2.calcOpticalFlowPyrLK(
                self._prev_frame, gray, self._prev_features, None, **self.lk_params
            )

            if p2 is not None and len(p2) > 0:
                good_prev = self._prev_features[st == 1]
                good_curr = p2[st == 1]
            else:
                good_prev = np.empty((0, 2), dtype=np.float32)
                good_curr = np.empty((0, 2), dtype=np.float32)

            if len(good_prev) >= 4:
                # Estimate 2D similarity transform (translation + rotation + uniform scale)
                affine_mat, inliers = cv2.estimateAffinePartial2D(
                    good_prev, good_curr, method=cv2.RANSAC,
                    ransacReprojThreshold=self.ransac_threshold
                )

                if affine_mat is not None:
                    raw_tx = float(affine_mat[0, 2])
                    raw_ty = float(affine_mat[1, 2])
                    raw_da = float(np.arctan2(affine_mat[1, 0], affine_mat[0, 0]))

                    # Stability score from RANSAC inlier ratio + residual variance
                    num_inliers = int(np.sum(inliers))
                    inlier_ratio = num_inliers / len(inliers) if len(inliers) > 0 else 1.0

                    if num_inliers >= 4:
                        inlier_mask = inliers.ravel() == 1
                        prev_inliers = good_prev[inlier_mask]
                        curr_inliers = good_curr[inlier_mask]
                        expected = (affine_mat[:, :2] @ prev_inliers.T).T + affine_mat[:, 2]
                        residuals = np.linalg.norm(curr_inliers - expected, axis=1)
                        res_std = float(np.std(residuals)) if len(residuals) > 0 else 0.0

                        vibration_penalty = min(self.stability_penalty_cap, res_std / 10.0)
                        raw_stability = max(0.0, min(1.0, inlier_ratio - vibration_penalty))
                    else:
                        raw_stability = inlier_ratio

                    # Keep only inlier features for next frame tracking
                    self._prev_features = good_curr[inliers.ravel() == 1].reshape(-1, 1, 2)
                else:
                    self._prev_features = good_curr.reshape(-1, 1, 2)
            else:
                self._prev_features = np.empty((0, 1, 2), dtype=np.float32)

        # 3. Replenish features if below threshold
        if self._prev_features is None or len(self._prev_features) < self.min_features_to_track:
            mask = self._create_foreground_mask(gray.shape, person_bboxes)
            new_features = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)

            if new_features is not None:
                if self._prev_features is not None and len(self._prev_features) > 0:
                    existing_pts = self._prev_features.reshape(-1, 2)
                    filtered_new = []
                    for pt in new_features.reshape(-1, 2):
                        dists = np.linalg.norm(existing_pts - pt, axis=1)
                        if np.min(dists) > 10.0:
                            filtered_new.append(pt)
                    if filtered_new:
                        arr = np.array(filtered_new, dtype=np.float32).reshape(-1, 1, 2)
                        self._prev_features = np.vstack([self._prev_features, arr])
                else:
                    self._prev_features = new_features

        self._prev_frame = gray

        # ── 4. Temporal EMA Smoothing ──
        alpha = self._ema_alpha
        if not self._initialized:
            self._smooth_tx = raw_tx
            self._smooth_ty = raw_ty
            self._smooth_da = raw_da
            self._smooth_stability = raw_stability
            self._initialized = True
        else:
            self._smooth_tx = alpha * raw_tx + (1.0 - alpha) * self._smooth_tx
            self._smooth_ty = alpha * raw_ty + (1.0 - alpha) * self._smooth_ty
            self._smooth_da = alpha * raw_da + (1.0 - alpha) * self._smooth_da
            self._smooth_stability = alpha * raw_stability + (1.0 - alpha) * self._smooth_stability

        # ── 5. Noise-Floor Gating ──
        motion_magnitude = np.sqrt(self._smooth_tx ** 2 + self._smooth_ty ** 2)
        is_compensating = motion_magnitude > self._noise_floor or abs(self._smooth_da) > 0.001

        if is_compensating:
            # Build the smoothed affine matrix
            cos_a = np.cos(self._smooth_da)
            sin_a = np.sin(self._smooth_da)
            smoothed_affine = np.array([
                [cos_a, -sin_a, self._smooth_tx],
                [sin_a, cos_a, self._smooth_ty]
            ], dtype=np.float32)

            # ── 6. Confidence-Weighted Blend Toward Identity ──
            # When stability is low (noisy estimate), blend toward identity to prevent
            # injecting noise into velocity calculations.
            conf = max(0.0, min(1.0, self._smooth_stability))
            blended_affine = conf * smoothed_affine + (1.0 - conf) * identity
            self._affine_matrix = blended_affine
        else:
            # Below noise floor — return identity (no compensation needed)
            self._affine_matrix = identity.copy()

        motion_result["translation_x"] = self._smooth_tx
        motion_result["translation_y"] = self._smooth_ty
        motion_result["rotation"] = self._smooth_da
        motion_result["stability_score"] = float(self._smooth_stability)
        motion_result["affine_matrix"] = self._affine_matrix.copy()
        motion_result["is_compensating"] = bool(is_compensating)

        return motion_result

    def get_affine_matrix(self) -> np.ndarray:
        """Returns the latest smoothed, confidence-weighted 2x3 affine matrix."""
        return self._affine_matrix.copy()

    def reset(self):
        """Reset all internal state for a new video/session."""
        self._prev_frame = None
        self._prev_features = None
        self._smooth_tx = 0.0
        self._smooth_ty = 0.0
        self._smooth_da = 0.0
        self._smooth_stability = 1.0
        self._initialized = False
        self._affine_matrix = np.eye(2, 3, dtype=np.float32)
