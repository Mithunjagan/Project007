"""
PROJECT 007 — Drone Motion Audit (P6.0-A)
Runs evaluation over the drone dataset (hovering, slow pan, fast yaw, windy, etc.) and
compares raw velocities with ego-motion compensated velocities.
The baseline is fight_test.mp4 (static camera), which serves as the physical ground truth.

Proves that:
1. Raw velocities spike or drift wildly under drone camera motion.
2. Compensated velocities closely match the static baseline ground truth, minimizing MAE.
"""

import os
import json
import time
from pathlib import Path
import cv2
import numpy as np

from pipeline.core import PipelineRunner
from pipeline.egomotion import EgomotionEstimator
from utils.logger import get_logger

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DRONE_DIR = BASE_DIR / "dataset/drone"
STATIC_VIDEO = BASE_DIR / "dataset/interaction/fight_test.mp4"
REPORT_PATH = BASE_DIR / "evaluation/reports/drone_motion_audit.json"

# List of drone videos to audit
DRONE_VIDEOS = [
    "hovering",
    "slow_pan",
    "fast_yaw",
    "windy_footage",
    "people_walking_below"
]


def compose_affines_for_audit(ego_history, t_prev, t_curr):
    if t_prev is None or t_curr is None or t_prev >= t_curr:
        return None
    H = np.eye(3, dtype=np.float32)
    has_comp = False
    for fid in range(t_prev + 1, t_curr + 1):
        A = ego_history.get(fid)
        if A is not None:
            H_i = np.eye(3, dtype=np.float32)
            H_i[:2, :] = A
            H = H_i @ H
            has_comp = True
    if has_comp:
        return H[:2, :]
    return None


def calculate_arm_velocity(curr_kps, prev_kps, dt, bbox_h, affine_mat=None):
    """
    Calculates arm swing velocity.
    If affine_mat (2x3) is provided, compensates prev_kps coordinates before computing velocity.
    """
    arm_keys = ["left_elbow", "right_elbow", "left_wrist", "right_wrist"]
    dx_list = []
    dy_list = []

    for key in arm_keys:
        curr = curr_kps.get(key)
        prev = prev_kps.get(key)

        if curr and prev and curr.get("visibility", 0) > 0.3 and prev.get("visibility", 0) > 0.3:
            px, py = prev["x"], prev["y"]
            
            if affine_mat is not None:
                # Project prev point using the estimated affine matrix
                ex = affine_mat[0, 0] * px + affine_mat[0, 1] * py + affine_mat[0, 2]
                ey = affine_mat[1, 0] * px + affine_mat[1, 1] * py + affine_mat[1, 2]
            else:
                ex, ey = px, py
                
            dx_list.append(curr["x"] - ex)
            dy_list.append(curr["y"] - ey)

    if not dx_list:
        return 0.0

    mean_dx = float(np.mean(dx_list))
    mean_dy = float(np.mean(dy_list))

    speed_px_per_sec = np.sqrt(mean_dx**2 + mean_dy**2) / dt
    return float(speed_px_per_sec / bbox_h)


def calculate_body_velocity(curr_center, prev_center, dt, bbox_h, affine_mat=None):
    """
    Calculates body displacement velocity.
    If affine_mat is provided, compensates prev_center coordinates.
    """
    px, py = prev_center[0], prev_center[1]
    
    if affine_mat is not None:
        ex = affine_mat[0, 0] * px + affine_mat[0, 1] * py + affine_mat[0, 2]
        ey = affine_mat[1, 0] * px + affine_mat[1, 1] * py + affine_mat[1, 2]
    else:
        ex, ey = px, py
        
    dx = curr_center[0] - ex
    dy = curr_center[1] - ey
    speed_px_per_sec = np.sqrt(dx**2 + dy**2) / dt
    return float(speed_px_per_sec / bbox_h)


def calculate_approach_velocity(curr_a, curr_b, prev_a, prev_b, dt, avg_bbox_h, affine_mat=None):
    """
    Calculates pairwise approach velocity.
    If affine_mat is provided, compensates previous distances.
    """
    ca_now = np.array(curr_a["center"])
    cb_now = np.array(curr_b["center"])
    ca_prev = np.array(prev_a["center"])
    cb_prev = np.array(prev_b["center"])

    dist_now_px = float(np.linalg.norm(ca_now - cb_now))
    
    if affine_mat is not None:
        # Project previous centroids to current frame scale/coords
        pa_ex = np.array([
            affine_mat[0, 0] * ca_prev[0] + affine_mat[0, 1] * ca_prev[1] + affine_mat[0, 2],
            affine_mat[1, 0] * ca_prev[0] + affine_mat[1, 1] * ca_prev[1] + affine_mat[1, 2]
        ])
        pb_ex = np.array([
            affine_mat[0, 0] * cb_prev[0] + affine_mat[0, 1] * cb_prev[1] + affine_mat[0, 2],
            affine_mat[1, 0] * cb_prev[0] + affine_mat[1, 1] * cb_prev[1] + affine_mat[1, 2]
        ])
        dist_prev_px = float(np.linalg.norm(pa_ex - pb_ex))
    else:
        dist_prev_px = float(np.linalg.norm(ca_prev - cb_prev))

    approach_vel_px = (dist_prev_px - dist_now_px) / dt
    return float(approach_vel_px / avg_bbox_h)


class DroneMotionAuditor:
    def __init__(self):
        self.baseline_data = {}

    def extract_velocities(self, video_path: str, use_egomotion: bool = False) -> dict:
        """
        Runs the pipeline runner over a video and extracts raw & compensated velocities.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video {video_path}")
            return {}

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

        runner = PipelineRunner(
            frame_width, frame_height,
            sync_mode=True, enable_recording=False, log_telemetry=False
        )
        
        egomotion_estimator = EgomotionEstimator()
        
        # Frame index -> metrics dict
        video_metrics = {}
        frame_count = 0
        ego_history = {}  # frame_count -> affine_matrix

        try:
            while frame_count < 500:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                dt = 1.0 / fps
                now_mono = frame_count * dt
                now_wall = time.time() + now_mono

                # Run detection and tracking
                _, _, _ = runner.step(frame, frame_count, now_mono, now_wall)

                # Get active tracks from the pipeline's track buffer
                active_ids = runner.track_buffer.get_active_ids()
                
                # Get person bboxes for egomotion masking
                person_bboxes = []
                for det in runner.current_detections:
                    person_bboxes.append(det.bbox)

                # Compute egomotion affine matrix (A_prev_to_curr)
                motion_result = egomotion_estimator.update(frame, person_bboxes)
                
                # Use the pre-built smoothed, confidence-weighted affine directly
                if motion_result.get("is_compensating", False):
                    ego_history[frame_count] = motion_result["affine_matrix"]
                else:
                    ego_history[frame_count] = None
                
                tx = motion_result["translation_x"]
                ty = motion_result["translation_y"]
                da = motion_result["rotation"]

                frame_arm_raw = []
                frame_arm_comp = []
                frame_body_raw = []
                frame_body_comp = []

                # 1. Compute Arm and Body velocities for each active track
                for tid in active_ids:
                    history = runner.track_buffer.get_history(tid)
                    if len(history) < 2:
                        continue

                    curr = history[-1]
                    prev = history[-2]

                    # Dynamic dt based on the track history frames
                    dt_track = curr["timestamp"] - prev["timestamp"]
                    if dt_track <= 0:
                        dt_track = dt

                    # Composed affine matrix composition for arm velocity (pose interval)
                    composed_affine_pose = None
                    if use_egomotion:
                        t_prev_pose = prev.get("pose_frame_id")
                        t_curr_pose = curr.get("pose_frame_id")
                        composed_affine_pose = compose_affines_for_audit(ego_history, t_prev_pose, t_curr_pose)

                    # 1-frame affine matrix composition for body displacement velocity
                    composed_affine_body = None
                    if use_egomotion:
                        composed_affine_body = compose_affines_for_audit(ego_history, prev.get("frame_id"), curr.get("frame_id"))

                    bbox = curr["bbox"]
                    bbox_h = max(1.0, bbox[3] - bbox[1])

                    # Check if pose was updated on this frame
                    curr_pose_fid = curr.get("pose_frame_id")
                    prev_pose_fid = prev.get("pose_frame_id")

                    if curr_pose_fid is not None and prev_pose_fid is not None and curr_pose_fid == prev_pose_fid:
                        # Non-pose frame: Arm velocity is 0
                        v_arm_raw = 0.0
                        v_arm_comp = 0.0
                    else:
                        v_arm_raw = calculate_arm_velocity(curr["keypoints"], prev["keypoints"], dt_track, bbox_h)
                        v_arm_comp = calculate_arm_velocity(
                            curr["keypoints"], prev["keypoints"], dt_track, bbox_h, 
                            composed_affine_pose
                        )

                    frame_arm_raw.append(v_arm_raw)
                    frame_arm_comp.append(v_arm_comp)

                    # Raw Body velocity
                    v_body_raw = calculate_body_velocity(curr["center"], prev["center"], dt_track, bbox_h)
                    frame_body_raw.append(v_body_raw)

                    # Compensated Body velocity
                    v_body_comp = calculate_body_velocity(
                        curr["center"], prev["center"], dt_track, bbox_h,
                        composed_affine_body
                    )
                    frame_body_comp.append(v_body_comp)

                # 2. Compute Approach velocities for pairs of active tracks
                frame_app_raw = []
                frame_app_comp = []
                
                sorted_ids = sorted(active_ids)
                for i in range(len(sorted_ids)):
                    for j in range(i + 1, len(sorted_ids)):
                        id_a, id_b = sorted_ids[i], sorted_ids[j]
                        hist_a = runner.track_buffer.get_history(id_a)
                        hist_b = runner.track_buffer.get_history(id_b)

                        if len(hist_a) < 2 or len(hist_b) < 2:
                            continue

                        curr_a, curr_b = hist_a[-1], hist_b[-1]
                        prev_a, prev_b = hist_a[-2], hist_b[-2]

                        dt_pair = curr_a["timestamp"] - prev_a["timestamp"]
                        if dt_pair <= 0:
                            dt_pair = dt

                        composed_affine_pair = None
                        if use_egomotion:
                            t_prev = prev_a.get("frame_id")
                            t_curr = curr_a.get("frame_id")
                            composed_affine_pair = compose_affines_for_audit(ego_history, t_prev, t_curr)

                        bbox_h_a = max(1.0, curr_a["bbox"][3] - curr_a["bbox"][1])
                        bbox_h_b = max(1.0, curr_b["bbox"][3] - curr_b["bbox"][1])
                        avg_bbox_h = (bbox_h_a + bbox_h_b) / 2.0

                        # Raw approach
                        v_app_raw = calculate_approach_velocity(curr_a, curr_b, prev_a, prev_b, dt_pair, avg_bbox_h)
                        frame_app_raw.append(v_app_raw)

                        # Compensated approach
                        v_app_comp = calculate_approach_velocity(
                            curr_a, curr_b, prev_a, prev_b, dt_pair, avg_bbox_h,
                            composed_affine_pair
                        )
                        frame_app_comp.append(v_app_comp)

                video_metrics[frame_count] = {
                    "arm_raw": float(np.mean(frame_arm_raw)) if frame_arm_raw else 0.0,
                    "arm_comp": float(np.mean(frame_arm_comp)) if frame_arm_comp else 0.0,
                    "body_raw": float(np.mean(frame_body_raw)) if frame_body_raw else 0.0,
                    "body_comp": float(np.mean(frame_body_comp)) if frame_body_comp else 0.0,
                    "app_raw": float(np.mean(frame_app_raw)) if frame_app_raw else 0.0,
                    "app_comp": float(np.mean(frame_app_comp)) if frame_app_comp else 0.0,
                    "stability_score": motion_result["stability_score"],
                    "translation_mag": np.sqrt(tx**2 + ty**2),
                    "rotation_mag": abs(da)
                }

        finally:
            runner.cleanup()
            cap.release()

        return video_metrics

    def run_audit(self):
        logger.info("[*] Step 1: Processing Static Baseline Video (fight_test.mp4)...")
        # Run baseline *without* motion compensation (it's already static, so raw is baseline)
        self.baseline_data = self.extract_velocities(STATIC_VIDEO, use_egomotion=False)
        logger.info(f"[+] Static baseline processed: {len(self.baseline_data)} frames.")

        audit_results = {}

        # Process each drone video
        for name in DRONE_VIDEOS:
            video_path = DRONE_DIR / f"{name}.mp4"
            if not video_path.exists():
                logger.warning(f"[-] Video {video_path} does not exist. Skipping.")
                continue

            logger.info(f"[*] Auditing Video: {name}.mp4...")
            # Extract both raw and compensated velocities
            metrics = self.extract_velocities(video_path, use_egomotion=True)
            
            # Compare metrics frame-by-frame to baseline static video
            raw_arm_errors = []
            comp_arm_errors = []
            raw_body_errors = []
            comp_body_errors = []
            raw_app_errors = []
            comp_app_errors = []

            for f_idx, frame_data in metrics.items():
                if f_idx not in self.baseline_data:
                    continue
                
                base_data = self.baseline_data[f_idx]
                
                # Error is difference from static camera ground truth
                raw_arm_errors.append(abs(frame_data["arm_raw"] - base_data["arm_raw"]))
                comp_arm_errors.append(abs(frame_data["arm_comp"] - base_data["arm_raw"]))

                raw_body_errors.append(abs(frame_data["body_raw"] - base_data["body_raw"]))
                comp_body_errors.append(abs(frame_data["body_comp"] - base_data["body_raw"]))

                raw_app_errors.append(abs(frame_data["app_raw"] - base_data["app_raw"]))
                comp_app_errors.append(abs(frame_data["app_comp"] - base_data["app_raw"]))

            arm_raw_mae = float(np.mean(raw_arm_errors)) if raw_arm_errors else 0.0
            arm_comp_mae = float(np.mean(comp_arm_errors)) if comp_arm_errors else 0.0
            
            body_raw_mae = float(np.mean(raw_body_errors)) if raw_body_errors else 0.0
            body_comp_mae = float(np.mean(comp_body_errors)) if comp_body_errors else 0.0

            app_raw_mae = float(np.mean(raw_app_errors)) if raw_app_errors else 0.0
            app_comp_mae = float(np.mean(comp_app_errors)) if comp_app_errors else 0.0

            arm_reduction = (arm_raw_mae - arm_comp_mae) / arm_raw_mae * 100.0 if arm_raw_mae > 0 else 0.0
            body_reduction = (body_raw_mae - body_comp_mae) / body_raw_mae * 100.0 if body_raw_mae > 0 else 0.0
            app_reduction = (app_raw_mae - app_comp_mae) / app_raw_mae * 100.0 if app_raw_mae > 0 else 0.0

            audit_results[name] = {
                "arm_swing_velocity": {
                    "raw_mae": round(arm_raw_mae, 4),
                    "compensated_mae": round(arm_comp_mae, 4),
                    "reduction_pct": round(arm_reduction, 2)
                },
                "body_displacement_velocity": {
                    "raw_mae": round(body_raw_mae, 4),
                    "compensated_mae": round(body_comp_mae, 4),
                    "reduction_pct": round(body_reduction, 2)
                },
                "pairwise_approach_velocity": {
                    "raw_mae": round(app_raw_mae, 4),
                    "compensated_mae": round(app_comp_mae, 4),
                    "reduction_pct": round(app_reduction, 2)
                }
            }

        # Print comparison report table
        print("\n" + "="*80)
        print("                 PROJECT 007 — DRONE EGOMOTION COMPENSATION AUDIT")
        print("="*80)
        print(f"{'Video Profile':<22} | {'Metric Type':<20} | {'Raw MAE':>9} | {'Comp MAE':>9} | {'Error Red. %':>12}")
        print("-"*80)
        
        for v_name, result in audit_results.items():
            for m_name, m_data in result.items():
                m_label = m_name.replace("_", " ").title()
                print(f"{v_name:<22} | {m_label:<20} | {m_data['raw_mae']:>9.4f} | {m_data['compensated_mae']:>9.4f} | {m_data['reduction_pct']:>11.1f}%")
            print("-"*80)

        # Save JSON report
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(REPORT_PATH, "w") as f:
            json.dump({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "results": audit_results
            }, f, indent=4)
        logger.info(f"Audit report saved to {REPORT_PATH}")


if __name__ == "__main__":
    auditor = DroneMotionAuditor()
    auditor.run_audit()
