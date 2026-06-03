"""
PROJECT 007 — P4.6 Pose Pipeline Audit
Traces the entire data flow from YOLO to Motion Engine to find where
pose-derived features disappear.
"""

import argparse
import json
import time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

from pipeline.core import PipelineRunner
from utils.logger import get_logger

logger = get_logger(__name__)

def audit_pose_pipeline(video_path: str):
    logger.info(f"Starting Pose Pipeline Audit on: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    runner = PipelineRunner(width, height, sync_mode=True, enable_recording=False, log_telemetry=False)

    stats = {
        "yolo_detections": 0,
        "crops_submitted": 0,
        "successful_poses": 0,
        "failed_poses": 0,
        "keypoint_validation": {
            "valid_left_wrist": 0,
            "valid_right_wrist": 0,
            "valid_shoulders": 0,
            "visible_10_plus": 0
        },
        "track_buffer_updates": 0,
        "updates_per_track": defaultdict(int),
        "orphaned_poses": 0,
        "motion_failures": {
            "no_previous_frame": 0,
            "missing_keypoints": 0,
            "calculation_resulted_in_zero": 0
        }
    }
    
    debug_dir = Path("debug/pose_samples")
    debug_dir.mkdir(parents=True, exist_ok=True)
    samples_saved = 0
    
    # Hooks
    orig_extract = runner.pose_extractor._extract_keypoints
    def hooked_extract(crop, bbox):
        stats["crops_submitted"] += 1
        res = orig_extract(crop, bbox)
        if res is not None:
            stats["successful_poses"] += 1
            # Validation
            vis = sum(1 for kp in res.values() if isinstance(kp, dict) and kp.get("visibility", 0) > 0.3)
            if vis >= 10:
                stats["keypoint_validation"]["visible_10_plus"] += 1
            if res.get("left_wrist", {}).get("visibility", 0) > 0.3:
                stats["keypoint_validation"]["valid_left_wrist"] += 1
            if res.get("right_wrist", {}).get("visibility", 0) > 0.3:
                stats["keypoint_validation"]["valid_right_wrist"] += 1
            if res.get("left_shoulder", {}).get("visibility", 0) > 0.3 and res.get("right_shoulder", {}).get("visibility", 0) > 0.3:
                stats["keypoint_validation"]["valid_shoulders"] += 1
        else:
            stats["failed_poses"] += 1
        return res
    runner.pose_extractor._extract_keypoints = hooked_extract
    
    orig_update = runner.track_buffer.update
    def hooked_update(track_id, keypoints, bbox, timestamp=None):
        stats["track_buffer_updates"] += 1
        stats["updates_per_track"][track_id] += 1
        return orig_update(track_id, keypoints, bbox, timestamp)
    runner.track_buffer.update = hooked_update
    
    orig_compute = runner.motion_engine.compute
    def hooked_compute(track_id, buffer):
        history = buffer.get_history(track_id)
        if len(history) < 2:
            stats["motion_failures"]["no_previous_frame"] += 1
            return orig_compute(track_id, buffer)
            
        current = history[-1]
        previous = history[-2]
        
        arm_keys = ["left_elbow", "right_elbow", "left_wrist", "right_wrist"]
        has_curr = any(current["keypoints"].get(k, {}).get("visibility", 0) > 0.3 for k in arm_keys)
        has_prev = any(previous["keypoints"].get(k, {}).get("visibility", 0) > 0.3 for k in arm_keys)
        
        if not has_curr or not has_prev:
            stats["motion_failures"]["missing_keypoints"] += 1
            
        res = orig_compute(track_id, buffer)
        if res["arm_velocity"] == 0.0:
            stats["motion_failures"]["calculation_resulted_in_zero"] += 1
            
        return res
    runner.motion_engine.compute = hooked_compute

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1
            now = time.time()
            
            annotated, snap, fused = runner.step(frame, frame_count, now, now)
            
            stats["yolo_detections"] += snap.get("person_count", 0)
            
            # Orphan check (for sync mode, mostly just tracking deletions)
            # In PipelineRunner: active_ids = {det.track_id for det in current_detections}
            # Orphaned poses are those where we got a pose but track_id isn't in current_detections
            # Actually, this is more relevant for async mode. In sync mode, we only extract for current_detections.
            
            if samples_saved < 20 and len(runner.latest_keypoints) > 0:
                cv2.imwrite(str(debug_dir / f"sample_{samples_saved}.jpg"), annotated)
                samples_saved += 1
    finally:
        runner.cleanup()
        cap.release()

    total_crops = stats["crops_submitted"]
    pct = (stats["successful_poses"] / total_crops * 100) if total_crops > 0 else 0
    
    # Save JSON report
    out_dir = Path("evaluation/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "pose_pipeline_audit.json", "w") as f:
        json.dump(stats, f, indent=4)
        
    print("\n" + "="*50)
    print("POSE PIPELINE TRACE AUDIT")
    print("="*50)
    print("A) Is MediaPipe generating landmarks?")
    if stats["successful_poses"] > 0:
        print(f"   Yes. Success rate: {pct:.1f}% ({stats['successful_poses']}/{total_crops} crops)")
        print(f"   Valid left_wrist: {stats['keypoint_validation']['valid_left_wrist']}")
        print(f"   Valid right_wrist: {stats['keypoint_validation']['valid_right_wrist']}")
        print(f"   Valid shoulders: {stats['keypoint_validation']['valid_shoulders']}")
        print(f"   Visible 10+ points: {stats['keypoint_validation']['visible_10_plus']}")
    else:
        print("   No, MediaPipe is failing to generate landmarks on all crops.")
        
    print("\nB) Are landmarks reaching TrackBuffer?")
    if stats["track_buffer_updates"] > 0:
        print(f"   Yes, {stats['track_buffer_updates']} updates reached TrackBuffer.")
    else:
        print("   No, landmarks are not reaching TrackBuffer.")
        
    print("\nC) Are track IDs changing and breaking temporal continuity?")
    updates = list(stats["updates_per_track"].values())
    avg_updates = sum(updates) / len(updates) if updates else 0
    print(f"   Total unique track IDs: {len(updates)}")
    print(f"   Average updates per track: {avg_updates:.1f}")
    if avg_updates < 2:
        print("   Yes, track IDs are extremely short-lived (fragmenting).")
    elif avg_updates < 10:
        print("   Tracks are somewhat short-lived. High fragmentation likely.")
    else:
        print("   Tracks seem somewhat stable.")
        
    print("\nD) Why is arm_velocity always zero?")
    print(f"   - No previous frame (buffer < 2): {stats['motion_failures']['no_previous_frame']}")
    print(f"   - Missing arm keypoints (low visibility): {stats['motion_failures']['missing_keypoints']}")
    print(f"   - Calculation naturally resulted in zero: {stats['motion_failures']['calculation_resulted_in_zero']}")
    
    print("\nE) What exact line/module is the first point where valid pose data disappears?")
    if stats["yolo_detections"] == 0:
        print("   -> YOLO detector (pipeline/core.py: _sync_detect). Nobody is detected.")
    elif stats["crops_submitted"] == 0:
        print("   -> Pose crop logic (pipeline/core.py: _sync_pose). Crops are rejected before MediaPipe.")
    elif stats["successful_poses"] == 0:
        print("   -> PoseExtractor (pipeline/pose.py: _extract_keypoints). MediaPipe fails on all crops.")
    elif stats["track_buffer_updates"] == 0:
        print("   -> Buffer linking (pipeline/core.py: track_buffer.update). Keypoints aren't matched to Track IDs.")
    elif stats["motion_failures"]["no_previous_frame"] > stats["motion_failures"]["missing_keypoints"]:
        print("   -> ByteTrack (YOLO Tracker). Track IDs fragment so quickly the buffer never reaches length=2.")
    elif stats["motion_failures"]["missing_keypoints"] > 0:
        print("   -> MediaPipe Visibility Threshold. Landmarks are generated, but visibility is consistently < 0.3.")
    else:
        print("   -> Unknown. Data is there but motion calculation outputs 0.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str)
    args = parser.parse_args()
    audit_pose_pipeline(args.video_path)
