"""
PROJECT 007 — P4.6 Detection Audit Mode
Instruments the existing pipeline to diagnose rule engine behavior.

Usage:
    python -m evaluation.detection_audit <video_path>
"""

import argparse
import csv
import json
import time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

from pipeline.core import PipelineRunner
from utils.logger import get_logger

logger = get_logger(__name__)

def audit_video(video_path: str):
    logger.info(f"Starting Detection Audit Mode on: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    runner = PipelineRunner(width, height, sync_mode=True, enable_recording=False, log_telemetry=False)
    
    # ── Hook into FusionEngine to capture raw rules ──
    captured_raw_rules = []
    original_update = runner.fusion_engine.update
    
    def hooked_update(raw_events, *args, **kwargs):
        captured_raw_rules.clear()
        captured_raw_rules.extend(raw_events)
        return original_update(raw_events, *args, **kwargs)
        
    runner.fusion_engine.update = hooked_update
    
    # Audit tracking variables
    out_dir = Path("evaluation/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = out_dir / "detection_audit.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame_id", "track_count", "active_track_ids", 
        "arm_velocity", "body_velocity", "fall_score", 
        "approach_velocity", "risk_score", "fused_score", 
        "state", "active_rules", "persistence_scores"
    ])
    
    rule_stats = defaultdict(lambda: {
        "evaluated_count": 0,
        "passed_threshold_count": 0,
        "persistence_promoted_count": 0,
        "cooldown_blocked_count": 0
    })
    
    maxima = {
        "max_arm_velocity": 0.0,
        "max_approach_velocity": 0.0,
        "max_fall_score": 0.0,
        "max_proximity_ratio": 0.0,
        "max_optical_flow": 0.0
    }
    
    frame_count = 0
    now_mono = time.perf_counter()
    now_wall = time.time()
    
    previous_states = {} # track_ids -> state
    
    print("\n" + "-"*50)
    print("STATE TRANSITIONS")
    print("-"*50)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            dt = 1.0 / fps
            now_mono += dt
            now_wall += dt
            
            _, snap, fused = runner.step(frame, frame_count, now_mono, now_wall)
            
            # --- Extract Features for Maxima & CSV ---
            track_count = snap.get("person_count", 0)
            active_ids = [str(t.track_id) for t in runner.current_detections]
            
            arm_vels = []
            body_vels = []
            fall_scores = []
            approach_vels = []
            
            for det in runner.current_detections:
                raw_m = runner.motion_engine.compute(det.track_id, runner.track_buffer)
                arm_vels.append(raw_m.get("arm_velocity", 0.0))
                body_vels.append(raw_m.get("body_displacement", 0.0))
                fall_scores.append(raw_m.get("fall_score", 0.0))
                
            pm = runner.motion_engine.compute_pairwise(runner.track_buffer)
            for pair, data in pm.items():
                approach_vels.append(data.get("approach_velocity", 0.0))
                
            max_arm_vel = max(arm_vels) if arm_vels else 0.0
            max_body_vel = max(body_vels) if body_vels else 0.0
            max_fall = max(fall_scores) if fall_scores else 0.0
            max_app_vel = max(approach_vels) if approach_vels else 0.0
            
            flow = snap.get("global_flow_magnitude", 0.0)
            prox_ratio = snap.get("occupancy_ratio", 0.0)
            
            # Update Maxima
            maxima["max_arm_velocity"] = max(maxima["max_arm_velocity"], max_arm_vel)
            maxima["max_approach_velocity"] = max(maxima["max_approach_velocity"], max_app_vel)
            maxima["max_fall_score"] = max(maxima["max_fall_score"], max_fall)
            maxima["max_optical_flow"] = max(maxima["max_optical_flow"], flow)
            maxima["max_proximity_ratio"] = max(maxima["max_proximity_ratio"], prox_ratio)
            
            # --- Rule Counters ---
            all_possible_rules = [
                "DIRECTED_ARM_SWING", "RAPID_APPROACH", "FALL_EVENT", 
                "SUSTAINED_CONTACT", "CROWD_DISPERSION", "CAMERA_SHAKE", 
                "LENS_OCCLUSION", "CAMERA_BLOCKAGE", "CAMERA_RUSH", 
                "PROXIMITY_INTRUSION", "ABNORMAL_SINGLE_SUBJECT_ENERGY"
            ]
            
            # Count passed
            passed_rules = [r.rule_type for r in captured_raw_rules]
            
            # Count promoted
            promoted_rules = set()
            for fe in fused:
                promoted_rules.update(fe.contributing_rules)
                
            for r in all_possible_rules:
                if track_count > 0 or r in ["CAMERA_SHAKE", "LENS_OCCLUSION", "CAMERA_BLOCKAGE"]:
                    rule_stats[r]["evaluated_count"] += 1
                
                count_in_passed = passed_rules.count(r)
                if count_in_passed > 0:
                    rule_stats[r]["passed_threshold_count"] += count_in_passed
                    
                if r in promoted_rules:
                    rule_stats[r]["persistence_promoted_count"] += 1
                    
                # If passed but not promoted
                if count_in_passed > 0 and r not in promoted_rules:
                    rule_stats[r]["cooldown_blocked_count"] += count_in_passed
                    
            # --- Extract Scores & States ---
            fused_score = 0.0
            current_state = "NORMAL"
            persistence_scores = {}
            active_rules_list = []
            
            for fe in fused:
                if fe.evidence_score > fused_score:
                    fused_score = fe.evidence_score
                    current_state = fe.state
                
                tid_key = str(sorted(list(fe.track_ids)))
                persistence_scores[tid_key] = fe.evidence_score
                active_rules_list.extend(fe.contributing_rules)
                
                # Print State Transitions
                prev = previous_states.get(tid_key, "NORMAL")
                if prev != fe.state:
                    time_str = f"[{int((frame_count/fps) // 60):02d}:{int((frame_count/fps) % 60):02d}]"
                    print(f"{time_str} Frame {frame_count:04d} | {prev} -> {fe.state} (Tracks: {tid_key}, Risk: {fe.evidence_score:.2f})")
                    previous_states[tid_key] = fe.state
                    
            # CSV Record
            csv_writer.writerow([
                frame_count,
                track_count,
                ";".join(active_ids),
                f"{max_arm_vel:.4f}",
                f"{max_body_vel:.4f}",
                f"{max_fall:.4f}",
                f"{max_app_vel:.4f}",
                f"{fused_score:.4f}",
                f"{fused_score:.4f}",
                current_state,
                ";".join(set(active_rules_list)),
                json.dumps(persistence_scores)
            ])
            
    finally:
        csv_file.close()
        runner.cleanup()
        cap.release()
        
    # Write JSONs
    with open(out_dir / "rule_statistics.json", "w") as f:
        json.dump(rule_stats, f, indent=4)
        
    with open(out_dir / "feature_extremes.json", "w") as f:
        json.dump(maxima, f, indent=4)
        
    # Print summary
    print("\n" + "="*50)
    print("RULE ENGINE HEALTH")
    print("="*50)
    for rule, stats in rule_stats.items():
        if stats["evaluated_count"] > 0:
            print(f"\n{rule}:")
            print(f"  evaluated = {stats['evaluated_count']}")
            print(f"  threshold_pass = {stats['passed_threshold_count']}")
            print(f"  promoted = {stats['persistence_promoted_count']}")
            print(f"  blocked/suppressed = {stats['cooldown_blocked_count']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str, help="Path to the video file to audit.")
    args = parser.parse_args()
    audit_video(args.video_path)
