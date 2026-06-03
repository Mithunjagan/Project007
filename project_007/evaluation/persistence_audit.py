"""
PROJECT 007 — P4.7 Persistence Audit Mode
Verifies that Scene-Level Evidence Persistence correctly accumulates risk across track ID fragmentations.

Usage:
    python -m evaluation.persistence_audit <video_path>
"""

import argparse
import json
import time
from pathlib import Path

import cv2

from pipeline.core import PipelineRunner
from utils.logger import get_logger

logger = get_logger(__name__)

def audit_persistence(video_path: str):
    logger.info(f"Starting Persistence Audit on: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    runner = PipelineRunner(width, height, sync_mode=True, enable_recording=False, log_telemetry=False)
    
    out_dir = Path("evaluation/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    frame_count = 0
    now_mono = time.perf_counter()
    now_wall = time.time()
    
    # We will track risk per state
    max_risk_score = 0.0
    highest_state = "NORMAL"
    event_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            dt = 1.0 / fps
            now_mono += dt
            now_wall += dt
            
            _, _, fused = runner.step(frame, frame_count, now_mono, now_wall)
            
            for fe in fused:
                if fe.evidence_score > max_risk_score:
                    max_risk_score = fe.evidence_score
                
                # State ordering hack
                state_order = {"NORMAL": 0, "SUSPICIOUS": 1, "HIGH_RISK": 2, "CRITICAL": 3}
                if state_order[fe.state] > state_order[highest_state]:
                    highest_state = fe.state
                    logger.info(f"Frame {frame_count} - STATE ESCALATION: {fe.state} (Risk: {fe.evidence_score:.2f})")
                    
                if fe.state in ["HIGH_RISK", "CRITICAL"]:
                    event_count += 1
                    
    finally:
        runner.cleanup()
        cap.release()
        
    stats = runner.fusion_engine.interaction_manager.stats
    
    # Calculate lifetime statistics
    lifetimes = stats.get("interaction_lifetimes", [])
    if lifetimes:
        avg_lifetime = sum(lifetimes) / len(lifetimes)
        max_lifetime = max(lifetimes)
    else:
        avg_lifetime = 0.0
        max_lifetime = 0.0
        
    report = {
        "video": video_path,
        "total_frames": frame_count,
        "max_risk_score": round(max_risk_score, 4),
        "highest_state": highest_state,
        "event_frames": event_count,
        "interaction_stats": {
            "merges": stats["interaction_merges"],
            "expirations": stats["interaction_expirations"],
            "track_churn_events": stats["track_churn_events"],
            "avg_lifetime_sec": round(avg_lifetime, 2),
            "max_lifetime_sec": round(max_lifetime, 2)
        }
    }
    
    report_path = out_dir / "interaction_persistence_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    logger.info("=" * 50)
    logger.info("PERSISTENCE AUDIT SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Highest State Achieved : {highest_state}")
    logger.info(f"Max Risk Score         : {max_risk_score:.4f}")
    logger.info(f"Event Frames           : {event_count}")
    logger.info(f"Interaction Merges     : {stats['interaction_merges']}")
    logger.info(f"Track Churn Events     : {stats['track_churn_events']}")
    logger.info(f"Avg Interaction Life   : {avg_lifetime:.2f}s")
    logger.info(f"Max Interaction Life   : {max_lifetime:.2f}s")
    logger.info("=" * 50)
    logger.info(f"Report saved to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str, help="Path to video")
    args = parser.parse_args()
    audit_persistence(args.video_path)
