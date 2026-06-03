"""
PROJECT 007 — Drone Dataset Generator (P6.0-A)
Generates simulated drone videos from fight_test.mp4 by applying various camera trajectories:
1. Hovering (low-frequency drift + rotation)
2. Slow Pan (linear translation sweep)
3. Fast Yaw (rapid horizontal pan)
4. Windy Footage (high-frequency random jitter)
5. People Walking Below (hovering drift + altitude zoom)

Also duplicates the fight_test.json annotations for these files to enable replay auditing.
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path

# Ensure paths are correct
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_VIDEO = BASE_DIR / "dataset/interaction/fight_test.mp4"
INPUT_ANN = BASE_DIR / "dataset/annotations/fight_test.json"
OUTPUT_DIR = BASE_DIR / "dataset/drone"
ANN_DIR = BASE_DIR / "dataset/annotations"

print(f"Base Directory: {BASE_DIR}")
print(f"Input Video: {INPUT_VIDEO}")
print(f"Input Annotations: {INPUT_ANN}")

def get_warp_matrix(w, h, tx, ty, da, scale=1.0):
    """
    Computes warp matrix to simulate camera translation, rotation, and scaling.
    Note: A camera translation of +tx shifts pixels by -tx.
    """
    # Shift center to origin, apply rotation/scale, then shift back with translation
    center_x, center_y = w / 2.0, h / 2.0
    
    # 2D similarity transform components
    cos_a = np.cos(da)
    sin_a = np.sin(da)
    
    a = scale * cos_a
    b = -scale * sin_a
    c = center_x * (1.0 - a) - center_y * b - tx
    
    d = scale * sin_a
    e = scale * cos_a
    f = center_y * (1.0 - e) - center_x * d - ty
    
    return np.array([[a, b, c], [d, e, f]], dtype=np.float32)

def generate_drone_video(name, motion_fn):
    """
    Reads INPUT_VIDEO, applies motion warp frame by frame, and saves to drone directory.
    """
    cap = cv2.VideoCapture(str(INPUT_VIDEO))
    if not cap.isOpened():
        print(f"[-] Error: Cannot open input video {INPUT_VIDEO}")
        return False
        
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    out_path = OUTPUT_DIR / f"{name}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    
    print(f"[+] Generating {name}.mp4 ({total_frames} frames)...")
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Get translation, rotation, and scale for this frame
        tx, ty, da, scale = motion_fn(frame_idx, total_frames, fps)
        
        # Get warp matrix
        M = get_warp_matrix(w, h, tx, ty, da, scale)
        
        # Warp the frame using BORDER_REPLICATE to avoid black edges
        warped = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        out.write(warped)
        frame_idx += 1
        
    cap.release()
    out.release()
    print(f"[+] Saved: {out_path}")
    
    # Generate matching annotations JSON
    ann_path = ANN_DIR / f"{name}.json"
    if INPUT_ANN.exists():
        with open(INPUT_ANN, "r") as f:
            ann_data = json.load(f)
        ann_data["video_id"] = name
        with open(ann_path, "w") as f:
            json.dump(ann_data, f, indent=4)
        print(f"[+] Copied Annotations: {ann_path}")
    return True

# ── Motion Trajectory Profiles ───────────────────────────────────────────────

def motion_hovering(idx, total, fps):
    # Sinusoidal drift: 2s horizontal period, 3s vertical period
    t = idx / fps
    tx = 15.0 * np.sin(2.0 * np.pi * 0.5 * t)
    ty = 10.0 * np.cos(2.0 * np.pi * 0.33 * t)
    da = 0.015 * np.sin(2.0 * np.pi * 0.2 * t)  # small rotation (yaw drift)
    return tx, ty, da, 1.0

def motion_slow_pan(idx, total, fps):
    # Linear pan translation: moves left to right and slightly down
    t = idx / fps
    tx = 1.5 * idx * (1.0 if idx < total / 2 else -1.0)  # Panning back and forth
    ty = 0.3 * idx * (1.0 if idx < total / 2 else -1.0)
    da = 0.0
    return tx, ty, da, 1.0

def motion_fast_yaw(idx, total, fps):
    # Rapid pan back and forth
    t = idx / fps
    tx = 120.0 * np.sin(2.0 * np.pi * 0.1 * t)  # High amplitude sweep
    ty = 10.0 * np.cos(2.0 * np.pi * 0.2 * t)
    da = 0.04 * np.sin(2.0 * np.pi * 0.1 * t)
    return tx, ty, da, 1.0

def motion_windy(idx, total, fps):
    # High-frequency jitter using random walk / white noise
    # We can seed it for reproducibility
    np.random.seed(idx)
    tx = np.random.normal(0.0, 5.0)
    ty = np.random.normal(0.0, 3.5)
    da = np.random.normal(0.0, 0.01)
    return tx, ty, da, 1.0

def motion_walking_below(idx, total, fps):
    # Centered vertical view with hovering drift + altitude zoom
    t = idx / fps
    tx = 8.0 * np.sin(2.0 * np.pi * 0.4 * t)
    ty = 6.0 * np.cos(2.0 * np.pi * 0.25 * t)
    da = 0.01 * np.sin(2.0 * np.pi * 0.15 * t)
    scale = 1.0 + 0.05 * np.sin(2.0 * np.pi * 0.08 * t) # +/- 5% scale changes
    return tx, ty, da, scale

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ANN_DIR.mkdir(parents=True, exist_ok=True)
    
    if not INPUT_VIDEO.exists():
        print(f"[-] Error: Base video {INPUT_VIDEO} does not exist!")
        return
        
    generate_drone_video("hovering", motion_hovering)
    generate_drone_video("slow_pan", motion_slow_pan)
    generate_drone_video("fast_yaw", motion_fast_yaw)
    generate_drone_video("windy_footage", motion_windy)
    generate_drone_video("people_walking_below", motion_walking_below)
    
    print("[+] Drone dataset generation completed successfully!")

if __name__ == "__main__":
    main()
