"""
PROJECT 007 — P3.0 Annotation GUI
Keyboard-driven video annotation tool for ground-truth labeling.

Usage:
    python -m evaluation.annotator <video_path>

Controls:
    Space    : Play / Pause
    N        : Next frame (paused)
    P        : Previous frame (paused)
    S        : Mark event START at current frame
    E        : Mark event END at current frame
    1-6      : Assign label to current event
    Enter    : Save current event
    Backspace: Delete last event
    W        : Write annotations to disk
    Q        : Quit

Labels:
    1 = normal
    2 = camera_rush
    3 = proximity_intrusion
    4 = camera_shake
    5 = lens_occlusion
    6 = high_energy_interaction
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

from dataset_tools.annotation_template import (
    EVENT_LABELS,
    create_empty_annotation,
    add_event,
    save_annotation,
    load_annotation,
)
from utils.logger import get_logger

logger = get_logger(__name__)

LABEL_KEYS = {
    ord("1"): "normal",
    ord("2"): "camera_rush",
    ord("3"): "proximity_intrusion",
    ord("4"): "camera_shake",
    ord("5"): "lens_occlusion",
    ord("6"): "high_energy_interaction",
}


class AnnotationGUI:
    """Keyboard-driven annotation GUI using OpenCV."""

    def __init__(self, video_path: str, annotations_dir: str = "dataset/annotations"):
        self.video_path = video_path
        self.video_id = Path(video_path).stem
        self.annotations_dir = annotations_dir

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.current_frame_idx = 0
        self.current_frame = None
        self.is_paused = True

        # Event marking state
        self.mark_start = -1
        self.mark_end = -1
        self.current_label = ""

        # Load existing annotation or create empty
        existing = load_annotation(self.video_id, self.annotations_dir)
        if existing:
            self.annotation = existing
            logger.info(f"Loaded existing annotation with {len(existing.get('events', []))} events")
        else:
            self.annotation = create_empty_annotation(
                self.video_id, self.total_frames, self.fps
            )

    def _read_frame(self, idx: int):
        """Seek to a specific frame index and read it."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            self.current_frame_idx = idx
        return ret

    def _draw_hud(self, frame):
        """Draw annotation HUD overlay."""
        display = frame.copy()
        h = display.shape[0]

        # Top bar: frame info
        current_time = self.current_frame_idx / self.fps
        status = "PAUSED" if self.is_paused else "PLAYING"
        cv2.putText(
            display,
            f"[{status}] Frame {self.current_frame_idx}/{self.total_frames}  "
            f"Time: {current_time:.2f}s",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        # Marking state
        mark_text = ""
        if self.mark_start >= 0:
            start_t = self.mark_start / self.fps
            mark_text = f"START: {start_t:.2f}s (f{self.mark_start})"
            if self.mark_end >= 0:
                end_t = self.mark_end / self.fps
                mark_text += f"  END: {end_t:.2f}s (f{self.mark_end})"
            if self.current_label:
                mark_text += f"  LABEL: {self.current_label}"
        else:
            mark_text = "Press S to mark event start"

        cv2.putText(
            display, mark_text,
            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1,
        )

        # Event count
        event_count = len(self.annotation.get("events", []))
        cv2.putText(
            display,
            f"Events: {event_count}  |  W=save  Q=quit",
            (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        # Label legend
        legend_y = 85
        for i, label in enumerate(EVENT_LABELS):
            color = (0, 255, 0) if label == self.current_label else (180, 180, 180)
            cv2.putText(
                display,
                f"{i + 1}={label}",
                (10, legend_y + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )

        # Draw event markers on a timeline bar
        bar_y = h - 40
        bar_h = 10
        cv2.rectangle(display, (10, bar_y), (self.frame_width - 10, bar_y + bar_h), (60, 60, 60), -1)

        # Current position
        if self.total_frames > 0:
            pos_x = 10 + int((self.frame_width - 20) * self.current_frame_idx / self.total_frames)
            cv2.line(display, (pos_x, bar_y - 3), (pos_x, bar_y + bar_h + 3), (0, 255, 255), 2)

        # Existing events
        for ev in self.annotation.get("events", []):
            sf = ev.get("start_frame", 0)
            ef = ev.get("end_frame", 0)
            if self.total_frames > 0:
                sx = 10 + int((self.frame_width - 20) * sf / self.total_frames)
                ex = 10 + int((self.frame_width - 20) * ef / self.total_frames)
                cv2.rectangle(display, (sx, bar_y), (ex, bar_y + bar_h), (0, 0, 255), -1)

        return display

    def run(self):
        """Main annotation loop."""
        self._read_frame(0)
        logger.info(f"Annotator opened: {self.video_path} ({self.total_frames} frames, {self.fps:.1f} fps)")

        while True:
            if self.current_frame is not None:
                display = self._draw_hud(self.current_frame)
                cv2.imshow("PROJECT 007 - ANNOTATOR", display)

            wait_ms = 1 if self.is_paused else max(1, int(1000 / self.fps))
            key = cv2.waitKey(wait_ms) & 0xFF

            if key == ord("q"):
                break

            elif key == ord(" "):  # Play/Pause
                self.is_paused = not self.is_paused

            elif key == ord("n") and self.is_paused:  # Next frame
                if self.current_frame_idx < self.total_frames - 1:
                    self._read_frame(self.current_frame_idx + 1)

            elif key == ord("p") and self.is_paused:  # Previous frame
                if self.current_frame_idx > 0:
                    self._read_frame(self.current_frame_idx - 1)

            elif key == ord("s"):  # Mark start
                self.mark_start = self.current_frame_idx
                self.mark_end = -1
                self.current_label = ""
                logger.info(f"Event START marked at frame {self.mark_start}")

            elif key == ord("e"):  # Mark end
                if self.mark_start >= 0:
                    self.mark_end = self.current_frame_idx
                    logger.info(f"Event END marked at frame {self.mark_end}")

            elif key in LABEL_KEYS:  # Assign label
                self.current_label = LABEL_KEYS[key]
                logger.info(f"Label selected: {self.current_label}")

            elif key == 13:  # Enter — commit event
                if self.mark_start >= 0 and self.mark_end >= 0 and self.current_label:
                    start_t = self.mark_start / self.fps
                    end_t = self.mark_end / self.fps
                    if end_t > start_t:
                        add_event(self.annotation, start_t, end_t, self.current_label)
                        logger.info(
                            f"Event added: {self.current_label} "
                            f"[{start_t:.2f}s - {end_t:.2f}s]"
                        )
                        self.mark_start = -1
                        self.mark_end = -1
                        self.current_label = ""
                    else:
                        logger.warning("End must be after start")
                else:
                    logger.warning("Set start (S), end (E), and label (1-6) before committing")

            elif key == 8:  # Backspace — delete last event
                events = self.annotation.get("events", [])
                if events:
                    removed = events.pop()
                    logger.info(f"Removed event: {removed['label']}")

            elif key == ord("w"):  # Write to disk
                save_annotation(self.annotation, self.annotations_dir)
                logger.info("Annotations saved to disk")

            # Auto-advance when playing
            if not self.is_paused:
                next_idx = self.current_frame_idx + 1
                if next_idx < self.total_frames:
                    self._read_frame(next_idx)
                else:
                    self.is_paused = True

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PROJECT 007 — Annotation GUI")
    parser.add_argument("video", help="Path to the video file")
    parser.add_argument("--annotations-dir", default="dataset/annotations", help="Annotations directory")
    args = parser.parse_args()

    gui = AnnotationGUI(args.video, args.annotations_dir)
    gui.run()
