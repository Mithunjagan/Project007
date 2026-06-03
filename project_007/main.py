"""
PROJECT 007 — Main Entry Point (P2.5)
Real-time human motion analysis pipeline.

Live camera mode uses async threads (sync_mode=False).
For evaluation/replay, use evaluation/replay_engine.py (sync_mode=True).
"""

import sys
import time

import cv2
import torch

from config import (
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_FALLBACK_WIDTH,
    CAMERA_FALLBACK_HEIGHT,
    WINDOW_TITLE,
)
from pipeline.core import PipelineRunner
from utils.logger import get_logger

logger = get_logger("main")

def _print_gpu_diagnostics() -> None:
    logger.info("=" * 55)
    logger.info("GPU DIAGNOSTICS")
    logger.info("=" * 55)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024 ** 3)
        logger.info(f"  CUDA available   : True")
        logger.info(f"  GPU              : {props.name}")
        logger.info(f"  VRAM (total)     : {vram_gb:.1f} GB")
        logger.info(f"  CUDA version     : {torch.version.cuda}")
    else:
        logger.warning("  CUDA NOT available — falling back to CPU.")
        logger.warning("  Performance will be significantly reduced.")

    logger.info("=" * 55)


def _open_webcam() -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        logger.error(f"Cannot open webcam at index {CAMERA_INDEX}")
        return None

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if actual_w != CAMERA_WIDTH or actual_h != CAMERA_HEIGHT:
        logger.warning(
            f"Requested {CAMERA_WIDTH}x{CAMERA_HEIGHT}, "
            f"got {actual_w}x{actual_h}. "
            f"Falling back to {CAMERA_FALLBACK_WIDTH}x{CAMERA_FALLBACK_HEIGHT}."
        )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FALLBACK_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FALLBACK_HEIGHT)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(f"Webcam opened: {actual_w}x{actual_h} (buffer=1)")
    return cap


def main() -> None:
    logger.info("PROJECT 007 — Motion Analysis Pipeline starting (P2.5) …")
    _print_gpu_diagnostics()

    cap = _open_webcam()
    if cap is None:
        logger.error("Failed to open webcam. Exiting.")
        sys.exit(1)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    runner = PipelineRunner(frame_width, frame_height)

    frame_count: int = 0
    logger.info("Pipeline running.  Press 'q' to exit.")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                logger.warning("Frame read failed — retrying …")
                time.sleep(0.01)
                continue

            frame_count += 1
            now_mono = time.perf_counter()
            now_wall = time.time()

            annotated, _, _ = runner.step(frame, frame_count, now_mono, now_wall)

            cv2.imshow(WINDOW_TITLE, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                logger.info("Exit requested by user.")
                break

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — shutting down …")
    except Exception as exc:
        logger.error(f"Unexpected error in main loop: {exc}", exc_info=True)
    finally:
        logger.info("Cleaning up resources …")
        runner.cleanup()
        cap.release()
        cv2.destroyAllWindows()
        logger.info("PROJECT 007 — Pipeline shut down cleanly.")


if __name__ == "__main__":
    main()
