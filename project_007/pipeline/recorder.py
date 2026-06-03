"""
PROJECT 007 — Video Clip Recorder
Ring-buffer video recorder that saves event clips with pre/post buffering.
Runs I/O in a background thread to prevent pipeline blocking.
"""

import os
import queue
import threading
import time
from collections import deque
import cv2

from config import (
    PRE_EVENT_BUFFER_SECONDS,
    POST_EVENT_BUFFER_SECONDS,
    SAVE_EVENT_CLIPS,
    TARGET_FPS,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class ClipRecorder:
    """
    Maintains a rolling buffer of frames. When triggered, saves the buffered
    frames plus future frames (post-buffer) to an MP4 file.
    """

    def __init__(self):
        self._enabled = SAVE_EVENT_CLIPS
        self._clips_dir = "clips"
        if self._enabled and not os.path.exists(self._clips_dir):
            os.makedirs(self._clips_dir)

        # Pre-event buffer (rolling)
        self._pre_buffer_size = int(TARGET_FPS * PRE_EVENT_BUFFER_SECONDS)
        self._frame_buffer = deque(maxlen=self._pre_buffer_size)

        # Post-event state
        self._is_recording = False
        self._post_frames_needed = 0
        self._current_clip_queue = None
        self._current_writer_thread = None

        logger.info(f"ClipRecorder initialised (enabled={self._enabled}, pre={PRE_EVENT_BUFFER_SECONDS}s, post={POST_EVENT_BUFFER_SECONDS}s)")

    def update(self, frame, timestamp: float) -> None:
        """
        Add a frame to the buffer. If currently recording an event, send it to the writer.
        """
        if not self._enabled:
            return

        # Keep a copy for the buffer to prevent drawing overlay artifacts if frame is modified later
        frame_copy = frame.copy()
        
        if self._is_recording:
            # Send to writer
            if self._current_clip_queue is not None:
                self._current_clip_queue.put(frame_copy)
            
            self._post_frames_needed -= 1
            if self._post_frames_needed <= 0:
                self._finish_recording()
        else:
            # Just roll the pre-buffer
            self._frame_buffer.append(frame_copy)

    def trigger(self, timestamp: float) -> None:
        """
        Trigger an event recording.
        """
        if not self._enabled or self._is_recording:
            return

        self._is_recording = True
        self._post_frames_needed = int(TARGET_FPS * POST_EVENT_BUFFER_SECONDS)
        
        clip_path = os.path.join(self._clips_dir, f"event_{int(timestamp)}.mp4")
        
        # Start a new writer thread
        self._current_clip_queue = queue.Queue()
        
        # Drain the current pre-buffer into the queue
        for f in self._frame_buffer:
            self._current_clip_queue.put(f)
            
        # We don't clear the frame buffer, it just naturally gets overwritten later.
        # But since we are recording, we stop appending to it until recording finishes.

        if len(self._frame_buffer) > 0:
            h, w = self._frame_buffer[0].shape[:2]
        else:
            h, w = 720, 1280  # Default fallback

        self._current_writer_thread = threading.Thread(
            target=self._writer_loop,
            args=(clip_path, w, h, self._current_clip_queue),
            daemon=True
        )
        self._current_writer_thread.start()
        logger.info(f"Event triggered! Recording clip to {clip_path}")

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    def _finish_recording(self) -> None:
        """Close out the current recording session."""
        self._is_recording = False
        if self._current_clip_queue is not None:
            self._current_clip_queue.put(None)  # Sentinel to stop writer
        self._current_clip_queue = None
        self._current_writer_thread = None
        logger.info("Clip recording finished.")

    def _writer_loop(self, filepath: str, width: int, height: int, q: queue.Queue):
        """Background thread to write frames to MP4."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(filepath, fourcc, TARGET_FPS, (width, height))

        if not writer.isOpened():
            logger.error(f"Failed to open VideoWriter for {filepath}")
            return

        try:
            while True:
                frame = q.get()
                if frame is None:
                    break
                writer.write(frame)
        except Exception as e:
            logger.error(f"Error writing clip: {e}")
        finally:
            writer.release()

    def close(self):
        """Ensure writer is stopped on shutdown."""
        if self._is_recording:
            self._finish_recording()
