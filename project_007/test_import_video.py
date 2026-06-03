"""
Tests for dataset_tools.import_video
"""

import os
import shutil
from pathlib import Path

import cv2
import numpy as np

from dataset_tools.import_video import import_video
from dataset_tools.session_manager import SessionManager


def test_import_video():
    # Setup test video
    test_video = "test_import_dummy.mp4"
    test_dataset = "test_dataset"
    
    if Path(test_dataset).exists():
        shutil.rmtree(test_dataset)

    # Create dummy video
    out = cv2.VideoWriter(test_video, cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
    for _ in range(30):
        out.write(np.zeros((480, 640, 3), dtype=np.uint8))
    out.release()

    try:
        # Test 1: Invalid category
        success = import_video(test_video, "invalid_category", test_dataset)
        assert not success, "Import should fail for invalid category"

        # Test 2: Invalid file
        success = import_video("nonexistent.mp4", "interaction", test_dataset)
        assert not success, "Import should fail for non-existent file"

        # Test 3: Successful import
        success = import_video(test_video, "interaction", test_dataset)
        assert success, "Import should succeed for valid video and category"

        # Verify output
        sm = SessionManager(test_dataset)
        sessions = sm.list_sessions("interaction")
        assert len(sessions) == 1, "There should be 1 session in interaction category"
        
        sess = sessions[0]
        assert sess["category"] == "interaction"
        assert sess["original_filename"] == test_video
        assert sess["fps"] == 30.0
        assert sess["total_frames"] == 30
        assert "duration_seconds" in sess
        assert "resolution" in sess
        
        # Verify manifest was generated
        manifest_path = Path(test_dataset) / "metadata" / "dataset_manifest.json"
        assert manifest_path.exists(), "Manifest should be generated"

        print("test_import_video passed successfully.")
    finally:
        # Cleanup
        if Path(test_video).exists():
            os.remove(test_video)
        if Path(test_dataset).exists():
            shutil.rmtree(test_dataset)

if __name__ == "__main__":
    test_import_video()
