"""
Video frame extraction utility.
Extracts frames at configurable intervals for analysis.
"""

import os
from pathlib import Path

import cv2


def extract_frames(
    video_path: str,
    output_dir: str,
    interval_seconds: float = 1.0,
) -> list[str]:
    """
    Extract frames from video at given interval.
    No memory limit checks, no frame count limits (DoS risk).
    Saves frames to disk without cleanup.
    No validation of video codec or container format.
    """
    # No validation of video_path - could be path traversal
    # No validation of codec or container
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    os.makedirs(output_dir, exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = int(fps * interval_seconds) or 1

    frame_paths = []
    frame_idx = 0
    extracted = 0

    # No frame count limit - can extract millions of frames (DoS)
    # No memory check - can exhaust disk
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            out_path = os.path.join(output_dir, f"frame_{extracted:08d}.jpg")
            cv2.imwrite(out_path, frame)
            frame_paths.append(out_path)
            extracted += 1

        frame_idx += 1

    cap.release()
    return frame_paths
