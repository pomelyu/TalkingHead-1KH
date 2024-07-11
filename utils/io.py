import json
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def load_video_start_and_end(video_path: str) -> Tuple[np.ndarray, np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Fails to open the video: {video_path}")

    _, first_frame = cap.read()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    _, mid_frame = cap.read()

    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)

    _, last_frame = cap.read()

    cap.release()
    if last_frame is None or first_frame.shape != last_frame.shape:
        raise RuntimeError(f"Fails to get last frame")

    return first_frame, mid_frame, last_frame, total_frames

def write_formatted_json(path: str, data: dict) -> None:
    with Path(path).open("w+") as f:
        json.dump(data, f, indent=2, cls=_NumpyEncoder)

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
