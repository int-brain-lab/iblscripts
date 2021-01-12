#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Tuesday, January 12th 2021, 5:48:08 pm
"""
Given a specific video session_path will count and printout the number of frames for the video
the GPIO pin states and the frame counter files
"""

import sys
from pathlib import Path
import numpy as np
import cv2


def load_embedded_frame_data(session_path, camera: str, raw=False):
    """
    :param session_path:
    :param camera: The specific camera to load, one of ('left', 'right', 'body')
    :param raw: If True the raw data are returned without preprocessing (thresholding, etc.)
    :return: The frame counter, the pin state
    """
    if session_path is None:
        return None, None
    raw_path = Path(session_path).joinpath('raw_video_data')
    # Load frame count
    count_file = raw_path / f'_iblrig_{camera}Camera.frame_counter.bin'
    count = np.fromfile(count_file, dtype=np.float64).astype(int) if count_file.exists() else None
    if not (count is None or raw):
        count -= count[0]  # start from zero
    # Load pin state
    pin_file = raw_path / f'_iblrig_{camera}Camera.GPIO.bin'
    pin_state = np.fromfile(pin_file, dtype=np.float64).astype(int) if pin_file.exists() else None
    if not (pin_state is None or raw):
        pin_state = pin_state > PIN_STATE_THRESHOLD
    return count, pin_state


def get_video_length(video_path):
    """
    Returns video length
    :param video_path: A path to the video
    :return:
    """
    is_url = isinstance(video_path, str) and video_path.startswith('http')
    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), f'Failed to open video file {video_path}'
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return length


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("I need a session_path as input...")
    session_path = Path(sys.argv[1])
    video_lengths = [get_video_length(p) for p in session_path.rglob('*.avi')]
    array_lengths = [(a.size, b.size) for a, b in [load_embedded_frame_data(session_path, cam, raw=True) for cam in ('left', 'right', 'body')]]
    frame_counter_lengths = [x[0] for x in array_lengths]
    GPIO_state_lengths = [x[1] for x in array_lengths]
    print('\n',
        sys.argv[1], '\n',
        sorted(video_lengths), '<-- Video lengths', '\n',
        sorted(frame_counter_lengths), '<-- Frame counter lengths', '\n',
        sorted(GPIO_state_lengths), '<-- GPIO state lengths', '\n',
    )