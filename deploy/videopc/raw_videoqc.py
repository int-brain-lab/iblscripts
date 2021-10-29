#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @File: videopc\raw_videoqc.py
# @Author: Niccolo' Bonacchi (@nbonacchi)
# @Date: Thursday, October 28th 2021, 3:37:12 pm
import argparse
import sys
from pathlib import Path

from ibllib.qc.camera import CameraQC
from one.api import ONE


def main(session_path, display=False):
    qc = CameraQC(session_path, 'left', one=ONE(mode='local', silent=True), stream=False)
    qc.video_path = session_path.joinpath('raw_video_data', f'_iblrig_{qc.label}Camera.raw.avi')
    qc.load_video_data()
    # Run frame checks
    bright_outcome = qc.check_brightness(display=display)
    pos_outcome = qc.check_position(display=display)
    focus_outcome = qc.check_focus(display=display)
    print(f"Brightness: {bright_outcome}\nPosition: {pos_outcome}\nFocus: {focus_outcome}")
    # Run meta data checks
    fh_outcome = qc.check_file_headers()
    fr_outcome = qc.check_framerate()
    res_outcome = qc.check_resolution()
    print(f"File headers: {fh_outcome}\nFrame rate: {fr_outcome}\nResolution: {res_outcome}")


if __name__ == '__main__':
    # session_path = Path('C:\\iblrig_data\\Subjects\\_iblrig_test_mouse\\2021-03-30\\007')
    # main(session_path)
    parser = argparse.ArgumentParser(
        description='Run video QC on raw files')
    parser.add_argument('session_path', help='Local session path')
    parser.add_argument(
        '-d', '--display',
        action="store_true",
        default=False,
        required=False,
        help='Whether to display plots'
    )
    args = parser.parse_args()
    main(Path(args.session_path), display=args.display)

