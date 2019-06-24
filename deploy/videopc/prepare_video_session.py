#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Thursday, May 2nd 2019, 5:41:56 pm
import argparse
import datetime
from pathlib import Path
import subprocess

import ibllib.io.params as params
from alf.folders import next_num_folder
import config_cameras as cams


VIDEOPC_PARAMS_FILE = Path(params.getfile('videopc_params'))


def load_videopc_params():
    if not VIDEOPC_PARAMS_FILE.exists():
        create_videopc_params()

    return params.read('videopc_params')


def create_videopc_params():
    if VIDEOPC_PARAMS_FILE.exists():
        print(f"{VIDEOPC_PARAMS_FILE} exists already, exiting...")
        return
    else:
        default = " [default: {}]: "
        data_folder_path = input(
            r"Where's your 'Subjects' data folder?" +
            default.format(r"C:\iblrig_data\Subjects")) or r"C:\iblrig_data\Subjects"
        body_cam_idx = input(
            "Please select the index of the BODY camera" + default.format(0)) or 0
        left_cam_idx = input(
            "Please select the index of the LEFT camera" + default.format(1)) or 1
        right_cam_idx = input(
            "Please select the index of the RIGHT camera" + default.format(2)) or 2

        param_dict = {
            'DATA_FOLDER_PATH': data_folder_path,
            'BODY_CAM_IDX': body_cam_idx,
            'LEFT_CAM_IDX': left_cam_idx,
            'RIGHT_CAM_IDX': right_cam_idx,
        }
        params.write('videopc_params', param_dict)
        print(f"Created {VIDEOPC_PARAMS_FILE}")
        return


def main(mouse):
    SUBJECT_NAME = mouse
    PARAMS = load_videopc_params()
    DATA_FOLDER = Path(PARAMS.DATA_FOLDER_PATH)
    VIDEOPC_FOLDER_PATH = Path(__file__).parent

    BONSAI = VIDEOPC_FOLDER_PATH / 'bonsai' / 'bin' / 'Bonsai64.exe'
    BONSAI_WORKFLOWS_PATH = BONSAI.parent.parent / 'workflows'
    STREAM_FILE = BONSAI_WORKFLOWS_PATH / 'three_cameras_stream.bonsai'
    RECORD_FILE = BONSAI_WORKFLOWS_PATH / 'three_cameras_record.bonsai'

    DATE = datetime.datetime.now().date().isoformat()
    NUM = next_num_folder(DATA_FOLDER / SUBJECT_NAME / DATE)

    SESSION_FOLDER = DATA_FOLDER / SUBJECT_NAME / DATE / NUM / 'raw_video_data'
    SESSION_FOLDER.mkdir(parents=True, exist_ok=True)
    print(f"Created {SESSION_FOLDER}")
    # Force trigger mode on all cams
    cams.enable_trigger_mode()
    print(f"Found {cams.NUM_CAMERAS} cameras. Trigger mode - ON")
    # Create filenames to call Bonsai
    filename = '_iblrig_{}Camera.raw.avi'
    # Open n start Bonsai view
    body = "-p:FileNameBody=" + SESSION_FOLDER / filename.format('body')
    left = "-p:FileNameLeft=" + SESSION_FOLDER / filename.format('left')
    right = "-p:FileNameRight=" + SESSION_FOLDER / filename.format('right')

    bodyidx = "-p:BodyCameraIndex=" + PARAMS.BODY_CAM_IDX
    leftidx = "-p:LeftCameraIndex=" + PARAMS.LEFT_CAM_IDX
    rightidx = "-p:RightCameraIndex=" + PARAMS.RIGHT_CAM_IDX

    start = '--start'  # --start-no-debug
    noboot = '--no-boot'
    # Open the streaming file and start
    subprocess.call([str(BONSAI), str(STREAM_FILE), start, noboot,
                     bodyidx, leftidx, rightidx])
    # Open the record_file no start
    subprocess.call([str(BONSAI), str(RECORD_FILE), noboot, body, left, right,
                     bodyidx, leftidx, rightidx])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Prepare video PC for ephys recording session')
    parser.add_argument('mouse', help='Mouse name')
    args = parser.parse_args()

    main(args.mouse)
