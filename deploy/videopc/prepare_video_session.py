#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Thursday, May 2nd 2019, 5:41:56 pm
import argparse
import datetime
import json
from pathlib import Path

import ibllib.io.params as params
from alf.folders import next_num_folder

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
        bonsai_path = input(
            r"Where's Bonsai64.exe? " +
            default.format(r"C:\iblrig\Bonsai2.3\Bonsai64.exe")
        ) or r"C:\iblrig\Bonsai2.3\Bonsai64.exe"
        bonsai_workflows_path = input(
            r"Where's the videopc workflows folder? " +
            default.format(r"C:\iblrig\Bonsai2.3\Bonsai64.exe")
        ) or r"C:\iblrig\Bonsai2.3\Bonsai64.exe"
        body_cam_idx = input(
            "Please select the index of the BODY camera" + default.format(0)) or 0
        left_cam_idx = input(
            "Please select the index of the LEFT camera" + default.format(1)) or 1
        right_cam_idx = input(
            "Please select the index of the RIGHT camera" + default.format(2)) or 2

        param_dict = {
            'DATA_FOLDER_PATH': data_folder_path,
            'BONSAI_PATH': bonsai_path,
            'BONSAI_WORKFLOWS_PATH' : bonsai_workflows_path
            'BODY_CAM_IDX': body_cam_idx,
            'LEFT_CAM_IDX': left_cam_idx,
            'RIGHT_CAM_IDX': right_cam_idx,
        }
        params.write('videopc_params', param_dict)
        print(f"Created {VIDEOPC_PARAMS_FILE}")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare videoPC for session')
    parser.add_argument('mouse', help='Mouse name')
    args = parser.parse_args()

    # get from config file?
    PARAMS = load_videopc_params()
    DATA_FOLDER = Path(PARAMS.DATA_FOLDER_PATH)
    # get from user
    SUBJECT_NAME = args.mouse

    DATE = datetime.datetime.now().date().isoformat()
    NUM = next_num_folder(DATA_FOLDER / SUBJECT_NAME / DATE)

    SESSION_FOLDER = DATA_FOLDER / SUBJECT_NAME / DATE / NUM / 'raw_video_data'
    SESSION_FOLDER.mkdir(parents=True, exist_ok=True)
    print(f"Created {SESSION_FOLDER}")
    # Force trigger mode on all cams
    import ibllib.pipes.videopc.config_cameras as cams
    cams.enable_trigger_mode(cams.CAM_LIST)
    # Create filenames to call Bonsai
    filename = '_iblrig_{}Camera.raw.avi'
    body_fname = SESSION_FOLDER / filename.format('body')
    left_fname = SESSION_FOLDER / filename.format('left')
    right_fname = SESSION_FOLDER / filename.format('right')
    # get Bonsai install path
    BONSAI = PARAMS.BONSAI_PATH
    # get idxs for cams
    body_cam_idx = 0
    left_cam_idx = 1
    right_cam_idx = 2
    # Open n start Bonsai view
    # Open don't start Bonsai recording
