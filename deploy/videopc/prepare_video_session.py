#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Thursday, May 2nd 2019, 5:41:56 pm
import argparse
import datetime
import os
import subprocess
from pathlib import Path

from one.alf.io import next_num_folder
from ibllib.pipes.misc import load_videopc_params

import config_cameras as cams


def main(mouse: str, training_session: bool = False) -> None:
    SUBJECT_NAME = mouse
    PARAMS = load_videopc_params()
    DATA_FOLDER = Path(PARAMS['DATA_FOLDER_PATH'])
    VIDEOPC_FOLDER_PATH = Path(__file__).absolute().parent

    BONSAI = VIDEOPC_FOLDER_PATH / 'bonsai' / 'bin' / 'Bonsai64.exe'
    BONSAI_WORKFLOWS_PATH = BONSAI.parent.parent / 'workflows'
    SETUP_FILE = BONSAI_WORKFLOWS_PATH / 'three_cameras_setup.bonsai'
    RECORD_FILE = BONSAI_WORKFLOWS_PATH / 'three_cameras_record.bonsai'
    if training_session:
        RECORD_FILE = BONSAI_WORKFLOWS_PATH / 'three_cameras_record_biasedCW.bonsai'

    DATE = datetime.datetime.now().date().isoformat()
    NUM = next_num_folder(DATA_FOLDER / SUBJECT_NAME / DATE)

    SESSION_FOLDER = DATA_FOLDER / SUBJECT_NAME / DATE / NUM / 'raw_video_data'
    SESSION_FOLDER.mkdir(parents=True, exist_ok=True)
    print(f"Created {SESSION_FOLDER}")
    # Create filenames to call Bonsai
    filename = '_iblrig_{}Camera.raw.avi'
    filenamets = '_iblrig_{}Camera.timestamps.ssv'
    filenamefc = '_iblrig_{}Camera.frame_counter.bin'
    filenameGPIO = '_iblrig_{}Camera.GPIO.bin'
    # Open n start Bonsai view
    body = "-p:FileNameBody=" + str(SESSION_FOLDER / filename.format('body'))
    left = "-p:FileNameLeft=" + str(SESSION_FOLDER / filename.format('left'))
    right = "-p:FileNameRight=" + str(SESSION_FOLDER / filename.format('right'))
    bodyts = "-p:FileNameBodyTimestamps=" + str(SESSION_FOLDER / filenamets.format('body'))
    leftts = "-p:FileNameLeftTimestamps=" + str(SESSION_FOLDER / filenamets.format('left'))
    rightts = "-p:FileNameRightTimestamps=" + str(SESSION_FOLDER / filenamets.format('right'))
    bodyfc = "-p:FileNameBodyFrameCounter=" + str(SESSION_FOLDER / filenamefc.format('body'))
    leftfc = "-p:FileNameLeftFrameCounter=" + str(SESSION_FOLDER / filenamefc.format('left'))
    rightfc = "-p:FileNameRightFrameCounter=" + str(SESSION_FOLDER / filenamefc.format('right'))
    bodyGPIO = "-p:FileNameBodyGPIO=" + str(SESSION_FOLDER / filenameGPIO.format('body'))
    leftGPIO = "-p:FileNameLeftGPIO=" + str(SESSION_FOLDER / filenameGPIO.format('left'))
    rightGPIO = "-p:FileNameRightGPIO=" + str(SESSION_FOLDER / filenameGPIO.format('right'))

    bodyidx = "-p:BodyCameraIndex=" + str(PARAMS['BODY_CAM_IDX'])
    leftidx = "-p:LeftCameraIndex=" + str(PARAMS['LEFT_CAM_IDX'])
    rightidx = "-p:RightCameraIndex=" + str(PARAMS['RIGHT_CAM_IDX'])

    start = '--start'  # --start-no-debug
    noboot = '--no-boot'

    # Force trigger mode on all cams
    cams.enable_trigger_mode()
    print(f"Found {cams.NUM_CAMERAS} cameras. Trigger mode - ON")
    # Open the streaming file and start
    here = os.getcwd()
    os.chdir(BONSAI_WORKFLOWS_PATH)
    subprocess.call([str(BONSAI), str(SETUP_FILE), start, noboot,
                     bodyidx, leftidx, rightidx])
    os.chdir(here)
    # Open the record_file no start
    subprocess.call([str(BONSAI), str(RECORD_FILE), noboot, body, left, right,
                     bodyidx, leftidx, rightidx, bodyts, leftts, rightts,
                     bodyfc, leftfc, rightfc, bodyGPIO, leftGPIO, rightGPIO])
    # Create a transfer_me.flag file
    open(SESSION_FOLDER.parent / 'transfer_me.flag', 'w')
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Prepare video PC for video recording session')
    parser.add_argument('mouse', help='Mouse name')
    parser.add_argument(
        '-t', '--training', default=False, required=False, action='store_true',
        help='Launch video workflow for biasedCW sessionon ephys rig.')
    args = parser.parse_args()
    # print(args)
    # print(type(args.mouse), type(args.training))
    main(args.mouse, training_session=args.training)
