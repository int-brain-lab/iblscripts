#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Thursday, May 2nd 2019, 5:41:56 pm
import argparse
import datetime
import subprocess
from pathlib import Path

from alf.folders import next_num_folder

from ibllib.pipes.misc import load_videopc_params
import config_cameras as cams


def main(mouse: str, training_session: bool = False, new: bool = False) -> None:
    SUBJECT_NAME = mouse
    PARAMS = load_videopc_params()
    DATA_FOLDER = Path(PARAMS['DATA_FOLDER_PATH'])
    VIDEOPC_FOLDER_PATH = Path(__file__).parent

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
    filenamevideo = '_iblrig_{}Camera.raw.avi'
    filenameframedata = '_iblrig_{}Camera.FrameData.csv'
    # Define parameters to call bonsai
    bodyidx = "-p:BodyCameraIndex=" + str(PARAMS['BODY_CAM_IDX'])
    leftidx = "-p:LeftCameraIndex=" + str(PARAMS['LEFT_CAM_IDX'])
    rightidx = "-p:RightCameraIndex=" + str(PARAMS['RIGHT_CAM_IDX'])

    body = "-p:FileNameBody=" + str(SESSION_FOLDER / filenamevideo.format('body'))
    left = "-p:FileNameLeft=" + str(SESSION_FOLDER / filenamevideo.format('left'))
    right = "-p:FileNameRight=" + str(SESSION_FOLDER / filenamevideo.format('right'))

    bodydata = "-p:FileNameBodyData=" + str(SESSION_FOLDER / filenameframedata.format('body'))
    leftdata = "-p:FileNameLefDatat=" + str(SESSION_FOLDER / filenameframedata.format('left'))
    rightdata = "-p:FileNameRightData=" + str(SESSION_FOLDER / filenameframedata.format('right'))

    start = '--start'  # --start-no-debug
    noboot = '--no-boot'
    noeditor = '--no-editor'
    # Force trigger mode on all cams
    cams.disable_trigger_mode()
    print(f"Found {cams.NUM_CAMERAS} cameras. Trigger mode - OFF")
    # Open the streaming file and start
    subprocess.call([str(BONSAI), str(SETUP_FILE), start, noboot,
                     bodyidx, leftidx, rightidx])
    # Force trigger mode on all cams
    cams.enable_trigger_mode()
    print(f"Found {cams.NUM_CAMERAS} cameras. Trigger mode - ON")
    # Open the record_file no start
    subprocess.call([str(BONSAI), str(RECORD_FILE), noboot, body, left, right,
                     bodyidx, leftidx, rightidx, bodydata, leftdata, rightdata])
    # subprocess.call(['python', '-c', 'import os; print(os.getcwd())'])
    subprocess.call(['python', 'video_lengths.py', str(SESSION_FOLDER.parent)])
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
