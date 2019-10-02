import argparse
import datetime
import json
from pathlib import Path

import ibllib.io.params as params

VIDEOPC_PARAMS_FILE = Path(params.getfile('videopc_params'))


def create_videopc_params(force=False):
    if VIDEOPC_PARAMS_FILE.exists() and not force:
        print(f"{VIDEOPC_PARAMS_FILE} exists already, exiting...")
        return
    else:
        default = " [default: {}]: "
        data_folder_path = input(
            r"Where's your 'Subjects' data folder?" +
            default.format(r"D:\iblrig_data\Subjects")) or r"D:\iblrig_data\Subjects"
        remote_data_folder_path = input(
            r"Where's your REMOTE 'Subjects' data folder?" +
            default.format(r"\\iblserver.champalimaud.pt\ibldata\Subjects")) or r"\\iblserver.champalimaud.pt\ibldata\Subjects"
        body_cam_idx = input(
            "Please select the index of the BODY camera" + default.format(0)) or 0
        left_cam_idx = input(
            "Please select the index of the LEFT camera" + default.format(1)) or 1
        right_cam_idx = input(
            "Please select the index of the RIGHT camera" + default.format(2)) or 2

        param_dict = {
            'DATA_FOLDER_PATH': data_folder_path,
            'REMOTE_DATA_FOLDER_PATH': remote_data_folder_path,
            'BODY_CAM_IDX': body_cam_idx,
            'LEFT_CAM_IDX': left_cam_idx,
            'RIGHT_CAM_IDX': right_cam_idx,
        }
        params.write('videopc_params', param_dict)
        print(f"Created {VIDEOPC_PARAMS_FILE}")
        return


def load_videopc_params():
    if not VIDEOPC_PARAMS_FILE.exists():
        create_videopc_params()
    with open(VIDEOPC_PARAMS_FILE, 'r') as f:
        pars = json.load(f)
    return pars


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Setup video parmas file')
    parser.add_argument('-f', '--force', default=False, required=False, action='store_true',
        help='Update parameters')
    args = parser.parse_args()
    create_videopc_params(force=args.force)
