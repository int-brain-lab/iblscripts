import argparse
import datetime
from pathlib import Path

import ibllib.io.params as params
from alf.folders import next_num_folder
import json


EPHYSPC_PARAMS_FILE = Path(params.getfile('ephyspc_params'))



def create_ephyspc_params(force=False):
    if EPHYSPC_PARAMS_FILE.exists() and not force:
        print(f"{EPHYSPC_PARAMS_FILE} exists already, exiting...")
        return
    else:
        default = " [default: {}]: "
        data_folder_path = input(
            r"Where's your LOCAL 'Subjects' data folder?" +
            default.format(r"D:\iblrig_data\Subjects")) or r"D:\iblrig_data\Subjects"
        remote_data_folder_path = input(
            r"Where's your REMOTE 'Subjects' data folder?" +
            default.format(r"\\iblserver.champalimaud.pt\ibldata\Subjects")) or r"\\iblserver.champalimaud.pt\ibldata\Subjects"
        param_dict = {
            'DATA_FOLDER_PATH': data_folder_path,
            'REMOTE_DATA_FOLDER_PATH': remote_data_folder_path,
        }
        params.write('ephyspc_params', param_dict)
        print(f"Created {EPHYSPC_PARAMS_FILE}")
        return


def load_ephyspc_params():
    if not EPHYSPC_PARAMS_FILE.exists():
        create_ephyspc_params()
    with open(EPHYSPC_PARAMS_FILE, 'r') as f:
        pars = json.load(f)
    return pars


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Setup ephys parmas file')
    parser.add_argument('-f', '--force', default=False, required=False, action='store_true',
        help='Update parameters')
    args = parser.parse_args()
    create_ephyspc_params(force=args.force)