#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Thursday, May 2nd 2019, 5:41:56 pm
import argparse
import datetime
from pathlib import Path

import ibllib.io.params as params
from alf.folders import next_num_folder


EPHYSPC_PARAMS_FILE = Path(params.getfile('ephyspc_params'))


def load_ephyspc_params():
    if not EPHYSPC_PARAMS_FILE.exists():
        create_ephyspc_params()

    return params.read('ephyspc_params')


def create_ephyspc_params():
    if EPHYSPC_PARAMS_FILE.exists():
        print(f"{EPHYSPC_PARAMS_FILE} exists already, exiting...")
        return
    else:
        default = " [default: {}]: "
        data_folder_path = input(
            r"Where's your 'Subjects' data folder?" +
            default.format(r"C:\iblrig_data\Subjects")) or r"C:\iblrig_data\Subjects"

        param_dict = {
            'DATA_FOLDER_PATH': data_folder_path,
        }
        params.write('ephyspc_params', param_dict)
        print(f"Created {EPHYSPC_PARAMS_FILE}")
        return


def main(mouse):
    SUBJECT_NAME = mouse
    PARAMS = load_ephyspc_params()
    DATA_FOLDER = Path(PARAMS.DATA_FOLDER_PATH)

    DATE = datetime.datetime.now().date().isoformat()
    NUM = next_num_folder(DATA_FOLDER / SUBJECT_NAME / DATE)

    SESSION_FOLDER = DATA_FOLDER / SUBJECT_NAME / DATE / NUM / 'raw_ephys_data'
    SESSION_FOLDER.mkdir(parents=True, exist_ok=True)
    print(f"Created {SESSION_FOLDER}")
    PROBE_00_FOLDER = SESSION_FOLDER / 'probe_00'
    PROBE_00_FOLDER.mkdir(parents=True, exist_ok=True)
    print(f"Created {PROBE_00_FOLDER}")
    PROBE_01_FOLDER = SESSION_FOLDER / 'probe_01'
    PROBE_01_FOLDER.mkdir(parents=True, exist_ok=True)
    print(f"Created {PROBE_01_FOLDER}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Prepare ephys PC for ephys recording session')
    parser.add_argument('mouse', help='Mouse name')
    args = parser.parse_args()

    main(args.mouse)
