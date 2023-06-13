#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Thursday, May 2nd 2019, 5:41:56 pm
import argparse
import datetime
import subprocess
from pathlib import Path

import ibllib
from ibllib.pipes.misc import load_ephyspc_params
from ibllib.pipes.dynamic_pipeline import get_acquisition_description
import ibllib.io.session_params as sess_params
from one.alf.io import next_num_folder
from packaging.version import parse


def check_ibllib_version(ignore=False):
    bla = subprocess.run(
        "pip install ibllib==ver",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    ble = [x.decode("utf-8") for x in bla.stderr.rsplit()]
    # Latest version is at the end of the error message before the close parens
    latest_ibllib = parse([x.strip(")") for x in ble if ")" in x][0])
    if latest_ibllib != parse(ibllib.__version__):
        msg = (
            f"You are using ibllib {ibllib.__version__}, but the latest version is {latest_ibllib}"
        )
        print(f"{msg} - Please update ibllib")
        print("To update run: [conda activate iblenv] and [pip install -U ibllib]")
        if ignore:
            return
        raise Exception(msg)


def check_iblscripts_version(ignore=False):
    ps = subprocess.run(
        "git fetch; git status", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    cmd = subprocess.run(
        "git fetch && git status", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    psmsg = ""
    cmdmsg = ""
    if b"On branch master" not in ps.stdout:
        psmsg = psmsg + " You are not on the master branch. Please switch to the master branch"
    if b"On branch master" not in cmd.stdout:
        cmdmsg = cmdmsg + " You are not on the master branch. Please switch to the master branch"
    if b"Your branch is up to date" not in ps.stdout:
        psmsg = psmsg + " Your branch is not up to date. Please update your branch"
    if b"Your branch is up to date" not in cmd.stdout:
        cmdmsg = cmdmsg + " Your branch is not up to date. Please update your branch"

    if ignore:
        return
    if (psmsg == cmdmsg) and psmsg != "":
        raise Exception(psmsg)
    elif (psmsg != cmdmsg) and (psmsg == "" or cmdmsg == ""):
        return


def main(mouse, stub=None):
    SUBJECT_NAME = mouse
    PARAMS = load_ephyspc_params()
    DATA_FOLDER = Path(PARAMS["DATA_FOLDER_PATH"])

    DATE = datetime.datetime.now().date().isoformat()
    NUM = next_num_folder(DATA_FOLDER / SUBJECT_NAME / DATE)

    SESSION_FOLDER = DATA_FOLDER / SUBJECT_NAME / DATE / NUM / "raw_ephys_data"
    SESSION_FOLDER.mkdir(parents=True, exist_ok=True)
    print(f"Created {SESSION_FOLDER}")
    PROBE_00_FOLDER = SESSION_FOLDER / "probe00"
    PROBE_00_FOLDER.mkdir(parents=True, exist_ok=True)
    print(f"Created {PROBE_00_FOLDER}")
    PROBE_01_FOLDER = SESSION_FOLDER / "probe01"
    PROBE_01_FOLDER.mkdir(parents=True, exist_ok=True)
    print(f"Created {PROBE_01_FOLDER}")
    # Save the experiment description file (used for extraction and by the copy script)
    if stub:
        params = sess_params.read_params(stub)
    else:
        protocol = 'choice_world_recording'
        params = get_acquisition_description(protocol)
        # Remove tasks and other devices to make ephys PC stub
        params.pop('tasks')
        params['devices'] = {k: v for k, v in params['devices'].items() if k == 'neuropixel'}
    sess_params.prepare_experiment(f'{SUBJECT_NAME}/{DATE}/{NUM}', params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare ephys PC for ephys recording session")
    parser.add_argument("mouse", help="Mouse name")
    parser.add_argument(
        "--ignore-checks",
        default=False,
        required=False,
        action="store_true",
        help="Ignore ibllib and iblscripts checks",
    )
    parser.add_argument(
        "--stub",
        type=Path,
        required=False,
        help="Path to an experiment description stub file.",
    )
    args = parser.parse_args()

    check_ibllib_version(ignore=args.ignore_checks)
    check_iblscripts_version(ignore=args.ignore_checks)
    main(args.mouse, stub=args.stub)
