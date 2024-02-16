#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Thursday, May 2nd 2019, 5:41:56 pm
import argparse
import datetime
import subprocess
from pathlib import Path
from packaging import version
import warnings
import os

from one.alf.io import next_num_folder
from iblutil.util import setup_logger

import ibllib
from ibllib.pipes.misc import load_ephyspc_params


def check_ibllib_version(ignore=False):
    bla = subprocess.run(
        "pip install ibllib==ver",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    ble = [x.decode("utf-8") for x in bla.stderr.rsplit()]
    # Latest version is at the end of the error message before the close parens
    latest_ibllib = version.parse([x.strip(")") for x in ble if ")" in x][0])
    if latest_ibllib != version.parse(ibllib.__version__):
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


def _v8_check():
    """Return True if iblrigv8 installed."""
    try:
        import iblrig
        return version.parse(iblrig.__version__) >= version.Version('8.0.0')
    except ModuleNotFoundError:
        return False


def main_v8(mouse, debug=False):
    # from iblrig.base_tasks import EmptySession
    from iblrig.transfer_experiments import EphysCopier

    log = setup_logger(name='iblrig', level=10 if debug else 20)

    PARAMS = load_ephyspc_params()
    iblrig_settings = {
        'iblrig_local_data_path': Path(PARAMS['DATA_FOLDER_PATH']),
        'iblrig_local_subjects_path': Path(PARAMS['DATA_FOLDER_PATH']),
        'iblrig_remote_data_path': Path(PARAMS['REMOTE_DATA_FOLDER_PATH']),
        'iblrig_remote_subjects_path': Path(PARAMS['REMOTE_DATA_FOLDER_PATH'], 'Subjects'),
    }

    # if PARAMS.get('PROBE_TYPE_00', '3B') != '3B' or PARAMS.get('PROBE_TYPE_01', '3B') != '3B':
    #     raise NotImplementedError('Only 3B probes supported.')
    n_probes = sum(k.lower().startswith('probe_type_') for k in PARAMS)

    # FIXME this isn't working!
    # session = EmptySession(subject=mouse, interactive=False, iblrig_settings=iblrig_settings)
    # session_path = session.paths.SESSION_FOLDER
    date = datetime.datetime.now().date().isoformat()
    num = next_num_folder(iblrig_settings['iblrig_local_subjects_path'] / mouse / date)
    session_path = iblrig_settings['iblrig_local_subjects_path'] / mouse / date / num
    raw_data_folder = session_path.joinpath('raw_ephys_data')
    raw_data_folder.mkdir(parents=True, exist_ok=True)

    log.info('Created %s', raw_data_folder)
    REMOTE_SUBJECT_FOLDER = iblrig_settings['iblrig_remote_subjects_path']

    for n in range(n_probes):
        probe_folder = raw_data_folder / f'probe{n:02}'
        probe_folder.mkdir(exist_ok=True)
        log.info('Created %s', probe_folder)

    # Save the stub files locally and in the remote repo for future copy script to use
    copier = EphysCopier(session_path=session_path, remote_subjects_folder=REMOTE_SUBJECT_FOLDER)
    copier.initialize_experiment(nprobes=n_probes)

    ans = input('Type "abort" to cancel or just press return to finalize\n')
    if ans.lower().strip() == 'abort' and not any(filter(Path.is_file, raw_data_folder.rglob('*'))):
        log.warning('Removing %s', raw_data_folder)
        for d in raw_data_folder.iterdir():  # Remove probe folders
            d.rmdir()
        raw_data_folder.rmdir()  # remove collection
        # Delete whole session folder?
        session_files = list(session_path.rglob('*'))
        if len(session_files) == 1 and session_files[0].name.startswith('_ibl_experiment.description'):
            ans = input(f'Remove empty session {"/".join(session_path.parts[-3:])}? [y/N]\n')
            if (ans.strip().lower() or 'n')[0] == 'y':
                log.warning('Removing %s', session_path)
                log.debug('Removing %s', session_files[0])
                session_files[0].unlink()
                session_path.rmdir()
                # Remove remote exp description file
                log.debug('Removing %s', copier.file_remote_experiment_description)
                copier.file_remote_experiment_description.unlink()
    else:
        session_path.joinpath('transfer_me.flag').touch()


def main(mouse, **_):
    warnings.warn('For iblrigv8 behaviour sessions, install iblrigv8 on this PC also', FutureWarning)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare ephys PC for ephys recording session")
    parser.add_argument("mouse", help="Mouse name")
    parser.add_argument("--debug", help="Debug mode", action="store_true")
    parser.add_argument(
        "--ignore-checks",
        default=False,
        required=False,
        action="store_true",
        help="Ignore ibllib and iblscripts checks",
    )
    args = parser.parse_args()

    check_ibllib_version(ignore=args.ignore_checks)
    check_iblscripts_version(ignore=args.ignore_checks)
    fcn = main_v8 if _v8_check() else main
    fcn(args.mouse, debug=args.debug)
