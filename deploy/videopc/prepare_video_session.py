#!/usr/bin/env python
"""Prepare video PC for video recording session.

NB: This requires iblrigv8 installed.
"""
import argparse
import datetime
import subprocess
from pathlib import Path
import traceback

import ibllib
from ibllib.pipes.misc import load_videopc_params
from one.alf.io import next_num_folder
from packaging.version import parse
from iblrig.transfer_experiments import VideoCopier
from iblrig.path_helper import load_settings_yaml

import config_cameras as cams
from video_lengths import main as len_files


def get_activated_environment(ignore=False):
    envs = (
        subprocess.run(
            'conda env list',
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        .stdout.decode('utf-8')
        .strip()
        .split()
    )
    current_env = envs[envs.index('*') - 1]

    return current_env


def check_ibllib_version(ignore=False):
    try:
        bla = subprocess.run(
            'pip install ibllib==ver',
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        ble = [x.decode('utf-8') for x in bla.stderr.rsplit()]
        # Latest version is at the end of the error message before the close parens
        latest_ibllib = parse([x.strip(')') for x in ble if ')' in x][0])
        if latest_ibllib != parse(ibllib.__version__):
            msg = (
                f'You are using ibllib {ibllib.__version__}, but the latest version is {latest_ibllib}'
            )
            print(f'{msg} - Please update ibllib')
            print('To update run: [conda activate iblenv] and [pip install -U ibllib]')
            if ignore:
                return
            raise Exception(msg)
    except Exception:
        print(traceback.format_exc())


def check_iblscripts_version(ignore=False):
    ps = subprocess.run(
        'git fetch; git status', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    cmd = subprocess.run(
        'git fetch && git status', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    psmsg = ''
    cmdmsg = ''
    if b'On branch master' not in ps.stdout:
        psmsg = psmsg + ' You are not on the master branch. Please switch to the master branch'
    if b'On branch master' not in cmd.stdout:
        cmdmsg = cmdmsg + ' You are not on the master branch. Please switch to the master branch'
    if b'Your branch is up to date' not in ps.stdout:
        psmsg = psmsg + ' Your branch is not up to date. Please update your branch'
    if b'Your branch is up to date' not in cmd.stdout:
        cmdmsg = cmdmsg + ' Your branch is not up to date. Please update your branch'

    if ignore:
        return
    if (psmsg == cmdmsg) and psmsg != '':
        raise Exception(psmsg)
    elif (psmsg != cmdmsg) and (psmsg == '' or cmdmsg == ''):
        return


def update_repo():
    subprocess.run(
        'git fetch; git checkout master; git pull',
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def update_ibllib(env='iblenv'):
    subprocess.run(
        f'bash -c "conda activate {env}; pip install -U ibllib"',
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def launch_three_videos_acquisition(mouse: str, training_session: bool = False, new: bool = False) -> None:
    """

    Parameters
    ----------
    mouse : str
        The subject name.
    training_session : bool
        If true, assumes single camera setup.
    new : bool
        Currently unused. I assume this would determine whether to iterate the session number.

    TODO drop videopc params in favour of iblrig hardware params
    """
    SUBJECT_NAME = mouse
    PARAMS = load_videopc_params()
    DATA_FOLDER = Path(PARAMS['DATA_FOLDER_PATH'])
    rig_settings = load_settings_yaml('iblrig_settings')
    assert DATA_FOLDER == Path(rig_settings['iblrig_local_data_path'])
    VIDEOPC_FOLDER_PATH = Path(__file__).absolute().parent

    BONSAI = VIDEOPC_FOLDER_PATH / 'bonsai' / 'bin' / 'Bonsai.exe'
    BONSAI_WORKFLOWS_PATH = BONSAI.parent.parent / 'workflows'
    SETUP_FILE = BONSAI_WORKFLOWS_PATH / 'EphysRig_SetupCameras.bonsai'
    RECORD_FILE = BONSAI_WORKFLOWS_PATH / 'EphysRig_SaveVideo_EphysTasks.bonsai'
    if training_session:
        RECORD_FILE = BONSAI_WORKFLOWS_PATH / 'EphysRig_SaveVideo_TrainingTasks.bonsai'

    DATE = datetime.datetime.now().date().isoformat()
    NUM = next_num_folder(DATA_FOLDER / SUBJECT_NAME / DATE)

    session_path = DATA_FOLDER / SUBJECT_NAME / DATE / NUM
    collection_folder = session_path / 'raw_video_data'
    collection_folder.mkdir(parents=True, exist_ok=True)
    print(f'Created {collection_folder}')

    # Save the stub files locally and in the remote repo for future copy script to use
    acq_desc = {
        'devices': {
            'cameras': {
                'right': {'collection': 'raw_video_data', 'sync_label': 'audio'},
                'body': {'collection': 'raw_video_data', 'sync_label': 'audio'},
                'left': {'collection': 'raw_video_data', 'sync_label': 'audio'},
            },
        },
        'version': '1.0.0'
    }
    remote = Path(rig_settings['iblrig_remote_data_path'])
    copier = VideoCopier(session_path=session_path, remote_subjects_folder=remote)
    copier.initialize_experiment(acquisition_description=acq_desc)

    # Create filenames to call Bonsai
    filenamevideo = '_iblrig_{}Camera.raw.avi'
    filenameframedata = '_iblrig_{}Camera.frameData.bin'
    # Define parameters to call bonsai
    bodyidx = '-p:BodyCameraIndex=' + str(PARAMS['BODY_CAM_IDX'])
    leftidx = '-p:LeftCameraIndex=' + str(PARAMS['LEFT_CAM_IDX'])
    rightidx = '-p:RightCameraIndex=' + str(PARAMS['RIGHT_CAM_IDX'])

    body = '-p:FileNameBody=' + str(collection_folder / filenamevideo.format('body'))
    left = '-p:FileNameLeft=' + str(collection_folder / filenamevideo.format('left'))
    right = '-p:FileNameRight=' + str(collection_folder / filenamevideo.format('right'))

    bodydata = '-p:FileNameBodyData=' + str(collection_folder / filenameframedata.format('body'))
    leftdata = '-p:FileNameLeftData=' + str(collection_folder / filenameframedata.format('left'))
    rightdata = '-p:FileNameRightData=' + str(collection_folder / filenameframedata.format('right'))

    start = '--start'  # --start-no-debug
    noboot = '--no-boot'
    # noeditor = '--no-editor'
    # Force trigger mode on all cams
    cams.disable_trigger_mode()
    # Open the streaming file and start
    subprocess.call([str(BONSAI), str(SETUP_FILE), start, noboot, bodyidx, leftidx, rightidx], cwd=str(BONSAI_WORKFLOWS_PATH))
    # Force trigger mode on all cams
    cams.enable_trigger_mode()
    # Open the record_file start and wait for manual trigger mode disabling
    rec = subprocess.Popen(
        [
            str(BONSAI),
            str(RECORD_FILE),
            noboot,
            start,
            body,
            left,
            right,
            bodyidx,
            leftidx,
            rightidx,
            bodydata,
            leftdata,
            rightdata,
        ],
        cwd=str(BONSAI_WORKFLOWS_PATH)
    )
    print('\nPRESS ENTER TO START CAMERAS' * 10)
    untrigger = input('') or 1
    print('ENTER key press detected, starting cameras...')
    if untrigger:
        cams.disable_trigger_mode()
        print('\nTo terminate video acquisition, please stop and close Bonsai workflow.')
    rec.wait()
    # Check lengths
    len_files(collection_folder.parent, display=True)  # Will printout the results
    # XXX: Consider not creating the transfer flag if lengths are not good:
    #       will impact the transfer script as it requires both transfers to be completed before
    #       creating the raw_session.flag
    # Create a transfer_me.flag file
    open(collection_folder.parent / 'transfer_me.flag', 'w')
    print(f'\nCreated transfer flag for session {collection_folder.parent}')
    print('Video acquisition session finished.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare video PC for video recording session')
    parser.add_argument('mouse', help='Mouse name')
    parser.add_argument(
        '-t',
        '--training',
        default=False,
        required=False,
        action='store_true',
        help='Launch video workflow for biasedCW session on ephys rig.',
    )
    parser.add_argument(
        '--ignore-checks',
        default=False,
        required=False,
        action='store_true',
        help='Ignore ibllib and iblscripts checks',
    )
    args = parser.parse_args()
    # print(args)
    # print(type(args.mouse), type(args.training))
    check_ibllib_version(ignore=args.ignore_checks)
    check_iblscripts_version(ignore=args.ignore_checks)
    launch_three_videos_acquisition(args.mouse, training_session=args.training)
