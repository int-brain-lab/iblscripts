from pathlib import Path
import logging
import os
from datetime import datetime
import shutil
from oneibl.one import ONE
import alf.io
import ibllib.io.raw_data_loaders as raw
from ibllib.pipes.local_server import _get_lab
ROOT_PATH = Path('/mnt/s0/Data/Subjects')

_logger = logging.getLogger('ibllib')


def correct_ephys_manual_video_copies():
    """
    """
    for flag in ROOT_PATH.rglob('ephys_data_transferred.flag'):
        video = True
        passive = True
        behaviour = True
        session_path = alf.io.get_session_path(flag)
        avi_files = list(session_path.joinpath('raw_video_data').glob('*.avi'))

        if len(avi_files) < 3:
            video = False
        if not session_path.joinpath('raw_behavior_data').exists():
            behaviour = False
        if not session_path.joinpath('raw_passive_data').exists():
            passive = False
        _logger.info(f"{session_path} V{video}, B{behaviour}, P{passive}")


def correct_flags_biased_in_ephys_rig():
    """
    Biased sessions acquired on ephys rigs do not convert video transferred flag
    To not interfere with ongoing transfers, only handle sessions that are older than 7 days
    """
    N_DAYS = 7
    for flag in ROOT_PATH.rglob('video_data_transferred.flag'):
        session_path = alf.io.get_session_path(flag)
        ses_date = datetime.strptime(session_path.parts[-2], "%Y-%M-%d")
        if (datetime.now() - ses_date).days > N_DAYS:
            settings = raw.load_settings(session_path)
            if 'ephys' in settings['PYBPOD_BOARD'] and settings['PYBPOD_PROTOCOL']\
                    == '_iblrig_tasks_biasedChoiceWorld':
                _logger.info(session_path)
                flag.unlink()
                session_path.joinpath('raw_session.flag').touch()


def correct_passive_in_wrong_folder():
    """
    Finds the occasions where the data has been transferred manually and the passive folder has
    has not been moved and got the correct file structure
    """
    one = ONE()
    lab = _get_lab(one)
    if lab[0] == 'wittenlab':

        for flag in ROOT_PATH.rglob('passive_data_for_ephys.flag'):
            passive_data_path = alf.io.get_session_path(flag)
            passive_session = passive_data_path.stem
            passive_folder = passive_data_path.joinpath('raw_behavior_data')
            sessions = os.listdir(passive_data_path.parent)

            # find the session number that isn't
            data_sess = [sess for sess in sessions if sess != passive_session]
            if len(data_sess) == 1:
                session_path = passive_data_path.parent.joinpath(data_sess[0])
            else:
                # If more than one we register passive to the latest one
                data_sess.sort()
                session_path = passive_data_path.parent.joinpath(data_sess[-1])

            # copy the file
            data_path = session_path.joinpath('raw_passive_data')
            shutil.copytree(passive_folder, data_path)
            _logger.info(f'moved {passive_folder} to {data_path}')

            # find the tasks for this session and set it to waiting
            eid = one.eid_from_path(session_path)
            if eid:
                tasks = one.alyx.rest('tasks', 'list', session=eid, name='TrainingRegisterRaw')
                if len(tasks) > 0:
                    stat = {'status': 'Waiting'}
                    one.alyx.rest('tasks', 'partial_update', id=tasks[0]['id'], data=stat)

    else:
        return


if __name__ == "__main__":
    correct_flags_biased_in_ephys_rig()
    correct_ephys_manual_video_copies()
