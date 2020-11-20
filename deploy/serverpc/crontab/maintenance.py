from pathlib import Path
import logging
from datetime import datetime

import alf.io
import ibllib.io.raw_data_loaders as raw
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


if __name__ == "__main__":
    correct_flags_biased_in_ephys_rig()
    correct_ephys_manual_video_copies()
