from pathlib import Path
import logging
import os
from datetime import datetime
import shutil
from oneibl.one import ONE
import alf.io
import ibllib.io.raw_data_loaders as raw
from ibllib.ephys import spikes
from ibllib.pipes.local_server import _get_lab
from ibllib.io import spikeglx
from ibllib.pipes.ephys_preprocessing import SpikeSorting_KS2_Matlab
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


def spike_amplitude_patching():
    one = ONE()
    ks2_out = ROOT_PATH.rglob('spike_sorting_ks2.log')
    # need to get the spike sorting path
    # need to look for the alf path, if the alf path contains templates.amps
    # do a quick check on alyx to confirm that the dataset exists on the flatiron
    # if it does we can skip
    # if it doesn't then we need to find get the collection from flatiron cos of witten fuck up
    # make that the outpath
    # spike sorting in path
    # make template model
    # convert things
    # register datasets - you know what just do the whole shabbam as we don't know if it for real

    for ks2_out in ROOT_PATH.rglob('spike_sorting_ks2.log'):
        session_path = alf.io.get_session_path(ks2_out)
        ks2_path = Path(ks2_out).parent
        probe = ks2_path.stem
        alf_path = session_path.joinpath('alf', probe)
        assert(alf_path.exists())
        templates_file = alf_path.joinpath('templates.amps.npy')
        if templates_file.exists():
            eid = one.eid_from_path(session_path)
            dset = one.alyx.rest('datasets', 'list', session=eid, name='templates.amps.npy')
            if len(dset) > 0:
            # then we all good
                continue
            else:
                print('mismatch server and database for ' + str(alf_path))
                # see what going on
        else:
            print('temps.amps dont exist for ' + str(alf_path))
            # need to specify bin path
            files = spikeglx.glob_ephys_files(session_path, suffix='.meta', ext='meta')
            ap_files = [(ef.get("ap"), ef.get("label")) for ef in files if "ap" in ef.keys()]

            meta_file = next(ap[0] for ap in ap_files if ap[1] == probe)
            ap_file = meta_file.with_suffix('.cbin')

            spikes.ks2_to_alf(
                ks2_path,
                bin_path=meta_file.parent,
                out_path=alf_path,
                bin_file=None,
                ampfactor=SpikeSorting_KS2_Matlab._sample2v(ap_file),
            )
            # we want ap meta file for our probe label

            out, _ = spikes.sync_spike_sorting(ap_file=ap_file, out_path=alf_path)

            # now we need to register these datasets




if __name__ == "__main__":
    correct_flags_biased_in_ephys_rig()
    correct_ephys_manual_video_copies()
