from pathlib import Path
import logging
import os
from datetime import datetime
import shutil

from packaging import version
import numpy as np
from one.api import ONE
from one.remote.globus import get_lab_from_endpoint_id
from one.alf.path import get_session_path, session_path_parts
import spikeglx

import ibllib.io.raw_data_loaders as raw
from ibllib.ephys import spikes
from ibllib.pipes.ephys_preprocessing import SpikeSorting, EphysCellsQc
from ibllib.oneibl.registration import register_dataset
from ibllib.pipes.local_server import _get_volume_usage
import ibllib.io.session_params as session_params
from ibllib.pipes.dynamic_pipeline import acquisition_description_legacy_session


ROOT_PATH = Path('/mnt/s0/Data/Subjects')

_logger = logging.getLogger('ibllib')


def correct_flags_biased_in_ephys_rig():
    """
    Biased sessions acquired on ephys rigs do not convert video transferred flag
    To not interfere with ongoing transfers, only handle sessions that are older than 7 days
    """
    N_DAYS = 7
    for flag in ROOT_PATH.rglob('video_data_transferred.flag'):
        session_path = get_session_path(flag)
        ses_date = datetime.strptime(session_path.parts[-2], "%Y-%M-%d")
        if (datetime.now() - ses_date).days > N_DAYS:
            settings = raw.load_settings(session_path)
            if settings is not None:
                if ('ephys' in settings['PYBPOD_BOARD'] and
                        settings['PYBPOD_PROTOCOL'] == '_iblrig_tasks_biasedChoiceWorld'):
                    _logger.info(session_path)
                    flag.unlink()
                    session_path.joinpath('raw_session.flag').touch()


def correct_passive_in_wrong_folder():
    """
    Finds the occasions where the data has been transferred manually and the passive folder has
    has not been moved and got the correct file structure
    """
    one = ONE(cache_rest=None)
    lab = get_lab_from_endpoint_id(alyx=one.alyx)
    if lab[0] == 'wittenlab':

        for flag in ROOT_PATH.rglob('passive_data_for_ephys.flag'):
            passive_data_path = get_session_path(flag)
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

            # remove the passive flag
            flag.unlink()

            # find the tasks for this session and set it to waiting
            eid = one.path2eid(session_path)
            if eid:
                tasks = one.alyx.rest('tasks', 'list', session=eid, name='TrainingRegisterRaw')
                if len(tasks) > 0:
                    stat = {'status': 'Waiting'}
                    one.alyx.rest('tasks', 'partial_update', id=tasks[0]['id'], data=stat)

    else:
        return


def correct_passive_params():
    """Patch parameter files of passive sessions, renaming the session number and collection.

    Note: This should only apply for sessions acquired on iblrig v7.2.4 or earlier.
    """
    for flag_file in ROOT_PATH.glob('**/raw_session.flag'):
        session_path = flag_file.parent
        if session_path.joinpath('raw_passive_data', '_iblrig_taskSettings.raw.json').exists():
            settings = raw.load_settings(session_path, 'raw_passive_data')
            ver = settings.get('IBLRIG_VERSION', settings.get('IBLRIG_VERSION_TAG')) or '0.0.0'
            if version.parse(ver) < version.parse('7.2.5'):
                parts = session_path_parts(session_path, as_dict=True, assert_valid=True)
                parts.pop('lab')
                new_settings = raw.patch_settings(
                    session_path, collection='raw_passive_data', new_collection='raw_passive_data', **parts
                )
                # Log changes just in case they were made in error
                info = ('SUBJECT_NAME', 'SESSION_DATE', 'SESSION_NUMBER')
                patched = ['/'.join([*map(s.get, info), s.get('SESSION_RAW_DATA_FOLDER').split('\\')[-1]])
                           for s in (settings, new_settings)]
                if patched[0] != patched[1]:
                    _logger.info('SETTINGS PATCHED: %s -> %s', *patched)


def spike_amplitude_patching():
    """
    Patch the datasets that have incorrect spikes.amplitude datasets. While doing it also look for
    sessions that have spikesorting/ alf folders but for some reason haven't been registered and
    uploaded to flatiron for some reason (normally because .cbin file is missing).

    Five different scenarios to consider
    1. Data extracted properly, is on flatiron and has templates.amps - do nothing
    2. Data extracted properly, is on flatiron but doesn't have templates.amps - phy convert
       and register
    3. Data extracted properly with templates.amps , but not on flatiron - phy convert and
       register (don't necessarily need to phy convert but double check in case it was the
       syncing that errored)
    4. Data extracted properly without templates.amps, but non on flatiron - phy convert and
       register
    5. Data spike sorted but not extracted - phy convert and register

    """

    def phy2alf_conversion(session_path, ks2_path, alf_path, probe_label):
        try:
            # Find spikeglx meta data files associated with the session and probe
            files = spikeglx.glob_ephys_files(session_path, ext='meta')
            ap_files = [(ef.get("ap"), ef.get("label")) for ef in files if "ap" in ef.keys()]
            meta_file = next(ap[0] for ap in ap_files if ap[1] == probe_label)

            # The .cbin file doesn't always still exist on server so point to it from meta
            ap_file = meta_file.with_suffix('.cbin')

            # Convert to alf format
            spikes.ks2_to_alf(
                ks2_path,
                bin_path=meta_file.parent,
                out_path=alf_path,
                bin_file=None,
                ampfactor=SpikeSorting._sample2v(ap_file))

            # Sync the probes
            out_files, _ = spikes.sync_spike_sorting(ap_file=ap_file, out_path=alf_path)

            return 0, out_files, None

        except BaseException as err:
            _logger.error(f'{session_path} and {probe_label} errored with message: {err}')

            return -1, None, err

    def add_note_to_insertion(eid, probe, one, msg=None):
        insertion = one.alyx.rest('insertions', 'list', session=eid, name=probe)

        if len(insertion) > 0:
            probe_id = insertion[0]['id']
            status_note = {'user': one._par.ALYX_LOGIN,
                           'content_type': 'probeinsertion',
                           'object_id': probe_id,
                           'text': f'amps_patching_local_server2: {msg}'}
            _ = one.alyx.rest('notes', 'create', data=status_note)
        else:
            # If the probe insertion doesn't exist, make a session note
            status_note = {'user': one._par.ALYX_LOGIN,
                           'content_type': 'session',
                           'object_id': eid,
                           'text': f'amps_patching_local_server2: {probe}: {msg}'}
            _ = one.alyx.rest('notes', 'create', data=status_note)

    one = ONE(cache_rest=None)

    for ks2_out in ROOT_PATH.rglob('spike_sorting_ks2.log'):
        ks2_path = Path(ks2_out).parent

        # Clean up old flags if they exist
        if ks2_path.joinpath('amps_patching_local_server.flag').exists():
            ks2_path.joinpath('amps_patching_local_server.flag').unlink()

        # If we already looked at this session previously, no need to try again
        if ks2_path.joinpath('amps_patching_local_server2.flag').exists():
            continue

        # Make the flag if it is the first time looking into session
        ks2_path.joinpath('amps_patching_local_server2.flag').touch()

        # Now proceed with everything else
        session_path = get_session_path(ks2_out)
        eid = one.path2eid(session_path)
        if eid is None:
            # Skip sessions that don't exist on alyx!
            continue
        probe = ks2_path.stem
        alf_path = session_path.joinpath('alf', probe)
        alf_path.mkdir(parents=True, exist_ok=True)

        # If a clusters.metrics file exists in the alf_path, delete it. Causes registration error!
        cluster_metrics = alf_path.joinpath('clusters.metrics.csv')
        if cluster_metrics.exists():
            os.remove(cluster_metrics)

        # templates.amps file only exists if it is new phy extractor
        templates_file = alf_path.joinpath('templates.amps.npy')
        if templates_file.exists():
            dset = one.alyx.rest('datasets', 'list', session=eid, name='templates.amps.npy')
            # check if it has been registered for this probe specifically
            collection = [ds['collection'].rsplit('/', 1)[-1] for ds in dset]
            if probe in collection:
                continue

        # Otherwise we need to extract alf files and register datasets
        status, out, err = phy2alf_conversion(session_path, ks2_path, alf_path, probe)
        if status == 0:
            try:
                cluster_qc = EphysCellsQc(session_path, one=one)
                qc_file, df_units, drift = cluster_qc._compute_cell_qc(alf_path)
                out.append(qc_file)
                cluster_qc._label_probe_qc(alf_path, df_units, drift)
                register_dataset(out, one=one)
                add_note_to_insertion(eid, probe, one, msg='completed')
                _logger.info(f'All good: {session_path} and {probe}')
            except BaseException as err2:
                _logger.info(f'Errored at qc/ registration stage: {session_path} and {probe}')
                add_note_to_insertion(eid, probe, one, msg=err2)
        else:
            # Log the error
            add_note_to_insertion(eid, probe, one, msg=err)
            continue


def upload_ks2_output():
    """
    Copy ks2 output to a .tar file and upload to flatiron for all past sessions that have
    spike sorting output
    """
    # if the space on the disk > 500Gb continue, otherwise, don't bother
    usage = _get_volume_usage('/mnt/s0/Data', 'disk')
    if usage['disk_available'] < 500:
        return

    one = ONE(cache_rest=None)

    for ilog, ks2_out in enumerate(ROOT_PATH.rglob('spike_sorting_ks2.log')):
        # check space on disk after every 25 extractions. stop if we are running low!
        if np.mod(ilog, 25) == 0:
            usage = _get_volume_usage('/mnt/s0/Data', 'disk')
            if usage['disk_available'] < 500:
                return

        ks2_path = Path(ks2_out).parent
        session_path = get_session_path(ks2_out)

        probe = ks2_path.stem
        tar_dir = session_path.joinpath('spike_sorters', 'ks2_matlab', probe)
        tar_dir.mkdir(exist_ok=True, parents=True)

        # If the flag exists it means we have already extracted
        if tar_dir.joinpath('tar_existed.flag').exists():
            # We already done this, no need to repeat!!
            continue

        eid = one.path2eid(session_path)

        # For latest sessions tar file will be created by task and automatically registered so we
        # may have a case where tar file already registered and uploaded but no tar_existed.flag
        if tar_dir.joinpath('_kilosort_raw.output.tar').exists():
            # double check it indeed has been registered for this probe
            dset = one.alyx.rest('datasets', 'list', session=eid, name='_kilosort_raw.output.tar')
            collection = [ds['collection'].rsplit('/', 1)[-1] for ds in dset]
            if probe in collection:
                tar_dir.joinpath('tar_existed.flag').touch()
                continue

        if eid is None:
            # Skip sessions that don't exist on alyx!
            continue

        out = spikes.ks2_to_tar(ks2_path, tar_dir)
        register_dataset(out, one=one)
        # Make flag to indicate data already registered for this session
        tar_dir.joinpath('tar_existed.flag').touch()


def remove_old_spike_sortings_outputs():

    ks2_output = ['amplitudes.npy',
                  'channel_map.npy',
                  'channel_positions.npy',
                  'cluster_Amplitude.tsv',
                  'cluster_ContamPct.tsv',
                  'cluster_group.tsv',
                  'cluster_KSLabel.tsv',
                  'params.py',
                  'pc_feature_ind.npy',
                  'pc_features.npy',
                  'similar_templates.npy',
                  'spike_clusters.npy',
                  'spike_sorting_ks2.log',
                  'spike_templates.npy',
                  'spike_times.npy',
                  'template_feature_ind.npy',
                  'template_features.npy',
                  'templates.npy',
                  'templates_ind.npy',
                  'whitening_mat.npy',
                  'whitening_mat_inv.npy']
    siz = 0
    for ks2_flag in ROOT_PATH.rglob('spike_sorting_ks2.log'):
        session_path = get_session_path(ks2_flag)
        ks2_path = ks2_flag.parent
        probe = ks2_path.parts[-1]
        tar_dir = session_path.joinpath('spike_sorters', 'ks2_matlab', probe)
        if not any((tar_dir.joinpath('tar_existed.flag').exists(), tar_dir.joinpath('_kilosort_raw.output.tar'))):
            continue
        for fn in ks2_output:
            fil = next(tar_dir.glob(fn), None)
            if fil is None:
                continue
            siz += fil.stat().st_size
            fil.unlink()
    _logger.info(f'remove old spike sorting outputs removed {siz / 1024 ** 3} Go')


def glob_sessions_fast(pattern, root_path=ROOT_PATH):
    for sub_dir in root_path.glob('*'):
        if not sub_dir.is_dir():
            continue
        for date_dir in sub_dir.glob('20*'):
            if not date_dir.is_dir():
                continue
            for session_path in date_dir.glob('0*'):
                if not session_path.is_dir():
                    continue
                for fn in session_path.glob(pattern):
                    yield fn


def dynamic_pipeline_transition_photometry():
    """
    Looks for a _device/photometry_00.yaml file if found create an acquisition description file and add
    a raw_session_flag
    """
    for photometry_yaml in glob_sessions_fast("_device/photometry_00.yaml"):
        session_path = get_session_path(photometry_yaml)
        if session_path.joinpath('_ibl_experiment.description.yaml').exists():
            continue
        print(f"Found photometry yaml: {photometry_yaml}, create acquisition description file and raw session flag")
        fp_description = session_params.read_params(photometry_yaml)
        description = acquisition_description_legacy_session(session_path)
        description['devices']['photometry'] = fp_description['devices']['photometry']
        description['procedures'] = list(set(description['procedures'] + ['Fiber photometry']))
        session_params.write_params(session_path, description)
        session_path.joinpath('raw_session.flag').touch()


if __name__ == "__main__":
    correct_flags_biased_in_ephys_rig()
    # correct_ephys_manual_video_copies()
    # remove_iti_duration()
    # spike_amplitude_patching()
    # upload_ks2_output()
    correct_passive_in_wrong_folder()
    correct_passive_params()
    # remove_old_spike_sortings_outputs()
    dynamic_pipeline_transition_photometry()
