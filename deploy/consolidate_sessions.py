"""A script for consolidating multiple behaviour sessions into one session.

NB: This is intended to be run only on the behaviour PC for sessions where the video, main sync,
etc. are acquired on another device.
"""
from pathlib import Path
from itertools import chain
import warnings
import argparse
import shutil
import tempfile

from ibllib.io import session_params
from ibllib.io.raw_data_loaders import patch_settings
from ibllib.pipes.misc import create_basic_transfer_params
from one.alf.io import iter_sessions
from one.alf.files import get_session_path


def ensure_absolute(paths, transfer_pars):
    """Ensure input paths are absolute pathlib objects.

    This converts relative session paths, e.g. subject/data/num to absolute paths.

    Parameters
    ----------
    paths : list of str, list of pathlib.Path
        An iterable of relative and/or absolute session paths.
    transfer_pars : dict
        The transfer parameters containing a DATA_FOLDER_PATH key for converting relative paths.

    Returns
    -------
    list of pathlib.Path
        The input paths as absolute Path objects.
    """
    abs_paths = []
    for path in map(Path, paths):
        if not path.is_absolute():
            assert (root := transfer_pars.get('DATA_FOLDER_PATH'))
            abs_paths.append(Path(root) / path)
        else:
            abs_paths.append(path)
    return abs_paths


def replace_device_collection(params, old_collection, new_collections):
    """
    Replace instances of old_collection with new_collection in the devices map of an experiment
    description dict. NB: This doesn't support sub-collections.

    Parameters
    ----------
    params : dict
        A loaded experiment.description file.
    old_collection : str
        The old collection name to replace.
    new_collections : str
        The new collection name.

    Returns
    -------
    dict
        The same experiment.description dict with the collections replaced (NB: the dict is
        modified in place; a copy is not returned).
    """
    for d in chain(*map(dict.values, params.get('devices', {}).values())):
        if d and d.get('collection') == old_collection:
            d['collection'] = new_collections
    return params


def main(*args, stub=None):
    transfer_pars = create_basic_transfer_params()
    skipped = 0  # number of sessions that were skipped due to missing behaviour data
    if not args:
        raise ValueError('Please provide at least one session path to consolidate')
    elif len(args) == 1:
        if session_path := get_session_path(args[0]):
            # e.g. subject/date/num or subject/date/num/collection
            sessions = iter_sessions(session_path.parent)
        else:
            # e.g. subject/date or subject
            sessions = iter_sessions(args[0])
    else:
        sessions = args
    sessions = ensure_absolute(sessions, transfer_pars)
    if not sessions:
        warnings.warn('No sessions found')
        return
    if stub:  # If a stub file was given, copy it into each session
        if (stub := Path(stub)).is_dir():
            stub = next(stub.glob('*experiment.description*'))
        for session in sessions:
            if not next(session.glob('*experiment.description*'), False):
                shutil.copy(stub, session)
    dst_session = sessions[0]
    for i, session in enumerate(sessions):
        collection = f'raw_task_data_{i - skipped:02}'
        if (raw_behaviour_data := session.joinpath('raw_behavior_data')).exists():
            # Rename raw_behaviour_data folder
            new_name = dst_session.joinpath(collection)
            raw_behaviour_data = raw_behaviour_data.rename(new_name)
        else:
            warnings.warn(f'Skipping session {session}: no raw_behavior_data folder')
            skipped += 1
            continue
        # Patch the settings file with the new paths
        new_settings = patch_settings(dst_session, collection,
                                      new_collection=collection, number=dst_session.parts[-1])

        # Aggregate experiment description file
        yaml_name = f'_ibl_experiment.description_{transfer_pars["TRANSFER_LABEL"]}.yaml'
        yaml_file = next(session.glob('_ibl_experiment.description*'), session / yaml_name)
        # If experiment.description doesn't exist, create one and insert task name from settings
        if not yaml_file.exists():
            task = new_settings.get('PYBPOD_PROTOCOL')
            # If settings exist and contain task, insert into experiment description
            params = {'tasks': [{task: {}}], 'version': '1.0.0'} if task else {}
        else:
            params = session_params.read_params(yaml_file)
        # If tasks key exists and is non empty, update the collection key
        if params and params.get('tasks'):
            next(iter(params['tasks'][0].values()))['collection'] = raw_behaviour_data.name
        replace_device_collection(params, 'raw_behavior_data', raw_behaviour_data.name)
        session_params.write_yaml(yaml_file, params)  # Save parameters before aggregation

        if i != 0:
            main_acq_desc = next(dst_session.glob('_ibl_experiment.description*.yaml'))
            session_params.aggregate_device(yaml_file, main_acq_desc, unlink=True)
            if any(session.rglob('*')):
                # If directory is not empty, move it to tmp
                dst = Path(tempfile.gettempdir()).joinpath('deleted_local_sessions', *session.parts[-3:-1])
                dst.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.move(session, dst)
            else:
                session.rmdir()


if __name__ == '__main__':
    """
    Examples
    --------
    >>> python consolidate_sessions.py --session subject/2022-01-01/001 subject/2022-01-01/003
    >>> python consolidate_sessions.py --session subject/2022-01-01/001
    >>> python consolidate_sessions.py --session subject/2022-01-01
    >>> python consolidate_sessions.py --session subject/2022-01-01 --stub /path/to/_ibl_experiment.description.yaml
    """
    parser = argparse.ArgumentParser(description='Consolidate sessions')
    parser.add_argument('--session', action='extend', nargs='+', type=str, help='One or more sessions to consolidate')
    parser.add_argument('--stub', type=str, help='A path to a stub experiment.description file to copy into each session')
    args = parser.parse_args()
    main(*args.session, stub=args.stub)
