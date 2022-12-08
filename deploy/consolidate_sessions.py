from pathlib import Path
import warnings
import argparse
import shutil
import tempfile

from ibllib.io import session_params, raw_data_loaders
from ibllib.pipes.misc import create_basic_transfer_params
from one.alf.io import iter_sessions
from one.alf.files import get_session_path


def ensure_absolute(paths, transfer_pars):
    abs_paths = []
    for path in paths:
        if not path.is_absolute():
            assert (root := transfer_pars.get('DATA_FOLDER_PATH'))
            abs_paths.append(Path(root) / path)
        else:
            abs_paths.append(path)
    return abs_paths


def main(*args):
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
    dst_session = sessions[0]
    for i, session in enumerate(sessions):
        if (raw_behaviour_data := session.joinpath('raw_behavior_data')).exists():
            # Rename raw_behaviour_data folder
            new_name = dst_session.joinpath(f'raw_task_data_{i - skipped:02}')
            raw_behaviour_data = raw_behaviour_data.rename(new_name)
        else:
            warnings.warn(f'Skipping session {session}: no raw_behavior_data folder')
            skipped += 1
            continue

        # Aggregate experiment description file
        yaml_name = f'_ibl_experiment.description_{transfer_pars["TRANSFER_LABEL"]}.yaml'
        yaml_file = next(session.glob('_ibl_experiment.description*'), session / yaml_name)
        if not yaml_file.exists():
            settings = raw_data_loaders.load_settings(dst_session, task_collection=raw_behaviour_data.name) or {}
            task = settings.get('PYBPOD_PROTOCOL')
            if task:
                params = {'tasks': [{task: {}}], 'version': '1.0.0'}
            else:
                params = {}
        else:
            params = session_params.read_params(yaml_file)
        if params and params.get('tasks'):
            next(iter(params['tasks'][0].values()))['collection'] = raw_behaviour_data.name
        session_params.write_yaml(yaml_file, params)

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
    parser = argparse.ArgumentParser(description='Consolidate sessions')
    parser.add_argument('--session', action='extend', nargs='+', type=str, help='One or more sessions to consolidate')
    args = parser.parse_args()
    main(*args.session)
