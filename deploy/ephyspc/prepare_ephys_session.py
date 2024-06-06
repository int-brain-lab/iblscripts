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
import asyncio

from one.alf.io import next_num_folder
from one.api import OneAlyx
from iblutil.util import setup_logger
from iblutil.io import net

import ibllib
from ibllib.pipes.misc import load_ephyspc_params


def check_ibllib_version(ignore=False):
    bla = subprocess.run(
        'pip install ibllib==ver',
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    ble = [x.decode('utf-8') for x in bla.stderr.rsplit()]
    # Latest version is at the end of the error message before the close parens
    latest_ibllib = version.parse([x.strip(')') for x in ble if ')' in x][0])
    if latest_ibllib != version.parse(ibllib.__version__):
        msg = (
            f'You are using ibllib {ibllib.__version__}, but the latest version is {latest_ibllib}'
        )
        print(f'{msg} - Please update ibllib')
        print('To update run: [conda activate iblenv] and [pip install -U ibllib]')
        if ignore:
            return
        raise Exception(msg)


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


def _v8_check():
    """Return True if iblrigv8 installed."""
    try:
        import iblrig
        return version.parse(iblrig.__version__) >= version.Version('8.0.0')
    except ModuleNotFoundError:
        return False


def main_v8(mouse, debug=False, n_probes=None):
    # from iblrig.base_tasks import EmptySession
    from iblrig.transfer_experiments import EphysCopier

    log = setup_logger(name='iblrig', level=10 if debug else 20)

    PARAMS = load_ephyspc_params()
    iblrig_settings = {
        'iblrig_local_data_path': Path(PARAMS['DATA_FOLDER_PATH']),
        'iblrig_local_subjects_path': Path(PARAMS['DATA_FOLDER_PATH']),
        'iblrig_remote_data_path': Path(PARAMS['REMOTE_DATA_FOLDER_PATH']),
        'iblrig_remote_subjects_path': Path(PARAMS['REMOTE_DATA_FOLDER_PATH']),
    }

    # if PARAMS.get('PROBE_TYPE_00', '3B') != '3B' or PARAMS.get('PROBE_TYPE_01', '3B') != '3B':
    #     raise NotImplementedError('Only 3B probes supported.')
    if n_probes is None:
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


async def main_v8_networked(mouse, debug=False, n_probes=None, service_uri=None):
    # from iblrig.base_tasks import EmptySession
    from iblrig.transfer_experiments import EphysCopier
    from iblrig.net import get_server_communicator, update_alyx_token, read_stdin

    log = setup_logger(name='iblrig', level=10 if debug else 20)
    PARAMS = load_ephyspc_params()
    iblrig_settings = {
        'iblrig_local_data_path': Path(PARAMS['DATA_FOLDER_PATH']),
        'iblrig_local_subjects_path': Path(PARAMS['DATA_FOLDER_PATH']),
        'iblrig_remote_data_path': Path(PARAMS['REMOTE_DATA_FOLDER_PATH']),
        'iblrig_remote_subjects_path': Path(PARAMS['REMOTE_DATA_FOLDER_PATH']),
    }

    # if PARAMS.get('PROBE_TYPE_00', '3B') != '3B' or PARAMS.get('PROBE_TYPE_01', '3B') != '3B':
    #     raise NotImplementedError('Only 3B probes supported.')
    if n_probes is None:
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
    communicator, _ = await get_server_communicator(service_uri, 'neuropixel')
    copier.initialize_experiment(nprobes=n_probes)

    one = OneAlyx(silent=True, mode='local')
    exp_ref = one.path2ref(session_path)
    tasks = set()

    log.info('Type "abort" to cancel or just press return to finalize')
    while True:
        # Ensure we are awaiting a message from the remote rig.
        # This task must be re-added each time a message is received.
        if not any(t.get_name() == 'remote' for t in tasks) and communicator and communicator.is_connected:
            task = asyncio.create_task(communicator.on_event(net.base.ExpMessage.any()), name='remote')
            tasks.add(task)
        if not any(t.get_name() == 'keyboard' for t in tasks):
            tasks.add(asyncio.create_task(anext(read_stdin()), name='keyboard'))
        # Await the next task outcome
        done, _ = await asyncio.wait(tasks, timeout=None, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            match task.get_name():
                case 'keyboard':
                    if net.base.is_success(task):
                        line = task.result().strip().lower()
                        if line == 'abort' and not any(filter(Path.is_file, raw_data_folder.rglob('*'))):
                            log.warning('Removing %s', raw_data_folder)
                            for d in raw_data_folder.iterdir():  # Remove probe folders
                                d.rmdir()
                            raw_data_folder.rmdir()  # remove collection
                            # Delete whole session folder?
                            session_files = list(session_path.rglob('*'))
                            if len(session_files) == 1 and session_files[0].name.startswith(
                                    '_ibl_experiment.description'):
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
                        communicator.close()
                        for task in tasks:
                            task.cancel()
                        tasks.clear()
                        return
                case 'remote':
                    if task.cancelled():
                        log.debug('Remote com await cancelled')
                        log.error('Remote communicator closed')
                    else:
                        data, addr, event = task.result()
                        S = net.base.ExpMessage
                        match event:
                            case S.EXPINFO:
                                reponse_data = {'exp_ref': one.dict2ref(exp_ref), 'main_sync': True}
                                await communicator.info(net.base.ExpStatus.RUNNING, reponse_data, addr=addr)
                            case S.EXPSTATUS:
                                await communicator.status(net.base.ExpStatus.RUNNING, addr=addr)
                            case S.EXPINIT:
                                ...  # TODO
                            case S.EXPSTART:
                                await communicator.start(exp_ref, addr=addr)
                            case S.ALYX:
                                base_url, token = data
                                if base_url and token and next(iter(token)):
                                    # Install alyx token
                                    update_alyx_token(data, addr, one.alyx)
                                elif one.alyx.is_logged_in and (base_url or one.alyx.base_url) == one.alyx.base_url:
                                    # Return alyx token
                                    await communicator.alyx(one.alyx, addr=addr)
                            case _:
                                # Do nothing for the others  # TODO Change iblrig mixin to not await on stop and cleanups
                                pass
                case _:
                    raise NotImplementedError(f'Unexpected task "{task.get_name()}"')
            tasks.remove(task)


def main(mouse, **_):
    warnings.warn('For iblrigv8 behaviour sessions, install iblrigv8 on this PC also', FutureWarning)
    SUBJECT_NAME = mouse
    PARAMS = load_ephyspc_params()
    DATA_FOLDER = Path(PARAMS['DATA_FOLDER_PATH'])

    DATE = datetime.datetime.now().date().isoformat()
    NUM = next_num_folder(DATA_FOLDER / SUBJECT_NAME / DATE)

    SESSION_FOLDER = DATA_FOLDER / SUBJECT_NAME / DATE / NUM / 'raw_ephys_data'
    SESSION_FOLDER.mkdir(parents=True, exist_ok=True)
    print(f'Created {SESSION_FOLDER}')
    PROBE_00_FOLDER = SESSION_FOLDER / 'probe00'
    PROBE_00_FOLDER.mkdir(parents=True, exist_ok=True)
    print(f'Created {PROBE_00_FOLDER}')
    PROBE_01_FOLDER = SESSION_FOLDER / 'probe01'
    PROBE_01_FOLDER.mkdir(parents=True, exist_ok=True)
    print(f'Created {PROBE_01_FOLDER}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare ephys PC for ephys recording session')
    parser.add_argument('mouse', help='Mouse name')
    parser.add_argument('--debug', help='Debug mode', action='store_true')
    parser.add_argument('--n_probes', '-n', help='Number of probes in use', type=int)
    parser.add_argument(
        '--ignore-checks',
        default=False,
        required=False,
        action='store_true',
        help='Ignore ibllib and iblscripts checks',
    )
    parser.add_argument('--service_uri', required=False, nargs='?', default=None,
                        help='the service URI to listen to messages on. pass ":<port>" to specify port only.')
    args = parser.parse_args()
    service_uri = getattr(args, 'service_uri', 'service_uri' in args)

    check_ibllib_version(ignore=args.ignore_checks)
    check_iblscripts_version(ignore=args.ignore_checks)
    fcn = main_v8 if _v8_check() else main
    fcn(args.mouse, debug=args.debug, n_probes=args.n_probes, service_uri=service_uri)
