"""
Entry point to system commands for IBL pipeline.
python rerun.py 04_audio_training /mnt/s0/Data/Subjects --dry=True
python rerun.py 21_qc_ephys /mnt/s0/Data/Subjects --dry=True
python rerun.py 22_audio_ephys /mnt/s0/Data/Subjects --dry=True
python rerun.py 23_compress_ephys /mnt/s0/Data/Subjects --dry=True
python rerun.py 26_merge_sync_ephys /mnt/s0/Data/Subjects --dry=True
python rerun.py 27_compress_ephys_video /mnt/s0/Data/Subjects --dry=True
"""

# Per dataset type
import logging
from pathlib import Path
from dateutil.parser import parse
import re
import argparse

import alf.io
from ibllib.io import flags, spikeglx
import ibllib.pipes.experimental_data as pipes
import ibllib.pipes.extract_session as extract_session


logger = logging.getLogger('ibllib')
DRANGE = ('2000-01-01', '2100-01-01')  # default date range


def rerun_01_extract_training(ses_path, drange, dry=True):
    files_error, files_error_date = _order_glob_by_session_date(ses_path.rglob('extract_me.error'))
    for file_error, date in zip(files_error, files_error_date):
        if not(date >= drange[0] and (date <= drange[1])):
            continue
        print(file_error)
        if dry:
            continue
        file_error.unlink()
        flags.create_extract_flags(file_error.parent, force=True)
        pipes.extract(file_error.parent)


def rerun_02_register(ses_path, drange, dry=True):
    # compute the date range including both bounds
    files_error, files_error_date = _order_glob_by_session_date(ses_path.rglob(
        'register_me.error'))
    for file_error, date in zip(files_error, files_error_date):
        if not(date >= drange[0] and (date <= drange[1])):
            continue
        print(file_error)
        if dry:
            continue
        file_error.unlink()
        flags.create_register_flags(file_error.parent, force=True)
        pipes.register(file_error.parent)


def rerun_03_compress_video(ses_path, drange, dry=True):
    # for a failed compression there is an `extract.error` file in the raw_video folder
    files_error, files_error_date = _order_glob_by_session_date(ses_path.rglob('extract.error'))
    for file_error, date in zip(files_error, files_error_date):
        if not(date >= drange[0] and (date <= drange[1])):
            continue
        print(file_error)
        if dry:
            continue
        file_error.unlink()
        flags.create_compress_video_flags(file_error.parents[1])
    if not dry:
        logger.warning("Flags created, to compress videos, launch the compress script from deploy")


def rerun_04_audio_training(root_path, drange, **kwargs):
    """
    This job looks for wav files and create `audio_training.flag` for each wav file found
    """
    _rerun_wav_files(root_path, drange=drange, flag_name='audio_training.flag',
                     task_excludes=['ephys', 'ephys_sync'], **kwargs)


def rerun_05_dlc_training(root_path, drange, dry=True):
    pass


def rerun_20_extract_ephys(ses_path, drange, dry=True):
    _rerun_ephys(ses_path, drange, dry=dry, pipefunc=pipes.extract_ephys,
                 flagstr='extract_ephys.flag')


def rerun_21_qc_ephys(ses_path, drange, dry=True):
    _rerun_ephys(ses_path, drange, dry=dry, pipefunc=pipes.raw_ephys_qc,
                 flagstr='raw_ephys_qc.flag')


def rerun_22_audio_ephys(root_path, drange, **kwargs):
    """
    This job looks for wav files and create `audio_ephys.flag` for each wav file found
    """
    _rerun_wav_files(root_path, drange=drange, flag_name='audio_ephys.flag',
                     task_includes=['ephys', 'ephys_sync'], **kwargs)


def rerun_23_compress_ephys(root_path, dry=True):
    """
    Looks for uncompressed 'ap.bin', 'lf.bin' and 'nidq.bin' files and creates 'compress_me.flags`
    For ap and lf, creates flags only if the ks2_alf folder exists at the same level.
    For nidq files, creates the flag regardless.
    """

    def _create_compress_flag(bin_file):
        print(bin_file)
        flag_file = bin_file.parent.joinpath('compress_ephys.flag')
        if not dry:
            flag_file.touch()

    ephys_files = spikeglx.glob_ephys_files(root_path)
    for ef in ephys_files:
        if ef.get('ap', None):
            if ef.ap.parent.joinpath('spike_sorting_ks2.log').exists():
                _create_compress_flag(ef.ap)
        if ef.get('lf', None):
            if ef.ap.parent.joinpath('spike_sorting_ks2.log').exists():
                _create_compress_flag(ef.lf)
        if ef.get('nidq', None):
            _create_compress_flag(ef.nidq)


def rerun_26_sync_merge_ephys(root_path, dry=True):
    """
    Looks for 'ap.bin', 'lf.bin' and files and creates '26_sync_merge_ephys.flags`
    only if the spike sorting is done: ie. `spike_sorting_*.log` exists
    """
    ephys_files = spikeglx.glob_ephys_files(root_path)
    for ef in ephys_files:
        if ef.get('ap'):
            bin_file = ef.get('ap')
            if not next(bin_file.parent.glob('spike_sorting_*.log'), None):
                return
            print(bin_file)
            flag_file = bin_file.parent.joinpath('sync_merge_ephys.flag')
            if not dry:
                flag_file.touch()


def rerun_27_compress_ephys_video(root_path, drange=DRANGE, dry=True):
    _rerun_avi_files(root_path, flag_name='compress_video_ephys.flag',
                     task_includes=['ephys', 'ephys_sync'], dry=dry, drange=drange)


def _rerun_avi_files(root_path, flag_name, task_excludes=None, task_includes=None,
                     drange=DRANGE, dry=True):
    avi_files = _glob_date_range(root_path, task_excludes=task_excludes,
                                 task_includes=task_includes,
                                 glob_pattern='_iblrig_*Camera.raw.avi', drange=drange)
    avi_folders = list(set([str(af.parent) for af in avi_files]))
    for af in avi_folders:
        print(af)
        if dry:
            continue
        flags.create_compress_video_flags(Path(af).parent, flag_name)


def _rerun_wav_files(root_path, flag_name, task_excludes=None, task_includes=None,
                     drange=DRANGE, dry=True):
    audio_files = _glob_date_range(root_path, task_excludes=task_excludes,
                                   task_includes=task_includes,
                                   glob_pattern='_iblrig_micData.raw.wav', drange=drange)
    for af in audio_files:
        print(af)
        if dry:
            continue
        flags.create_audio_flags(af.parents[1], flag_name)


def _rerun_ephys(ses_path, drange=DRANGE, dry=True, pipefunc=None, flagstr=None):
    """
    Creates a flag at the session root and run the associated job
    """
    ephys_files = _glob_date_range(ses_path, glob_pattern='*.ap.*bin', drange=drange)
    for ef in ephys_files:
        if dry:
            print(ef)
            continue
        session_path = alf.io.get_session_path(ef)
        flags.create_other_flags(session_path, flagstr, force=True)
    if not dry:
        pipefunc(session_path)


def _glob_date_range(root_path, glob_pattern, task_excludes=None, task_includes=None,
                     drange=DRANGE):
    files, files_date = _order_glob_by_session_date(root_path.rglob(glob_pattern))
    sessions = [f for f, d in zip(files, files_date) if drange[0] <= d <= drange[1]]

    def _test_task(f):
        # check if task is included or excluded
        task = extract_session.get_task_extractor_type(f)
        if not task:
            return False
        if task_includes and task not in task_includes:
            return False
        if task_excludes and task in task_excludes:
            return False
        return True

    return [f for f in sessions if _test_task(f)]


def _order_glob_by_session_date(flag_files):
    """
    Given a list/generator of PurePaths below an ALF session folder, outtput a list of of PurePaths
    sorted by date in reverse order.
    :param flag_files: list/generator of PurePaths
    :return: list of PurePaths
    """
    flag_files = list(flag_files)

    def _fdate(fl):
        dat = [parse(fp) for fp in fl.parts if re.match(r'\d{4}-\d{2}-\d{2}', fp)]
        if dat:
            return dat[0]
        else:
            return parse('1999-12-12')

    t = [_fdate(fil) for fil in flag_files]
    return [f for _, f in sorted(zip(t, flag_files), reverse=True)], sorted(t, reverse=True)


if __name__ == "__main__":
    ALLOWED_ACTIONS = ['04_audio_training', '21_qc_ephys', '22_audio_ephys', '23_compress_ephys',
                       '26_sync_merge_ephys', '27_compress_ephys_video']
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('action', help='Action: ' + ','.join(ALLOWED_ACTIONS))
    parser.add_argument('folder', help='A Folder containing a session')
    parser.add_argument('--dry', help='Dry Run', required=False, default=False, type=str)
    parser.add_argument('--first', help='yyyy-mm-dd date', required=False,
                        default='1999-12-12', type=str)
    parser.add_argument('--last', help='yyyy-mm-dd date', required=False,
                        default='2050-12-12', type=str)
    args = parser.parse_args()  # returns data from the options specified (echo)

    if args.dry and args.dry.lower() == 'false':
        args.dry = False
    assert(Path(args.folder).exists())

    date_range = [parse(args.first), parse(args.last)]
    ses_path = Path(args.folder)
    if args.action == '04_audio_training':
        rerun_04_audio_training(ses_path, date_range, dry=args.dry)
    elif args.action == '21_qc_ephys':
        rerun_21_qc_ephys(ses_path, date_range, dry=args.dry)
    elif args.action == '22_audio_ephys':
        rerun_22_audio_ephys(ses_path, date_range, dry=args.dry)
    elif args.action == '23_compress_ephys':
        rerun_23_compress_ephys(ses_path, dry=args.dry)
    elif args.action == '26_sync_merge_ephys':
        rerun_26_sync_merge_ephys(ses_path, dry=args.dry)
    elif args.action == '27_compress_ephys_video':
        rerun_27_compress_ephys_video(ses_path, date_range, args.dry)
    else:
        logger.error('Allowed actions are: ' + ', '.join(ALLOWED_ACTIONS))
