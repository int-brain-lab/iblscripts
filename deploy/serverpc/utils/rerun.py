"""
Entry point to system commands for IBL pipeline.

"""

# Per dataset type
import logging
from pathlib import Path
from dateutil.parser import parse
import re
import argparse

from ibllib.io import flags
import ibllib.pipes.experimental_data as pipes

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
        flags.create_compress_flags(file_error.parents[1])
    if not dry:
        logger.warning("Flags created, to compress videos, launch the compress script from deploy")


def rerun_04_audio_training(root_path, drange=DRANGE, dry=True):
    """
    This job removes the audio files so the re-run is only about the remaining wav files in
    training sessions
    """
    audio_files = _glob_date_range(ses_path, glob_pattern='_iblrig_micData.raw.wav', drange=drange)
    for af in audio_files:
        print(af)
        if dry:
            continue


def rerun_05_dlc_training(root_path, drange, dry=True):
    pass


def rerun_20_extract_ephys(ses_path, drange, dry=True):
    _rerun_ephys(ses_path, drange, dry=dry, pipefunc=pipes.extract_ephys,
                 flagstr='extract_ephys.flag')


def rerun_21_qc_ephys(ses_path, drange, dry=True):
    _rerun_ephys(ses_path, drange, dry=dry, pipefunc=pipes.raw_ephys_qc,
                 flagstr='raw_ephys_qc.flag')


def _rerun_ephys(ses_path, drange=DRANGE, dry=True, pipefunc=None, flagstr=None):
    ephys_files = _glob_date_range(ses_path, glob_pattern='*.ap.bin', drange=drange)
    for ef in ephys_files:
        if dry:
            print(ef)
            continue
        flags.create_other_flags(ef.parents[2], flagstr, force=True)
    if not dry:
        pipefunc(ef.parents[2])


def _glob_date_range(ses_path, glob_pattern, drange=DRANGE):
    files, files_date = _order_glob_by_session_date(ses_path.rglob(glob_pattern))
    return [f for f, d in zip(files, files_date) if drange[0] <= d <= drange[1]]


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
    ALLOWED_ACTIONS = ['extract', 'register', 'compress_video', 'extract_ephys']
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
    if args.action == 'extract':
        rerun_extract_training(ses_path, date_range, dry=args.dry)
    elif args.action == 'register':
        rerun_register(ses_path, date_range, dry=args.dry)
    elif args.action == 'compress_video':
        rerun_compress_video(ses_path, date_range, dry=args.dry)
    elif args.action == 'extract_ephys':
        rerun_extract_ephys(ses_path, date_range, dry=args.dry)
    else:
        logger.error('Allowed actions are: ' + ', '.join(ALLOWED_ACTIONS))
