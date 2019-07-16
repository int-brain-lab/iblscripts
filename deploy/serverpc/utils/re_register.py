"""
Entry point to system commands for IBL pipeline.

>>> python re_register.py /mnt/s0/Data/Subjects/ [--dry=True --first=2019-07-10 --last=2019-07-11]
"""

# Per dataset type
from pathlib import Path
from dateutil.parser import parse
import re
import argparse

from ibllib.io import flags
from ibllib.pipes.experimental_data import register

dry = True


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
    ALLOWED_ACTIONS = ['create', 'extract', 'register', 'compress_video']
    parser = argparse.ArgumentParser(description='Description of your program')
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

    ses_path = Path(args.folder)

    # compute the date range including both bounds
    files_error, files_error_date = _order_glob_by_session_date(ses_path.rglob(
        'register_me.error'))
    drange = [parse(args.first), parse(args.last)]

    for file_error, date in zip(files_error, files_error_date):
        if not(date >= drange[0] and (date <= drange[1])):
            continue
        print(file_error)
        if args.dry:
            continue
        file_error.unlink()
        flags.create_register_flags(file_error.parent, force=True)
        register(file_error.parent)
