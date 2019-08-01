"""
Entry point to system commands for IBL pipeline.

>>> python ephys.py extract /mnt/s0/Data/Subjects --dry=True --count=10
>>> python ephys.py qc /mnt/s0/Data/Subjects --dry=True --count=10
"""

import argparse
import logging
from pathlib import Path

import ibllib.pipes.experimental_data as pipes

logger = logging.getLogger('ibllib')


def extract(ses_path, dry=True, max_sessions=10):
    pipes.extract_ephys(ses_path, max_sessions=max_sessions)


def qc(ses_path, dry=True, max_sessions=5):
    pipes.qc_ephys(ses_path, dry=dry, max_sessions=max_sessions)


if __name__ == "__main__":
    ALLOWED_ACTIONS = ['extract', 'qc']
    parser = argparse.ArgumentParser(description='Ephys Pipeline')
    parser.add_argument('action', help='Action: ' + ','.join(ALLOWED_ACTIONS))
    parser.add_argument('folder', help='A Folder containing a session')
    parser.add_argument('--dry', help='Dry Run', required=False, default=False, type=str)
    parser.add_argument('--count', help='Max number of sessions to run this on',
                        required=False, default=False, type=int)
    args = parser.parse_args()  # returns data from the options specified (echo)
    if args.dry and args.dry.lower() == 'false':
        args.dry = False
    assert(Path(args.folder).exists())
    if args.action == 'extract':
        extract(ses_path=args.folder, dry=args.dry, max_sessions=args.count)
    elif args.action == 'qc':
        qc(ses_path=args.folder, dry=args.dry, max_sessions=args.count)
    else:
        logger.error(f'Action "{args.action}" not valid. Allowed actions are: ' +
                     ', '.join(ALLOWED_ACTIONS))
