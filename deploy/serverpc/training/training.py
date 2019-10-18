"""
Entry point to system commands for IBL pipeline.

>>> python experimental_data.py extract /path/to/my/session/ [--dry=True]
>>> python experimental_data.py register /path/to/my/session/ [--dry=True]
>>> python experimental_data.py create /path/to/my/session/ [--dry=True]
>>> python experimental_data.py compress_video /path/to/my/session/ [--dry=True --count=4]
>>> python experimental_data.py audio_training /path/to/my/session/ [--dry=True --count=4]
"""

import argparse
import logging
from pathlib import Path

from ibllib.pipes.experimental_data import (extract, register, create, compress_video,
                                            audio_training)

logger = logging.getLogger('ibllib')

if __name__ == "__main__":
    ALLOWED_ACTIONS = ['create', 'extract', 'register', 'compress_video', 'audio_training']
    parser = argparse.ArgumentParser(description='Description of your program')
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
        extract(args.folder, dry=args.dry)
    elif args.action == 'register':
        register(args.folder, dry=args.dry)
    elif args.action == 'create':
        create(args.folder, dry=args.dry)
    elif args.action == 'compress_video':
        compress_video(args.folder, dry=args.dry, max_sessions=args.count)
    elif args.action == 'audio_training':
        audio_training(args.folder, dry=args.dry, max_sessions=args.count)
    else:
        logger.error('Allowed actions are: ' + ', '.join(ALLOWED_ACTIONS))
