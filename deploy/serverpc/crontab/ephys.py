"""
Entry point to system commands for IBL pipeline.

>>> python ephys.py extract_ephys /mnt/s0/Data/Subjects --dry=True --count=10
>>> python ephys.py raw_ephys_qc /mnt/s0/Data/Subjects --dry=True --count=10
>>> python ephys.py audio_ephys /mnt/s0/Data/Subjects --dry=True --count=5
>>> python ephys.py compress_ephys /mnt/s0/Data/Subjects --dry=True --count=5
>>> python ephys.py spike_sorting_qc /mnt/s0/Data/Subjects/KS002/2019-06-24/raw_ephys_data/probe00
>>> python ephys.py sync_merge_ephys /mnt/s0/Data/Subjects --dry=True
"""

import argparse
import logging
from pathlib import Path

import ibllib.pipes.experimental_data as pipes
from ibllib.ephys import ephysqc

logger = logging.getLogger('ibllib')


def _20_extract_ephys(ses_path, dry=True, max_sessions=10):
    pipes.extract_ephys(ses_path, dry=dry, max_sessions=max_sessions)


def _21_raw_ephys_qc(ses_path, dry=True, max_sessions=5):
    pipes.raw_ephys_qc(ses_path, dry=dry, max_sessions=max_sessions)


def _23_compress_ephys(ses_path, dry=True, max_sessions=5):
    pipes.compress_ephys(ses_path, dry=dry, max_sessions=max_sessions)


def _25_spike_sorting_qc(ks2_path):
    ephysqc._spike_sorting_metrics(ks2_path, save=True)


def _26_sync_merge_ephys(ses_path, dry=True):
    pipes.sync_merge_ephys(ses_path, dry=dry)


def _27_compress_ephys_videos(root_path, dry=True, max_sessions=20):
    pipes.compress_ephys_video(root_path, dry=dry, max_sessions=max_sessions)


if __name__ == "__main__":
    ALLOWED_ACTIONS = ['extract_ephys', 'raw_ephys_qc', 'audio_ephys', 'compress_ephys',
                       'spike_sorting_qc', 'sync_merge_ephys', 'compress_ephys_videos']
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
    if args.action == 'extract_ephys':
        _20_extract_ephys(ses_path=args.folder, dry=args.dry, max_sessions=args.count)
    elif args.action == 'raw_ephys_qc':
        _21_raw_ephys_qc(ses_path=args.folder, dry=args.dry, max_sessions=args.count)
    elif args.action == 'audio_ephys':
        pipes.compress_audio(args.folder, dry=args.dry, max_sessions=args.count)
    elif args.action == 'compress_ephys':
        _23_compress_ephys(ses_path=args.folder, dry=args.dry, max_sessions=args.count)
    elif args.action == 'spike_sorting_qc':
        _25_spike_sorting_qc(ks2_path=args.folder)
    elif args.action == 'sync_merge_ephys':
        _26_sync_merge_ephys(ses_path=args.folder, dry=args.dry)
    elif args.action == 'compress_ephys_videos':
        _27_compress_ephys_videos(root_path=args.folder, dry=args.dry, max_sessions=args.count)
    else:
        logger.error(f'Action "{args.action}" not valid. Allowed actions are: ' +
                     ', '.join(ALLOWED_ACTIONS))
