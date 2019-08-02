"""
Entry point to system commands for IBL pipeline.

>>> python ephys.py extract /mnt/s0/Data/Subjects --dry=True --count=10
>>> python ephys.py qc /mnt/s0/Data/Subjects --dry=True --count=10
>>> python ephys.py ks2ibl /mnt/s0/Data/Subjects/ZM_1887/2019-07-19/001/raw_ephys_data/probe_left/_iblrig_ephysData.raw_g0_t0.imec.ap.bin  # NOQA
"""

import argparse
import logging
from pathlib import Path

import ibllib.pipes.experimental_data as pipes

logger = logging.getLogger('ibllib')


def extract(ses_path, dry=True, max_sessions=10):
    pipes.extract_ephys(ses_path, dry=dry, max_sessions=max_sessions)


def qc(ses_path, dry=True, max_sessions=5):
    pipes.qc_ephys(ses_path, dry=dry, max_sessions=max_sessions)


def ks2ibl(ap_file_path):
    # TODO this only a prototype, for now takes the bin ap file as input
    from phylib.io import alf, model
    ap_file_path = Path(ap_file_path)
    chem_out = ap_file_path.parents[2] / 'alf'
    # Path('link-to-textfile.txt').symlink_to(Path('textfile.txt'))
    m = model.TemplateModel(dir_path=ap_file_path.parent,
                            dat_path=ap_file_path,
                            sample_rate=30000,
                            n_channels_dat=385)
    ac = alf.EphysAlfCreator(m)
    ac.convert(chem_out)


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
    elif args.action == 'ks2ibl':
        ks2ibl(ap_file_path=args.folder)
    else:
        logger.error(f'Action "{args.action}" not valid. Allowed actions are: ' +
                     ', '.join(ALLOWED_ACTIONS))
