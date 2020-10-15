from pathlib import Path
import argparse
import logging

from ibllib.pipes.scan_fix_passive_files import execute

log = logging.getLogger('ibllib')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scan and fix badly transfered passive sessions')
    parser.add_argument('root_data_folder', default='/mnt/s0/Data', help='Root data folder [/mnt/s0/Data]')
    parser.add_argument('--dry', required=False, default=False,
                        action='store_true', help='Dry run? default: False')
    args = parser.parse_args()  # returns data from the options specified (echo)
    root_path = Path(args.root_data_folder)
    if not root_path.exists():
        log.error(f"{root_path} does not exist")
    from_to_pairs, moved_ok = execute(root_path, dry=args.dry)
    if args.dry:
        log.info(from_to_pairs)