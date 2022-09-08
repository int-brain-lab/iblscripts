from pathlib import Path
import argparse
import logging

from ibllib.io import spikeglx

_logger = logging.getLogger('ibllib')


def check_ephys_file(root_path, hash=False):
    root_path = Path(root_path)
    efiles = spikeglx.glob_ephys_files(root_path)
    for ef in efiles:
        for lab in ['nidq', 'ap', 'lf']:
            if not ef.get(lab, None):
                continue
            try:
                sr = spikeglx.Reader(ef[lab])
                if hash:
                    ok = sr.verify_hash()
                    if not ok:
                        raise ValueError("hashes don't match")
                _logger.info(f"PASS {ef[lab]}")
            except Exception:
                _logger.error(f"FAILED {ef[lab]} is corrupt !!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check ephys files')
    parser.add_argument('folder', help='Any folder')
    parser.add_argument('--hash', dest='hash', help='Computes hash and verify with meta-data',
                        const=True, default=False, nargs='?')
    args = parser.parse_args()  # returns data from the options specified (echo)
    root_path = Path(args.folder)
    assert root_path.exists()
    check_ephys_file(root_path, hash=args.hash)
