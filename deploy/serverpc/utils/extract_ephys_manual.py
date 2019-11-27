from pathlib import Path
import shutil
import argparse

from ibllib.io import spikeglx
from ibllib.pipes import experimental_data as jobs


def re_extract_session(session_path):
    session_path = Path(session_path)
    DRY = False
    DELETE_PATTERNS = ['alf/channels.*.npy',
                       'alf/**/spikes.times*.npy',
                       'alf/clusters.*.npy',
                       'alf/params.py',
                       'alf/spikes.*.npy',
                       'alf/templates.*.npy',
                       'alf/whitening_mat_inv.npy',
                       'logs/*.*',
                       'extract_register.log',
                       'flatiron.flag',
                        'raw_ephys_data/**/_spikeglx_ephysQcFreqAP.freq*.npy',
                        'raw_ephys_data/**/_spikeglx_ephysQcFreqAP.power*.npy',
                        'raw_ephys_data/**/_spikeglx_ephysQcFreqLF.freq*.npy',
                        'raw_ephys_data/**/_spikeglx_ephysQcFreqLF.power*.npy',
                        'raw_ephys_data/**/_spikeglx_ephysQcTimeAP.rms*.npy',
                        'raw_ephys_data/**/_spikeglx_ephysQcTimeAP.times*.npy',
                        'raw_ephys_data/**/_spikeglx_ephysQcTimeLF.rms*.npy',
                        'raw_ephys_data/**/_spikeglx_ephysQcTimeLF.times*.npy',
                        'raw_ephys_data/**/*.sync.npy',
                        'raw_ephys_data/**/*.timestamps.npy',
                       ]

    RMTREE_PATTERNS = ['raw_ephys_data/**/ks2_alf',
                       'alf/tmp_merge']

    RENAMES = []

    ephys_files = spikeglx.glob_ephys_files(session_path)

    for dp in DELETE_PATTERNS:
        for match in session_path.glob(dp):
            print(match)
            if not DRY:
                match.unlink()

    for rmt in RMTREE_PATTERNS:
        for match in session_path.glob(rmt):
            print(match)
            if not DRY:
                shutil.rmtree(match)

    for ft in RENAMES:
        for f in session_path.rglob(ft[0]):
            print(f, ft[1])
            if not DRY:
                f.rename(f.parent.joinpath(ft[1]))

    ## 20_extract_ephys.sh
    # this should return SGLX found for probes if extraction was done previously
    session_path.joinpath('extract_ephys.flag').touch()
    jobs.extract_ephys(session_path)

    ## 21_raw_ephys_qc.sh
    # here we just create the flags (too long to run on the same process)
    session_path.joinpath('raw_ephys_qc.flag').touch()
    # jobs.raw_ephys_qc(session_path, dry=True)

    ## 26_sync_merge_ephys.sh
    for ef in ephys_files:
        if ef.get('ap'):
            if not ef.ap.parent.joinpath('spike_templates.npy').exists():
                continue
            print(ef.ap)
            ef.ap.parent.joinpath('sync_merge_ephys.flag').touch()
    jobs.sync_merge_ephys(session_path)

    # 23_compress_ephys
    # here we just create the flags (too long to run on the same process)
    for ef in ephys_files:
        if ef.get('ap'):
            print(ef.ap)
            if not ef.ap.parent.joinpath('spike_templates.npy').exists():
                continue
            ef.ap.parent.joinpath('compress_ephys.flag').touch()
    # jobs.compress_ephys(session_path, dry=True)
    # 22_audio_ephys.sh
    # 27_compress_ephys_videos.sh
    # 28_dlc_ephys.sh


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Re-extraction of ephys sessions')
    parser.add_argument('folder', help='Session folder: /path/to/subject/yyyy-mm-dd/XXX')
    args = parser.parse_args()  # returns data from the options specified (echo)
    ses_path = Path(args.folder)
    assert(ses_path.exists())
    re_extract_session(ses_path)
