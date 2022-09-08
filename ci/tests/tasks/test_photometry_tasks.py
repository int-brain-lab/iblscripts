from pathlib import Path

from ibllib.io.extractors import fibrephotometry

# LOAD THE CSV FILES
FOLDER_RAW_PHOTOMETRY = Path("/home/ibladmin/Documents/olivier/fp_sync/rigs_data/photometry")
daily_folders = [f for f in FOLDER_RAW_PHOTOMETRY.glob('20*') if f.is_dir()]

for daily_folder in daily_folders:
    daq_files = list(daily_folder.glob("sync_*.tdms"))
    photometry_files = list(daily_folder.glob("raw_photometry*.csv"))
    daq_files.sort()
    photometry_files.sort()
    assert len(daq_files) == len(photometry_files)
    n_run = len(daq_files)
    for n in range(n_run):
        daq_file = daq_files[n]
        photometry_file = photometry_files[n]
        fibrephotometry.check_timestamps(daq_file, photometry_file)
        fibrephotometry.sync_photometry_to_daq(daq_file, photometry_file)
