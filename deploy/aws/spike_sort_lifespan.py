import argparse
from one.api import ONE
from ibllib.pipes.ephys_tasks import SpikeSorting, EphysPulses
from pathlib import Path


from ibllib.ephys import sync_probes
if __name__ == "__main__":
    # parse arguments with argparse, the first is the eid, the second is the probe name
    parser = argparse.ArgumentParser(description='Run spike sorting on a session')
    parser.add_argument('pid', help='The pid of the recording to sort')
    parser.add_argument('pname', help='The name of the probe')
    # add option for a dry run
    parser.add_argument('--dry-run', action='store_true', help='Do not run the spike sorting')


    args = parser.parse_args()
    eid = args.eid
    pname = args.pname

    one = ONE(base_url='https://alyx.internationalbrainlab.org')
    session_path = one.eid2path(eid)
    session_path = Path('/mnt/s0/Data/Subjects').joinpath(*session_path.parts[-3:])
    complete = len(list(session_path.joinpath(f'alf/{pname}/iblsorter').glob('waveforms*'))) >= 4
    if complete:
        print(eid, pname, 'complete - skip')
        exit()
    else:
        print(eid, pname)
    if args.dry_run:
        print("Dry run - exiting...")
        exit()

    sync_job = EphysPulses(session_path, one=one, pname=pname,
                           sync_collection='raw_ephys_data/probe00', location="aws")
    sync_job.run()
    ssjob = SpikeSorting(session_path, one=one, pname=pname,
                         device_collection='raw_ephys_data', location="aws")
    ssjob.run()
    ssjob.register_datasets()
    sync_job.register_datasets()
