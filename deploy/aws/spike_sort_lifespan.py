# sudo docker compose exec spikesorter python /root/Documents/PYTHON/iblscripts/deploy/aws/spike_sort_lifespan.py 556b57db-e4bd-456b-b0bf-d0ddc56603ff
import argparse
from one.api import ONE
from ibllib.pipes.ephys_tasks import SpikeSorting, EphysPulses
from brainbox.io.one import SpikeSortingLoader
from ibllib.ephys import sync_probes


if __name__ == "__main__":
    # parse arguments with argparse, the first is the eid, the second is the probe name
    parser = argparse.ArgumentParser(description='Run spike sorting on a session')
    parser.add_argument('pid', help='The pid of the recording to sort')
    # add option for a dry run
    parser.add_argument('--dry-run', action='store_true', help='Do not run the spike sorting')
    args = parser.parse_args()
    eid = args.pid

    one = ONE(base_url='https://alyx.internationalbrainlab.org')
    ssl = SpikeSortingLoader(one=one, pid=args.pid)

    sync_job = EphysPulses(ssl.session_path, one=one, pname=ssl.pname,
                           sync_collection='raw_ephys_data/probe00', location="local")
    sync_job.run()
    ssjob = SpikeSorting(ssl.session_path, one=one, pname=ssl.pname,
                         device_collection='raw_ephys_data', location="local")
    ssjob.run()
    ssjob.register_datasets()
    sync_job.register_datasets()
