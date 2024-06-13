import argparse
import os
from pathlib import Path
import shutil

from one.api import ONE

from ibllib.pipes.sdsc_tasks import RegisterSpikeSortingSDSC


quarantine_paths = [
    Path('/mnt/ibl/quarantine/tasks/SpikeSorting'),
    Path('/mnt/ibl/quarantine/tasks_olivier/SpikeSorting'),
    Path('/mnt/ibl/quarantine/tasks_owinter/SpikeSorting'),
    Path('/mnt/ibl/quarantine/tasks_mfaulkner/SpikeSorting'),
    Path('/mnt/ibl/quarantine/tasks_mwells/SpikeSorting'),
    Path('/mnt/ibl/quarantine/tasks_clangfield/SpikeSorting'),
    Path('/mnt/ibl/quarantine/tasks_external/SpikeSorting'),
]

revision_label = '#2024-05-06#'
one = ONE()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pid")
    args = parser.parse_args()
    pid = args.pid
    eid, probe = one.pid2eid(pid)
    session_path = one.eid2path(eid)

    QUAR_PATH = None
    for quarantine_path in quarantine_paths:
        data_path = quarantine_path.joinpath(session_path.relative_to(one.cache_dir)).joinpath("alf", probe, "pykilosort")
        if data_path.exists():
            QUAR_PATH = quarantine_path
            break
    if QUAR_PATH is None:
        raise FileNotFoundError(f"Spike sorting for {pid} was nowhere to be found")

    os.environ["SDSC_PATCH_PATH"] = str(QUAR_PATH.parent)

    new_path = Path('/home/datauser/temp/RegisterSpikeSortingSDSC/').joinpath(
        *session_path.parts[-5:], 'alf', probe, 'pykilosort', revision_label)
    if new_path.exists():
        shutil.rmtree(new_path)
    shutil.copytree(data_path, new_path)

    new_path.joinpath('params.py').unlink()
    new_path.joinpath('whitening_mat_inv.npy').unlink()
    new_path.joinpath('cluster_KSLabel.tsv').unlink()

    task = RegisterSpikeSortingSDSC(session_path, pname=probe, location="SDSC", one=one, revision_label=revision_label)
    print(task.session_path)
    out = task.run()
    print(task.outputs)
    response = task.register_datasets(default=False, force=True)
    task.cleanUp()

    print(f"\t\tDONE: {pid}")
