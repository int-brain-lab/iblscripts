from pathlib import Path

from oneibl.one import ONE
import alf.io
from ibllib.pipes import training_preprocessing
import ibllib.io.raw_data_loaders as rawio

ROOT_PATH = Path('/mnt/s0/Data/Subjects')
DRY = False

avi_files = ROOT_PATH.rglob('_iblrig_leftCamera.raw.avi')
one = ONE()

for avi_file in avi_files:
    session_path = alf.io.get_session_path(avi_file)
    session_type = rawio.get_session_extractor_type(session_path)
    print(session_path, session_type)
    if DRY:
        continue
    if session_type in [False, 'biased', 'biased', 'habituation', 'training']:
        task = training_preprocessing.TrainingVideoCompress(session_path)
        status = task.run()
        if status == 0 and task.outputs is not None:
            # on a successful run, if there is no data to register, set status to Empty
            registered_dsets = task.register_datasets(one=one, job_id='gnagna')
