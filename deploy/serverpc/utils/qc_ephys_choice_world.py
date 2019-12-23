import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from oneibl.one import ONE
import alf.io
from brainbox.core import Bunch

from ibllib.ephys import ephysqc
from ibllib.io.extractors import ephys_fpga, training_wheel, ephys_trials, biased_trials

# To log errors :
import logging
_logger = logging.getLogger('ibllib')

one = ONE()
# eid = one.search(subject='KS005', date_range='2019-08-30', number=1)[0]
# eid = one.search(subject='KS016', date_range='2019-12-05', number=1)[0]
# eid = one.search(subject='CSP004', date_range='2019-11-27', number=1)[0]
# eid = one.search(subject='CSHL028', date_range='2019-12-17', number=3)[0]

dtypes = [
    '_spikeglx_sync.channels',
    '_spikeglx_sync.polarities',
    '_spikeglx_sync.times',
    '_iblrig_taskSettings.raw',
    '_iblrig_taskData.raw',
    '_iblrig_encoderEvents.raw',
    '_iblrig_encoderPositions.raw',
    '_iblrig_encoderTrialInfo.raw',
    '_iblrig_Camera.timestamps',
    'ephysData.raw.meta',
    'ephysData.raw.wiring',
]

i = 0

def qc_ephys_session(eid, display=True):
    files = one.load(eid, dataset_types=dtypes, download_only=True)
    sess_path = alf.io.get_session_path(files[0])
    _logger.info(f"{i}, {sess_path}")

    temp_alf_folder = sess_path.joinpath('fpga_test', 'alf')
    temp_alf_folder.mkdir(parents=True, exist_ok=True)


    sync, chmap = ephys_fpga._get_main_probe_sync(sess_path, bin_exists=False)
    bpod_trials = ephys_trials.extract_all(sess_path, output_path=temp_alf_folder, save=True)
    # check that the output is complete
    fpga_trials = ephys_fpga.extract_behaviour_sync(sync, output_path=temp_alf_folder,
                                                    chmap=chmap, save=True, display=display)
    # align with the bpod
    ephys_fpga.align_with_bpod(temp_alf_folder.parent)
    alf_trials = alf.io.load_object(temp_alf_folder, '_ibl_trials')

    # do the QC
    qcs, qct = ephysqc.qc_fpga_task(fpga_trials, alf_trials)

    """
    Some sanity checks that are used as testing (not QC)
    """
    # checks that all matrices in qc_fpga_task have the same number of trials
    assert (np.size(np.unique([fpga_trials[k].shape[0] for k in fpga_trials])) == 1)
    # all trials have either valve open or error tone in and are mutually exclusive
    assert np.all(np.isnan(fpga_trials['valve_open'] * fpga_trials['error_tone_in']))
    assert np.all(np.logical_xor(np.isnan(fpga_trials['valve_open']),
                                 np.isnan(fpga_trials['error_tone_in'])))

eids = one.search(task_protocol='ephyschoice', dataset_types=dtypes)

OFFSET = 1  # extract the last trial when there is the passive protocol trailing...
for i, eid in enumerate(eids):
    if i < OFFSET:
        continue
    qc_ephys_session(eid, display=True)

# '53738f95-bd08-4d9d-9133-483fdb19e8da' valve_open / error_tone in: check channel maps as it's 3A
