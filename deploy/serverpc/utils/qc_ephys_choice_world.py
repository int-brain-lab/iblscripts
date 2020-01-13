from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from oneibl.one import ONE
import alf.io
from brainbox.core import Bunch

from ibllib.ephys import ephysqc
from ibllib.io.extractors import ephys_fpga, training_wheel, ephys_trials, biased_trials
import ibllib.io.raw_data_loaders as rawio

# To log errors :
import logging
_logger = logging.getLogger('ibllib')

one = ONE()
# eid = one.search(subject='KS005', date_range='2019-08-30', number=1)[0]
# eid = one.search(subject='KS016', date_range='2019-12-05', number=1)[0]
# eid = one.search(subject='CSP004', date_range='2019-11-27', number=1)[0]
# eid = one.search(subject='CSHL028', date_range='2019-12-17', number=3)[0]

dtypes_search = [
    '_spikeglx_sync.channels',
    '_spikeglx_sync.polarities',
    '_spikeglx_sync.times',
    '_iblrig_taskSettings.raw',
    '_iblrig_taskData.raw',
    ]

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


def qc_ephys_session(sess_path, display=True):

    sess_path = Path(sess_path)
    temp_alf_folder = sess_path.joinpath('fpga_test', 'alf')
    temp_alf_folder.mkdir(parents=True, exist_ok=True)

    raw_trials = rawio.load_data(sess_path)
    tmax = raw_trials[-1]['behavior_data']['States timestamps']['exit_state'][0][-1] + 60

    sync, chmap = ephys_fpga._get_main_probe_sync(sess_path, bin_exists=False)
    bpod_trials = ephys_trials.extract_all(sess_path, output_path=temp_alf_folder, save=True)
    # check that the output is complete
    fpga_trials = ephys_fpga.extract_behaviour_sync(sync, output_path=temp_alf_folder, tmax=tmax,
                                                    chmap=chmap, save=True, display=display)
    # align with the bpod
    bpod2fpga = ephys_fpga.align_with_bpod(temp_alf_folder.parent)
    alf_trials = alf.io.load_object(temp_alf_folder, '_ibl_trials')

    # do the QC
    qcs, qct = ephysqc.qc_fpga_task(fpga_trials, alf_trials)

    # do the wheel part
    bpod_wheel = training_wheel.get_wheel_data(sess_path, save=False)
    fpga_wheel = ephys_fpga.extract_wheel_sync(sync, chmap=chmap, save=False)

    if display:

        t0 = max(np.min(bpod2fpga(bpod_wheel['re_ts'])), np.min(fpga_wheel['re_ts']))
        dy = np.interp(t0, fpga_wheel['re_ts'], fpga_wheel['re_pos']) - np.interp(
            t0, bpod2fpga(bpod_wheel['re_ts']), bpod_wheel['re_pos'])

        fix, axes = plt.subplots(nrows=2, sharex='all', sharey='all')
        # axes[0].plot(t, pos), axes[0].title.set_text('Extracted')
        axes[0].plot(bpod2fpga(bpod_wheel['re_ts']), bpod_wheel['re_pos'] + dy)
        axes[0].plot(fpga_wheel['re_ts'], fpga_wheel['re_pos'])
        axes[0].title.set_text('FPGA')
        axes[1].plot(bpod2fpga(bpod_wheel['re_ts']), bpod_wheel['re_pos'] + dy)
        axes[1].title.set_text('Bpod')

    return alf.io.dataframe({**fpga_trials, **alf_trials, **qct})


eids = one.search(task_protocol='ephyschoice', dataset_types=dtypes_search)
# eids = one.search(subject='CSHL049', date_range=['2020-01-11'], number=1)
DISPLAY = True
OFFSET = 0  # extract the last trial when there is the passive protocol trailing...
for i, eid in enumerate(eids):
    if i < OFFSET:
        continue
    files = one.load(eid, dataset_types=dtypes, download_only=True)
    if not any(files):
        continue
    sess_path = alf.io.get_session_path(files[0])
    try:
        _logger.info(f"{i}/{len(eids)} {eid} {sess_path}")
        qc_frame = qc_ephys_session(sess_path, display=DISPLAY)
    except Exception as e:
        _logger.error(f"{i}/{len(eids)} {eid} {sess_path}")
        _logger.error(str(e))
    break
# '53738f95-bd08-4d9d-9133-483fdb19e8da' passive stimulus issue ZM_1898 2019-12-11 002 : TODO integration test
#  'c6d5cea7-e1c4-48e1-8898-78e039fabf2b' trial 470 (t 2048-2052) has no feedback: KS023/2019-12-11/001

# import ibllib.plots as iplt
# plt.figure()
# dt = trials['intervals'][0, 0]  # first match
# dt = trials['intervals'][-1, 0] - trials['intervals_bpod'][-1, 0]  # last-match

# iplt.vertical_lines(trials['intervals_bpod'][:, 0] + dt)
# iplt.vertical_lines(trials['intervals'][:, 0])

# plt.figure()
# plt.plot(trials['intervals_bpod'][:500, 0] + trials['intervals'][0, 0] - trials['intervals'][:500, 0])


import ViewEphysQC
ViewEphysQC.viewqc(qc_frame)

a = 1
