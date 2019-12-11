import warnings # to debug

import numpy as np
import matplotlib.pyplot as plt

from oneibl.one import ONE
import alf.io

import ibllib.io.extractors
from ibllib.io import spikeglx
import ibllib.plots as iblplots

one = ONE()
eid = one.search(subject='KS005', date_range='2019-08-30', number=1)[0]
eid = one.search(subject='KS016', date_range='2019-12-05', number=1)[0]
# eid = one.search(subject='CSHL_020', date_range='2019-12-04', number=5)[0]

one.alyx.rest('sessions', 'read', id=eid)['task_protocol']

one.list(eid)
dtypes = [
         '_spikeglx_sync.channels',
         '_spikeglx_sync.polarities',
         '_spikeglx_sync.times',
         '_iblrig_taskSettings.raw',
         '_iblrig_taskData.raw',
         '_iblrig_encoderEvents.raw',
         '_iblrig_encoderPositions.raw',
         '_iblrig_encoderTrialInfo.raw',
]

files = one.load(eid, dataset_types=dtypes, download_only=True)
sess_path = alf.io.get_session_path(files[0])

chmap = ibllib.io.extractors.ephys_fpga.CHMAPS['3B']['nidq']
# chmap = ibllib.io.extractors.ephys_fpga.CHMAPS['3A']['ap']

"""get the sync pulses dealing with 3A and 3B revisions"""
if next(sess_path.joinpath('raw_ephys_data').glob('_spikeglx_sync.*'), None):
    # if there is nidq sync it's a 3B session
    sync_path = sess_path.joinpath(r'raw_ephys_data')
else:  # otherwise it's a 3A
    # TODO find the main sync probe
    # sync_path = sess_path.joinpath(r'raw_ephys_data', 'probe00')
    pass
sync = alf.io.load_object(sync_path, '_spikeglx_sync', short_keys=True)

"""get the wheel data for both fpga and bpod"""
fpga_wheel = ibllib.io.extractors.ephys_fpga.extract_wheel_sync(sync, chmap=chmap, save=False)
bpod_wheel = ibllib.io.extractors.training_wheel.get_wheel_data(sess_path, save=False)

"""get the behaviour data for both fpga and bpod"""
# -- Out FPGA : 
# dict_keys(['ready_tone_in', 'error_tone_in', 'valve_open', 'stim_freeze', 'stimOn_times',
# 'iti_in', 'goCue_times', 'feedback_times', 'intervals', 'response_times'])
ibllib.io.extractors.ephys_trials.extract_all(sess_path, save=True)
fpga_behaviour = ibllib.io.extractors.ephys_fpga.extract_behaviour_sync(
    sync, output_path=sess_path.joinpath('alf'), chmap=chmap, save=True)

# -- Out BPOD :
# dict_keys(['feedbackType', 'contrastLeft', 'contrastRight', 'probabilityLeft',
# 'session_path', 'choice', 'rewardVolume', 'feedback_times', 'stimOn_times', 'intervals',
# 'response_times', 'camera_timestamps', 'goCue_times', 'goCueTrigger_times',
# 'stimOnTrigger_times', 'included'])
bpod_behaviour = ibllib.io.extractors.biased_trials.extract_all(sess_path, save=False)

"""get the sync between behaviour and bpod"""
bpod_offset = ibllib.io.extractors.ephys_fpga.align_with_bpod(sess_path)


## -----   PLOTS    -----
fix, axes = plt.subplots(nrows=2, sharex='all', sharey='all')
# axes[0].plot(t, pos), axes[0].title.set_text('Extracted')
axes[0].plot(fpga_wheel['re_ts'], fpga_wheel['re_pos']), axes[0].title.set_text('FPGA')
axes[0].plot(bpod_wheel['re_ts'] + bpod_offset, bpod_wheel['re_pos'])
axes[1].plot(bpod_wheel['re_ts'] + bpod_offset, bpod_wheel['re_pos'])
axes[1].title.set_text('Bpod')

# plt.figure(4)
# plt.plot(fpga_behaviour['intervals'][:, 0], bpod_behaviour['stimOn_times'] -
#         fpga_behaviour['stimOn_times'] + bpod_offset)

# plt.figure(5)
# plt.plot(fpga_behaviour['stimOn_times'] - fpga_behaviour['intervals'][:, 0] )

# ------------------------------------------------------
#          Start the QC part (Ephys only)
# ------------------------------------------------------

# TEST  Response times should be increasing continuously and non negative
#       Note: RT are not durations but time stamps
assert np.all(np.diff(fpga_behaviour['response_times']) > 0)
assert np.all(fpga_behaviour['response_times'] > 0)

# TEST  StimOn and GoCue should all be within a very small tolerance of each other
#       1. check for non-Nans
assert not np.any(np.isnan(fpga_behaviour['stimOn_times']))
assert not np.any(np.isnan(fpga_behaviour['goCue_times']))

#       2. check for similar size
array_size = np.zeros((2, 1))
array_size[0] = np.size(fpga_behaviour['stimOn_times'])
array_size[1] = np.size(fpga_behaviour['goCue_times'])
assert np.size(np.unique(array_size)) == 1

#       3. test if closeby value
dtimes_stimOn_goCue = {}
dtimes_stimOn_goCue = fpga_behaviour['goCue_times'] - fpga_behaviour['stimOn_times']
assert np.all(dtimes_stimOn_goCue < 0.05)

# ------------------------------------------------------
#          Start the QC part (Bpod+Ephys)
# ------------------------------------------------------

# TEST  Compare times from the bpod behaviour extraction to the Ephys extraction
dbpod_fpga = {}
for k in ['goCue_times', 'stimOn_times']:
    dbpod_fpga[k] = bpod_behaviour[k] - fpga_behaviour[k] + bpod_offset
    # we should use the diff from trial start for a more accurate test but this is good enough for now
    assert np.all(dbpod_fpga[k] < 0.05)

# ------------------------------------------------------
#          Start the QC PART (Bpod only)
# ------------------------------------------------------

# TEST  StimOn, StimOnTrigger, GoCue and GoCueTrigger should all be within a very small tolerance of each other
#       1. check for non-Nans
assert not np.any(np.isnan(bpod_behaviour['stimOn_times']))
assert not np.any(np.isnan(bpod_behaviour['goCue_times']))
assert not np.any(np.isnan(bpod_behaviour['stimOnTrigger_times']))
assert not np.any(np.isnan(bpod_behaviour['goCueTrigger_times']))

#       2. check for similar size
array_size = np.zeros((4, 1))
array_size[0] = np.size(bpod_behaviour['stimOn_times'])
array_size[1] = np.size(bpod_behaviour['goCue_times'])
array_size[2] = np.size(bpod_behaviour['stimOnTrigger_times'])
array_size[3] = np.size(bpod_behaviour['goCueTrigger_times'])
assert np.size(np.unique(array_size)) == 1