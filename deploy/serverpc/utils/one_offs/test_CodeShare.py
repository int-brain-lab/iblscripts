import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from oneibl.one import ONE
import alf.io
from brainbox.core import Bunch

from ibllib.ephys import ephysqc
from ibllib.io.extractors import ephys_fpga, training_wheel, ephys_trials, biased_trials
from ibllib.io import spikeglx
import ibllib.plots as iblplots

# To log errors : 
import logging
_logger = logging.getLogger('ibllib')
def _single_test(assertion, str_ok, str_ko):
    if assertion:
        _logger.info(str_ok)
        return True
    else:
        _logger.error(str_ko)
        return False

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
         '_iblrig_Camera.timestamps',
]

files = one.load(eid, dataset_types=dtypes, download_only=True)
sess_path = alf.io.get_session_path(files[0])

chmap = ephys_fpga.CHMAPS['3B']['nidq']
# chmap = ibllib.io.extractors.ephys_fpga.CHMAPS['3A']['ap']

"""get the sync pulses dealing with 3A and 3B revisions"""
if next(sess_path.joinpath('raw_ephys_data').glob('_spikeglx_sync.*'), None):
    # if there is nidq sync it's a 3B session
    sync_path = sess_path.joinpath(r'raw_ephys_data')
else:  # otherwise it's a 3A
    # TODO find the main sync probe
    # sync_path = sess_path.joinpath(r'raw_ephys_data', 'probe00')
    raise NotImplementedError("TODO: 3A detect main probe")
sync = alf.io.load_object(sync_path, '_spikeglx_sync', short_keys=True)

"""get the wheel data for both fpga and bpod"""
fpga_wheel = ephys_fpga.extract_wheel_sync(sync, chmap=chmap, save=False)
bpod_wheel = training_wheel.get_wheel_data(sess_path, save=False)

"""get the behaviour data for both fpga and bpod"""
# -- Out FPGA : 
# dict_keys(['ready_tone_in', 'error_tone_in', 'valve_open', 'stim_freeze', 'stimOn_times',
# 'iti_in', 'goCue_times', 'feedback_times', 'intervals', 'response_times'])
ephys_trials.extract_all(sess_path, save=True)
fpga_behaviour = ephys_fpga.extract_behaviour_sync(
    sync, output_path=sess_path.joinpath('alf'), chmap=chmap, save=True, display=True)
# -- Out BPOD :
# dict_keys(['feedbackType', 'contrastLeft', 'contrastRight', 'probabilityLeft',
# 'session_path', 'choice', 'rewardVolume', 'feedback_times', 'stimOn_times', 'intervals',
# 'response_times', 'camera_timestamps', 'goCue_times', 'goCueTrigger_times',
# 'stimOnTrigger_times', 'included'])
bpod_behaviour = biased_trials.extract_all(sess_path, save=False)

"""get the sync between behaviour and bpod"""
bpod_offset = ephys_fpga.align_with_bpod(sess_path)

"""get the camera pulses from FPGA"""
fpga_cam = ephys_fpga.extract_camera_sync(sync, output_path=sess_path.joinpath('alf'),
                                          chmap=chmap, save=False)

# ------------------------------------------------------
#          Start the QC part (Ephys only)
# ------------------------------------------------------
session_ephys_qc, trials_ephys_qc = ephysqc.fpga_behaviour(fpga_behaviour)



# TEST  Wheel should not move xx amount of time (quiescent period) before go cue
#       Wheel should move before feedback
# TODO ingest code from Michael S : https://github.com/int-brain-lab/ibllib/blob/brainbox/brainbox/examples/count_wheel_time_impossibilities.py 

# TEST  stim freeze,response_time,feedback_time should be very close in time
# TODO awaiting values from Nicco

# TEST  No frame2ttl change between stim off and go cue
# fpga_behavior['stimOff_times']

# TEST  No frame2ttl signal between stim freeze and stim off

# TEST  Number of Bonsai command to change screen should match Number of state change of frame2ttl (do test per trial)

# TEST  Between go tone and feedback (noise or reward, not no-go), frame2ttl should be changing at ~60Hz if wheel moves

# TEST  Order of events -- TODO Olivier you may have something ?

# ------------------------------------------------------
#          Start the QC PART (Bpod only)
# ------------------------------------------------------

array_size = np.zeros((4, 1))
array_size[0] = np.size(bpod_behaviour['stimOn_times'])
array_size[1] = np.size(bpod_behaviour['goCue_times'])
array_size[2] = np.size(bpod_behaviour['stimOnTrigger_times'])
array_size[3] = np.size(bpod_behaviour['goCueTrigger_times'])

trials_bpod_qc = Bunch({
    # TEST  StimOn, StimOnTrigger, GoCue and GoCueTrigger should all be within a very small tolerance of each other
    #       1. Check for Nans
    'stimOn_times_nan': ~np.any(np.isnan(bpod_behaviour['stimOn_times'])),
    'goCue_times_nan': ~np.any(np.isnan(bpod_behaviour['goCue_times'])),
    'stimOnTrigger_times_nan': ~np.any(np.isnan(bpod_behaviour['stimOnTrigger_times'])),
    'goCueTrigger_times_nan':~np.any(np.isnan(bpod_behaviour['goCueTrigger_times'])),
    #       2. Delay
    # TODO

})

# Test output at session level
session_bpod_qc = {k:np.all(trials_bpod_qc[k]) for k in trials_bpod_qc}

#  Data size test -- COULD REMOVE TODO
size_stimOn_goCue = [np.size(bpod_behaviour['stimOn_times']),
                     np.size(bpod_behaviour['goCue_times']),
                     np.size(bpod_behaviour['stimOnTrigger_times']),
                     np.size(bpod_behaviour['goCueTrigger_times'])]

session_bpod_qc['stimOn_times_goCue_times_size']= np.size(np.unique(size_stimOn_goCue)) == 1


# ------------------------------------------------------
#          Start the QC part (Bpod+Ephys)
# ------------------------------------------------------

# TEST  Compare times from the bpod behaviour extraction to the Ephys extraction
#dbpod_fpga = {}
#for k in ['goCue_times', 'stimOn_times']:
#    dbpod_fpga[k] = bpod_behaviour[k] - fpga_behaviour[k] + bpod_offset
#    # we should use the diff from trial start for a more accurate test but this is good enough for now
#    assert np.all(dbpod_fpga[k] < 0.05)

# TEST  if those numbers are equal: 
#       TTL for camera received by Bpod / FPGA
#       saved camera timestamps
#       saved camera frames
