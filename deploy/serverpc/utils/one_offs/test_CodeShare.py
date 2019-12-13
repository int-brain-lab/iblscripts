import warnings # to debug

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from oneibl.one import ONE
import alf.io

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
fpga_cam = ephys_fpga.extract_camera_sync(sync, output_path=sess_path.joinpath('alf'), chmap=chmap, save=False)


# ________________________________ FCT
def  session_test_on_trial(trials_qc):
    session_qc = {k:np.all(trials_qc[k]) for k in trials_qc}
    return session_qc
 

# ------------------------------------------------------
#          Start the QC part (Ephys only)
# ------------------------------------------------------

# Make a bunch gathering all trial QC
from brainbox.core import Bunch



trials_ephys_qc = Bunch({
    # TEST  StimOn and GoCue should all be within a very small tolerance of each other
    #       1. check for non-Nans
    'stimOn_times_nan': ~np.isnan(fpga_behaviour['stimOn_times']),  
    'goCue_times_nan': ~np.isnan(fpga_behaviour['goCue_times']),
    #       2. check goCue is after stimOn 
    'stimOn_times_before_goCue_times': fpga_behaviour['stimOn_times'] - fpga_behaviour['goCue_times'] > 0,
    #       3. check if closeby value
    'stimOn_times_goCue_times_diff': fpga_behaviour['stimOn_times'] - fpga_behaviour['goCue_times'] < 0.010,
    # TEST  Response times (from session start) should be increasing continuously
    #       Note: RT are not durations but time stamps from session start
    #       1. check for non-Nans
    'response_times_nan': ~np.isnan(fpga_behaviour['response_times']),
    #       2. check for positive increase
    'response_times_increase': np.diff(np.append([0], fpga_behaviour['response_times'])) > 0,
    # TEST  Response times (from goCue) should be positive
    'response_times_goCue_times_diff': fpga_behaviour['response_times'] - fpga_behaviour['goCue_times'] > 0,
    # TEST  1. Stim freeze should happen before feedback
    'stim_freeze_before_feedback': fpga_behaviour['stim_freeze'] - fpga_behaviour['feedback_times'] > 0,
    #       2. Delay between stim freeze and feedback <10ms
    'stim_freeze_delay_feedback': np.abs(fpga_behaviour['stim_freeze'] - fpga_behaviour['feedback_times']) < 0.010,
    # TEST  1. StimOff open should happen after valve
    'stimOff_after_valve': fpga_behaviour['stimOff_times'] - fpga_behaviour['valve_open'] > 0,
    #       2. Delay between valve and stim off should be 1s, added 0.1 as acceptable jitter
    'stimOff_delay_valve': fpga_behaviour['stimOff_times'] - fpga_behaviour['valve_open'] < 1.1,
    # TEST  Start of iti_in should be within a very small tolerance of the stim off
    'iti_in_delay_stim_off': np.abs(fpga_behaviour['stimOff_times'] - fpga_behaviour['iti_in']) < 0.01,
    # TEST  1. StimOff open should happen after noise
    'stimOff_after_noise': fpga_behaviour['stimOff_times'] - fpga_behaviour['error_tone_in'] > 0,
    #       2. Delay between noise and stim off should be 2s, added 0.1 as acceptable jitter
    'stimOff_delay_noise': fpga_behaviour['stimOff_times'] - fpga_behaviour['error_tone_in'] < 2.1,
    # TEST  1. Response_times should be before feedback
    'response_before_feedback': fpga_behaviour['feedback_times'] - fpga_behaviour['response_times'] > 0,
    #       2. Delay between wheel reaches threshold (response time) and feedback is 100us, acceptable jitter 500 us
    'response_feedback_delay': fpga_behaviour['feedback_times'] - fpga_behaviour['response_times'] < 0.0005,
    })


# Test output at session level
session_ephys_qc = {k:np.all(trials_ephys_qc[k]) for k in trials_ephys_qc}

#  Data size test  -- COULD REMOVE TODO
size_stimOn_goCue = [np.size(fpga_behaviour['stimOn_times']), np.size(fpga_behaviour['goCue_times'])]
size_response_goCue = [np.size(fpga_behaviour['response_times']), np.size(fpga_behaviour['goCue_times'])]

session_ephys_qc['stimOn_times_goCue_times_size'] = np.size(np.unique(size_stimOn_goCue)) == 1
session_ephys_qc['response_times_goCue_times_size'] = np.size(np.unique(size_response_goCue)) == 1


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
ephys_fpga.extract_camera_sync(sync, alf_path, save=save, chmap=sync_chmap)