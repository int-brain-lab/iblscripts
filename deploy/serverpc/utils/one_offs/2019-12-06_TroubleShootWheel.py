import matplotlib.pyplot as plt
import ibllib.io.extractors

from oneibl.one import ONE
import alf.io

one = ONE()
eid = one.search(subject='KS005', date_range='2019-08-30', number=1)[0]

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
"""get the sync pulses"""
sync_path = sess_path.joinpath(r'raw_ephys_data')
sync = alf.io.load_object(sync_path, '_spikeglx_sync')
"""get the wheel data for both fpga and bpod"""
fpga_wheel = ibllib.io.extractors.ephys_fpga.extract_wheel_sync(sync, chmap=chmap, save=False)
bpod_wheel = ibllib.io.extractors.training_wheel.get_wheel_data(sess_path, save=False)
"""get the behaviour data for both fpga and bpod"""
# Out[5]: dict_keys(['ready_tone_in', 'error_tone_in', 'valve_open', 'stim_freeze', 'stimOn_times',
# 'iti_in', 'goCue_times', 'feedback_times', 'intervals', 'response_times'])
ibllib.io.extractors.ephys_trials.extract_all(sess_path, save=True)
fpga_behaviour = ibllib.io.extractors.ephys_fpga.extract_behaviour_sync(
    sync, output_path=sess_path.joinpath('alf'), chmap=chmap, save=True)
# Out[8]: dict_keys(['feedbackType', 'contrastLeft', 'contrastRight', 'probabilityLeft',
# 'session_path', 'choice', 'rewardVolume', 'feedback_times', 'stimOn_times', 'intervals',
# 'response_times', 'camera_timestamps', 'goCue_times', 'goCueTrigger_times',
# 'stimOnTrigger_times', 'included'])
bpod_behaviour = ibllib.io.extractors.biased_trials.extract_all(sess_path, save=False)
"""get the sync between behaviour and bpod"""
bpod_offset = ibllib.io.extractors.ephys_fpga.align_with_bpod(sess_path)


fix, axes = plt.subplots(nrows=2, sharex='all', sharey='all')
# axes[0].plot(t, pos), axes[0].title.set_text('Extracted')
axes[0].plot(fpga_wheel['re_ts'], fpga_wheel['re_pos']), axes[0].title.set_text('FPGA')
axes[0].plot(bpod_wheel['re_ts'] + bpod_offset, bpod_wheel['re_pos'])
axes[1].plot(bpod_wheel['re_ts'] + bpod_offset, bpod_wheel['re_pos'])
axes[1].title.set_text('Bpod')

plt.figure(2)
plt.plot(bpod_behaviour['stimOn_times'] - fpga_behaviour['stimOn_times'] + bpod_offset)
