import numpy as np
# TRAINING PARAMETERS ESTIMATIONS
N_LABS = np.array([10, 12])  # 10-12 experimental labs
N_MICE_PER_BATCH = np.array([4, 6])  # 4-6 mice at a time
N_TRAINING_DAYS_PER_YEAR = round(365*5/7)  # 261 days per year training
# 2 Go/camera/hour of recording at 25 Hz, cropped, mpeg compression, 850/570 framesize, 2 cams
# in practice we have 424Mb for 48 mins compressed at 29
# in practice we have 1320 for 70 mins compressed at 23
SIZE_TRAINING_VIDEO_HOURLY_GB = np.array([2, 4])  # size for all cameras (one)
TRAINING_SESSION_DURATION_HOURS = 1
# EPHYS PARAMETERS ESTIMATIONS
RECORDED_MICE_PER_BATCH = np.array([1, 2])  # 1-2 mice recorded per batch
TRAINING_CYCLE_DURATION_DAYS = 7 * 8  # 8 weeks from training to recording
RECORDINGS_PER_MOUSE = np.array([4, 6])  # recordings sessions per mouse
EPHYS_SESSION_DURATION_HOURS = 1
SIZE_RECORDING_VIDEO_HOURLY_GB = np.array([4, 8]) * 3

# Training sessions
# TODO behaviour ephys (including audio)
n_ses_per_year = N_TRAINING_DAYS_PER_YEAR * N_LABS * N_MICE_PER_BATCH
size_training_videos_yearly_Tb = n_ses_per_year * SIZE_TRAINING_VIDEO_HOURLY_GB * TRAINING_SESSION_DURATION_HOURS / 1024
print('Training')
print(str(n_ses_per_year), ' training sessions per year')
print(size_training_videos_yearly_Tb, 'Tb of training videos')

# Ephys

n_mice_rec_per_year = RECORDED_MICE_PER_BATCH * np.round(365 / TRAINING_CYCLE_DURATION_DAYS * N_LABS)  # IBL mice recorded per year
n_rec_per_year = RECORDINGS_PER_MOUSE * n_mice_rec_per_year

# video ephys
size_recording_video_yearly_Tb = SIZE_RECORDING_VIDEO_HOURLY_GB * n_rec_per_year * EPHYS_SESSION_DURATION_HOURS / 1024

# raw ephys
nprobes = 2
nchannels = 385
#Neuropixel 30kHz + 2kHz, 16bits, 385 channels, 2 probes
size_recording_neuropixel_hourly_Gb = 32000 * nchannels * 2 * 3600 / 1024 / 1024 / 1024 * nprobes
size_recording_neuropixel_yearly_Tb = size_recording_neuropixel_hourly_Gb * EPHYS_SESSION_DURATION_HOURS * n_rec_per_year * 1 / 1024

# histology
histology_size_per_mouse_Gb = 30
size_histology_yearly_Tb = n_mice_rec_per_year * histology_size_per_mouse_Gb / 1024

# TODO behaviour ephys (including audio)
# TODO processed spikes ephys

print('\n Recording')
print(n_rec_per_year, ' IBL recording sessions in a year')
print(size_recording_video_yearly_Tb, ' Tb of recording videos')
print(size_recording_neuropixel_yearly_Tb, ' Tb of recording neuropixel data')


# Training
# [10440 18792]  training sessions per year
# [20.390625 73.40625 ] Tb of training videos
#  Recording
# [260. 936.]  IBL recording sessions in a year
# [ 3.046875 21.9375  ]  Tb of recording videos
# [ 41.95142537 151.02513134]  Tb of recording neuropixel data
