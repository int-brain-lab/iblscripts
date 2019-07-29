from pathlib import Path
from datetime import datetime
import csv
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cbook import flatten

import cv2  # pip install opencv-python

import alf.io
import ibllib.dsp as dsp
import ibllib.io.spikeglx
import ibllib.plots
import ibllib.io.extractors.ephys_fpga as ephys_fpga

_logger = logging.getLogger('ibllib')

"""
Entry point to system commands for IBL pipeline.
Select your environment and run the
>>> cd ~/Documents/PYTHON/iblscripts/deploy/serverpc/ephys/
>>> source ~/Documents/PYTHON/envs/iblenv/bin/activate
>>> python synchronization_protocol.py /datadisk/Local/20190710_sync_test

pip install opencv-python to install cv2 dependency on top of ibl environment

This script test temporal synchronisation of
the bpod, cameras and neuropixels probes in saline.
There are 500 square pulses coming from the fpga,
each of lenghts 500 ms, send through 4 system components:
fpga (the sync signal), an LED light (picked up by cameras),
2 Neuropixels probes in saline and bpod.
Each block tests if the component detected
all 500 square pulses of the test signal
and asserts that the temporal jitter is below a threshold.
It is further tested if the camera time stamps between
the two probes equal in number and have a small temporal jitter.
The script assumes Guido's folder structure.

Include '_iblrig_leftCamera.raw.avi' in vids;
(I took it out as this video was faulty in
Guido's data).

This is the expected input tree for the sync check to run properly:
Pay particular attention to the naming of ephys files.
/datadisk/Local/20190710_sync_test
├── bpod
│   ├── _iblrig_taskCodeFiles.raw.zip
│   ├── _iblrig_taskData.raw.jsonable
│   └── _iblrig_taskSettings.raw.json
├── ephys
│   ├── 20190709_sync_left_g0_t0.imec.ap.bin
│   ├── 20190709_sync_left_g0_t0.imec.ap.meta
│   ├── 20190709_sync_left_g0_t0.imec.lf.bin
│   ├── 20190709_sync_left_g0_t0.imec.lf.meta
│   ├── 20190709_sync_right_g0_t0.imec.ap.bin
│   ├── 20190709_sync_right_g0_t0.imec.ap.meta
│   ├── 20190709_sync_right_g0_t0.imec.lf.bin
│   ├── 20190709_sync_right_g0_t0.imec.lf.meta
└── video
    ├── _iblrig_bodyCamera.raw.avi
    ├── _iblrig_bodyCamera.raw_timestamps.ssv
    ├── _iblrig_leftCamera.raw.avi
    ├── _iblrig_leftCamera.raw_timestamps.ssv
    ├── _iblrig_rightCamera.raw.avi
    └── _iblrig_rightCamera.raw_timestamps.ssv
"""

SHOW_PLOTS = True  # of histograms and wavefronts

###########
'''
ephys - for both probes
'''
###########


def get_ephys_data(raw_ephys_apfile, label=''):
    """
    E.g.
    raw_ephys_apfile =
    sync_test_folder + '/ephys/20190709_sync_right_g0_t0.imec.ap.bin'
    """

    if alf.io.exists(raw_ephys_apfile.parent, '_spikeglx_sync', glob=[label]):
        sync = alf.io.load_object(raw_ephys_apfile.parent, '_spikeglx_sync',
                                  glob=[label], short_keys=True)
    else:
        sync = ephys_fpga._sync_to_alf(raw_ephys_apfile, parts=label, save=True)
    # load reader object, and extract sync traces
    sr = ibllib.io.spikeglx.Reader(raw_ephys_apfile)
    assert sr.fs == 30000, 'sampling rate is not 30 kHz, adjust script!'
    _logger.info('extracted %s' % raw_ephys_apfile)
    return sr, sync


def compare_camera_timestamps_between_two_probes(sync_right, sync_left):
    """
    sync_left has no square signal
    """
    # using the probe 3a channel map:
    '''
    0: Arduino synchronization signal
    2: 150 Hz camera
    3: 30 Hz camera
    4: 60 Hz camera
    7: Bpod
    11: Frame2TTL
    12 & 13: Rotary Encoder
    15: Audio
    '''

    for cam_code in [2, 3, 4]:

        cam_times_left = ephys_fpga._get_sync_fronts(sync_left, cam_code)['times']
        cam_times_right = ephys_fpga._get_sync_fronts(sync_right, cam_code)['times']

        assert len(cam_times_left) == len(
            cam_times_right), "# time stamps don't match between probes"

        D = abs(np.array(cam_times_left) - np.array(cam_times_right))

        assert max(D) - min(D) < 0.005, 'cam_code %s; Temporal jitter \
                between probes is large!!' % cam_code

        print('Sync check for cam %s time stamps \
                of left and right probe passed' % cam_code)

        print('mean = ', np.round(np.mean(D), 6),
              'sec ; std = ', np.round(np.std(D), 6),
              'sec ; max - min = ', np.round(max(D) - min(D), 6), 'sec')


def first_occ_index(array, n_at_least):
    """
    Getting index of first occurence in boolean array
    with at least n consecutive False entries
    """
    curr_found_false = 0
    curr_index = 0
    for index, elem in enumerate(array):
        if not elem:
            if curr_found_false == 0:
                curr_index = index
            curr_found_false += 1
            if curr_found_false == n_at_least:
                return curr_index
        else:
            curr_found_false = 0


def event_extraction_and_comparison(sr):

    # it took 8 min to run that for 6 min of data, all 300 ish channels
    # silent channels for Guido's set:
    # [36,75,112,151,188,227,264,303,317,340,379,384]

    # sr,sync=get_ephys_data(sync_test_folder)
    """
    this function first finds the times of square signal fronts in ephys and
    compares them to corresponding ones in the sync signal.
    Iteratively for small data chunks
    """

    print('starting event_extraction_and_comparison')
    period_duration = 30000  # in observations, 30 kHz
    BATCH_SIZE_SAMPLES = period_duration  # in observations, 30 kHz

    # only used to find first pulse
    wg = dsp.WindowGenerator(sr.ns, BATCH_SIZE_SAMPLES, overlap=1)

    # if the data is needed as well, loop over the file
    # raw data contains raw ephys traces, while raw_sync contains the 16 sync
    # traces

    rawdata, _ = sr.read_samples(0, BATCH_SIZE_SAMPLES)
    _, chans = rawdata.shape

    chan_fronts = {}
    sync_fronts = {}

    sync_fronts['fpga up fronts'] = []
    sync_fronts['fpga down fronts'] = []

    for j in range(chans):
        chan_fronts[j] = {}
        chan_fronts[j]['ephys up fronts'] = []
        chan_fronts[j]['ephys down fronts'] = []

    # find time of first pulse (take first channel with square signal)
    for i in range(chans):

        try:

            # assuming a signal in the first minute
            for first, last in list(
                    wg.firstlast)[:60]:

                rawdata, rawsync = sr.read_samples(first, last)

                diffs = np.diff(rawsync.T[0])
                sync_up_fronts = np.where(diffs == 1)[0] + first

                if len(sync_up_fronts) != 0:
                    break

            assert len(sync_up_fronts) != 0
            Channel = i
            break

        except BaseException:
            print('channel %s shows no pulse signal, checking next' % i)
            assert i < 10, \
                "something wrong, \
                the first 10 channels don't show a square signal"
            continue

    start_of_chopping = sync_up_fronts[0] - period_duration / 4

    k = 0

    # assure there is exactly one pulse per cut segment

    for pulse in range(500):  # there are 500 square pulses
        first = int(start_of_chopping + period_duration * pulse)
        last = int(first + period_duration)

        if k % 100 == 0:
            print('segment %s of %s' % (k, 500))

        k += 1

        rawdata, rawsync = sr.read_samples(first, last)

        # get fronts for sync signal
        diffs = np.diff(rawsync.T[0])  # can that thing be a global variable?
        sync_up_fronts = np.where(diffs == 1)[0] + first
        sync_down_fronts = np.where(diffs == -1)[0] + first
        sync_fronts['fpga up fronts'].append(sync_up_fronts)
        sync_fronts['fpga down fronts'].append(sync_down_fronts)

        # get fronts for only one valid ephys channel
        obs, chans = rawdata.shape

        i = Channel

        Mean = np.median(rawdata.T[i])
        Std = np.std(rawdata.T[i])

        ups = np.invert(rawdata.T[i] > Mean + 6 * Std)
        downs = np.invert(rawdata.T[i] < Mean - 6 * Std)

        up_fronts = []
        down_fronts = []
        # Activity front at least 10 samples long (empirical)

        up_fronts.append(first_occ_index(ups, 20) + first)
        down_fronts.append(first_occ_index(downs, 20) + first)

        chan_fronts[i]['ephys up fronts'].append(up_fronts)
        chan_fronts[i]['ephys down fronts'].append(down_fronts)

    return chan_fronts, sync_fronts  # all differences


def evaluate_ephys(chan_fronts, sync_fronts):
    """
    check number of detected square pulses and temporal jitter
    """

    # check if all signals have been detected
    L_sync_up = list(flatten(sync_fronts['fpga up fronts']))
    L_sync_down = list(flatten(sync_fronts['fpga down fronts']))

    assert len(L_sync_up) == 500, 'not all fpga up fronts detected'
    assert len(L_sync_down) == 500, 'not all fpga down fronts detected'

    for i in range(len(chan_fronts)):

        try:

            L_chan_up = list(flatten(chan_fronts[i]['ephys up fronts']))
            L_chan_down = list(flatten(chan_fronts[i]['ephys down fronts']))

            assert len(L_chan_up) == 500, \
                'not all ephys up fronts detected'
            assert len(L_chan_down) == 500, \
                'not all ephys down fronts detected'

            break

        except BaseException:

            continue

    ups_errors = np.array(L_chan_up) - np.array(L_sync_up)
    downs_errors = np.array(L_chan_down) - np.array(L_sync_down)

    MAX = max([max(ups_errors), max(downs_errors)])

    if MAX > 20:
        print('ATTENTION, the maximal error is unusually high, %s sec' %
              str(MAX / 30000.))

    print('ephys test passed')

    if SHOW_PLOTS:
        plt.figure('histogram')

        #  pool up front and down front temporal errors
        Er = [np.array(ups_errors), np.array(downs_errors)]
        f = np.reshape(Er, 1000) / 30000.

        plt.hist(f)
        plt.xlabel('error between fpga fronts and ephys fronts in sec')


###########
'''
video
'''
###########


def convert_pgts(time):
    """Convert PointGray cameras timestamps to seconds.
    Use convert then uncycle"""
    # offset = time & 0xFFF
    cycle1 = (time >> 12) & 0x1FFF
    cycle2 = (time >> 25) & 0x7F
    seconds = cycle2 + cycle1 / 8000.
    return seconds


def uncycle_pgts(time):
    """Unwrap the converted seconds of a PointGray camera timestamp series."""
    cycles = np.insert(np.diff(time) < 0, 0, False)
    cycleindex = np.cumsum(cycles)
    return time + cycleindex * 128


def get_video_stamps_and_brightness(sync_test_folder):

    # for each frame in the video, set 1 or zero corresponding to LED status
    startTime = datetime.now()

    d = {}
    # No '_iblrig_leftCamera.raw.avi',
    # took it out, as it was faulty in Guido's data
    vids = ['_iblrig_bodyCamera.raw.avi', '_iblrig_rightCamera.raw.avi']

    # maybe 12 min in total for the 30 Hz and 60 Hz videos

    for vid in vids:
        video_path = sync_test_folder + '/video/' + vid

        print('Loading video, this takes some minuts:', video_path)
        cap = cv2.VideoCapture(video_path)
        # fps = cap.get(cv2.CAP_PROP_FPS)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        brightness = np.zeros(frameCount)

        # for each frame, save brightness in array
        for i in range(frameCount):
            cap.set(1, i)
            _, frame = cap.read()
            brightness[i] = np.sum(frame)

        with open(video_path[:-4] + '_timestamps.ssv', 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=' ')
            ssv_times = np.array([line for line in csv_reader])

        ssv_times_sec = [convert_pgts(int(time)) for time in ssv_times[:, 0]]
        uncycle_pgts(ssv_times_sec)

        d[vid] = [brightness, uncycle_pgts(ssv_times_sec)]

    cap.release()
    print(datetime.now() - startTime)
    np.save(sync_test_folder + '/video/brightness.npy', d)
    return d


def evaluate_camera_sync(d, sync):

    # d=get_video_stamps_and_brightness(sync_test_folder)
    # sr, sync, rawdata, rawsync=get_ephys_data(sync_test_folder)

    # using the probe 3a channel map:
    '''
    0: Arduino synchronization signal
    2: 150 Hz camera
    3: 30 Hz camera
    4: 60 Hz camera
    7: Bpod
    11: Frame2TTL
    12 & 13: Rotary Encoder
    15: Audio
    '''
    y = {
        '_iblrig_bodyCamera.raw.avi': 3,
        '_iblrig_rightCamera.raw.avi': 4,
        '_iblrig_leftCamera.raw.avi': 2}

    s3 = ephys_fpga._get_sync_fronts(sync, 0)  # get arduino sync signal

    for vid in d:
        # threshold brightness time-series of the camera to have it in {-1,1}
        r3 = [1 if x > np.mean(d[vid][0]) else -1 for x in d[vid][0]]
        # fpga cam time stamps
        cam_times = ephys_fpga._get_sync_fronts(sync, y[vid])['times']

        # assuming at the end the frames are dropped
        drops = len(cam_times) - len(r3) * 2

        # check if an extremely high number of frames is dropped at the end
        assert drops < 500, '%s frames dropped for %s!!!' % (drops, vid)

        # get fronts of video brightness square signal
        diffr3 = np.diff(r3)  # get signal jumps via differentiation
        fronts_brightness = []
        for i in range(len(diffr3)):
            if diffr3[i] != 0:
                fronts_brightness.append(cam_times[:-drops][0::2][i])

        # check if all 500 square pulses are detected
        assert len(fronts_brightness) == len(
            s3['times']), 'Not all square signals detected in %s!' % vid

        # get temporal difference between fpga wave fronts and brightness wave
        # fronts
        D = fronts_brightness - s3['times']

        assert len(
            D) == 1000, \
            'not all 500 pulses were detected \
            by fpga and brightness in %s!' % vid

        print(' %s, Wave fronts temp. diff, in sec: \
            mean = %s, std = %s, max = %s' % (vid,
                                              np.round(np.mean(abs(D)), 4),
                                              np.round(np.std(abs(D)), 4),
                                              np.round(max(abs(D)), 4)))

        # check if temporal jitter between fpga and brightness wavefronts is
        # below 100 ms
        assert max(
            abs(D)) < 0.200, \
            'Jitter between fpga and brightness fronts is large!!'

        if SHOW_PLOTS:

            plt.figure('wavefronts, ' + vid)
            ibllib.plots.squares(
                s3['times'],
                s3['polarities'],
                label='fpga square signal',
                marker='o')
            plt.plot(cam_times[:-drops][0::2],
                     r3,
                     alpha=0.5,
                     label='thresholded video brightness',
                     linewidth=2,
                     marker='x')

            plt.legend()
            plt.title('wavefronts for fpga and brightness of %s' % vid)
            plt.show()

            plt.figure('histogram of front differences, %s' % vid)
            plt.title('histogram of temporal errors of fronts')
            plt.hist(D)
            plt.xlabel('error between fpga fronts and ephys fronts in sec')
            plt.show()


##########
'''
BPod
'''
##########


def compare_bpod_json_with_fpga(sync_test_folder, sync):
    '''
    sr, sync=get_ephys_data(sync_test_folder)
    '''

    #  get the bpod signal from the jasonable file
    import json
    with open(sync_test_folder + '/bpod/_iblrig_taskData.raw.jsonable') as fid:
        out = json.load(fid)

    ins = out['Events timestamps']['BNC1High']
    outs = out['Events timestamps']['BNC1Low']

    assert len(ins) == 500, 'not all pulses detected in bpod!'
    assert len(ins) == len(outs), 'not all fronts detected in bpod signal!'

    # get the fpga signal from the sync object
    s3 = ephys_fpga._get_sync_fronts(sync, 0)  # 3b channel map

    assert len(s3['times']) == 1000, 'not all fronts detected in fpga signal!'

    offset_on = np.mean(
        np.array(s3['times'][1::2]) - np.array(outs))  # get delay
    offset_off = np.mean(np.array(s3['times'][0::2]) - np.array(ins))

    jitter_on = np.std(
        np.array(s3['times'][1::2]) - np.array(outs))  # get jitter
    jitter_off = np.std(np.array(s3['times'][0::2]) - np.array(ins))

    inter_pulse_interval_bpod = abs(np.array(ins) - np.array(outs))
    inter_pulse_interval_fpga = abs(
        np.array(s3['times'][1::2]) - np.array(s3['times'][0::2]))

    print(
        'maximal bpod jitter in sec: ',
        np.round(
            max(inter_pulse_interval_bpod) -
            min(inter_pulse_interval_bpod),
            6))
    print(
        'maximal fpga jitter in sec: ',
        np.round(
            max(inter_pulse_interval_fpga) -
            min(inter_pulse_interval_fpga),
            6))
    print('maximal bpod-fpga in sec: ',
          np.round(max(abs(np.array(s3['times'][1::2]) - np.array(outs))) -
                   min(abs(np.array(s3['times'][1::2]) - np.array(outs)), 6)))

    print('The fpga 500 ms square signal and \
          the bpod 500 ms square signal are offset',
          'by %s sec and the difference between them has std %s sec'
          % (np.round(np.mean([offset_on, offset_off]), 6),
             np.round(np.mean([jitter_on, jitter_off]), 6)))

    if SHOW_PLOTS:

        plt.figure('wavefronts')
        plt.plot(s3['times'], s3['polarities'], label='fpga')
        plt.plot(
            ins,
            np.ones(
                len(ins)),
            linestyle='',
            marker='o',
            label='pbod on')
        plt.plot(
            outs,
            np.ones(
                len(outs)),
            linestyle='',
            marker='x',
            label='bpod off')
        plt.legend()
        plt.show()

        plt.figure('histogram of wavefront differences, bpod and fpga')

        plt.hist(np.array(s3['times'][1::2]) - np.array(outs))
        plt.xlabel('error between fpga fronts and ephys fronts in sec')
        plt.show()


def run_synchronization_protocol(sync_test_folder):
    # running all tests took 12 min for Guido's example data
    startTime = datetime.now()

    ap_file_left = list(Path(sync_test_folder).rglob('*left*.imec.ap.bin'))[0]
    ap_file_right = list(Path(sync_test_folder).rglob('*right*.imec.ap.bin'))[0]

    # load and compare sync signals between the two ephys probes
    sr_left, sync_left = get_ephys_data(ap_file_left, 'left')
    sr_right, sync_right = get_ephys_data(ap_file_right, 'right')
    _logger.info('Compare the camera timestamps between the two probes')
    compare_camera_timestamps_between_two_probes(sync_right, sync_left)

    # compare ephys fronts with fpga pulse signal for right probe
    _logger.info('compare ephys fronts with fpga pulse signal for right probe')
    chan_fronts, sync_fronts = event_extraction_and_comparison(sr_right)
    evaluate_ephys(chan_fronts, sync_fronts)

    # do camera check
    _logger.info('Evaluate Camera sync')
    d = get_video_stamps_and_brightness(sync_test_folder)
    evaluate_camera_sync(d, sync_right)

    # do bpod check
    _logger.info('Evaluate Bpod sync')
    compare_bpod_json_with_fpga(sync_test_folder, sync_right)

    _logger.warning('All tests passed !!', datetime.now() - startTime)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('folder', help='A Folder containing a session')
    parser.add_argument('--dry', help='Dry Run', required=False, default=False, type=str)
    parser.add_argument('--count', help='Max number of sessions to run this on',
                        required=False, default=False, type=int)
    args = parser.parse_args()  # returns data from the options specified (echo)
    if args.dry and args.dry.lower() == 'false':
        args.dry = False
    assert(Path(args.folder).exists())
