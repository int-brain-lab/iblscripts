from pathlib import Path
from datetime import datetime
import csv
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cbook import flatten

import cv2  # pip install opencv-python

import ibllib.io.spikeglx
import ibllib.plots

from DemoReadSGLXData.readSGLX import readMeta, SampRate, makeMemMapRaw, ExtractDigital

SHOW_PLOTS = False
_logger = logging.getLogger('ibllib')

"""
That's for 3b probes! Here only up fronts are compared.
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

20190801_sync_test_SWC/
    video/
        _iblrig_rightCamera.raw.avi
        _iblrig_rightCamera.raw_timestamps.ssv
        _iblrig_bodyCamera.raw_timestamps.ssv
        _iblrig_leftCamera.raw_timestamps.ssv
        _iblrig_bodyCamera.raw.avi
        _iblrig_leftCamera.raw.avi
    ephys/
        sync_right_g0_t0.imec.ap.meta
        sync_LED_g1_t0.nidq.bin
        sync_LED_g1_t0.nidq.meta
        sync_left_g0_t0.imec.ap.bin
        sync_left_g0_t0.imec.ap.meta
        sync_right_g0_t0.imec.ap.bin
    bpod/
        _iblrig_taskSettings.raw.json
        _iblrig_taskCodeFiles.raw.zip
        _iblrig_taskData.raw.jsonable

"""

###########
'''
ephys - for both probes
'''
###########


def get_3b_sync_signal(binFullPath):

    # For a digital channel: zero based index of the digital word in
    # the saved file. For imec data there is never more than one digital word.
    dw = 0

    # Zero-based Line indicies to read from the digital word and plot.
    # For 3B2 imec data: the sync pulse is stored in line 6.
    dLineList = [0, 1, 2, 3, 7]
    dlabel = ['Cam60Hz', 'Cam150Hz', 'Cam30Hz', 'Imec', 'Arduino']

    # Read in metadata; returns a dictionary with string for values
    meta = readMeta(binFullPath)

    tStart = 0
    tEnd = int(float(meta['fileTimeSecs']))

    # parameters common to NI and imec data
    sRate = SampRate(meta)
    firstSamp = int(sRate * tStart)
    lastSamp = int(sRate * tEnd)

    rawData = makeMemMapRaw(binFullPath, meta)
    digArray = ExtractDigital(rawData, firstSamp, lastSamp, dw,
                              dLineList, meta)
    sync = {}

    # save it in sec
    for i in range(len(dlabel)):
        sync[dlabel[i]] = {}
        sync[dlabel[i]]['timeStampsSec'] = np.arange(len(digArray[i])) / sRate
        sync[dlabel[i]]['values'] = digArray[i]
    return sync


def get_ephys_data(raw_ephys_apfile):
    """
    That's the analog signal from the ap.bin file
    """

    # load reader object, and extract sync traces
    sr = ibllib.io.spikeglx.Reader(raw_ephys_apfile)
    assert int(sr.fs) == 30000, 'sampling rate is not 30 kHz, adjust script!'
    _logger.info('extracted %s' % raw_ephys_apfile)
    return sr


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


def front_extraction_from_arduino_and_ephys(sr, sync):

    # it took 8 min to run that for 6 min of data, all 300 ish channels
    # silent channels for Guido's set:
    # [36,75,112,151,188,227,264,303,317,340,379,384]

    # sr,sync=get_ephys_data(sync_test_folder)
    """
    this function first finds the times of square signal fronts in ephys and
    compares them to corresponding ones in the sync signal.
    It assumes the sync signal is within ms aligned to the rawdata signal.
    Iteratively for small data chunks
    """

    _logger.info('starting event_extraction_and_comparison')
    period_duration = 30000  # in observations, 30 kHz
    BATCH_SIZE_SAMPLES = period_duration  # in observations, 30 kHz

    rawdata, _ = sr.read_samples(0, BATCH_SIZE_SAMPLES)
    _, chans = rawdata.shape

    chan_fronts = {}
    sync_fronts = {}

    # This is to correct for different sampling rates
    diffs = np.diff(sync['Arduino']['values'])
    sync_fronts['fpga up fronts'] = sync['Arduino']['timeStampsSec'][np.where(diffs == 1)[0]]

    # get times of fronts in smaples at 30 kHz for comparison
    sync_fronts['fpga up fronts'] = sync_fronts['fpga up fronts'] * sr.fs

    for j in range(chans):
        chan_fronts[j] = {}
        chan_fronts[j]['ephys up fronts'] = []

    sync_up_fronts = sync_fronts['fpga up fronts']

    assert len(sync_up_fronts) != 0, 'No starting pulse found'

    k = 0

    # assure there is exactly one pulse per cut segment

    for pulse in range(500):  # there are 500 square pulses

        if pulse == 0:

            first = int(sync_up_fronts[0] - period_duration / 4)
            last = int(first + period_duration)

        else:

            first = int(sync_up_fronts[0] - period_duration / 4 + period_duration)
            last = int(first + period_duration)

        if k % 100 == 0:
            print('segment %s of %s' % (k, 500))
        k += 1

        rawdata, _ = sr.read_samples(first, last)

        # get fronts for only one valid ephys channel
        obs, chans = rawdata.shape

        # check first channel

        i = 0  # assume channel 0 is valid (to be generalized maybe)

        Mean = np.median(rawdata.T[i])
        Std = np.std(rawdata.T[i])

        ups = np.invert(rawdata.T[i] > Mean + 6 * Std)

        up_fronts = []
        up_fronts.append(first_occ_index(ups, 3) + first)
        chan_fronts[i]['ephys up fronts'].append(up_fronts)

    return chan_fronts, sync_fronts


def evaluate_ephys(chan_fronts, sync_fronts, show_plots=SHOW_PLOTS):
    """
    check number of detected square pulses and temporal jitter
    """

    # check if all signals have been detected
    L_sync_up = list(flatten(sync_fronts['fpga up fronts']))

    assert len(L_sync_up) == 500, 'not all fpga up fronts detected'

    for i in range(len(chan_fronts)):

        try:

            L_chan_up = list(flatten(chan_fronts[i]['ephys up fronts']))

            assert len(L_chan_up) == 500, \
                'not all ephys up fronts detected'

            break

        except BaseException:

            continue

    ups_errors = np.array(L_chan_up) - np.array(L_sync_up)

    MAX = max(ups_errors)

    if MAX > 20:
        print('ATTENTION, the maximal error is unusually high, %s sec' %
              str(MAX / 30000.))

    print('ephys test passed')

    if show_plots:
        plt.figure('histogram')
        f = np.array(ups_errors) / 30000.
        plt.hist(f)
        plt.xlabel('error between fpga and ephys up fronts in sec')


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

    try:
        d = np.load(sync_test_folder + '/video/brightness.npy', allow_pickle=True).flat[0]
        return d

    except BaseException:

        # for each frame in the video, set 1 or zero corresponding to LED status
        startTime = datetime.now()

        d = {}

        vids = ['_iblrig_bodyCamera.raw.avi',
                '_iblrig_rightCamera.raw.avi',
                '_iblrig_leftCamera.raw.avi']

        for vid in vids:
            video_path = sync_test_folder + '/video/' + vid

            print('Loading video, this takes some minutes:', video_path)
            cap = cv2.VideoCapture(video_path)
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


def evaluate_camera_sync(d, sync, show_plots=SHOW_PLOTS):

    y = {
        '_iblrig_bodyCamera.raw.avi': 'Cam30Hz',
        '_iblrig_rightCamera.raw.avi': 'Cam150Hz',
        '_iblrig_leftCamera.raw.avi': 'Cam60Hz'}

    # get arduino sync signal in sec
    diffs = np.diff(sync['Arduino']['values'])
    s3 = sync['Arduino']['timeStampsSec'][np.where(diffs == 1)[0]]

    for vid in d:
        # threshold brightness time-series of the camera to have it in {-1,1}
        r3 = [1 if x > np.mean(d[vid][0]) else -1 for x in d[vid][0]]

        # fpga cam time stamps
        diffs = np.diff(sync[y[vid]]['values'])
        cam_times = sync[y[vid]]['timeStampsSec'][np.where(diffs == 1)[0]]

        # assuming at the end the frames are dropped
        drops = len(cam_times) - len(r3)

        # check if an extremely high number of frames is dropped at the end
        assert drops < 500, '%s frames dropped for %s!!!' % (drops, vid)

        # get fronts of video brightness square signal
        diffr3 = np.diff(r3)  # get signal jumps via differentiation
        brightness_ups = cam_times[:-drops][np.where(diffr3 != 0)[0][::2]]

        # check if all 500 square pulses are detected
        assert len(brightness_ups) == len(s3), 'Not all 500 up fronts detected in %s!' % vid

        # get temporal difference between fpga wave fronts and brightness wave
        # fronts
        D = brightness_ups - s3

        assert len(
            D) == 500, \
            'not all 500 pulses were detected \
            by fpga and brightness in %s!' % y[vid]

        print(' %s, Wave fronts temp. diff, in sec: mean = %s, std = %s, max = %s' % (y[vid],
              np.round(np.mean(abs(D)), 4),
              np.round(np.std(abs(D)), 4),
              np.round(max(abs(D)), 4)))

        # Temporal jitter between fpga and brightness up fronts small?
        assert max(
            abs(D)) < 0.200, \
            'Jitter between fpga and brightness fronts is large!!'

        if show_plots:

            plt.figure('wavefronts, ' + y[vid])
            ibllib.plots.squares(
                s3['times'],
                s3['polarities'],
                label='fpga square signal',
                marker='o')
            plt.plot(cam_times[:-drops],
                     r3,
                     alpha=0.5,
                     label='thresholded video brightness',
                     linewidth=2,
                     marker='x')

            plt.legend()
            plt.title('wavefronts for fpga and brightness of %s' % y[vid])
            plt.show()

            plt.figure('histogram of front differences, %s' % y[vid])
            plt.title('histogram of temporal errors of fronts')
            plt.hist(D)
            plt.xlabel('error between fpga fronts and ephys fronts in sec')
            plt.show()

    print('Video test passed')


##########
'''
BPod
'''
##########


def compare_bpod_json_with_fpga(sync_test_folder, sync, show_plots=SHOW_PLOTS):
    '''
    sr, sync=get_ephys_data(sync_test_folder)
    '''

    #  get the bpod signal from the jasonable file
    import json
    with open(sync_test_folder + '/bpod/_iblrig_taskData.raw.jsonable') as fid:
        out = json.load(fid)

    ups = out['Events timestamps']['BNC1High']

    assert len(ups) == 500, 'not all pulses detected in bpod!'

    # get the fpga signal from sync
    diffs = np.diff(sync['Arduino']['values'])
    s3 = sync['Arduino']['timeStampsSec'][np.where(diffs == 1)[0]]

    assert len(s3) == 500, 'not all fronts detected in fpga signal!'

    D = np.array(s3) - np.array(ups)

    offset_on = np.mean(D)
    jitter_on = np.std(D)
    ipi_bpod = np.abs(np.diff(ups))  # inter pulse interval = ipi
    ipi_fpga = np.abs(np.diff(s3))

    print('maximal bpod jitter in sec: ',
          np.round(np.max(ipi_bpod) - np.min(ipi_bpod), 6))

    print('maximal fpga jitter in sec: ',
          np.round(np.max(ipi_fpga) - np.min(ipi_fpga), 6))

    print('maximal bpod-fpga in sec: ',
          np.round(np.max(np.abs(D)) - np.min(np.abs(D)), 6))

    print('fpga and bpod signal offset in sec: ', np.round(offset_on, 6))

    print('std of fpga and bpod difference in sec: ', np.round(jitter_on, 6))

    IntervalDurationDifferences = np.diff(np.array(s3)) - np.diff(np.array(ups))
    R = max(abs(IntervalDurationDifferences))

    print('maximal interval duration difference, fpga - bpod, [sec]:', R)

    assert R < 0.0002, 'Too high temporal jitter bpod - fpga!'

    if show_plots:

        plt.figure('wavefronts')
        plt.plot(s3['times'], s3['polarities'], label='fpga')
        plt.plot(
            ups,
            np.ones(
                len(ups)),
            linestyle='',
            marker='o',
            label='pbod on')

        plt.legend()
        plt.show()

        plt.figure('histogram of wavefront differences, bpod and fpga')

        plt.hist(np.array(s3) - np.array(ups))
        plt.xlabel('error between fpga fronts and ephys fronts in sec')
        plt.show()


def run_synchronization_protocol(sync_test_folder, display=SHOW_PLOTS):
    # running all tests took 12 min for Guido's example data
    startTime = datetime.now()

    ephys_nidq_file = list(Path(sync_test_folder).rglob('*nidq.bin'))[0]
    ap_file_left = list(Path(sync_test_folder).rglob('*left*.imec.ap.bin'))[0]
    ap_file_right = list(Path(sync_test_folder).rglob('*right*.imec.ap.bin'))[0]

    # load sync signals from nidq file (one for both probes)
    sync = get_3b_sync_signal(ephys_nidq_file)

    # load ephys signals from both probes
    sr_left = get_ephys_data(ap_file_left)
    sr_right = get_ephys_data(ap_file_right)

    # compare ephys fronts with fpga pulse signal for right probe
    _logger.info('compare ephys fronts with fpga pulse signal for right probe')
    chan_fronts, sync_fronts = front_extraction_from_arduino_and_ephys(sr_right, sync)
    evaluate_ephys(chan_fronts, sync_fronts, show_plots=display)

    # compare ephys fronts with fpga pulse signal for left probe
    _logger.info('compare ephys fronts with fpga pulse signal for left probe')
    chan_fronts, sync_fronts = front_extraction_from_arduino_and_ephys(sr_left, sync)
    evaluate_ephys(chan_fronts, sync_fronts, show_plots=display)

    # do camera check
    _logger.info('Evaluate Camera sync')
    d = get_video_stamps_and_brightness(sync_test_folder)
    evaluate_camera_sync(d, sync, show_plots=display)

    # do bpod check
    _logger.info('Evaluate Bpod sync')
    compare_bpod_json_with_fpga(sync_test_folder, sync, show_plots=display)

    _logger.info(f'All tests passed !!, took: {datetime.now() - startTime} seconds')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Synchronization protocol analysis')
    parser.add_argument('folder', help='A Folder containing a session')
    parser.add_argument('--display', help='Show Plots', required=False, default=False, type=str)
    args = parser.parse_args()  # returns data from the options specified (echo)
    if args.display and args.display.lower() == 'false':
        args.display = False
    assert(Path(args.folder).exists())
    run_synchronization_protocol(args.folder, display=args.display)
