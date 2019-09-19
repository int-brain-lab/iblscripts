import os
import glob
import json
import numpy as np


def get_stim_num_from_name(stim_ids, stim_name):
    """
    "VISUAL_STIMULI": {
    "0": "SPACER",
    "1": "receptive_field_mapping",
    "2": "orientation-direction_selectivity",
    "3": "contrast_reversal",
    "4": "task_stimuli",
    "5": "spontaneous_activity"}

    :param stim_ids: map from number (as string) to stimulus name
    :type stim_ids: dict
    :param stim_name: name of stimulus type
    :type stim_name: str
    :return: the number associated with the stimulus type
    :rtype: int
    """
    idx = None
    for key in stim_ids.keys():
        if stim_ids[key].lower() == stim_name.lower():
            idx = key
            break
    return int(idx)


def get_rf_mapping_stimulus(stim_metadata_file):
    """
    extract frames of rf mapping stimulus
    :param stim_metadata_file: _iblrig_taskSettings.raw.json file
    :type stim_metadata_file: str
    :return: stimulus frames
    :rtype: np.ndarray of shape (y_pix, x_pix, n_frames)
    """
    import json
    with open(stim_metadata_file, 'r') as f:
        meta = json.load(f)
    idx_rfm = get_stim_num_from_name(
        meta['VISUAL_STIMULI'], 'receptive_field_mapping')
    if idx_rfm is not None:
        # load rf mapping stimulus (sparse noise)
        file_name = meta['VISUAL_STIM_%i' % idx_rfm]['stim_data_file_name']
        # assumes stim data is in same directory
        bin_file = os.path.join(os.path.dirname(stim_metadata_file), file_name)
        frame_array = np.fromfile(bin_file, dtype='uint8')
        y_pix, x_pix, _ = meta['VISUAL_STIM_%i' % idx_rfm]['stim_file_shape']
        frames = np.transpose(
            np.reshape(frame_array, [y_pix, x_pix, -1], order='F'), [1, 0, 2])
    else:
        frames = np.array([])
    return frames


def get_rf_ttl_pulses(ttl_signal):
    """
    Find where ttl_signal increases or decreases

    :param ttl_signal:
    :type ttl_signal: array-like
    :return: where signal increases/decreases
    :rtype: tuple (np.ndarray, np.ndarray) of ttl (rise, fall) indices
    """
    # Convert values to 0, 1, -1 for simplicity
    assert len(np.unique(ttl_signal)) == 3
    ttl_sig = np.zeros(shape=ttl_signal.shape)
    ttl_sig[ttl_signal == np.max(ttl_signal)] = 1
    ttl_sig[ttl_signal == np.min(ttl_signal)] = -1
    # Find number of passage from 0->-1 and 0->+1
    d_ttl_sig = np.concatenate([np.diff(ttl_sig), [0]])
    idxs_up = np.where((ttl_sig == 0) & (d_ttl_sig == 1))[0]
    idxs_dn = np.where((ttl_sig == 0) & (d_ttl_sig == -1))[0]
    return idxs_up, idxs_dn


def get_expected_ttl_pulses(stim_order, stim_meta, ttl_signal_rf_map):
    """
    Get expected number of ttl pulses for each stimulus

    :param stim_order: list of stimulus ids throughout protocol
    :type stim_order: array-like
    :param stim_meta: dictionary containing stim metadata; from _iblrig_taskSettings json
    :type stim_meta: dict
    :param ttl_signal_rf_map: ttl signal during receptive field mapping with locally sparse noise
    :type ttl_signal_rf_map: array-like
    :return: list of ttl pulses for each stimulus class
    :rtype: list
    """
    n_expected_ttl_pulses = np.zeros(shape=(len(stim_order)))
    for i, stim_id in enumerate(stim_order):
        if stim_meta['VISUAL_STIMULI'][str(stim_id)] == 'receptive_field_mapping':
            n_instances = np.sum((np.array(stim_order) == stim_id) * 1)
            if n_instances > 1:
                raise ValueError('Extractor expects a single rf mapping presentation')
            # number of TTL pulses expected in frame2ttl trace for rf mapping
            idxs_up, idxs_dn = get_rf_ttl_pulses(ttl_signal_rf_map)
            n_expected_ttl_pulses[i] = len(idxs_up) + len(idxs_dn)
        else:
            key = str('VISUAL_STIM_%i' % stim_id)
            if key in stim_meta:
                n_expected_ttl_pulses[i] = stim_meta[key]['ttl_num']
            else:
                # spontaneous activity, no stimulus info in metadata
                n_expected_ttl_pulses[i] = 0
    return n_expected_ttl_pulses


def get_spacer_times(spacer_template, jitter, ttl_signal, t_quiet):
    """
    :param spacer_template: list of indices where ttl signal changes
    :type spacer_template: array-like
    :param jitter: jitter (in seconds) for matching ttl_signal with spacer_template
    :type jitter: float
    :param ttl_signal:
    :type ttl_signal: array-like
    :param t_quiet: seconds between spacer and next stim
    :type t_quiet: float
    :return: times of spacer onset/offset
    :rtype: n_spacer x 2 np.ndarray; first col onset times, second col offset
    """
    # spacer_times = get_spacer_times()
    diff_spacer_template = np.diff(spacer_template)
    # add jitter;
    # remove extreme values (shouldn't be a problem with iblrig versions >= 5.2.10)
    spacer_model = jitter + diff_spacer_template[2:-2]
    # diff ttl signal to compare to spacer_model
    dttl = np.diff(ttl_signal)
    # remove diffs larger than max diff in model to clean up signal
    dttl[dttl > np.max(spacer_model)] = 0
    # convolve cleaned diff ttl signal w/ spacer model
    conv_dttl = np.correlate(dttl, spacer_model, mode='full')
    # find spacer location
    thresh = 3.0
    idxs_spacer_middle = np.where(
        (conv_dttl[1:-2] < thresh) &
        (conv_dttl[2:-1] > thresh) &
        (conv_dttl[3:] < thresh))[0]
    # adjust indices for
    # - `np.where` call above
    # - length of spacer_model
    idxs_spacer_middle += 2 - int((np.floor(len(spacer_model) / 2)))
    # pull out spacer times (middle)
    ts_spacer_middle = ttl_signal[idxs_spacer_middle]
    # put beginning/end of spacer times into an array
    spacer_length = np.max(spacer_template)
    spacer_times = np.zeros(shape=(ts_spacer_middle.shape[0], 2))
    for i, t in enumerate(ts_spacer_middle):
        spacer_times[i, 0] = t - (spacer_length / 2) - t_quiet
        spacer_times[i, 1] = t + (spacer_length / 2) + t_quiet
    return spacer_times, conv_dttl


def export_to_alf(session_path, stim_ts, stim_datas, stim_names):
    counters = {stim_name: 0 for stim_name in np.unique(stim_names)}
    for stim_t, stim_data, stim_name in zip(stim_ts, stim_datas, stim_names):
        if stim_name == 'spacer':
            continue
        if stim_name == 'receptive_field_mapping':
            stim_short = 'rfmap'
        elif stim_name == 'orientation-direction_selectivity':
            stim_short = 'odsgratings'
        elif stim_name == 'contrast_reversal':
            stim_short = 'reversal'
        elif stim_name == 'task_stimuli':
            stim_short = 'task'
        elif stim_name == 'spontaneous_activity':
            stim_short = 'spontaneous'
        else:
            raise ValueError('"%s" is an unknown stimulus protocol' % stim_name)
        filename_times = str('_iblcertif_.%s.times.%02i.npy' % (stim_short, counters[stim_name]))
        filename_stims = str('_iblcertif_.%s.stims.%02i.npy' % (stim_short, counters[stim_name]))
        counters[stim_name] += 1
        if not os.path.exists(os.path.join(session_path, 'alf')):
            os.mkdir(os.path.join(session_path, 'alf'))
        shape_t = stim_t.shape[0]
        shape_d = stim_data.shape[0]
        if shape_t != 0 and shape_d != 0:
            assert shape_t == shape_d
        if shape_t != 0:
            np.save(os.path.join(session_path, 'alf', filename_times), stim_t)
        if shape_d != 0:
            np.save(os.path.join(session_path, 'alf', filename_stims), stim_data)


def extract_stimulus_info_to_alf(session_path):
    """
    Extract the stimulus information stored in metadata and export to alf files

    :param session_path: absolute path of a session, i.e. /mnt/data/Subjects/ZM_1887/2019-07-10/001
    :type session_path: str
    :return: None; instead stimulus info is stored in the following alf files:
            `_iblcertification_.stimtype.times.00.npy`
            `_iblcertification_.stimtype.frames.00.npy`
        where `stimtype` are `odsgratings`, `sparsenoise`, `contrastreversal`, and `taskstimulus`
    """
    # define ttl channel that controls the stimulus
    fr2ttl_ch = 12
    # Ipad screen refresh rate 60 Hz
    t_bin = 1 / 60
    # jitter allowance in spacer template matching (units of time bins)
    bin_jitter = 3
    iblrig_version_min = [5, 2, 9]

    # load metadata
    json_file = glob.glob(os.path.join(session_path, '*taskSettings.raw*'))[0]
    with open(json_file, 'r') as f:
        meta = json.load(f)
    # load sync data (can move to ONE soon)
    sync_ch = np.load(glob.glob(os.path.join(session_path, '*sync.channels*'))[0])
    sync_pol = np.load(glob.glob(os.path.join(session_path, '*sync.polarities*'))[0])
    sync_times = np.load(glob.glob(os.path.join(session_path, '*sync.times*'))[0])

    # find times of when ttl polarity changes on fr2ttl channel
    sync_pol_ = sync_pol[sync_ch == fr2ttl_ch]
    sync_times_ = sync_times[sync_ch == fr2ttl_ch]
    sync_rise_times = sync_times_[sync_pol_ == 1]
    sync_fall_times = sync_times_[sync_pol_ == -1]
    ttl_sig = np.sort(np.concatenate([sync_rise_times, sync_fall_times]))

    # pull useful fields out of metadata
    protocol = meta['VISUAL_STIMULUS_TYPE']
    if protocol != 'ephys_certification':
        raise ValueError(
            '"%s" is an invalid protocol; function only works for ephys_certification' % protocol)
    iblrig_version = [int(i) for i in meta['IBLRIG_VERSION_TAG'].split('.')]
    if not np.all(iblrig_version >= np.array(iblrig_version_min)):
        raise ValueError(
            'Special extractor needed for code version {}; minimum supported version is {}'.format(
                iblrig_version, iblrig_version_min))
    stim_ids = meta['VISUAL_STIMULI']
    stim_order = np.array(meta['STIM_ORDER'])
    id_spacer = get_stim_num_from_name(stim_ids, 'spacer')
    spacer_template = t_bin * np.array(meta['VISUAL_STIM_%i' % id_spacer]['ttl_frame_nums'])
    frames = get_rf_mapping_stimulus(json_file)
    n_frames = frames.shape[-1]

    # get number of expected ttl pulses from upper left stim pixel (rf mapping) and
    # metadata (all other stims)
    if len(frames) != 0:
        frame_ttl_signal = frames[0, 0, :]
    else:
        frame_ttl_signal = None
    n_expected_ttl_pulses = get_expected_ttl_pulses(stim_order, meta, frame_ttl_signal)

    stim_ts = [[] for _ in stim_order]
    stim_datas = [[] for _ in stim_order]
    stim_names = [[] for _ in stim_order]
    # start with spacers
    spacer_times, conv_sig = get_spacer_times(spacer_template, t_bin * bin_jitter, ttl_sig, 1.5)
    idxs_spacer = np.where(stim_order == get_stim_num_from_name(stim_ids, 'spacer'))[0]
    n_expected_spacers = len(idxs_spacer)
    n_spacers = spacer_times.shape[0]
    if n_spacers != n_expected_spacers:
        raise ValueError(
            '%i is an invalid number of spacer templates in ttl signal; expected %i' %
            (n_spacers, n_expected_spacers))
    # now stimuli
    for i, stim_id in enumerate(stim_order):
        stim_names[i] = stim_ids[str(stim_id)]
        if i not in idxs_spacer:
            # assumes all non-spacers are preceded by a spacer
            ttl_times = np.where(
                (ttl_sig > spacer_times[int(i / 2), 1]) &
                (ttl_sig < spacer_times[int((i + 1) / 2), 0]))[0]
            if stim_ids[str(stim_id)] == 'receptive_field_mapping':
                # we only recorded rise times earlier; ttl_stim contains rise and
                # fall times
                # beg/end offsets b/c Bonsai generates 1 pulse at beginning and end
                stim_ts[i] = ttl_sig[ttl_times[1:-2:2]]
                stim_datas[i] = None
            elif stim_ids[str(stim_id)] == 'orientation-direction_selectivity':
                # offset by 2 at beginning; rapid transient artifact due to Bonsai loading
                stim_ts[i] = ttl_sig[ttl_times[2:]]
                stim_datas[i] = None
            else:
                stim_ts[i] = ttl_sig[ttl_times]
                stim_datas[i] = None
            # check ttl pulses against expected ttl pulses from upper left stim pixel
            # (rf mapping) and metadata (all other stims)
            if len(stim_ts[i]) != n_expected_ttl_pulses[i]:
                raise ValueError('TTL pulses inconsistent for %i' % stim_ids[str(stim_id)])

    # assign proper timestamps for rf mapping
    if frame_ttl_signal is not None:
        beg_extrap_val = -10001
        end_extrap_val = -10000
        # TODO: hardcoded and assumes there is only 1 presentation of RF mapping
        idx_rf = np.where(
            stim_order == get_stim_num_from_name(stim_ids, 'receptive_field_mapping'))[0][0]
        idxs_up, idxs_dn = get_rf_ttl_pulses(frame_ttl_signal)
        X = np.sort(np.concatenate([idxs_up, idxs_dn]))
        T = stim_ts[idx_rf]
        Xq = np.arange(n_frames)
        # make left and right extrapolations distinctive to easily find later
        Tq = np.interp(Xq, X, T, left=beg_extrap_val, right=end_extrap_val)
        # uniform spacing outside boundaries of ttl signal
        # first values
        n_beg = len(np.where(Tq == beg_extrap_val)[0])
        Tq[:n_beg] = T[0] - np.arange(n_beg, 0, -1) * t_bin
        # end values
        n_end = len(np.where(Tq == end_extrap_val)[0])
        Tq[-n_end:] = T[-1] + np.arange(1, n_end + 1) * t_bin
        stim_ts[idx_rf] = Tq
        stim_datas[idx_rf] = frames

    export_to_alf(stim_ts, stim_datas, stim_names)
