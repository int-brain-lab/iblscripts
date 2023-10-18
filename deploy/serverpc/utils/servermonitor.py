from pathlib import Path
from one.api import ONE
from ibllib.io.raw_data_loaders import load_settings
from json import JSONDecodeError
import time
from datetime import datetime, timedelta

one = ONE(base_url='https://alyx.internationalbrainlab.org')
one.alyx.clear_rest_cache()
ROOT_PATH = Path('/mnt/s0/Data/Subjects')

to_remove_from_server = []
to_remove_from_alyx = []
no_video_create_raw_session_flag = []
other_create_raw_session_flag = []
video_tasks = []
audio_tasks = []
ephys_tasks = []

start = time.time()
info = []
avi = []
wav = []
ap_bin = []
nidq_bin = []
multiple_eids = []
other_files = []

subjects = list(ROOT_PATH.glob('*'))
for sub in subjects:
    if not sub.is_dir():
        other_files.append(str(sub))
        continue
    dates = list(sub.glob('*'))
    for dat in dates:
        if not sub.is_dir():
            other_files.append(str(dat))
            continue

        sess = list(dat.glob('*'))

        good_sess = []
        for se in sess:
            if not se.is_dir():
                other_files.append(str(se))
            elif not one.is_exp_ref(one.path2ref(se)):
                avi += [{str(a): 'Odd location'} for a in list(se.rglob('*.avi'))]
                ap_bin += [{str(a): 'Odd location'} for a in list(se.rglob('*ap.bin'))]
                nidq_bin += [{str(a): 'Odd location'} for a in list(se.rglob('*nidq.bin'))]
                wav += [{str(a): 'Odd location'} for a in list(se.rglob('*wav'))]
                other_files.append(str(se))
            else:
                good_sess.append(se)

        if len(good_sess) == 0:
            continue

        if len(good_sess) > 0:
            other_flag = True
            other_sess = {}
            for se in good_sess:
                task_settings = next(se.rglob('*taskSettings*'), None)
                # here we assume left video is always collected
                video_existence = next(se.rglob('*leftCamera*'), None)
                other_sess[se.name] = {'behavior': task_settings, 'video': video_existence}
        else:
            other_flag = False

        for se in good_sess:
            try:
                eid = one.path2eid(se)
            except AssertionError:
                multiple_eids.append({str(se): 'multiple eids for session'})
                continue

            if eid is None or len(one.list_datasets(eid)) == 0:
                if eid is None:
                    expl = 'Session has no eid'
                    extra = ''
                else:
                    expl = 'Session has no datasets'
                    extra = ' and delete session from alyx'

                files = list(Path(se).rglob('*'))
                folders = [str(f.name) for f in files if f.is_dir() and f.parent == se]
                files = [str(f.name) for f in [f for f in files if not f.is_dir()] if '.flag' not in str(f)]

                avi += [{str(a): expl} for a in list(se.rglob('*.avi'))]
                ap_bin += [{str(a): expl} for a in list(se.rglob('*ap.bin'))]
                nidq_bin += [{str(a): expl} for a in list(se.rglob('*nidq.bin'))]
                wav += [{str(a): expl} for a in list(se.rglob('*wav'))]

                compare_date = datetime.today().date() - timedelta(days=2)
                date = one.path2ref(se)['date']
                if date >= compare_date:
                    info.append(
                        {str(se): expl + f': Sessions only acquired in last two days, alyx may not be updated yet - '
                                         f'folders found are {", ".join(folders)} '})
                    continue

                if len(files) == 0:

                    info.append({str(se): expl + ': No files found - can probably delete this session folder' + extra})
                    if eid is None:
                        to_remove_from_server.append(str(se))
                    else:
                        to_remove_from_alyx.append(str(se) + ' ' + eid)
                    continue
                if len(folders) == 1 and (folders[0] == 'logs' or folders[0] == '_devices'):
                    info.append({str(se): expl + f': No data found, folders are {", ".join(folders)} - '
                                f'can probably delete this session folder' + extra})
                    if eid is None:
                        to_remove_from_server.append(str(se))
                    else:
                        to_remove_from_alyx.append(str(se) + ' ' + eid)
                    continue

                if 'raw_behavior_data' not in folders and not any(['raw_task_data' in f for f in folders]):
                    lonely_behavior = False
                    if other_flag:
                        for k, v in other_sess.items():
                            if k == se.name:
                                continue
                            if v['behavior'] and not v['video']:
                                lonely_behavior = True

                    if lonely_behavior:
                        info.append({str(se): expl + f': No behavior folder found, folders are {", ".join(folders)} - '
                                    f'{k} for this date contains a lonely behavior session'})
                    else:
                        info.append({str(se): expl + f': No behavior folder found, folders are {", ".join(folders)}'})

                    continue

                if '_iblrig_taskSettings.raw.json' not in files:
                    info.append({str(se): expl + f': No taskSettings file found, folders are {", ".join(folders)}'})
                    continue

                try:
                    bpod_settings = load_settings(session_path=Path(se))
                except JSONDecodeError:
                    info.append({str(se): expl + ': The taskSettings file is corrupt'})
                    continue

                if bpod_settings is None:
                    info.append({str(se): expl + ': The taskSettings file is empty'})
                    continue

                sess_num = se.name
                date = se.parent.name
                subject = se.parent.parent.name
                settings_num = bpod_settings['SESSION_NUMBER']
                settings_user = bpod_settings['PYBPOD_CREATOR'][0]
                settings_sub = bpod_settings['SUBJECT_NAME']
                board = bpod_settings['PYBPOD_BOARD']
                protocol = '_'.join(bpod_settings['PYBPOD_PROTOCOL'].split('_')[3:])
                if sess_num != settings_num:
                    if eid is None:
                        eids = one.search(subject=subject, date=date)
                        # Here we have a session registered but that has no datasets, likely due to mismatch in
                        # session number
                        if len(eids) == 0:
                            info.append({str(se): expl + f': Session number mismatch - patch settings from '
                                                         f'{settings_num} to {sess_num}'})
                        elif len(eids) == 1 and len(one.list_datasets(eids[0])) == 0:
                            info.append({str(se): expl + f': Session number mismatch - move folder from '
                                                         f'{sess_num} to {settings_num}'})
                        else:
                            info.append({str(se): expl + f': Session number mismatch - task settings has number '
                                                         f'{settings_num}'})
                    else:
                        info.append({str(se): expl + f': Session number mismatch - task settings has number '
                                                     f'{settings_num}'})
                elif subject != settings_sub:
                    info.append({str(se): expl + f': Subject name mismatch - patch settings from '
                                                 f'{settings_sub} to {subject}'})
                elif settings_user == '_iblrig_test_user':
                    info.append({str(se): expl + f': User mismatch - patch settings from '
                                                 f'{settings_user} to correct username'})
                else:
                    alyx_sub = one.alyx.rest('subjects', 'list', nickname=subject)
                    if len(alyx_sub) == 0:
                        info.append({str(se): expl + f': Subject {subject} not found on alyx'})
                    else:

                        if 'raw_video_data' not in folders:
                            lonely_video = False
                            if other_flag:
                                for k, v in other_sess.items():
                                    if k == se.name:
                                        continue
                                    if not v['behavior'] and v['video']:
                                        lonely_video = True
                            if lonely_video:
                                info.append({str(se): expl + f': No video folder found, folders are {", ".join(folders)}'
                                                             f' - {k} for this date contains a lonely video session'})
                            else:
                                if 'ephys' in board and 'ephys' not in protocol:
                                    info.append({str(se): expl + f': No video folder found, folders are '
                                                                 f'{", ".join(folders)} - {protocol} was collected on '
                                                                 f'{board}, sometimes these sessions'
                                                                 f' do not have video data'})
                                    no_video_create_raw_session_flag.append(str(se))
                                else:
                                    info.append(
                                        {str(se): expl + f': No video folder found, folders are {", ".join(folders)}'})
                        else:

                            info.append({str(se): expl + f': Couldn"t determine problem - folders found are '
                                        f'{", ".join(folders)} - maybe we just need to add a raw_session.flag?'})
                            other_create_raw_session_flag.append(str(se))

            else:

                registered_files = one.list_datasets(eid)
                # First look for video files
                files = list(Path(se).rglob('*.avi'))
                if len(files) > 0:
                    mp4_files = [str(Path(f).name) for f in registered_files if '.mp4' in str(f)]
                    if len(mp4_files) > 0:
                        for f in files:
                            if f.with_suffix('.mp4').name in mp4_files:
                                avi.append({str(f): 'mp4 already seems to exist - check on flatiron that '
                                                    'they are the same'})
                            else:
                                avi.append(
                                    {str(f): 'mp4 does not exist - check tasks'})
                    else:
                        tasks = one.alyx.rest('tasks', 'list', session=eid)
                        if len(tasks) == 0:
                            avi += [{str(f): 'No tasks for session'} for f in files]
                        else:
                            video_task = next((t for t in tasks if 'VideoCompress' in t['name']), None)
                            if video_task is None:
                                avi += [{str(f): 'No video task found for session'} for f in files]
                            else:
                                video_status = video_task['status']
                                avi += [{str(f): f'{video_task["name"]} task has status - {video_status}'}
                                        for f in files]
                                video_tasks += [f'{str(f)} - {video_task["name"]} {video_status} task_id: '
                                                f'{video_task["id"]}' for f in files]

                # Now look at audio
                files = list(Path(se).rglob('*.wav'))
                if len(files) > 0:
                    tasks = one.alyx.rest('tasks', 'list', session=eid)
                    if len(tasks) == 0:
                        wav += [{str(f): 'No tasks for session'} for f in files]
                    else:
                        audio_task = next((t for t in tasks if 'Audio' in t['name']), None)
                        if audio_task is None:
                            wav += [{str(f): 'No audio task found for session'} for f in files]
                        else:
                            audio_status = audio_task['status']
                            wav += [{str(f): f'{audio_task["name"]} task has status - {audio_status}'} for f in files]
                            audio_tasks += [f'{str(f)} - {audio_task["name"]} {audio_status} task_id: '
                                            f'{audio_task["id"]}' for f in files]

                # Now look at nidq.bin
                files = list(Path(se).rglob('*.nidq.bin'))
                if len(files) > 0:
                    nidq_files = [str(Path(f).name) for f in registered_files if 'nidq.cbin' in str(f)]
                    if len(nidq_files) > 0:
                        for f in files:
                            if f.with_suffix('.cbin').name in nidq_files:
                                nidq_bin.append({str(f): 'nidq.cbin already seems to exist - check on flatiron that '
                                                         'they are the same'})
                            else:
                                nidq_bin.append(
                                    {str(f): 'nidq.cbin does not exist - check tasks'})
                    else:
                        tasks = one.alyx.rest('tasks', 'list', session=eid)
                        if len(tasks) == 0:
                            nidq_bin += [{str(f): 'No tasks for session'} for f in files]
                        else:
                            tasks = [t for t in tasks if 'ExperimentDescription' not in t['name']]

                            if tasks[0]['graph'] == 'TrainingExtractionPipeline':
                                nidq_bin += [{str(f): 'Old pipeline - training_preprocessing'} for f in files]
                            elif tasks[0]['graph'] == 'EphysExtractionPipeline':
                                mtscomp_task = next((t for t in tasks if 'EphysMtscomp' in t['name']), None)
                                if mtscomp_task is None:
                                    nidq_bin += [{str(f): 'No mtscomp task found for session'} for f in files]
                                else:
                                    mtscomp_status = mtscomp_task['status']
                                    nidq_bin += [{str(f): f'{mtscomp_task["name"]} task has status - {mtscomp_status}'}
                                                 for f in files]
                                    ephys_tasks += [
                                        f'{str(f)} - {mtscomp_task["name"]} {mtscomp_status} task_id: '
                                        f'{mtscomp_task["id"]}' for f in files]
                            else:
                                for t in tasks:
                                    sync = t.get('arguments', {}).get('sync', None)
                                    if sync is not None:
                                        break

                                if sync == 'bpod':
                                    nidq_bin += [{str(f): 'Dynamic pipeline - sync is bpod'} for f in files]

                                else:
                                    nidq_task = next((t for t in tasks if 'SyncRegisterRaw' in t['name']), None)
                                    if nidq_task is None:
                                        nidq_bin += [{str(f): 'No nidq compress task found for session'} for f in files]
                                    else:
                                        nidq_status = nidq_task['status']
                                        nidq_bin += [{str(f): f'{nidq_task["name"]} task has status - {nidq_status}'}
                                                     for f in files]
                                        ephys_tasks += [
                                            f'{str(f)} - {nidq_task["name"]} {nidq_status} task_id: {nidq_task["id"]}'
                                            for f in files]

                # Now look at ap.bin
                files = list(Path(se).rglob('*.ap.bin'))
                if len(files) > 0:
                    ap_files = [str(Path(f).name) for f in registered_files if 'ap.cbin' in str(f)]
                    if len(ap_files) > 0:
                        for f in files:
                            if f.with_suffix('.cbin').name in ap_files:
                                ap_bin.append({str(f): 'ap.cbin already seems to exist - check on flatiron that they '
                                                       'are the same'})
                            else:
                                ap_bin.append(
                                    {str(f): 'ap.cbin does not exist - check tasks'})
                    else:
                        tasks = one.alyx.rest('tasks', 'list', session=eid)
                        if len(tasks) == 0:
                            ap_bin += [{str(f): 'No tasks for session'} for f in files]
                        else:
                            tasks = [t for t in tasks if 'ExperimentDescription' not in t['name']]

                            if tasks[0]['graph'] == 'TrainingExtractionPipeline':
                                ap_bin += [{str(f): 'Old pipeline - training_preprocessing'} for f in files]
                            elif tasks[0]['graph'] == 'EphysExtractionPipeline':
                                mtscomp_task = next((t for t in tasks if 'EphysMtscomp' in t['name']), None)
                                if mtscomp_task is None:
                                    ap_bin += [{str(f): 'No mtscomp task found for session'} for f in files]
                                else:
                                    mtscomp_status = mtscomp_task['status']
                                    ap_bin += [{str(f): f'{mtscomp_task["name"]} task has status - {mtscomp_status}'}
                                               for f in files]
                                    ephys_tasks += [
                                        f'{str(f)} - {mtscomp_task["name"]} {mtscomp_status} '
                                        f'task_id: {mtscomp_task["id"]}' for f in files]
                            else:
                                for t in tasks:
                                    sync = t.get('arguments', {}).get('sync', None)
                                    if sync is not None:
                                        break

                                if sync == 'bpod':
                                    ap_bin += [{str(f): 'Dynamic pipeline - sync is bpod'} for f in files]

                                else:
                                    for f in files:
                                        probe = f.parent.name
                                        ap_task = next((t for t in tasks if ('EphyCompress' in t['name'] and
                                                                             probe in t['name'])), None)
                                        if ap_task is None:
                                            ap_bin += [{str(f): 'No ap compress task found for session'}]
                                        else:
                                            ap_status = ap_task['status']
                                            ap_bin += [{str(f): f'{ap_task["name"]} task has status - {ap_status}'}]
                                            ephys_tasks += [
                                                f'{str(f)} - {ap_task["name"]} {ap_status} task_id: {ap_task["id"]}']


def convert_to_dict(list_vals):
    dict_vals = {}
    key_list = []
    for li in list_vals:
        for k, v in li.items():
            dict_vals[k] = v
            key_list.append(k)

    return dict_vals, key_list


mult_eids_dict, mult_eids_key = convert_to_dict(multiple_eids)
info_dict, info_key = convert_to_dict(info)
avi_dict, avi_key = convert_to_dict(avi)
nidq_dict, nidq_key = convert_to_dict(nidq_bin)
ap_dict, ap_key = convert_to_dict(ap_bin)
wav_dict, wav_key = convert_to_dict(wav)


def dict_to_string(dict_vals, key_list):
    key_list.sort()
    full_string = ''
    prev_sub = ''
    for line in key_list:
        reason = dict_vals[line]
        if line.split('/')[5] != prev_sub:
            full_string += '\n'
        full_string += f'{line}: {reason}\n'
        prev_sub = line.split('/')[5]

    return full_string


def list_to_eids(list_vals):
    n_lines = len(list_vals)
    full_string = '\n'
    full_string += '['
    for i, line in enumerate(list_vals):
        if i == n_lines - 1:
            full_string += f'{line}]\n'
        else:
            full_string += f'{line},\n'

    return full_string


def list_to_string(list_vals):
    full_string = '\n'
    for line in list_vals:
        full_string += f'{line}\n'

    return full_string


# Write to text file
save_path = Path('/var/log/ibl/cleanup')
save_path.mkdir(exist_ok=True)
with open(save_path.joinpath('info.txt'), 'w') as f:
    f.write(dict_to_string(info_dict, info_key))

with open(save_path.joinpath('multiple_eids.txt'), 'w') as f:
    f.write(dict_to_string(mult_eids_dict, mult_eids_key))

with open(save_path.joinpath('avi_files.txt'), 'w') as f:
    f.write(dict_to_string(avi_dict, avi_key))

with open(save_path.joinpath('ap_bin_files.txt'), 'w') as f:
    f.write(dict_to_string(ap_dict, ap_key))

with open(save_path.joinpath('nidq_bin_files.txt'), 'w') as f:
    f.write(dict_to_string(nidq_dict, nidq_key))

with open(save_path.joinpath('wav_files.txt'), 'w') as f:
    f.write(dict_to_string(wav_dict, wav_key))

save_path = Path('/var/log/ibl/cleanup/summary')
save_path.mkdir(exist_ok=True)

with open(save_path.joinpath('folders_to_remove_from_server.txt'), 'w') as f:
    f.write(list_to_string(to_remove_from_server))

with open(save_path.joinpath('folders_to_remove_from_server_and_sessions_from_alyx.txt'), 'w') as f:
    f.write(list_to_string(to_remove_from_alyx))

with open(save_path.joinpath('sessions_with_no_video_create_raw_session_flag.txt'), 'w') as f:
    f.write(list_to_string(no_video_create_raw_session_flag))

with open(save_path.joinpath('sessions_nothing_detected_create_raw_session_flag.txt'), 'w') as f:
    f.write(list_to_string(other_create_raw_session_flag))

with open(save_path.joinpath('video_tasks.txt'), 'w') as f:
    ids = [f"'{i.split(':')[1][1:]}'" for i in video_tasks]
    ids = list(set(ids))
    f.write(list_to_string(video_tasks) + list_to_eids(ids))

with open(save_path.joinpath('audio_tasks.txt'), 'w') as f:
    ids = [f"'{i.split(':')[1][1:]}'" for i in audio_tasks]
    ids = list(set(ids))
    f.write(list_to_string(audio_tasks) + list_to_eids(ids))

with open(save_path.joinpath('ephys_tasks.txt'), 'w') as f:
    ids = [f"'{i.split(':')[1][1:]}'" for i in ephys_tasks]
    ids = list(set(ids))
    f.write(list_to_string(ephys_tasks) + list_to_eids(ids))


print(time.time() - start)


def load_file(file, save_path=None):
    save_path = Path('/var/log/ibl/cleanup') if save_path is None else save_path
    with open(save_path.joinpath(file), 'r') as f:
        info = [line.strip() for line in f.readlines() if line != "\n"]

    return info
