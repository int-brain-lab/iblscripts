# Adapted from
# https://github.com/int-brain-lab/ibldevtools/blob/master/olivier/archive/2022/2022-03-14_trials_tables.py
# https://github.com/int-brain-lab/ibldevtools/blob/master/miles/2022-01-17-alyx_trials_table_patch.py
# https://github.com/int-brain-lab/ibldevtools/blob/master/miles/2022-12-19_register-zainab-aggregates.py
"""
Generate per subject trials aggregate files for all culled subjects that have at least one session with an ibl project
and ibl task protocol.
1. Check if all sessions have trials tables. For those that don't, try to generate them.
   Log if it's not possible and skip those sessions.
2. Check for which subjects trial aggregate files need to be generated or updated
   (using hash of individual dataset uuids and hashes)
   a. If file exists and does not need updating, do nothing.
   b. If this is the first version of the file, generate and register dataset, create file records, sync to AWS
   c. If original file is protected, create and register new revision of dataset.
   d. If original file is not protected, overwrite it, update hash and file size of dataset.
3. Sync to AWS.
"""

'''
===========
SETTING UP
===========
'''

from django.db.models import Count, Q
from actions.models import Session
from subjects.models import Subject
from data.models import Dataset, DatasetType, DataFormat, DataRepository, FileRecord, Revision
from misc.models import LabMember

import logging
import datetime
import time
import hashlib
from pathlib import Path
from subprocess import Popen, PIPE, STDOUT

import pandas as pd
import globus_sdk as globus

from one.alf import io as alfio, files as alfiles
from iblutil.io import hashfile, params
from ibllib.io.extractors.training_trials import PhasePosQuiescence, StimOnTriggerTimes

# Settings
root_path = Path('/mnt/ibl')
output_path = Path('/mnt/ibl/aggregates/')
collection = 'Subjects'
file_name = '_ibl_subjectTrials.table.pqt'
alyx_user = 'julia.huntenburg'
version = 1.0
dry = False
force_overwrite =  True

# Set up
output_path.mkdir(exist_ok=True, parents=True)
alyx_user = LabMember.objects.get(username=alyx_user)
today_revision = datetime.datetime.today().strftime('%Y-%m-%d')

# Prepare logger
today = datetime.datetime.today().strftime('%Y%m%d')
logger = logging.getLogger('ibllib')
logger.setLevel(logging.INFO)
handler = logging.handlers.RotatingFileHandler(output_path.joinpath(f'subjectTrials_{today}.log'),
                                               maxBytes=(1024 * 1024 * 256), )
logger.addHandler(handler)


# Functions
def log_subprocess_output(pipe, log_function=print):
    for line in iter(pipe.readline, b''):
        log_function(line.decode().strip())


def login_auto(globus_client_id, str_app='globus/default'):
    token = params.read(str_app, {})
    required_fields = {'refresh_token', 'access_token', 'expires_at_seconds'}
    if not (token and required_fields.issubset(token.as_dict())):
        raise ValueError("Token file doesn't exist, run ibllib.io.globus.setup first")
    client = globus.NativeAppAuthClient(globus_client_id)
    client.oauth2_start_flow(refresh_tokens=True)
    authorizer = globus.RefreshTokenAuthorizer(token.refresh_token, client)
    return globus.TransferClient(authorizer=authorizer)


# Set up dictionaries to catch errors or other logs
status_agg = {}

""""
========================
SUBJECT AGGREGATE TABLES
========================
"""
# Now find all culled subjects with at least one session in an ibl project
sessions = Session.objects.filter(project__name__icontains='ibl')
subjects = Subject.objects.filter(id__in=sessions.values_list('subject'), cull__isnull=False
                                  ).exclude(nickname__icontains='test')
# Also make sure to only keep subjects that have at least one session with ibl task protocol and a trials table
sessions = Session.objects.filter(subject__in=subjects, task_protocol__icontains='ibl')
sessions = sessions.annotate(
    trials_table_count=Count('data_dataset_session_related',
                             filter=Q(data_dataset_session_related__name='_ibl_trials.table.pqt')))
sessions = sessions.exclude(trials_table_count=0)
subjects = Subject.objects.filter(id__in=sessions.values_list('subject'))

# dataset format, type and repos
dataset_format = DataFormat.objects.get(name='parquet')
dataset_type = DatasetType.objects.get(name='subjectTrials.table')
aws_repo = DataRepository.objects.get(name='aws_aggregates')
fi_repo = DataRepository.objects.get(name='flatiron_aggregates')

# Go through subjects and check if aggregate needs to be (re)created
logger.info('\n')
logger.info(f' {subjects.count()} SUBJECTS')
# existing files with this file name
all_ds = Dataset.objects.filter(name=file_name, default_dataset=True)

for i, sub in enumerate(subjects):
    try:
        print(f'{i}/{subjects.count()} {sub.nickname}')
        logger.info(f'Subject {sub.nickname} {sub.id}')
        # Find all sessions of this subject
        sub_sess = Session.objects.filter(subject=sub, task_protocol__icontains='ibl')
        # First create hash and check if aggregate needs to be (re)created
        trials_ds = Dataset.objects.filter(session__in=sub_sess, name='_ibl_trials.table.pqt', default_dataset=True)
        trials_ds = trials_ds.order_by('hash')
        # For sessions that have a trials table, add the task data files
        task_ds = Dataset.objects.filter(session__in=trials_ds.values_list('session', flat=True),
                                         name__in=['_iblrig_taskSettings.raw.json', '_iblrig_taskData.raw.jsonable'],
                                         default_dataset=True)
        # If we don't have task data for each session, we that's a problem
        if task_ds.count() / 2 < trials_ds.count():
            logger.info(f'...not all sessions have raw task data')
            status_agg[f'{sub.id}'] = 'ERROR: not all sessions have raw task data'
            continue
        else:
            hash_ds = trials_ds | task_ds
            hash_ds = hash_ds.order_by('hash')
        hash_str = ''.join([str(item) for pair in hash_ds.values_list('hash', 'id') for item in pair]).encode('utf-8')
        new_hash = hashlib.md5(hash_str).hexdigest()
        revision = None  # Only set if making a new revision is required
        # Check if this dataset exists
        ds_id = next((d.id for d in all_ds if d.content_object == sub), None)
        ds = Dataset.objects.filter(id=ds_id)
        # If there is exactly one default dataset, check if it needs updating
        if ds.count() == 1:
            if ds.first().revision is None:
                out_file = output_path.joinpath(collection, sub.lab.name, sub.nickname, file_name)
            else:
                out_file = output_path.joinpath(collection, sub.lab.name, sub.nickname, ds.first().revision, file_name)
            # If force overwrite, we will overwrite the existing file without any checks
            if force_overwrite:
                logger.info(f'...force overwrite')
                status_agg[f'{sub.id}'] = 'FORCE: force overwrite'
                # Add the uuid to the out file to overwrite the current file
                out_file = alfiles.add_uuid_string(out_file, ds.first().pk)
            # See if the file exists on disk (we are on SDSC so need to check with uuid in name)
            # If yes, create the expected hash and try to compare to the hash of the existing file
            elif alfiles.add_uuid_string(out_file, ds.first().pk).exists():
                try:
                    old_hash = ds.first().json['aggregate_hash']
                except TypeError:
                    # If the json doesn't have the hash, just set it to None, we recreate the file in this case
                    old_hash = None
                # If the hash is the same we don't need to do anything
                if old_hash == new_hash:
                    logger.info(f'...aggregate exists and is up to date')
                    status_agg[f'{sub.id}'] = 'EXIST: aggregate exists, hash match'
                    continue
                else:
                    # Otherwise check if the file is protected, if yes, create a revision, otherwise overwrite
                    if ds.first().is_protected:
                        logger.info(f'...aggregate already exists but is protected, hash mismatch, creating revision')
                        status_agg[f'{sub.id}'] = 'REVISION: aggregate exists protected, hash mismatch'
                        # Make revision other than None and add revision to file path
                        revision, _ = Revision.objects.get_or_create(name=today_revision)
                        if ds.first().revision is None:
                            out_file = out_file.parent.joinpath(f"#{today_revision}#", out_file.name)
                        else:
                            # If the current default is already a revision, remove the revision part of the path
                            out_file = out_file.parent.parent.joinpath(f"#{today_revision}#", out_file.name)
                    else:
                        logger.info(f'...aggregate already exists but is not protected, hash mismatch, overwriting')
                        status_agg[f'{sub.id}'] = 'OVERWRITE: aggregate exists not protected, hash mismatch'
                        # Add the uuid to the out file to overwrite the current file
                        out_file = alfiles.add_uuid_string(out_file, ds.first().pk)
            # If the dataset entry exist but the dataset cannot be found on disk, just recreate the dataset
            else:
                logger.info(f'...dataset entry exists but file is missing on disk, creating new')
                status_agg[f'{sub.id}'] = 'CREATE: aggregate dataset entry exists, file missing'
                # Here, too, update the file name with the uuid to create the file on disk
                out_file = alfiles.add_uuid_string(out_file, ds.first().pk)
        # If no dataset exists yet, create it
        elif ds.count() == 0:
            out_file = output_path.joinpath(collection, sub.lab.name, sub.nickname, file_name)
            logger.info(f'...aggregate does not yet exist, creating.')
            status_agg[f'{sub.id}'] = 'CREATE: aggregate does not exist'

        # If dry run, stop here
        if dry:
            logger.info(f'...DRY RUN would create {out_file}')
            continue
        # Create aggregate dataset and save to disk
        all_trials = []
        for t in trials_ds:
            # load trials table
            alf_path = root_path.joinpath(sub.lab.name, 'Subjects', t.file_records.filter(
                data_repository__name__startswith='flatiron').first().relative_path
                                          ).parent
            trials = alfio.load_object(alf_path, 'trials', attribute='table', short_keys=True)
            trials = trials.to_df()

            # Add to list of trials for subject
            trials['session'] = str(t.session.id)
            trials['session_start_time'] = t.session.start_time
            trials['session_number'] = t.session.number
            trials['task_protocol'] = t.session.task_protocol

            # Load quiescence and stimOn_trigger and add to the table
            (*_, quiescence), _ = PhasePosQuiescence(alf_path.parent).extract(save=False)
            stimon_trigger, _ = StimOnTriggerTimes(alf_path.parent).extract(save=False)
            trials['quiescence'] = quiescence
            trials['stimOnTrigger_times'] = stimon_trigger
            # Add to list of trials for subject
            all_trials.append(trials)

        # Concatenate trials from all sessions for subject and save
        df_trials = pd.concat(all_trials, ignore_index=True)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df_trials.to_parquet(out_file)
        assert out_file.exists(), f'Failed to save to {out_file}'
        assert not pd.read_parquet(out_file).empty, f'Failed to read {out_file}'
        logger.info(f"...Saved {out_file}")

        # Get file size and hash which we need in any case
        file_hash = hashfile.md5(out_file)
        file_size = out_file.stat().st_size
        # If we overwrote an existing file, update hashes and size in the dataset entry
        if ds.count() == 1 and revision is None:
            ds.update(hash=file_hash, file_size=file_size, json={'aggregate_hash': new_hash})
            logger.info(f"...Updated hash and size of existing dataset entry {ds.first().pk}")
        # If we made a new file or revision, create new dataset entry and file records
        else:
            # Create dataset entry (make default)
            new_ds = Dataset.objects.create(
                name=file_name,
                hash=file_hash,
                file_size=file_size,
                json={'aggregate_hash': new_hash},
                revision=revision,
                collection=collection,
                default_dataset=True,
                dataset_type=dataset_type,
                data_format=dataset_format,
                created_by=alyx_user,
                version=version,
                content_object=sub,
            )
            # Validate dataset
            new_ds.full_clean()
            new_ds.save()
            # Make previous default dataset not default anymore (if there was one)
            if ds.count() == 1:
                _ = ds.update(default_dataset=False)
            # Change name on disk to include dataset id
            new_out_file = out_file.rename(alfiles.add_uuid_string(out_file, new_ds.pk))
            assert new_out_file.exists(), f"Failed to save renamed file {new_out_file}"
            logger.info(f"...Renamed file to {new_out_file}")
            # Create one file record per repository
            for repo in [aws_repo, fi_repo]:
                record = {
                    'dataset': new_ds,
                    'data_repository': repo,
                    'relative_path': str(out_file.relative_to(output_path)),
                    'exists': False if repo.name.startswith('aws') else True
                }
                try:
                    _ = FileRecord.objects.get_or_create(**record)
                except BaseException as e:
                    logger.error(f'...ERROR: Failed to create file record on {repo.name}: {e}')
                    status_agg[f'{sub.id}'] = f'ERROR: Failed to create file record on {repo.name}: {e}'
                    continue

            logger.info(f"...Created new dataset entry {new_ds.pk} and file records")

    except Exception as e:
        logger.error(f"...Error for subject {sub.nickname}: {e}")
        status_agg[f'{sub.id}'] = f'ERROR: {e}'
        continue

# Save status to file
status_agg = pd.DataFrame.from_dict(status_agg, orient='index', columns=['status'])
status_agg.insert(0, 'subject_id', status_agg.index)
status_agg.reset_index(drop=True, inplace=True)
status_agg.to_csv(root_path.joinpath('subjects_trials_status.csv'))

if not dry:
    # Sync whole collection folder to AWS (for now)
    src_dir = str(output_path.joinpath(collection))
    dst_dir = f's3://ibl-brain-wide-map-private/aggregates/{collection}'
    cmd = ['aws', 's3', 'sync', src_dir, dst_dir, '--delete', '--profile', 'ibladmin', '--no-progress']
    logger.info(f"Syncing {src_dir} to AWS: " + " ".join(cmd))
    t0 = time.time()
    process = Popen(cmd, stdout=PIPE, stderr=STDOUT)
    with process.stdout:
        log_subprocess_output(process.stdout, logger.info)
    assert process.wait() == 0
    logger.debug(f'Session sync took {(time.time() - t0)} seconds')
    # Assume that everyting that existed in that folder on FI was synced and set file records to exist
    fi_frs = FileRecord.objects.filter(data_repository=fi_repo, relative_path__startswith=collection, exists=True)
    aws_frs = FileRecord.objects.filter(data_repository=aws_repo, dataset__in=fi_frs.values_list('dataset', flat=True))
    logger.info(f"Setting {aws_frs.count()} AWS file records to exists=True")
    aws_frs.update(exists=True)




157  c5dbd8a2-c0c9-4170-8845-b0e4d5bef961  OVERWRITE: aggregate exists not protected, has...
166  3ce5f63a-35b6-4f99-9b57-65cf464004af  OVERWRITE: aggregate exists not protected, has...
167  a139eac5-c21a-405b-9d5b-e5a1cef18f6d  OVERWRITE: aggregate exists not protected, has...
189  441ce657-3f91-46fb-b556-1490dd721f7e  OVERWRITE: aggregate exists not protected, has...
190  b5e826b1-afb5-4c60-8793-882b76bfa064  OVERWRITE: aggregate exists not protected, has...
204  1e287343-192b-4c4f-8737-e6eda82a41ac  OVERWRITE: aggregate exists not protected, has...
211  8efdd4c5-6e62-402a-b281-12173d0c3fc9  OVERWRITE: aggregate exists not protected, has...
212  9ebfa752-af72-4a77-819a-cc65ab23333a  OVERWRITE: aggregate exists not protected, has...
213  6ddb28e1-d663-4190-9796-708fe83efacb                   CREATE: aggregate does not exist
214  7f6d4bfe-c751-4ff5-89a9-ac201bb3ce6c                   CREATE: aggregate does not exist
215  17a44e6c-7831-4fbe-9c3d-d796982e2262                   CREATE: aggregate does not exist
216  81f860db-d7d0-4093-be86-7238008e89d7                   CREATE: aggregate does not exist
270  98d7b8e0-fba7-4bc7-a972-aa68dcc10658  OVERWRITE: aggregate exists not protected, has...
271  4d1edd1c-bcb0-4f99-99fd-ae7ea7d87d6d                   CREATE: aggregate does not exist
272  fe45957f-e466-4c63-b1d2-b0ee42f2a5ce                   CREATE: aggregate does not exist
273  cfa3cd6b-6730-4666-a79e-04d1d9b8a9d0                   CREATE: aggregate does not exist
286  79dc7d34-79cd-4bd1-9d7f-cbd0fc8d8f90  OVERWRITE: aggregate exists not protected, has...
366  db4552fe-a686-4823-bbdf-ad859efd8012  OVERWRITE: aggregate exists not protected, has...
377  1b0633a8-d168-44db-9275-3136ef75fec5  OVERWRITE: aggregate exists not protected, has...
391  5ba2d94a-1782-409b-947c-4d764544f604  OVERWRITE: aggregate exists not protected, has...
392  d5d7f730-08fc-429b-97c7-e9d556f92fef                   CREATE: aggregate does not exist
393  5b30991e-3140-4049-b000-34a7219072f0                   CREATE: aggregate does not exist
