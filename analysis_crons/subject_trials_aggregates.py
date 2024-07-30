"""
Generate per subject trials aggregate files for all culled subjects that have at least one session with an ibl project
and ibl task protocol.
Check for which subjects trial aggregate files need to be generated or updated
(using hash of individual dataset uuids and hashes)
a. If file exists and does not need updating, do nothing.
b. If this is the first version of the file, generate and register dataset, create file records, sync to AWS
c. If original file is protected, create and register new revision of dataset.
d. If original file is not protected, overwrite it, update hash and file size of dataset.
3. Sync to AWS.

# Adapted from
# https://github.com/int-brain-lab/ibldevtools/blob/master/olivier/archive/2022/2022-03-14_trials_tables.py
# https://github.com/int-brain-lab/ibldevtools/blob/master/miles/2022-01-17-alyx_trials_table_patch.py
# https://github.com/int-brain-lab/ibldevtools/blob/master/miles/2022-12-19_register-zainab-aggregates.py
"""

import datetime
import hashlib
import logging
import time
from pathlib import Path
from subprocess import Popen, PIPE, STDOUT

from django.db.models import Count, Q
import globus_sdk
import pandas as pd

from actions.models import Session
from data.models import Dataset, DatasetType, DataFormat, DataRepository, FileRecord, Revision
from iblutil.io import hashfile, params
from ibllib.io.extractors.training_trials import StimOnTriggerTimes
from misc.models import LabMember
from one.alf import io as alfio, files as alfiles
from subjects.models import Subject


"""
---------
FUNCTIONS
---------
"""


def log_subprocess_output(pipe, log_function=print):
    for line in iter(pipe.readline, b''):
        log_function(line.decode().strip())


def login_auto(globus_client_id, str_app='globus/default'):
    token = params.read(str_app, {})
    required_fields = {'refresh_token', 'access_token', 'expires_at_seconds'}
    if not (token and required_fields.issubset(token.as_dict())):
        raise ValueError("Token file doesn't exist, run ibllib.io.globus.setup first")
    client = globus_sdk.NativeAppAuthClient(globus_client_id)
    client.oauth2_start_flow(refresh_tokens=True)
    authorizer = globus_sdk.RefreshTokenAuthorizer(token.refresh_token, client)
    return globus_sdk.TransferClient(authorizer=authorizer)


def make_new_dataset(new_file, new_hash, sub, old_ds, revision, alyx_user):
    aws_repo = DataRepository.objects.get(name='aws_aggregates')
    fi_repo = DataRepository.objects.get(name='flatiron_aggregates')

    # If there was already a dataset, make it not be default anymore
    if old_ds is not None:
        old_ds.default_dataset = False
        old_ds.save()

    # Create dataset entry (make default)
    new_ds = Dataset.objects.create(
        name='_ibl_subjectTrials.table.pqt',
        hash=hashfile.md5(new_file),
        file_size=new_file.stat().st_size,
        json={'aggregate_hash': new_hash},
        revision=revision,
        collection='Subjects',
        default_dataset=True,
        dataset_type=DatasetType.objects.get(name='subjectTrials.table'),
        data_format=DataFormat.objects.get(name='parquet'),
        created_by=alyx_user,
        version=1.0,
        content_object=sub,
    )
    # Validate dataset
    new_ds.full_clean()
    new_ds.save()

    # Create one file record per repository
    for repo in [aws_repo, fi_repo]:
        record = {
            'dataset': new_ds,
            'data_repository': repo,
            'relative_path': str(out_file.relative_to(output_path)),
            'exists': False if repo.name.startswith('aws') else True
        }
        _ = FileRecord.objects.get_or_create(**record)

    logger.info(f"...Created new dataset entry {new_ds.pk} and file records")

    return new_ds


def make_new_aggregate(out_file, trials_ds):
    # Create aggregate dataset and save to disk
    all_trials = []
    for t in trials_ds:
        # load trials table
        alf_path = root_path.joinpath(sub.lab.name, 'Subjects', t.file_records.filter(
            data_repository__name__startswith='flatiron').first().relative_path).parent
        session_path = root_path.joinpath(sub.lab.name, 'Subjects', sub.nickname, str(t.session.start_time.date()),
                                          str(t.session.number).zfill(3))
        trials = alfio.load_object(alf_path, 'trials', attribute='table', short_keys=True)
        trials = trials.to_df()

        # Add to list of trials for subject
        trials['session'] = str(t.session.id)
        trials['session_start_time'] = t.session.start_time
        trials['session_number'] = t.session.number
        trials['task_protocol'] = t.session.task_protocol

        # Load quiescence and stimOn_trigger and add to the table
        quiescence = alfio.load_object(alf_path, 'trials',
                                       attribute='quiescencePeriod', short_keys=True)['quiescencePeriod']
        try:
            stimon_trigger, _ = StimOnTriggerTimes(session_path).extract(save=False)
        except TypeError:
            task_no = t.collection.split('/')[1].split('_')[1]
            stimon_trigger, _ = StimOnTriggerTimes(session_path).extract(task_collection=f'raw_task_data_{task_no}',
                                                                         save=False)
        trials['quiescence'] = quiescence
        trials['stimOnTrigger_times'] = stimon_trigger
        # TODO: Add protocol number
        # Add to list of trials for subject
        all_trials.append(trials)

    # Concatenate trials from all sessions for subject and save
    df_trials = pd.concat(all_trials, ignore_index=True)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df_trials.to_parquet(out_file)
    assert out_file.exists(), f'Failed to save to {out_file}'
    assert not pd.read_parquet(out_file).empty, f'Failed to read-after-write {out_file}'
    logger.info(f"...Saved {out_file}")

    return out_file


def update_dataset(ds, file, agg_hash):
    # Update dataset entry
    ds.hash = hashfile.md5(file)
    ds.file_size = file.stat().st_size
    ds.json['aggregate_hash'] = agg_hash
    ds.save()

    return ds


def make_aggregate_hash(trials_ds):
    # For sessions that have a trials table, add the task data files
    task_ds = Dataset.objects.filter(session__in=trials_ds.values_list('session', flat=True),
                                     name__in=['_iblrig_taskSettings.raw.json', '_iblrig_taskData.raw.jsonable'],
                                     default_dataset=True)
    # If we don't have task data for each session, that's a problem
    if task_ds.count() / 2 < trials_ds.count():
        raise AssertionError('Not all sessions have raw task data')

    # Get the hash
    hash_ds = trials_ds | task_ds
    hash_ds = hash_ds.order_by('hash')
    hash_str = ''.join([str(item) for pair in hash_ds.values_list('hash', 'id') for item in pair]).encode('utf-8')
    new_hash = hashlib.md5(hash_str).hexdigest()

    return new_hash


"""
------
SET UP
------
"""
dry = False  # Whether to do a dry run that doesn't create any files but just logs which ones would be created
only_new_subjects = False  # Whether to only create trial aggregates for subjects that don't have one
alyx_user = LabMember.objects.get(username='julia.huntenburg')
today = datetime.datetime.today().strftime('%Y-%m-%d')

# Paths
root_path = Path('/mnt/ibl')
output_path = Path('/mnt/ibl/aggregates/')
output_path.mkdir(exist_ok=True, parents=True)
log_path = output_path.joinpath('logs')
log_path.mkdir(exist_ok=True, parents=True)

# Set up logger
logger = logging.getLogger('ibllib')
logger.setLevel(logging.INFO)

# Set up dictionaries to catch errors or other logs
status_agg = {}


"""
-------------
FIND SUBJECTS
-------------
"""

# Find all culled subjects with at least one session in an ibl project
sessions = Session.objects.filter(projects__name__icontains='ibl')
subjects = Subject.objects.filter(id__in=sessions.values_list('subject'), cull__isnull=False
                                  ).exclude(nickname__icontains='test')
# Also make sure to only keep subjects that have at least one session with ibl task protocol and a trials table
sessions = Session.objects.filter(subject__in=subjects, task_protocol__icontains='ibl')
sessions = sessions.annotate(
    trials_table_count=Count('data_dataset_session_related',
                             filter=Q(data_dataset_session_related__name='_ibl_trials.table.pqt'))
)
sessions = sessions.exclude(trials_table_count=0)
subjects = Subject.objects.filter(id__in=sessions.values_list('subject'))

# If only new, exclude all subjects that already have an aggregate
# existing files with this file name
if only_new_subjects:
    table_exists = Dataset.objects.filter(name='_ibl_subjectTrials.table.pqt').values_list('object_id', flat=True)
    subjects = subjects.exclude(id__in=table_exists)

# Go through subjects and check if aggregate needs to be (re)created
logger.info('\n')
logger.info(f' {subjects.count()} SUBJECTS')

"""
--------------------------
CHECK / CREATE AGGREGATES
--------------------------
"""

for i, sub in enumerate(subjects):
    try:
        logger.info(f'{i}/{subjects.count()} {sub.nickname}')
        # Find all sessions of this subject
        sub_sess = Session.objects.filter(subject=sub, task_protocol__icontains='ibl')
        # Get all trials tables
        trials_ds = Dataset.objects.filter(session__in=sub_sess, name='_ibl_trials.table.pqt', default_dataset=True)
        trials_ds = trials_ds.order_by('hash')
        # Make the expected aggregate hash
        new_hash = make_aggregate_hash(trials_ds)

        # Check if a default dataset exists in the database
        ds = Dataset.objects.filter(name='_ibl_subjectTrials.table.pqt', object_id=sub.id, default_dataset=True)

        # If no dataset exists yet, we create it
        if ds.count() == 0:
            logger.info('...aggregate does not yet exist, creating.')
            status_agg[f'{sub.id}'] = 'CREATE: aggregate does not exist'
            out_file = output_path.joinpath('Subjects', sub.lab.name, sub.nickname, '_ibl_subjectTrials.table.pqt')
            if dry:
                logger.info(f'...DRY RUN would create {out_file}')
                continue
            else:
                # Create the file, then the dataset entry, then add the uuid to the file name
                out_file = make_new_aggregate(out_file=out_file, trials_ds=trials_ds)
                new_ds = make_new_dataset(
                    new_file=out_file, new_hash=new_hash, sub=sub, old_ds=None, revision=None, alyx_user=alyx_user
                )
                new_out_file = out_file.rename(alfiles.add_uuid_string(out_file, new_ds.pk))

        # If there is exactly one default dataset, check if it needs updating
        elif ds.count() == 1:
            out_file = output_path.joinpath(
                ds.first().file_records.get(data_repository__name__startswith='flatiron').relative_path
            )
            # See if the file exists on disk (we are on SDSC so need to check with uuid in name)
            if alfiles.add_uuid_string(out_file, ds.first().pk).exists():
                # If the hash is the same we don't need to do anything, the file is correct
                if ds.first().json['aggregate_hash'] == new_hash:
                    logger.info('...aggregate exists and is up to date')
                    status_agg[f'{sub.id}'] = 'EXIST: aggregate exists, hash match'
                    continue
                # If the hash differs, we want to either overwrite or create a revision
                else:
                    # If protected, create a revision
                    if ds.first().is_protected:
                        logger.info('...aggregate already exists but is protected, hash mismatch, creating revision')
                        status_agg[f'{sub.id}'] = 'REVISION: aggregate exists protected, hash mismatch'
                        # If the current default is not a revision yet, add the revision part to the path
                        if ds.first().revision is None:
                            out_file = out_file.parent.joinpath(f"#{today}#", out_file.name)
                        # If there has already been a revision today something is probably wrong
                        elif ds.first().revision.name == today:
                            raise AssertionError('Dataset already has a revision for today. Something is probably wrong.')
                        # If the current default is already a revision, replace the revision part of the path
                        else:
                            out_file = out_file.parent.parent.joinpath(f"#{today}#", out_file.name)
                        if dry:
                            logger.info(f'...DRY RUN would create {out_file}')
                            continue
                        else:
                            # Create the file, then the dataset entry, then add the uuid to the file name
                            out_file = make_new_aggregate(out_file=out_file, trials_ds=trials_ds)
                            today_revision, _ = Revision.objects.get_or_create(name=today)
                            new_ds = make_new_dataset(new_file=out_file, new_hash=new_hash, sub=sub, old_ds=ds.first(),
                                                      revision=today_revision, alyx_user=alyx_user)
                            new_out_file = out_file.rename(alfiles.add_uuid_string(out_file, new_ds.pk))
                    # If the existing dataset is not protected, we just overwrite it
                    else:
                        out_file = alfiles.add_uuid_string(out_file, ds.first().pk)
                        logger.info('...aggregate already exists but is not protected, hash mismatch, overwriting')
                        status_agg[f'{sub.id}'] = 'OVERWRITE: aggregate exists not protected, hash mismatch'
                        if dry:
                            logger.info(f'...DRY RUN would create {out_file}')
                            continue
                        new_out_file = make_new_aggregate(out_file=out_file, trials_ds=trials_ds)
                        new_ds = update_dataset(ds.first(), new_out_file, new_hash)
            # If the dataset entry exist but the dataset cannot be found on disk, just recreate the dataset
            else:
                out_file = alfiles.add_uuid_string(out_file, ds.first().pk)
                logger.info('...dataset entry exists but file is missing on disk, creating new')
                status_agg[f'{sub.id}'] = 'CREATE: aggregate dataset entry exists, file missing'
                if dry:
                    logger.info(f'...DRY RUN would create {out_file}')
                    continue
                new_out_file = make_new_aggregate(out_file=out_file, trials_ds=trials_ds)
                new_ds = update_dataset(ds.first(), new_out_file, new_hash)

        # If there is more than one default dataset, that's a problem
        elif ds.count() > 1:
            raise AssertionError('Multiple default datasets found.')

    except BaseException as e:
        logger.error(f"...Error for subject {sub.nickname}: {e}")
        status_agg[f'{sub.id}'] = f'ERROR: {e}'
        continue

# Save status to file
status_agg = pd.DataFrame.from_dict(status_agg, orient='index', columns=['status'])
status_agg.insert(0, 'subject_id', status_agg.index)
status_agg.reset_index(drop=True, inplace=True)
status_agg.to_csv(log_path.joinpath('subjectTrials.csv'))

if not dry:
    # Sync whole collection folder to AWS (for now)
    src_dir = str(output_path.joinpath('Subjects'))
    dst_dir = 's3://ibl-brain-wide-map-private/aggregates/Subjects'
    cmd = ['aws', 's3', 'sync', src_dir, dst_dir, '--delete', '--profile', 'ibladmin', '--no-progress']
    logger.info(f"Syncing {src_dir} to AWS: " + " ".join(cmd))
    t0 = time.time()
    process = Popen(cmd, stdout=PIPE, stderr=STDOUT)
    with process.stdout:
        log_subprocess_output(process.stdout, logger.info)
    assert process.wait() == 0
    logger.debug(f'Session sync took {(time.time() - t0)} seconds')
    # Assume that everything that existed in that folder on FI was synced and set file records to exist
    fi_frs = FileRecord.objects.filter(
        data_repository__name='flatiron_aggregates', relative_path__startswith='Subjects', exists=True
    )
    aws_frs = FileRecord.objects.filter(
        data_repository__name='aws_aggregates', dataset__in=fi_frs.values_list('dataset', flat=True)
    )
    logger.info(f"Setting {aws_frs.count()} AWS file records to exists=True")
    aws_frs.update(exists=True)
