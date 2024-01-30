from one.api import ONE
from ibllib.pipes.training_status import (get_training_info_for_session, find_earliest_recompute_date, save_dataframe,
                                          make_plots, compute_training_status)
import pandas as pd
import matplotlib.pyplot as plt
import gc

from subprocess import Popen, PIPE, STDOUT
from pathlib import Path
import logging
import shutil

from iblutil.io import hashfile
from one.alf import files as alfiles
from data.models import Dataset, DataRepository, DataFormat, DatasetType, FileRecord
from misc.models import LabMember
from subjects.models import Subject

one = ONE()
PLOT = False
REGISTER = False

save_path = Path('/home/datauser/temp/subject_training')
save_path.mkdir(exist_ok=True)

# COMPUTE THE SUBJECT TRAINING FOR THE SPECIFIED SUBJECTS
trials_ds = Dataset.objects.filter(name='_ibl_subjectTrials.table.pqt')
trials_subjects = Subject.objects.filter(id__in=trials_ds.values_list('object_id', flat=True))
training_ds = Dataset.objects.filter(name='_ibl_subjectTraining.table.pqt')
training_subjects = Subject.objects.filter(id__in=training_ds.values_list('object_id', flat=True))
subjects = trials_subjects.difference(training_subjects)

files = []

status_agg = {}
for sub in subjects:

    try:
        f = next(Path(f'/mnt/ibl/aggregates/Subjects/{sub.lab.name}/{sub.nickname}').glob('_ibl_subjectTrials.table*'))
        subj_path = save_path.joinpath(*f.parent.parts[-2:])
        subj_path.mkdir(parents=True, exist_ok=True)
        subj_df = pd.read_parquet(f)
        eids = subj_df.session.unique()

        missing_dates = pd.DataFrame()
        for eid in eids:
            session_path = one.eid2path(eid)
            session_path = Path('/mnt/ibl').joinpath(*session_path.parts[-5:])
            s_df = pd.DataFrame({'date': session_path.parts[-2], 'session_path': str(session_path)}, index=[0])
            missing_dates = pd.concat([missing_dates, s_df], ignore_index=True)

        missing_dates = missing_dates.sort_values('date')

        df = None
        # Iterate through the dates to fill up our training dataframe
        for _, grp in missing_dates.groupby('date'):
            sess_dicts = get_training_info_for_session(grp.session_path.values, one, force=False)
            if len(sess_dicts) == 0:
                continue

            for sess_dict in sess_dicts:
                if df is None:
                    df = pd.DataFrame.from_dict(sess_dict)
                else:
                    df = pd.concat([df, pd.DataFrame.from_dict(sess_dict)])

        # Sort values by date and reset the index
        df = df.sort_values('date')
        df = df.reset_index(drop=True)

        eids = []
        for sess in df.session_path.values:
            eids.append(one.path2eid(sess))

        df['session'] = eids
        # Save our dataframe
        save_dataframe(df, subj_path)

        # idx = np.where(df['task_protocol'].values == 'biased_opto')[0]
        # df['task_protocol'][idx] = 'biased'

        # Now go through the backlog and compute the training status for sessions.
        # If for example one was missing as it is cumulative
        # we need to go through and compute all the backlog
        # Find the earliest date in missing dates that we need to recompute the training status for
        missing_status = find_earliest_recompute_date(df.drop_duplicates('date').reset_index(drop=True))
        for date in missing_status:
            df = compute_training_status(df, date, one, force=False)

        # df = load_existing_dataframe(subj_path=subj_path)
        # df['task_protocol'][idx] = 'biased_opto'

        # Add in untrainable or unbiasable
        df_lim = df.drop_duplicates(subset='session', keep='first')
        # Detect untrainable
        un_df = df_lim[df_lim['training_status'] == 'in training'].sort_values('date')
        if len(un_df) >= 40:
            print('untrainable')
            sess = un_df.iloc[39].session
            df.loc[df['session'] == sess, 'training_status'] = 'untrainable'

        # Detect unbiasable
        un_df = df_lim[df_lim['task_protocol'] == 'biased'].sort_values('date')
        if len(un_df) >= 40:
            tr_st = un_df[0:40].training_status.unique()
            if 'ready4ephysrig' not in tr_st:
                print('unbiasable')
                sess = un_df.iloc[39].session
                df.loc[df['session'] == sess, 'training_status'] = 'unbiasable'

        save_dataframe(df, subj_path)

        # Need to get out sub dataframe
        status = df.set_index('date')[['training_status', 'session']].drop_duplicates(subset='training_status',
                                                                                      keep='first')
        status.to_parquet(subj_path.joinpath('_ibl_subjectTraining.table.pqt'))
        files.append(subj_path.joinpath('_ibl_subjectTraining.table.pqt'))
        status_dict = status.to_dict()

        date, sess = status_dict.items()
        data = {'trained_criteria': {v.replace(' ', '_'): (k, sess[1][k]) for k, v in date[1].items()}}

        one.alyx.json_field_update('subjects', sub.nickname, data=data)
    except BaseException as e:
        status_agg[sub.id] = e

    # Upload the plots to alyx
    if PLOT:
        session_path = subj_path.joinpath('2023-07-17', '001')
        make_plots(subj_path, one, df=df, save=True, upload=True)
        plt.close('all')
        gc.collect()


# TO register - this needs to be run in an django shell

if REGISTER:

    revision = None
    root_path = Path('/home/datauser/temp/subject_training')
    output_path = Path('/mnt/ibl/aggregates/')
    collection = 'Subjects'
    alyx_user = 'mayo'
    version = '1.0'

    root_dir = Path(f'/mnt/ibl/aggregates/{collection}/')
    fname = '_ibl_subjectTraining.table.pqt'
    dtype = 'subjectTraining.table'

    dataset_type = DatasetType.objects.get(name=dtype)
    dataset_format = DataFormat.objects.get(name='parquet')
    aws_repo = DataRepository.objects.get(name='aws_aggregates')
    fi_repo = DataRepository.objects.get(name='flatiron_aggregates')
    alyx_user = LabMember.objects.get(username=alyx_user)

    logger = logging.getLogger('ibllib')
    logger.setLevel(logging.INFO)

    def log_subprocess_output(pipe, log_function=print):
        for line in iter(pipe.readline, b''):
            log_function(line.decode().strip())

    for file in files:

        subject = file.parts[-2]
        out_file = output_path.joinpath(collection, *file.parts[-3:])
        # out_file.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy(file, out_file)

        sub = Subject.objects.get(nickname=subject)
        file_hash = hashfile.md5(out_file)
        file_size = out_file.stat().st_size
        # Create new dataset
        new_ds = Dataset.objects.create(
            name=fname,
            hash=file_hash,
            file_size=file_size,
            revision=revision,
            collection=collection,
            default_dataset=True,
            dataset_type=dataset_type,
            data_format=dataset_format,
            created_by=alyx_user,
            version=version,
            content_object=sub,  # content object is subject
        )
        # Validate dataset
        new_ds.full_clean()
        new_ds.save()

        # Change name on disk to include dataset id
        new_out_file = out_file.rename(alfiles.add_uuid_string(out_file, new_ds.pk))
        assert new_out_file.exists(), f"Failed to save renamed file {new_out_file}"
        logger.info(f"...Renamed file to {new_out_file}")
        # Create one file record per repository
        for repo in [aws_repo, fi_repo]:
            record = {
                'dataset': new_ds,
                'data_repository': repo,
                'relative_path': str(out_file.relative_to(root_dir.parent)),
                'exists': False if repo.name.startswith('aws') else True
            }
            try:
                _ = FileRecord.objects.get_or_create(**record)
            except BaseException as e:
                logger.error(f'...ERROR: Failed to create file record on {repo.name}: {e}')
                continue

        logger.info(f"...Created new dataset entry {new_ds.pk} and file records")

    # Sync to AWS
    src_dir = str(root_dir)
    dst_dir = f's3://ibl-brain-wide-map-private/aggregates/{collection}/'
    cmd = ['aws', 's3', 'sync', src_dir, dst_dir, '--delete', '--profile', 'ibladmin', '--no-progress']
    logger.info(f"Syncing {src_dir} to AWS: " + " ".join(cmd))
    process = Popen(cmd, stdout=PIPE, stderr=STDOUT)
    with process.stdout:
        log_subprocess_output(process.stdout, logger.info)
    assert process.wait() == 0
    # Assume that everything that existed in that folder on FI was synced and set file records to exist
    fi_frs = FileRecord.objects.filter(data_repository=fi_repo, relative_path__startswith=collection, exists=True,
                                       dataset__name=fname)
    aws_frs = FileRecord.objects.filter(data_repository=aws_repo, dataset__in=fi_frs.values_list('dataset', flat=True))
    logger.info(f"Setting {aws_frs.count()} AWS file records to exists=True")
    aws_frs.update(exists=True)
