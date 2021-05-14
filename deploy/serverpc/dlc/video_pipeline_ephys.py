"""
Entry point to system commands for IBL Videos pipeline.

>>> python video_pipelines.py create_flags /mnt/s0/Data/Subjects/ --dry=True
>>> python experimental_data.py dlc_training /mnt/s0/Data/Subjects/ [--dry=True --count=1]
"""

from pathlib import Path
import argparse
import subprocess
import logging
import time
import yaml
import shutil
import json
from dateutil.parser import parse
import re

import numpy as np
import pandas as pd

import cv2  # pip install opencv-python
import random
import os

import deeplabcut
import ibllib.io.flags

logger = logging.getLogger('ibllib')


# Cuda paths may need to be set
# In [11]: !export PATH="$PATH:/usr/local/cuda/bin"
#    ...: !export LD_LIBRARY_PATH="/usr/local/cuda/lib64"


def _order_glob_by_session_date(flag_files):
    """
    Given a list/generator of PurePaths below an ALF session folder, output a list of of PurePaths
    sorted by date in reverse order.
    :param flag_files: list/generator of PurePaths
    :return: list of PurePaths
    """
    flag_files = list(flag_files)

    def _fdate(fl):
        dat = [parse(fp)
               for fp in fl.parts if re.match(r'\d{4}-\d{2}-\d{2}', fp)]
        if dat:
            return dat[0]
        else:
            return parse('1999-12-12')

    t = [_fdate(fil) for fil in flag_files]
    return [f for _, f in sorted(zip(t, flag_files), reverse=True)]


def _set_dlc_paths(path_dlc):
    """
    OMG! Hard-coded paths everywhere ! Replace hard-coded paths.
    """
    for yaml_file in path_dlc.rglob('config.yaml'):
        # read the yaml config file
        with open(yaml_file) as fid:
            yaml_data = yaml.safe_load(fid)
        # if the path is correct skip to next
        if Path(yaml_data['project_path']) == yaml_file.parent:
            continue
        # else read the whole file
        with open(yaml_file) as fid:
            yaml_raw = fid.read()
        # patch the offending line and rewrite properly
        with open(yaml_file, 'w+') as fid:
            fid.writelines(
                yaml_raw.replace(
                    yaml_data['project_path'], str(
                        yaml_file.parent)))


def _run_command(command):
    """
    Runs a shell command using subprocess.

    :param command: command to run
    :return: dictionary with keys: process, stdout, stderr
    """
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    info, error = process.communicate()
    return {
        'process': process,
        'stdout': info.decode(),
        'stderr': error.decode()}


def _get_crop_window(df_crop, roi_name):
    """
    h5 is local file name only; get average position of a pivot point for autocropping

    :param df_crop: data frame from hdf5 file from video data
    :param roi_name: string among those: ('eye', 'nose_tip', 'tongue', 'paws',)
    :return: list of floats [width, height, x, y] defining window used for ffmpeg crop command
    """
    # choose parts to find pivot point which is used to crop around a ROI
    Q = {
        'eye': [
            'pupil_top_r'],
        'nose_tip': ['nose_tip'],
        'tongue': [
            'tube_top',
            'tube_bottom'],
        'paws': ['nose_tip'],
        'tail_start': ['tail_start']}
    parts = Q[roi_name]
    XYs = []
    for part in parts:
        x_values = df_crop[(df_crop.keys()[0][0], part, 'x')].values
        y_values = df_crop[(df_crop.keys()[0][0], part, 'y')].values
        likelyhoods = df_crop[(df_crop.keys()[0][0],
                               part, 'likelihood')].values

        mx = np.ma.masked_where(likelyhoods < 0.9, x_values)
        x = np.ma.compressed(mx)
        my = np.ma.masked_where(likelyhoods < 0.9, y_values)
        y = np.ma.compressed(my)

        XYs.append([int(np.nanmean(x)), int(np.nanmean(y))])

    xy = np.mean(XYs, axis=0)
    p = {'eye': [100, 100, xy[0] - 50, xy[1] - 50],
         'nose_tip': [100, 100, xy[0] - 50, xy[1] - 50],
         'tongue': [160, 160, xy[0] - 60, xy[1] - 100],
         'paws': [900, 800, xy[0], xy[1] - 100],
         'tail_start': [220, 220, xy[0] - 110, xy[1] - 110]}

    return p[roi_name]


def create_flags(root_path, dry=False):
    """
    Create flag files for the training DLC process
    """
    ibllib.io.flags.create_dlc_flags(
        root_path /
        'dlc_training.flag',
        clobber=True,
        dry=dry)


def dlc_ephys(file_mp4, force=False):
    """
    Analyse a leftCamera, rightCamera or bodyCamera video with DeepLabCut

    The process consists in 7 steps:
    0- Check if rightCamera video, then make it look like leftCamera video
    1- subsample video frames using ffmpeg
    2- run DLC to detect ROIS: 'eye', 'nose_tip', 'tongue', 'paws'
    3- crop videos for each ROIs using ffmpeg, subsample paws videos
    4- run DLC specialized networks on each ROIs
    5- output ALF dataset for the raw DLC output in ./session/alf/_ibl_leftCamera.dlc.json

    # This is a directory tree of the temporary files created
    # ./raw_video_data/dlc_tmp/  # tpath: temporary path
    #   _iblrig_leftCamera.raw.mp4'  # file_mp4
    #   _iblrig_leftCamera.subsampled.mp4  # file_temp['mp4_sub']
    #   _iblrig_leftCamera.subsampledDeepCut_resnet50_trainingRigFeb11shuffle1_550000.h5
    # tfile['h5_sub']
    #   _iblrig_leftCamera.eye.mp4 # tfile['eye']
    #   _iblrig_leftCamera.nose_tip.mp4 # tfile['nose_tip']
    #   _iblrig_leftCamera.tongue.mp4 # tfile['tongue']
    #   _iblrig_leftCamera.pose.mp4 # tfile['pose']

    :param file_mp4: file to run
    :return: None
    """
  
    if 'bodyCamera' in str(file_mp4):
        ROIs = ('tail_start',)
    else:
        ROIs = ('eye',  'tongue', 'paws', 'nose_tip',)

    file_mp4 = Path(file_mp4)

    def s00_transform_rightCam():
        '''
        The right cam is first flipped and rotated
        then the spatial resolution is increased
        such that the rightCamera video looks like the
        leftCamera video
        '''

        print('Flipping and turning rightCamera video')
        command_flip = ('ffmpeg -nostats -y -loglevel 0 -i {file_in} -vf '
                        '"transpose=1,transpose=1" -vf hflip {file_out}')
        file_out1 = str(file_mp4).replace('.raw.', '.flipped.')
        pop = _run_command(
            command_flip.format(
                file_in=file_mp4,
                file_out=file_out1))
        if pop['process'].returncode != 0:
            logger.error(f' DLC 0a/5: Flipping ffmpeg failed: {file_mp4}' + pop['stderr'])

        print('Increasing resolution of rightCamera video')
        command_reso_up = ('ffmpeg -nostats -y -loglevel 0 -i {file_in} '
                           '-vf scale=1280:1024 {file_out}')
        file_out2 = file_out1.replace('.flipped.', '.raw.transformed.')
        pop = _run_command(
            command_reso_up.format(
                file_in=file_out1,
                file_out=file_out2))
        if pop['process'].returncode != 0:
            logger.error(f' DLC 0b/5: Increase reso ffmpeg failed: {file_mp4}' + pop['stderr'])

        os.remove(file_out1)

        return file_out2

    # If the video is rightCamera, it is flipped and upsampled in resolution
    # first
    if 'rightCamera' in str(file_mp4):
        transformed_video = s00_transform_rightCam()
        file_mp4 = Path(transformed_video)

    file_label = file_mp4.stem.split('.')[0].split('_')[-1]
    file_alf_dlc = file_mp4.parents[0] / f'_ibl_{file_label}.dlc.npy'

    path_dlc = Path('/home/mic/DLC_weights')
    _set_dlc_paths(path_dlc)


    if 'bodyCamera' in str(file_mp4):
        dlc_params = {
            'roi_detect': path_dlc / 'tail-mic-2019-12-16' / 'config.yaml',            
            'tail_start': path_dlc / 'tail-mic-2019-12-16' / 'config.yaml'}        
    else:
        dlc_params = {
            'roi_detect': path_dlc / 'roi_detect' / 'config.yaml',
            'eye': path_dlc / 'eye-mic-2020-01-17' / 'config.yaml',
            'nose_tip': path_dlc / 'nose_tip' / 'config.yaml',
            'paws': path_dlc / 'paws-mic-2019-04-26' / 'config.yaml',
            'tongue': path_dlc / 'tongue-mic-2019-04-26' / 'config.yaml'}


    # create the paths of temporary files, see above for an example
    ps = 'dlc_tmp_%s' %str(file_mp4.name)[:-4]
    tpath = file_mp4.parent / ps
    tpath.mkdir(exist_ok=True)
    tfile = {'mp4_sub': tpath /
             str(file_mp4.name).replace('.raw.', '.subsampled.')}
    for roi in ROIs:
        tfile[roi] = tpath / str(file_mp4.name).replace('.raw.', f'.{roi}.')

    def s01_subsample():
        '''
        step 1 subsample video for detection
        put 500 uniformly sampled frames into new video
        '''

        if tfile['mp4_sub'].exists() and not force:
            return

        file_out = tfile['mp4_sub']

        cap = cv2.VideoCapture(str(file_mp4))
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        samples = random.sample(range(1, frameCount - 1), 500)
        size = (int(cap.get(3)), int(cap.get(4)))
        out = cv2.VideoWriter(
            str(file_out), cv2.VideoWriter_fourcc(
                *'mp4v'), 5, size)

        for i in samples:
            cap.set(1, i)
            _, frame = cap.read()
            out.write(frame)

        out.release()

    def s02_detect_rois():
        """
        step 2 run DLC to detect ROIS
        returns: df_crop, dataframe used to crop video
        """
        out = deeplabcut.analyze_videos(
            dlc_params['roi_detect'], [str(tfile['mp4_sub'])])
        deeplabcut.create_labeled_video(
            dlc_params['roi_detect'], [str(tfile['mp4_sub'])])
        tfile['h5_sub'] = next(tpath.glob(f'*{out}*.h5'), None)
        return pd.read_hdf(tfile['h5_sub'])

    def s03_crop_videos():
        """
        step 3 crop videos using ffmpeg
        returns: dictionary of cropping coordinates relative to upper left corner
        """
        crop_command = (
            'ffmpeg -nostats -y -loglevel 0  -i {file_in} -vf "crop={w[0]}:{w[1]}:'
            '{w[2]}:{w[3]}" -c:v libx264 -crf 11 -c:a copy {file_out}')
        whxy = {}
        for roi in ROIs:
            whxy[roi] = _get_crop_window(df_crop, roi)
            if tfile[roi].exists():
                continue
            pop = _run_command(crop_command.format(file_in=file_mp4,
                                                   file_out=tfile[roi],
                                                   w=whxy[roi]))
            logger.info('cropping ' + roi + ' video')
            if pop['process'].returncode != 0:
                logger.error(f'DLC 3/5: Cropping ffmpeg failed for ROI {roi}, file: {file_mp4}')
 
        if 'bodyCamera' not in str(file_mp4):
            # for paws spatial downsampling after cropping in order to speed up
            # processing x4
            
            tfile['paws_big'] = tfile['paws']
            tfile['paws'] = tfile['paws'].parent / \
                tfile['paws_big'].name.replace('paws', 'paws.small')
            if tfile['paws'].exists() and not force:
                return whxy
            cmd = (
                'ffmpeg -nostats -y -loglevel 0 -i {file_in} -vf scale=450:374 -c:v libx264 -crf 17'
                ' -c:a copy {file_out}').format(
                file_in=tfile['paws_big'],
                file_out=tfile['paws'])
            pop = _run_command(cmd)
            if pop['process'].returncode != 0:
                logger.error(f"DLC 3/5: Subsampling paws failed: {tfile['paws_big']}")

        return whxy

    def s04_run_dlc_specialized_neworks():
        for roi in ROIs:
            deeplabcut.analyze_videos(str(dlc_params[roi]), [str(tfile[roi])])
            deeplabcut.create_labeled_video(
                str(dlc_params[roi]), [str(tfile[roi])])

    def s05_extract_dlc_alf():
        """
        Output an ALF matrix with column names containing the full DLC results [nframes, nfeatures]
        """
        # tpath = Path('/mnt/s0/Data/Subjects/ZM_1374/2019-03-23/001/raw_video_data/dlc_tmp')
        columns = []
        for roi in ROIs:
            df = pd.read_hdf(next(tpath.glob(f'*{roi}*.h5')))
            # get the indices of this multi index hierarchical thing
            # translate and scale the specialized window in the full initial
            # frame
            indices = df.columns.to_flat_index()
            scale = 2 if roi == 'paws' else 1
            for ind in indices:
                if ind[-1] == 'x':
                    df[ind] = df[ind].apply(lambda x: x * scale + whxy[roi][2])
                elif ind[-1] == 'y':
                    df[ind] = df[ind].apply(lambda x: x * scale + whxy[roi][3])
            # concatenate this in a flat matrix
            columns.extend([f'{c[1]}_{c[2]}' for c in df.columns.to_flat_index()])
            if 'A' not in locals():
                A = np.zeros([df.shape[0], 0], float)
            A = np.c_[A, np.array(df).astype(float)]
        assert(len(columns) == A.shape[1])

        # write the ALF files without depending on ibllib
        file_meta_data = file_alf_dlc.parent / f'_ibl_{file_label}.dlc.metadata.json'
        np.save(file_alf_dlc, A)
        with open(file_meta_data, 'w+') as fid:
            fid.write(json.dumps({'columns': columns}, indent=1))
        return file_alf_dlc, file_meta_data
 
    # run steps one by one
    s01_subsample()
    df_crop = s02_detect_rois()
    whxy = s03_crop_videos()
    s04_run_dlc_specialized_neworks()
    files_alf = s05_extract_dlc_alf()
    # shutil.rmtree(tpath)  # and then clean up
    if '.raw.transformed' in str(file_mp4):
        os.remove(file_mp4)
        
    return files_alf


# if __name__ == "__main__":
#    ALLOWED_ACTIONS = ['dlc_training', 'create_flags']
#    parser = argparse.ArgumentParser(description='Description of your program')
#    parser.add_argument('action', help='Action: ' + ','.join(ALLOWED_ACTIONS))
#    parser.add_argument('folder', help='A Folder containing one or several sessions')
#    parser.add_argument('--dry', help='Dry Run', required=False, default=False, type=str)
#    parser.add_argument('--count', help='Max number of sessions to run this on',
#                        required=False, default=10, type=int)
#    args = parser.parse_args()  # returns data from the options specified (echo)
#    if args.dry and args.dry.lower() == 'false':
#        args.dry = False
#    assert(Path(args.folder).exists())
#    if args.action == 'dlc_training':
#        main_path = Path(args.folder)
#        c = 0
#        # look for dlc training flag files
#        flag_files = Path(main_path).rglob('dlc_training.flag')
#        # sort them according to the session date, so the more recent gets processed first
#        flag_files = _order_glob_by_session_date(flag_files)
#        for flag_file in flag_files:
#            # flag files should contain file names. If not delete them
#            rel_path = ibllib.io.flags.read_flag_file(flag_file)
#            if isinstance(rel_path, bool):
#                flag_file.unlink()
#                continue
#            for relative_path in rel_path:
#                video_file = flag_file.parent / relative_path
#                t0 = time.time()
#                c += 1
#                # stop the loop if the counter is exhausted
#                if c > args.count:
#                    break
#                logger.info(str(video_file))
#                # dry run only prints and exit
#                if args.dry:
#                    continue
#                # run the main job
#                files_alf = dlc_training(video_file)
#                # remove the dlc_compute flag
#                flag_file.unlink()
#                # create the register_me flag
#                file_list = [str(fil.relative_to(files_alf[0].parents[1])) for fil in files_alf]
#                ibllib.io.flags.write_flag_file(files_alf[0].parents[1] / 'register_me.flag',
#                                                file_list=file_list)
#                t1 = time.time()
#                logger.info(str(video_file) + 'Completed in ' + str(int(t1 - t0)) + ' secs')

#    elif args.action == 'create_flags':
#        create_flags(Path(args.folder), dry=args.dry)
#    else:
#        logger.error('Allowed actions are: ' + ', '.join(ALLOWED_ACTIONS))
