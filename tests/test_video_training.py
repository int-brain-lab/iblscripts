# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Tuesday, February 19th 2019, 11:45:24 am
# @Last Modified by: Niccolò Bonacchi
# @Last Modified time: 19-02-2019 11:46:07.077
import shutil
import unittest
import tempfile
from pathlib import Path

import ibllib.io.flags
import ibllib.pipes.experimental_data as iblrig_pipeline
from oneibl.one import ONE


class TestVideoEphys(unittest.TestCase):

    def test_compress_all_vids(self):
        self.init_folder = Path('/mnt/s0/Data/IntegrationTests/ephys/ephys_video_init')
        with tempfile.TemporaryDirectory() as tdir:
            root_path = Path(tdir).joinpath('Subjects')
            shutil.copytree(self.init_folder, root_path)
            # creates the flags
            ibllib.io.flags.create_compress_video_flags(root_path,
                                                        flag_name='compress_video_ephys.flag')
            iblrig_pipeline.compress_ephys_video(root_path, dry=False)
            # compress video flags is replaced by register me flag, and 3 mp4 files appeared
            self.assertIsNone(next(root_path.rglob('compress_video_ephys.flag'), None))
            self.assertIsNone(next(root_path.rglob('*.avi'), None))
            self.assertTrue(len(list(root_path.rglob('register_me.flag'))) == 1)
            self.assertTrue(len(list(root_path.rglob('*.mp4'))) == 3)
            """
            Do the audio compression test as well
            """
            ibllib.io.flags.create_audio_flags(root_path, flag_name='audio_ephys.flag')
            iblrig_pipeline.compress_audio(root_path, dry=False)
            self.assertIsNone(next(root_path.rglob('audio_ephys.flag'), None))
            self.assertIsNone(next(root_path.rglob('*.wav'), None))
            self.assertTrue(len(list(root_path.rglob('*.flac'))) == 1)


class TestVideoTraining(unittest.TestCase):

    def setUp(self):
        self.init_folder = Path('/mnt/s0/Data/IntegrationTests/Subjects_init')
        if not self.init_folder.exists():
            return
        # Set ONE to use the test database
        self.one = ONE(base_url='https://testdev.alyx.internationalbrainlab.org',  # testdev
                       username='test_user', password='TapetesBloc18')
        self.vidfiles = list(self.init_folder.rglob('*.avi'))
        # Init rig_folder
        self.server_folder = self.init_folder.parent / 'ServerSubjects'
        if self.server_folder.exists():
            shutil.rmtree(self.server_folder)
        shutil.copytree(self.init_folder, self.server_folder)
        for vidfile in self.server_folder.rglob('*.avi'):
            ibllib.io.flags.create_compress_video_flags(vidfile.parents[1])

    def _registration(self):
        iblrig_pipeline.register(self.server_folder, one=self.one)
        # Check for deletion of register_me.flag
        rflags = list(self.server_folder.rglob('register_me.flag'))
        self.assertTrue(rflags == [])

    def _compression(self):
        iblrig_pipeline.compress_video(self.server_folder)
        # Check for deletion of compress_video.flag
        cflags = list(self.server_folder.rglob('compress_video.flag'))
        self.assertTrue(cflags == [])
        # Check for creation of register_me.flag
        rflags = list(self.server_folder.rglob('register_me.flag'))
        self.assertTrue(rflags != [])
        # Check for flag in all sessions
        self.assertTrue(len(rflags) == len(self.vidfiles))

    def test_video_training(self):
        if not self.init_folder.exists():
            return
        self._compression()
        self._registration()

    def tearDown(self):
        if not self.init_folder.exists():
            return
        shutil.rmtree(self.server_folder, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(exit=False)
