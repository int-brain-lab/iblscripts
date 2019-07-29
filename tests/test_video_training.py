# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Tuesday, February 19th 2019, 11:45:24 am
# @Last Modified by: Niccolò Bonacchi
# @Last Modified time: 19-02-2019 11:46:07.077
import shutil
import unittest
from pathlib import Path

import ibllib.pipes.experimental_data as iblrig_pipeline
from oneibl.one import ONE


class TestVideo(unittest.TestCase):

    def setUp(self):
        self.init_folder = Path('/mnt/s0/Data/IntegrationTests/Subjects_init')
        if not self.init_folder.exists():
            return
        # Set ONE to use the test database
        self.one = ONE(base_url='https://testdev.alyx.internationalbrainlab.org',  # testdev
                       username='test_user', password='TapetesBloc18')

        self.sessions = [x.parent for x in self.init_folder.rglob(
            'create_me.flag')]
        self.rig_folder = self.init_folder.parent / 'RigSubjects'
        self.server_folder = self.init_folder.parent / 'ServerSubjects'
        self.vidfiles = list(self.init_folder.rglob('*.avi'))
        # Init rig_folder
        if self.rig_folder.exists():
            shutil.rmtree(self.rig_folder)
        shutil.copytree(self.init_folder, self.rig_folder)

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
        shutil.rmtree(self.rig_folder, ignore_errors=True)
        shutil.rmtree(self.server_folder, ignore_errors=True)
        # os.system("ssh -i ~/.ssh/alyx.internationalbrainlab.org.pem ubuntu@test.alyx.internationalbrainlab.org './02_rebuild_from_cache.sh'")  # noqa


if __name__ == "__main__":
    unittest.main(exit=False)
