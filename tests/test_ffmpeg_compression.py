import shutil
import unittest
import tempfile
from pathlib import Path

from ibllib.pipes import ephys_preprocessing, training_preprocessing

from . import base

class TestVideoAudioEphys(base.IntegrationTest):

    def test_compress_all_vids(self):
        EPHYS_INIT_FOLDER = self.data_path.joinpath('ephys', 'ephys_video_init')
        TRAINING_INIT_FOLDER = self.data_path.joinpath('Subjects_init')

        with tempfile.TemporaryDirectory() as tdir:
            shutil.copytree(EPHYS_INIT_FOLDER, Path(tdir).joinpath('Subjects'))
            for ts_file in Path(tdir).rglob("_iblrig_taskSettings.raw.json"):
                """
                Test the video compression
                """
                session_path = ts_file.parents[1]
                job = ephys_preprocessing.EphysVideoCompress(session_path)
                job.run()
                # check output files and non-existent inputs
                self.assertTrue(len(list(session_path.rglob('*.avi'))) == 0)
                self.assertTrue(len(list(session_path.rglob('*.mp4'))) == 3)
                self.assertTrue(len(job.outputs) == 3)
                # a second run should still output files for registration
                job.run()
                self.assertTrue(len(job.outputs) == 3)
                """
                Do the audio compression test as well
                """
                job_audio = ephys_preprocessing.EphysAudio(session_path)
                job_audio.run()
                self.assertIsNone(next(session_path.rglob('*.wav'), None))
                self.assertTrue(next(session_path.rglob('*.flac')) == job_audio.outputs[0])


class TestVideoTraining(base.IntegrationTest):

    def setUp(self) -> None:
        self.TRAINING_INIT_FOLDER = self.data_path.joinpath('Subjects_init')
        assert self.TRAINING_INIT_FOLDER.exists()

    def test_compress_training(self):
        vid_files = list(self.TRAINING_INIT_FOLDER.rglob('*.avi'))
        for vid_file in vid_files:
            init_session_path = vid_file.parents[1]
            with tempfile.TemporaryDirectory() as tdir:
                session_path = Path(tdir).joinpath(
                    init_session_path.relative_to(self.TRAINING_INIT_FOLDER))
                shutil.copytree(init_session_path, session_path)
                job = training_preprocessing.TrainingVideoCompress(session_path)
                job.run()
                self.assertIsNone(next(session_path.rglob('*.avi'), None))
                self.assertEqual(next(session_path.rglob('*.mp4')), job.outputs[0])
