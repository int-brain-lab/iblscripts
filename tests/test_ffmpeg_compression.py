import shutil
import unittest
import tempfile
from pathlib import Path

from ibllib.pipes import ephys_preprocessing, training_preprocessing

PATH_TESTS = Path('/mnt/s0/Data/IntegrationTests')
EPHYS_INIT_FOLDER = PATH_TESTS.joinpath('ephys', 'ephys_video_init')
TRAINING_INIT_FOLDER = PATH_TESTS.joinpath('Subjects_init')


class TestVideoAudioEphys(unittest.TestCase):

    def test_compress_all_vids(self):
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
                # a second run should not output anything
                job.run()
                self.assertTrue(len(job.outputs) == 0)
                """
                Do the audio compression test as well
                """
                job_audio = ephys_preprocessing.EphysAudio(session_path)
                job_audio.run()
                self.assertIsNone(next(session_path.rglob('*.wav'), None))
                self.assertTrue(next(session_path.rglob('*.flac')) == job_audio.outputs[0])


class TestVideoTraining(unittest.TestCase):

    def setUp(self):
        if not TRAINING_INIT_FOLDER.exists():
            return
        vid_files = list(TRAINING_INIT_FOLDER.rglob('*.avi'))
        for vid_file in vid_files:
            init_session_path = vid_file.parents[1]
            with tempfile.TemporaryDirectory() as tdir:
                session_path = Path(tdir).joinpath(
                    init_session_path.relative_to(TRAINING_INIT_FOLDER))
                shutil.copytree(init_session_path, session_path)
                job = training_preprocessing.TrainingVideoCompress(session_path)
                job.run()
                self.assertIsNone(next(session_path.rglob('*.avi'), None))
                self.assertEqual(next(session_path.rglob('*.mp4')), job.outputs[0])
