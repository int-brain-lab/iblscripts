import shutil
import tempfile
import unittest.mock
from pathlib import Path

from ibllib.pipes import ephys_preprocessing, training_preprocessing

from ci.tests import base


class TestVideoAudioEphys(base.IntegrationTest):

    @unittest.mock.patch('ibllib.qc.camera.CameraQC')
    @unittest.mock.patch('ibllib.io.extractors.camera.extract_all')
    def test_compress_all_vids(self, mock_ext, mock_qc):
        EPHYS_INIT_FOLDER = self.data_path.joinpath('ephys', 'ephys_video_init')

        mock_ext.return_value = (None, [])  # Return empty file list upon timestamp extraction
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
                mock_ext.assert_called()
                self.assertEqual(mock_qc.call_count, 3)
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

    @unittest.mock.patch('ibllib.pipes.training_preprocessing.CameraQC')
    @unittest.mock.patch('ibllib.io.extractors.camera.extract_all')
    def test_compress(self, mock_ext, mock_qc):
        """
        Here we're testing the video compression only.  The timestamp extraction and camera QC is
        stubbed and will not return any files.
        :param mock_ext: A stubbed camera extractor class.  Outputs no files.
        :param mock_qc: A stubbed camera qc class.  Does nothing.
        :return:
        """
        vid_files = list(self.TRAINING_INIT_FOLDER.rglob('*.avi'))
        mock_ext.return_value = (None, [])  # Return empty file list upon timestamp extraction
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
            mock_qc.assert_called_once()
            mock_qc.reset_mock()


if __name__ == "__main__":
    import unittest
    unittest.main(exit=False, verbosity=2)
