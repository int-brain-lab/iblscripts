import unittest

import alf.io
from ibllib.io.extractors import training_audio as audio

from ci.tests import base


class TestAudioExtraction(base.IntegrationTest):

    def setUp(self):
        file_wav = self.data_path.joinpath('Subjects_init', 'ZM_1085', '2019-06-24', '001',
                                           'raw_behavior_data', '_iblrig_micData.raw.wav')
        self.ses_path = file_wav.parents[1]
        if not self.ses_path.exists():
            return

    def test_qc_extract(self):
        # extract audio
        audio.extract_sound(self.ses_path, save=True)
        D = alf.io.load_object(self.ses_path / 'raw_behavior_data', 'audioSpectrogram')
        cues = alf.io.load_object(self.ses_path / 'raw_behavior_data', 'audioOnsetGoCue')
        self.assertEqual(cues['times_mic'].size, 4)
        self.assertEqual(D['power'].shape[0], D['times_mic'].shape[0])
        self.assertEqual(D['frequencies'].shape[1], D['power'].shape[1])
        # now test the registration of the data

    def tearDown(self):
        path_out = self.ses_path / 'raw_behavior_data'
        for f in path_out.glob('_iblmic_*'):
            f.unlink()


if __name__ == "__main__":
    unittest.main(exit=False)
