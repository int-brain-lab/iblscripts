import unittest
from pathlib import Path

import alf.io
from ibllib.io.extractors import training_audio as audio


class TestAudioExtraction(unittest.TestCase):

    def setUp(self):
        file_wav = Path('/mnt/s0/Data/IntegrationTests/Subjects_init/ZM_1085/2019-06-24/001/'
                        'raw_behavior_data/_iblrig_micData.raw.wav')
        self.ses_path = file_wav.parents[1]
        if not self.ses_path.exists():
            return

    def test_qc_extract(self):
        # extract audio
        audio.extract_sound(self.ses_path, save=True)
        D = alf.io.load_object(self.ses_path / 'raw_behavior_data', '_iblmic_audioSpectrogram')
        cues = alf.io.load_object(self.ses_path / 'raw_behavior_data',
                                  '_iblmic_audioOnsetGoCue.times_mic')
        self.assertEqual(cues['times_mic'].size, 5)
        self.assertEqual(D['power'].shape[0], D['times_mic'].shape[0])
        self.assertEqual(D['frequencies'].shape[1], D['power'].shape[1])
        # now test the registration of the data

    def tearDown(self):
        path_out = self.ses_path / 'raw_behavior_data'
        for f in path_out.glob('_iblmic_*'):
            f.unlink()


if __name__ == "__main__":
    unittest.main(exit=False)
