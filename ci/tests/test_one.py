import unittest

import numpy as np

import oneibl.params
from oneibl.stream import VideoStreamer


class TestONE(unittest.TestCase):
    def setUp(self) -> None:
        url = ('/integration/dlc/videos/CSHL_015/'
               'churchlandlab_CSHL_015_2019-11-12_001__iblrig_leftCamera.raw.short.mp4')
        root = oneibl.params.get().HTTP_DATA_SERVER
        self.url = f'{root}/{url}'

    def test_video_streamer(self):
        frame_id = 5
        vs = VideoStreamer(self.url)
        f, im = vs.get_frame(frame_id)
        assert vs.total_frames == 1200
        assert f
        # Test with data set dict
        dset = {'file_records': [{'data_url': self.url}]}
        vs = VideoStreamer(dset)
        f, im2 = vs.get_frame(frame_id)
        assert np.all(im == im2)
