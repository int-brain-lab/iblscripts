import numpy as np

from ibllib.io.video import VideoStreamer
from one.api import ONE

from ci.tests.base import IntegrationTest, TEST_DB

one = ONE(**TEST_DB)


class TestVideoStreamer(IntegrationTest):
    def setUp(self) -> None:
        self.eid, = one.search(subject='ZM_1743', number=1, date_range='2019-06-14')

    def test_video_streamer(self):
        dset = one.alyx.rest('datasets', 'list',
                             session=self.eid, name='_iblrig_leftCamera.raw.mp4')[0]
        url = next(fr['data_url'] for fr in dset['file_records'] if fr['data_url'])
        frame_id = 5
        vs = VideoStreamer(url)
        f, im = vs.get_frame(frame_id)
        assert f
        assert vs.total_frames == 144120
        # Test with data set dict
        vs = VideoStreamer(dset)
        f, im2 = vs.get_frame(frame_id)
        assert np.all(im == im2)
