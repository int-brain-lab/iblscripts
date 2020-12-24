import numpy as np

from oneibl.stream import VideoStreamer
from oneibl.one import ONE

one = ONE()
eid = "a9fb578a-9d7d-42b4-8dbc-3b419ce9f424"


def test_simple():
    frame_id = 4000
    dset = one.alyx.rest('datasets', 'list', session=eid, name='_iblrig_leftCamera.raw.mp4')[0]
    url = next(fr['data_url'] for fr in dset['file_records'] if fr['data_url'])
    vs = VideoStreamer(url)
    f, im = vs.get_frame(frame_id)
    assert vs.total_frames == 77897
    assert f
    vs = VideoStreamer(dset)
    f, im2 = vs.get_frame(frame_id)
    assert np.all(im == im2)
