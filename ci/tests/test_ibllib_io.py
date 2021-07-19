import unittest
import logging
import time

import numpy as np

import ibllib.io.video as vidio
from one.api import ONE
from ci.tests import base


class TestVideoIO(base.IntegrationTest):

    def setUp(self) -> None:
        root = self.data_path  # Path to integration data
        self.video_path = root.joinpath('ephys', 'choice_world_init', 'KS022', '2019-12-10',
                                        '001', 'raw_video_data', '_iblrig_leftCamera.raw.mp4')
        self.log = logging.getLogger('ibllib')

    def test_get_video_frame(self):
        n = 50  # Frame number to fetch
        frame = vidio.get_video_frame(self.video_path, n)
        expected_shape = (1024, 1280, 3)
        self.assertEqual(frame.shape, expected_shape)
        expected = np.array([[156, 222, 157, 75, 36, 15, 19, 20, 23]], dtype=np.uint8)
        np.testing.assert_array_equal(frame[:1, :9, 0], expected)

    def test_get_video_frames_preload(self):
        n = range(100, 103)  # Frame numbers to fetch

        # Test loading sequential frames without slice
        frames = vidio.get_video_frames_preload(self.video_path, n)
        expected_shape = (len(n), 1024, 1280, 3)
        self.assertEqual(frames.shape, expected_shape)
        self.assertEqual(frames.dtype, np.dtype(np.uint8))

        # Test loading frames with slice
        expected = np.array([[173, 133, 173, 216, 0],
                             [182, 133, 22, 241, 19],
                             [170, 152, 97, 48, 25]], dtype=np.uint8)
        frames = vidio.get_video_frames_preload(self.video_path, n, mask=np.s_[0, :5, 0])
        self.assertTrue(np.all(frames == expected))
        expected_shape = (len(n), 5)
        self.assertEqual(frames.shape, expected_shape)

        # Test loading frames as list
        frames = vidio.get_video_frames_preload(self.video_path, n, as_list=True)
        self.assertIsInstance(frames, list)
        self.assertEqual(frames[0].shape, (1024, 1280, 3))
        self.assertEqual(frames[0].dtype, np.dtype(np.uint8))
        self.assertEqual(len(frames), 3)

        # Test applying function
        frames = vidio.get_video_frames_preload(self.video_path, n,
                                                func=lambda x: np.mean(x, axis=2))
        expected_shape = (len(n), 1024, 1280)
        self.assertEqual(frames.shape, expected_shape)

    def test_get_video_frames_preload_perf(self):
        # Fetch x frames every 100 frames for y hundred frames total
        x = 5
        y = 3
        n = np.tile(np.arange(x * 10), (y, 1))
        n += np.arange(1, y * 100, 100).reshape(y, -1) - 1
        # Test loading sequential frames without slice
        t0 = time.time()
        vidio.get_video_frames_preload(self.video_path, n.flatten())
        elapsed = time.time() - t0
        self.log.info(f'fetching {n.size} frames with {y - 1} incontiguities took {elapsed:.2f}s')
        # self.assertLess(elapsed, 10, 'fetching frames took too long')

    def test_get_video_meta(self):
        # Check with local video path
        meta = vidio.get_video_meta(self.video_path)
        expected = {
            'length': 158377,
            'fps': 60,
            'width': 1280,
            'height': 1024,
            'size': 4257349100
        }
        self.assertTrue(expected.items() <= meta.items())
        self.assertEqual(meta.duration.total_seconds(), 2639.616667)

        # Check with remote path
        one = ONE(**base.TEST_DB)
        dset = one.alyx.rest('datasets', 'list', name='_iblrig_leftCamera.raw.mp4', exist=True)[0]
        video_url = next(fr['data_url'] for fr in dset['file_records'] if fr['data_url'])
        expected = {
            'length': 144120,
            'fps': 30,
            'width': 1280,
            'height': 1024,
            'size': 495090155
        }

        meta = vidio.get_video_meta(video_url, one=one)
        self.assertTrue(expected.items() <= meta.items())


if __name__ == '__main__':
    unittest.main()
