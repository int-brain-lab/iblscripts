import json
import logging

import ibllib.pipes.ephys_preprocessing as ephys_tasks
from one.api import ONE

from ci.tests import base


class TestEphysSignatures(base.IntegrationTest):
    def setUp(self):
        self.folder_path = self.data_path.joinpath('ephys', 'ephys_signatures')
        self.one = ONE(**base.TEST_DB, mode='local')

    def make_new_dataset(self):
        """helper function to use to create a new dataset"""
        folder_path = self.data_path.joinpath('ephys', 'ephys_signatures', 'RawEphysQC')
        for json_file in folder_path.rglob('result.json'):
            with open(json_file) as fid:
                result = json.load(fid)
            if result['outputs']:
                for bin_file in json_file.parent.rglob('*ap.meta'):
                    bin_file.parent.joinpath("_iblqc_ephysChannels.labels.npy").touch()
                    print(bin_file)

    @base.disable_log(level=logging.ERROR, quiet=True)
    def assert_task_inputs_outputs(self, session_paths, EphysTask):
        for session_path in session_paths.iterdir():
            with self.subTest(session=session_path):
                task = EphysTask(session_path, one=self.one)
                if EphysTask.__name__ == 'SpikeSorting':
                    task.signature['input_files'], task.signature['output_files'] = \
                        task.spike_sorting_signature()
                task.get_signatures()
                output_status, _ = task.assert_expected(task.output_files)
                input_status, _ = task.assert_expected_inputs(raise_error=False)
                with open(session_path.joinpath('result.json'), 'r') as f:
                    result = json.load(f)
                self.assertEqual(output_status, result['outputs'],
                                 f"test failed outputs {EphysTask}, {session_path}")
                self.assertEqual(input_status, result['inputs'],
                                 f"test failed inputs {EphysTask}, {session_path}")

    def test_EphysAudio_signatures(self):
        EphysTask = ephys_tasks.EphysAudio
        session_paths = self.folder_path.joinpath('EphysAudio')
        self.assert_task_inputs_outputs(session_paths, EphysTask)

    def test_EphysCellQC_signatures(self):
        EphysTask = ephys_tasks.EphysCellsQc
        session_paths = self.folder_path.joinpath('EphysCellsQc')
        self.assert_task_inputs_outputs(session_paths, EphysTask)

    def test_EphysMtscomp_signatures(self):
        EphysTask = ephys_tasks.EphysMtscomp
        session_paths = self.folder_path.joinpath('EphysMtscomp', '3A')
        self.assert_task_inputs_outputs(session_paths, EphysTask)

        session_paths = self.folder_path.joinpath('EphysMtscomp', '3B')
        self.assert_task_inputs_outputs(session_paths, EphysTask)

    def test_EphysPassive_signatures(self):
        EphysTask = ephys_tasks.EphysPassive
        session_paths = self.folder_path.joinpath('EphysPassive', '3A')
        self.assert_task_inputs_outputs(session_paths, EphysTask)

        session_paths = self.folder_path.joinpath('EphysPassive', '3B')
        self.assert_task_inputs_outputs(session_paths, EphysTask)

    def test_EphysPulses_signatures(self):
        EphysTask = ephys_tasks.EphysPulses
        session_paths = self.folder_path.joinpath('EphysPulses', '3A')
        self.assert_task_inputs_outputs(session_paths, EphysTask)

        session_paths = self.folder_path.joinpath('EphysPulses', '3B')
        self.assert_task_inputs_outputs(session_paths, EphysTask)

    def test_EphysTrials_signatures(self):
        EphysTask = ephys_tasks.EphysTrials
        session_paths = self.folder_path.joinpath('EphysTrials', '3A')
        self.assert_task_inputs_outputs(session_paths, EphysTask)

        session_paths = self.folder_path.joinpath('EphysTrials', '3B')
        self.assert_task_inputs_outputs(session_paths, EphysTask)

    def test_EphysVideoCompress_signatures(self):
        EphysTask = ephys_tasks.EphysVideoCompress
        session_paths = self.folder_path.joinpath('EphysVideoCompress')
        self.assert_task_inputs_outputs(session_paths, EphysTask)

    def test_EphysVideoSyncQC_signatures(self):
        EphysTask = ephys_tasks.EphysVideoSyncQc
        session_paths = self.folder_path.joinpath('EphysVideoSyncQc', '3A')
        self.assert_task_inputs_outputs(session_paths, EphysTask)

        session_paths = self.folder_path.joinpath('EphysVideoSyncQc', '3B')
        self.assert_task_inputs_outputs(session_paths, EphysTask)

    def test_RawEphysQC_signatures(self):
        EphysTask = ephys_tasks.RawEphysQC
        session_paths = self.folder_path.joinpath('RawEphysQC')
        self.assert_task_inputs_outputs(session_paths, EphysTask)

    def test_SpikeSorting_signatures(self):
        EphysTask = ephys_tasks.SpikeSorting
        session_paths = self.folder_path.joinpath('SpikeSorting', '3A')
        self.assert_task_inputs_outputs(session_paths, EphysTask)

        session_paths = self.folder_path.joinpath('SpikeSorting', '3B')
        self.assert_task_inputs_outputs(session_paths, EphysTask)


if __name__ == '__main__':
    import unittest
    unittest.main(exit=False)
