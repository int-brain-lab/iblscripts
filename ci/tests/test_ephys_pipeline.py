import json
import logging
import tempfile
import shutil
from pathlib import Path

import ibllib.pipes.ephys_preprocessing as ephys_tasks
from one.api import ONE

from ci.tests import base


class TestEphysSignatures(base.IntegrationTest):
    def setUp(self):
        self.folder_path = self.data_path.joinpath('ephys', 'ephys_signatures')
        self.one = ONE(**base.TEST_DB, mode='local')
        tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(tmp.name)
        self.addCleanup(tmp.cleanup)

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

    def mktree(self, test_path):
        """Replicate the test path tree in a temp dir session path."""
        session_path = self.tmp.joinpath('subject', '2020-01-01', test_path.name)
        files = map(lambda x: x.relative_to(test_path), test_path.rglob('*.*'))
        for file in map(session_path.joinpath, files):
            file.parent.mkdir(parents=True, exist_ok=True)
            if file.name == 'result.json':
                shutil.copy(test_path.joinpath(file.name), file.parent)
            else:
                file.touch()
        return session_path

    @base.disable_log(level=logging.ERROR, quiet=True)
    def assert_task_inputs_outputs(self, session_paths, EphysTask):
        for session_path in session_paths:
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
        session_paths = map(self.mktree, self.folder_path.joinpath('EphysAudio').iterdir())
        self.assert_task_inputs_outputs(session_paths, EphysTask)

    def test_EphysCellQC_signatures(self):
        EphysTask = ephys_tasks.EphysCellsQc
        session_paths = map(self.mktree, self.folder_path.joinpath('EphysCellsQc').iterdir())
        self.assert_task_inputs_outputs(session_paths, EphysTask)

    def test_EphysMtscomp_signatures(self):
        EphysTask = ephys_tasks.EphysMtscomp
        session_paths = map(self.mktree, self.folder_path.joinpath('EphysMtscomp', '3A').iterdir())
        self.assert_task_inputs_outputs(session_paths, EphysTask)

        session_paths = map(self.mktree, self.folder_path.joinpath('EphysMtscomp', '3B').iterdir())
        self.assert_task_inputs_outputs(session_paths, EphysTask)

    def test_EphysPassive_signatures(self):
        EphysTask = ephys_tasks.EphysPassive
        session_paths = map(self.mktree, self.folder_path.joinpath('EphysPassive', '3A').iterdir())
        self.assert_task_inputs_outputs(session_paths, EphysTask)

        session_paths = map(self.mktree, self.folder_path.joinpath('EphysPassive', '3B').iterdir())
        self.assert_task_inputs_outputs(session_paths, EphysTask)

    def test_EphysPulses_signatures(self):
        EphysTask = ephys_tasks.EphysPulses
        session_paths = map(self.mktree, self.folder_path.joinpath('EphysPulses', '3A').iterdir())
        self.assert_task_inputs_outputs(session_paths, EphysTask)

        session_paths = map(self.mktree, self.folder_path.joinpath('EphysPulses', '3B').iterdir())
        self.assert_task_inputs_outputs(session_paths, EphysTask)

    def test_EphysTrials_signatures(self):
        EphysTask = ephys_tasks.EphysTrials
        session_paths = map(self.mktree, self.folder_path.joinpath('EphysTrials', '3A').iterdir())
        self.assert_task_inputs_outputs(session_paths, EphysTask)

        session_paths = map(self.mktree, self.folder_path.joinpath('EphysTrials', '3B').iterdir())
        self.assert_task_inputs_outputs(session_paths, EphysTask)

    def test_EphysVideoCompress_signatures(self):
        EphysTask = ephys_tasks.EphysVideoCompress
        session_paths = map(self.mktree, self.folder_path.joinpath('EphysVideoCompress').iterdir())
        self.assert_task_inputs_outputs(session_paths, EphysTask)

    def test_EphysVideoSyncQC_signatures(self):
        EphysTask = ephys_tasks.EphysVideoSyncQc
        session_paths = map(self.mktree, self.folder_path.joinpath('EphysVideoSyncQc', '3A').iterdir())
        self.assert_task_inputs_outputs(session_paths, EphysTask)

        session_paths = map(self.mktree, self.folder_path.joinpath('EphysVideoSyncQc', '3B').iterdir())
        self.assert_task_inputs_outputs(session_paths, EphysTask)

    def test_RawEphysQC_signatures(self):
        EphysTask = ephys_tasks.RawEphysQC
        session_paths = map(self.mktree, self.folder_path.joinpath('RawEphysQC').iterdir())
        self.assert_task_inputs_outputs(session_paths, EphysTask)

    def test_SpikeSorting_signatures(self):
        EphysTask = ephys_tasks.SpikeSorting
        session_paths = map(self.mktree, self.folder_path.joinpath('SpikeSorting', '3A').iterdir())
        self.assert_task_inputs_outputs(session_paths, EphysTask)

        session_paths = map(self.mktree, self.folder_path.joinpath('SpikeSorting', '3B').iterdir())
        self.assert_task_inputs_outputs(session_paths, EphysTask)


if __name__ == '__main__':
    import unittest
    unittest.main(exit=False)
