from pathlib import Path
import logging
import shutil
import tempfile
import ibllib.pipes.behavior_tasks as btasks

from ci.tests import base

_logger = logging.getLogger('ibllib')


def make_sym_links(raw_session_path, extraction_path):
    """
    This creates symlinks to a scratch directory to start an extraction while leaving the
    raw data untouched
    :param raw_session_path: location containing the extraction fixture
    :param extraction_path: scratch location where the symlinks will end up,
    omitting the session parts like "/tmp"
    :return:
    """
    session_path = Path(extraction_path).joinpath(*raw_session_path.parts[-5:])

    for f in raw_session_path.rglob('*.*'):
        new_file = session_path.joinpath(f.relative_to(raw_session_path))
        if new_file.exists():
            continue
        new_file.parent.mkdir(exist_ok=True, parents=True)
        new_file.symlink_to(f)
    return session_path


class TestHabituationRegisterRaw(base.IntegrationTest):

    def setUp(self) -> None:
        self.raw_session_path = Path("/datadisk/scratch/raw_sessions/steinmetzlab/Subjects/NR_0020/2022-01-27/001")  # FIXME
        self.extraction_path = Path(tempfile.TemporaryDirectory().name)
        self.session_path = make_sym_links(self.raw_session_path, self.extraction_path)

    def tearDown(self):
        shutil.rmtree(self.extraction_path)

    def test_task(self):
        wf = btasks.HabituationRegisterRaw(self.session_path, collection='raw_behavior_data')
        status = wf.run()
        assert status == 0
        wf.assert_expected_outputs()


class TestTrialsRegisterRaw(base.IntegrationTest):

    def setUp(self) -> None:
        self.session_path = self.default_data_root().joinpath('ephys', 'passive_extraction', 'SWC_054', '2020-10-10', '001')

    def test_register(self):
        wf = btasks.TrialRegisterRaw(self.session_path, task_collection='raw_behavior_data')
        status = wf.run()

        assert status == 0

        for exp_files in wf.signature['output_files']:
            file = self.session_path.joinpath(exp_files[1], exp_files[0])
            assert file.exists()
            assert file in wf.outputs


class TestPassiveRegisterRaw(base.IntegrationTest):

    def setUp(self) -> None:
        self.session_path = self.default_data_root().joinpath('ephys', 'passive_extraction', 'SWC_054', '2020-10-10', '001')

    def test_register(self):
        wf = btasks.PassiveRegisterRaw(self.session_path, task_collection='raw_passive_data')
        status = wf.run()

        assert status == 0

        for exp_files in wf.signature['output_files']:
            file = self.session_path.joinpath(exp_files[1], exp_files[0])
            assert file.exists()
            assert file in wf.outputs


class TestPassiveTrials(base.IntegrationTest):
    pass
    # TODO test with normal passive and standalone passive (need to get the dataset on integration server)
