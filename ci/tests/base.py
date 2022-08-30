import unittest
from unittest.runner import TextTestResult, TextTestRunner
import time
import os
from pathlib import Path
from functools import wraps
import logging
import json
import tempfile

from iblutil.io import params
from one.alf.files import get_session_path
from one.api import ONE


class TimeLoggingTestResult(TextTestResult):
    """A class to record test durations"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_timings = []
        self._test_started_at = time.time()

    def startTest(self, test):
        self._test_started_at = time.time()
        super().startTest(test)

    def addSuccess(self, test):
        elapsed = time.time() - self._test_started_at
        name = str(test)  # self.getDescription(test) # includes first line of docstring
        self.test_timings.append((name, elapsed))
        super().addSuccess(test)

    def getTestDurations(self) -> list[tuple[str, int]]:
        """Returns list of tests and their durations, in reverse duration order"""
        return sorted(self.test_timings, key=lambda x: x[1], reverse=True)


class TimeLoggingTestRunner(TextTestRunner):
    """A class that prints a list of the slowest tests to the output stream"""
    def __init__(self, slow_test_threshold=0.3, *args, **kwargs):
        self.slow_test_threshold = slow_test_threshold
        super().__init__(resultclass=TimeLoggingTestResult, *args, **kwargs)

    def run(self, test):
        result = super().run(test)
        self.stream.writeln(f'\nSlow Tests (>{self.slow_test_threshold:.03}s):\n')
        for name, elapsed in result.getTestDurations():
            if elapsed > self.slow_test_threshold:
                self.stream.writeln(f'({elapsed:.03}s) {name}')
        return result


class IntegrationTest(unittest.TestCase):
    """A class for running integration tests"""
    def __init__(self, *args, data_path=None, **kwargs):
        """A base class for locating integration test data
        Upon initialization, loads the path to the integration test data.  The path is loaded from
        the '.ibl_ci' parameter file's 'data_root' parameter, or the current working directory.
        The data root may be overridden with the `data_path` keyword arg.  The data path must be an
        existing directory containing a 'Subjects_init' folder.
        :param data_path: The data root path to the integration data directory
        """
        super().__init__(*args, **kwargs)

        # Store the path to the integration data
        self.data_path = Path(data_path or self.default_data_root())
        data_present = (self.data_path.exists() and
                        self.data_path.is_dir() and
                        any(self.data_path.glob('Subjects_init')))
        if not data_present:
            raise FileNotFoundError(f'Invalid data root folder {self.data_path.absolute()}\n\t'
                                    'must contain a "Subjects_init" folder')

    @staticmethod
    def default_data_root():
        """Returns the path to the integration data.

        The path is loaded from the '.ibl_ci' parameter file's 'data_root' parameter,
        or the current working directory.
        """
        return Path(params.read('ibl_ci', {'data_root': '.'}).data_root)


def list_current_sessions(one=None):
    """
    Get the set of session eids used in integration tests.  When writing new tests, this can be
    a useful way of choosing which sessions to use.

    :param one: An ONE object for fetching session eid from path
    :return: Set of integration session eids
    """
    def not_null(itr):
        return filter(lambda x: x is not None, itr)
    one = one or ONE()
    root = IntegrationTest.default_data_root()
    folders = set(get_session_path(x[0]) for x in os.walk(root))
    eids = not_null(one.path2eid(x) for x in not_null(folders))
    return set(eids)


def disable_log(level=logging.CRITICAL, restore_level=None, quiet=False):
    """
    Decorator to temporarily disable the log.
    :param level: The minimum logging level to disable
    :param restore_level: The logging level to restore
    :param quiet: If false the fact that the log is disabled will be printed
    :return:
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            logging.disable(level)
            if not quiet:
                print('**Log disabled for test**')
            output = func(self, *args, **kwargs)
            if not quiet:
                print('**Log re-enabled**')
            logging.disable(restore_level or logging.NOTSET)
            return output
        return wrapper
    return decorator


def _get_test_db():
    db_json = os.getenv('TEST_DB_CONFIG', None)
    if db_json:
        with open(db_json, 'r') as f:
            return json.load(f)
    else:
        return {
            'base_url': 'https://test.alyx.internationalbrainlab.org',
            'username': 'test_user',
            'password': 'TapetesBloc18',
            'silent': True
        }


def make_sym_links(raw_session_path, extraction_path=None):
    """
    This creates symlinks to a scratch directory to start an extraction while leaving the
    raw data untouched.
    :param raw_session_path: location containing the extraction fixture, complying with alf convention
    :param extraction_path: (None) scratch location where the symlinks will end up,
    omitting the session parts example: "/tmp". If set to None, it will create a temporary
    directory using tempdir.
    :return:
    """
    if extraction_path is None:
        extraction_path = Path(tempfile.TemporaryDirectory().name)

    session_path = Path(extraction_path).joinpath(*raw_session_path.parts[-5:])

    for f in raw_session_path.rglob('*.*'):
        new_file = session_path.joinpath(f.relative_to(raw_session_path))
        if new_file.exists():
            continue
        new_file.parent.mkdir(exist_ok=True, parents=True)
        new_file.symlink_to(f)
    return session_path, extraction_path


TEST_DB = _get_test_db()
