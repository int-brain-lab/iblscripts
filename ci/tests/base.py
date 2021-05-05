import unittest
import os
from pathlib import Path
from functools import wraps
import logging

from ibllib.io import params
import alf.folders
from oneibl.one import ONE


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
    folders = set(alf.folders.session_path(x[0]) for x in os.walk(root))
    eids = not_null(one.eid_from_path(x) for x in not_null(folders))
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
