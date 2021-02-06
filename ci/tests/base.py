import unittest
from pathlib import Path
from ibllib.io import params


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
        data_present = (self.data_path.exists()
                        and self.data_path.is_dir()
                        and any(self.data_path.glob('Subjects_init')))
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
