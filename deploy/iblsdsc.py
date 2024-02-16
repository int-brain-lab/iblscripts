"""
This module provides a ONE class that uses the SDSC filesystem as a cache.
The purpose is to provide an API that is compatible with the standard ONE API for running on the Popeye cluster.

The specificity of this implementation arises from several factors:
-   the cache is read-only
-   the cache is a constant
-   each file is stored with an UUID between the file stem and the extension

The limitations of this implementation are:
-   alfio.load methods will load objects with long keys containing UUIDS

Recommended usage: just monkey patch the ONE import and run your code as usual on Popeye !
>>> from deploy.iblsdsc import OneSdsc as ONE
"""

from one.api import OneAlyx
import warnings
import logging
from datetime import datetime
from pathlib import Path, PurePosixPath

import pandas as pd
import numpy as np
from iblutil.io import hashfile

from one.converters import session_record2path
import one.util as util

_logger = logging.getLogger(__name__)
CACHE_DIR = Path("/mnt/sdceph/users/ibl/data")


class OneSdsc(OneAlyx):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # assign property here as it is set by the parent OneAlyx class at init
        self.uuid_filenames = True

    def load_object(self, *args, **kwargs):
        # call superclass method
        obj = super().load_object(*args, **kwargs)
        if isinstance(obj, list):
            return obj
        # pops the UUID in the key names
        keys = list(obj.keys())
        for k in keys:
            obj[k[:-37]] = obj.pop(k)
        return obj

    def _download_dataset(self, dset, cache_dir=None, update_cache=True, **kwargs):
        return

    def _check_filesystem(self, datasets, offline=None, update_exists=True, check_hash=True):
        """Update the local filesystem for the given datasets.

        Given a set of datasets, check whether records correctly reflect the filesystem.
        Called by load methods, this returns a list of file paths to load and return.

        Parameters
        ----------
        datasets : pandas.Series, pandas.DataFrame, list of dicts
            A list or DataFrame of dataset records
        offline : bool, None
            If false and Web client present, downloads the missing datasets from a remote
            repository
        update_exists : bool
            If true, the cache is updated to reflect the filesystem

        Returns
        -------
        A list of file paths for the datasets (None elements for non-existent datasets)
        """
        if isinstance(datasets, pd.Series):
            datasets = pd.DataFrame([datasets])
        elif not isinstance(datasets, pd.DataFrame):
            # Cast set of dicts (i.e. from REST datasets endpoint)
            datasets = util.datasets2records(list(datasets))
        indices_to_download = []  # indices of datasets that need (re)downloading
        files = []  # file path list to return
        # If the session_path field is missing from the datasets table, fetch from sessions table
        if 'session_path' not in datasets.columns:
            if 'eid' not in datasets.index.names:
                # Get slice of full frame with eid in index
                _dsets = self._cache['datasets'][
                    self._cache['datasets'].index.get_level_values(1).isin(datasets.index)
                ]
                idx = _dsets.index.get_level_values(1)
            else:
                _dsets = datasets
                idx = pd.IndexSlice[:, _dsets.index.get_level_values(1)]
            # Ugly but works over unique sessions, which should be quicker
            session_path = (self._cache['sessions']
                            .loc[_dsets.index.get_level_values(0).unique()]
                            .apply(session_record2path, axis=1))
            datasets.loc[idx, 'session_path'] = \
                pd.Series(_dsets.index.get_level_values(0)).map(session_path).values

        # First go through datasets and check if file exists and hash matches
        for i, rec in datasets.iterrows():
            file = Path(CACHE_DIR, *rec[['session_path', 'rel_path']])
            # CACHE_DIR
            file = next(file.parent.glob(f"{file.stem}.*{file.suffix}"))
            if file.exists():
                # Check if there's a hash mismatch
                # If so, add this index to list of datasets that need downloading
                if rec['file_size'] and file.stat().st_size != rec['file_size']:
                    _logger.warning('local file size mismatch on dataset: %s',
                                    PurePosixPath(rec.session_path, rec.rel_path))
                    indices_to_download.append(i)
                elif check_hash and rec['hash'] is not None:
                    if hashfile.md5(file) != rec['hash']:
                        _logger.warning('local md5 mismatch on dataset: %s',
                                        PurePosixPath(rec.session_path, rec.rel_path))
                        indices_to_download.append(i)
                files.append(file)  # File exists so add to file list
            else:
                raise FileExistsError('Dataset file not found: {}'.format(file))
            if rec['exists'] != file.exists():
                with warnings.catch_warnings():
                    # Suppress future warning: exist column should always be present
                    msg = '.*indexing on a MultiIndex with a nested sequence of labels.*'
                    warnings.filterwarnings('ignore', message=msg)
                    datasets.at[i, 'exists'] = not rec['exists']
                    if update_exists:
                        _logger.debug('Updating exists field')
                        if isinstance(i, tuple):
                            self._cache['datasets'].loc[i, 'exists'] = not rec['exists']
                        else:  # eid index level missing in datasets input
                            i = pd.IndexSlice[:, i]
                            self._cache['datasets'].loc[i, 'exists'] = not rec['exists']
                        self._cache['_meta']['modified_time'] = datetime.now()

        if self.record_loaded:
            loaded = np.fromiter(map(bool, files), bool)
            loaded_ids = np.array(datasets.index.to_list())[loaded]
            if '_loaded_datasets' not in self._cache:
                self._cache['_loaded_datasets'] = np.unique(loaded_ids)
            else:
                loaded_set = np.hstack([self._cache['_loaded_datasets'], loaded_ids])
                self._cache['_loaded_datasets'] = np.unique(loaded_set, axis=0)

        # Return full list of file paths
        return files


def _test_one_sdsc():
    """
    I have put the tests here
    :return:
    """
    from brainbox.io.one import SpikeSortingLoader, SessionLoader
    one = OneSdsc()
    pid = "069c2674-80b0-44b4-a3d9-28337512967f"
    eid, _ = one.pid2eid(pid)
    dsets = one.list_datasets(eid=eid)
    assert len(dsets) > 0
    # checks that this is indeed the short key version when using load object
    trials = one.load_object(eid, obj='trials')
    assert 'intervals' in trials
    # checks that this is indeed the short key version when using the session loader and spike sorting loader
    sl = SessionLoader(eid=eid, one=one)  # noqa
    sl.load_wheel()
    assert 'position' in sl.wheel.columns
    ssl = SpikeSortingLoader(pid=pid, one=one)
    spikes, clusters, channels = ssl.load_spike_sorting()  # noqa
    assert 'amps' in spikes
