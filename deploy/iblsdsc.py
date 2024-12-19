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
import logging
from pathlib import Path
from itertools import filterfalse

from one.api import OneAlyx
from one.alf.spec import is_uuid_string
import one.params

_logger = logging.getLogger(__name__)
CACHE_DIR = Path('/mnt/sdceph/users/ibl/data')
CACHE_DIR_FI = Path('/mnt/ibl')


class OneSdsc(OneAlyx):

    def __init__(self, *args, cache_dir=CACHE_DIR, **kwargs):
        if not kwargs.get('tables_dir'):
            # Ensure parquet tables downloaded to separate location to the dataset repo
            try:
                kwargs['tables_dir'] = one.params.get_cache_dir()  # by default this is user downloads
            except AttributeError:
                # ONE not set up
                if kwargs.get('mode') == 'remote':
                    raise RuntimeError('Database params not setup yet. Run one.params.setup first.')
                else:
                    _logger.warning('Database params not setup yet. REST queries will not work.')
                    kwargs['tables_dir'] = one.params.CACHE_DIR_DEFAULT
        super().__init__(*args, cache_dir=cache_dir, **kwargs)
        # assign property here as it is set by the parent OneAlyx class at init
        self.uuid_filenames = True

    def load_object(self, *args, **kwargs):
        # call superclass method
        obj = super().load_object(*args, **kwargs)
        if isinstance(obj, list) or not self.uuid_filenames:
            return obj
        # pops the UUID in the key names
        for k in obj.keys():
            new_key = '.'.join(filterfalse(is_uuid_string, k.split('.')))
            obj[new_key] = obj.pop(k)
        return obj

    def _download_datasets(self, dset, **kwargs):
        """Simply return list of None."""
        urls = self._dset2url(dset, update_cache=False)  # normalizes input to list
        return [None] * len(urls)


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
