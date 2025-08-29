"""Instructions and steps to get integration tests up and running
This script will set up the tokens and parameters for easy transfer of the integration data.

Requirements:
    The IBL Globus login credentials
    A Globus endpoint set up for downloading the integration data
    ibllib and iblscripts repositories
"""
from pathlib import Path

from iblutil.io import params
from one.remote.globus import Globus, as_globus_path
from one.alf.spec import is_uuid

# Set up Globus
Globus.setup('server')

print(
    """Setting up fixtures
You will now need to define a directory to which you will download the integration test data.
""")
params_id = 'ibl_ci'
pars = params.read(params_id, {'data_root': './'})
default = pars.data_root
data_root = input(
    f'Enter the desired location of the test data ({default}):'
)
data_root = Path(data_root.strip() or default).absolute()
pars = pars.set('data_root', as_globus_path(data_root))

remote_endpoint = input(
    'Enter the Globus endpoint ID of the remote test data:'
).strip()
assert is_uuid(remote_endpoint, (1,)), 'invalid Globus endpoint ID'
pars = pars.set('remote_endpoint', remote_endpoint)
params.write(params_id, pars)
print('You may now download the data by running `./download_data.py`')
