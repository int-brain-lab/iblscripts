"""Instructions and steps to get integration tests up and running
This script will set up the tokens and parameters for easy transfer of the integration data.

Requirements:
    The IBL Globus login credentials
    A Globus endpoint set up for downloading the integration data
    ibllib and iblscripts repositories
"""
from pathlib import Path

from ibllib.io import params
import oneibl.params

print(
    """Setting up Globus
1. Login to the Globus Website (ask devs for the login credentials)
2. Go to Endpoints and create a new endpoint for the local device (the one that will run this 
script).
3. In the new endpoint's overview page, copy the 'Endpoint UUID' field.  This is the LOCAL_REPO_ID.
4. Go to the 'IBL Top Level' endpoint overview page and copy the 'Endpoint UUID' field.  This is
the REMOTE_REPO_ID.
5. Copy your GLOBUS_CLIENT_ID (ask the software devs for this).
"""
)
params_id = 'globus'
pars = params.read(params_id, {'local_endpoint': None, 'remote_endpoint': None})
default = pars.local_endpoint
local_endpoint = input(
    f'Enter your LOCAL_REPO_ID ({default}):'
)
params.write(params_id, pars.set('local_endpoint', local_endpoint.strip() or default))

default = pars.remote_endpoint
remote_endpoint = input(
    f'Enter your REMOTE_REPO_ID ({default}):'
)
params.write(params_id, pars.set('remote_endpoint', remote_endpoint.strip() or default))

pars = oneibl.params.get()
default = pars.GLOBUS_CLIENT_ID
globus_client_id = input(
    f'Enter your GLOBUS_CLIENT_ID ({default}):'
).strip()
params.write(oneibl.params._PAR_ID_STR, pars.set('GLOBUS_CLIENT_ID', globus_client_id or default))

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
params.write(params_id, pars.set('data_root', str(data_root)))

print('You may now download the data by running `./download_data.py`')
