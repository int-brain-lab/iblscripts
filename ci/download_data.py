import time
from datetime import datetime, timedelta
import logging

from ibllib.io import params, globus
from ibllib.io.globus import as_globus_path
import oneibl.params

import globus_sdk
from globus_sdk.exc import TransferAPIError

logger = logging.getLogger('ibllib')

# Read in parameters
p = params.read('globus', {'local_endpoint': None, 'remote_endpoint': None})
LOCAL_REPO = p.local_endpoint  # Endpoint UUID from Website
SERVER_ID = p.remote_endpoint  # FlatIron
DST_DIR = params.read('ibl_ci', {'data_root': '.'}).data_root
GLOBUS_CLIENT_ID = oneibl.params.get().GLOBUS_CLIENT_ID
# Constants
SRC_DIR = '/integration'
POLL = (5, 60*60)  # min max seconds between pinging server
TIMEOUT = 24*60*60  # seconds before timeout

try:
    gtc = globus.login_auto(GLOBUS_CLIENT_ID)
except ValueError:
    logger.info('User authentication required...')
    globus.setup(GLOBUS_CLIENT_ID)
    gtc = globus.login_auto(GLOBUS_CLIENT_ID)

# Check path exists
try:
    gtc.operation_ls(SERVER_ID, path=SRC_DIR)
except TransferAPIError as ex:
    logger.error(f'Failed to query source endpoint path {SRC_DIR}')
    raise ex

# Create the destination path if it does not exist
dst_directory = as_globus_path(DST_DIR)

try:
    gtc.operation_ls(LOCAL_REPO, path=dst_directory)
except TransferAPIError as ex:
    if ex.http_status == 404:
        # Directory not found; create it
        try:
            gtc.operation_mkdir(LOCAL_REPO, dst_directory)
            logger.info(f'Created directory: {dst_directory}')
        except TransferAPIError as tapie:
            logger.error(f'Failed to create directory: {tapie.message}')
            raise tapie
    else:
        raise ex

# Create transfer object
transfer_object = globus_sdk.TransferData(
    gtc,
    source_endpoint=SERVER_ID,
    destination_endpoint=LOCAL_REPO,
    verify_checksum=False,
    delete_destination_extra=True,
    sync_level='mtime',
    label='integration data',
    deadline=datetime.now() + timedelta(0, TIMEOUT)
)

# add any number of items to the submission data
transfer_object.add_item(SRC_DIR, dst_directory, recursive=True)
response = gtc.submit_transfer(transfer_object)
assert round(response.http_status / 100) == 2  # Check for 20x status

# What for transfer to complete
task_id = response.data['task_id']
last_status = None
files_transferred = None
files_skipped = 0
poll = POLL[0]
MAX_WAIT = 60*60
# while not gtc.task_wait(task_id, timeout=WAIT):
running = True
while running:
    """Possible statuses = ('ACTIVE', 'INACTIVE', 'FAILED', 'SUCCEEDED')
    Nice statuses = (None, 'OK', 'Queued', 'PERMISSION_DENIED',
                     'ENDPOINT_ERROR', 'CONNECT_FAILED', 'PAUSED_BY_ADMIN')
    """
    tr = gtc.get_task(task_id)
    status = (
        'QUEUED'
        if tr.data['nice_status'] == 'Queued'
        else tr.data['status']
    )
    running = tr.data['status'] == 'ACTIVE'
    if files_skipped != tr.data['files_skipped']:
        files_skipped = tr.data['files_skipped']
        logger.info(f'Skipping {files_skipped} files....')
    if last_status != status or files_transferred != tr.data['files_transferred']:
        files_transferred = tr.data['files_transferred']
        total_files = tr.data['files'] - tr.data['files_skipped']
        if status == 'FAILED':
            logger.error(f'Transfer {status}: {tr.data["fatal_error"]}')
        else:
            logger.info(
                f'Transfer {status}: {files_transferred} of {total_files} files transferred')
        last_status = status
        poll = POLL[0]
    else:
        poll = min((poll * 2, POLL[1]))
    time.sleep(poll)

# Here we should exit
if __name__ == "__main__":
    exit(last_status != 'SUCCEEDED')
