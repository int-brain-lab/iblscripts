import time
from datetime import datetime, timedelta
import logging

from iblutil.io import params
from ibllib.io import globus

import globus_sdk
from globus_sdk import TransferAPIError

GLOBUS_PARAM_STRING = 'globus/admin'
DEFAULT_PAR = {'local_endpoint': None, 'remote_endpoint': None, 'GLOBUS_CLIENT_ID': None}
logger = logging.getLogger('ibllib')
logger.setLevel(logging.DEBUG)  # For logging transferred files

# Read in parameters
p = params.read(GLOBUS_PARAM_STRING, DEFAULT_PAR)
LOCAL_REPO = p.local_endpoint  # Endpoint UUID from Website
SERVER_ID = p.remote_endpoint  # FlatIron
GLOBUS_CLIENT_ID = p.GLOBUS_CLIENT_ID
DST_DIR = params.read('ibl_ci', {'data_root': '.'}).data_root
# Constants
SRC_DIR = '/integration'
POLL = (5, 60 * 2)  # min max seconds between pinging server
TIMEOUT = 24 * 60 * 60  # seconds before timeout
status_map = {
    'ACTIVE': ('QUEUED', 'ACTIVE', 'GC_NOT_CONNECTED'),
    'FAILED': ('ENDPOINT_ERROR', 'PERMISSION_DENIED', 'CONNECT_FAILED'),
    'INACTIVE': 'PAUSED_BY_ADMIN'
}

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
dst_directory = globus.as_globus_path(DST_DIR)

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
subtasks_failed = 0
poll = POLL[0]
MAX_WAIT = 60 * 60
# while not gtc.task_wait(task_id, timeout=WAIT):
running = True
prev_detail = None
while running:
    """Possible statuses = ('ACTIVE', 'INACTIVE', 'FAILED', 'SUCCEEDED')
    Nice statuses = (None, 'OK', 'Queued', 'PERMISSION_DENIED',
                     'ENDPOINT_ERROR', 'CONNECT_FAILED', 'PAUSED_BY_ADMIN')
    """
    tr = gtc.get_task(task_id)
    detail = (
        'ACTIVE'
        if (tr.data['nice_status']) == 'OK'
        else (tr.data['nice_status'] or tr.data['status']).upper()
    )
    status = next((k for k, v in status_map.items() if detail in v), tr.data['status'])
    running = tr.data['status'] == 'ACTIVE' and detail in status_map['ACTIVE']
    if files_skipped != tr.data['files_skipped']:
        files_skipped = tr.data['files_skipped']
        logger.info(f'Skipping {files_skipped} files....')
        poll = POLL[0]
    if last_status != status or files_transferred != tr.data['files_transferred']:
        files_transferred = tr.data['files_transferred']
        total_files = tr.data['files'] - tr.data['files_skipped']
        if status == 'FAILED' or detail in status_map['FAILED']:
            logger.error(f'Transfer {status}: {tr.data["fatal_error"] or detail}')
            # If still active and error unlikely to resolve by itself, cancel the task
            if tr.data['status'] == 'ACTIVE' and detail != 'CONNECT_FAILED':
                gtc.cancel_task(task_id)
                logger.warning('Transfer CANCELLED')
        elif status == 'INACTIVE' or detail == 'PAUSED_BY_ADMIN':
            logger.info(f'Transfer INACTIVE: {detail}')
        else:
            logger.info(
                f'Transfer {status}: {files_transferred} of {total_files} files transferred')
            # Report failed subtasks
            new_failed = tr['subtasks_expired'] + tr['subtasks_failed']
            if new_failed != subtasks_failed:
                logger.warning(f'{abs(new_failed - subtasks_failed)} sub-tasks expired or failed')
                subtasks_failed = new_failed
        last_status = status
        poll = POLL[0]
    elif detail == 'GC_NOT_CONNECTED' and prev_detail != detail:
        logger.warning('Globus Client not connected, this may be temporary')
        poll = POLL[0]
    else:
        poll = min((poll * 2, POLL[1]))
    prev_detail = detail
    time.sleep(poll) if running else logger.info(f'Final status: {last_status}')

if logger.level == 10:
    """Sometime Globus sets the status to SUCCEEDED but doesn't truly finish.
    The try/except handles an error thrown when querying task_successful_transfers too early"""
    try:
        for info in gtc.task_successful_transfers(task_id):
            src_file = info['source_path'].replace(SRC_DIR + '/', '')
            dst_file = info["destination_path"].replace(dst_directory + '/', '')
            logger.debug(f'{src_file} -> {dst_file}')
    except TransferAPIError:
        logger.debug('Failed to query transferred files')

# Here we should exit
if __name__ == "__main__":
    exit(last_status != 'SUCCEEDED')
