import time
from datetime import datetime, timedelta
import logging

from one.remote.globus import Globus
from iblutil.io import params

from globus_sdk import TransferAPIError

globus = Globus('server')
flatiron_id = 'ab2d064c-413d-11eb-b188-0ee0d5d9299f'
logger = logging.getLogger('ibllib')
logger.setLevel(logging.DEBUG)

# Read in parameters
globus.add_endpoint(flatiron_id, 'flatiron-integration', root_path='/integration')
globus.endpoints['local']['root_path'] = params.read('ibl_ci', {'data_root': '.'}).data_root
# Constants
POLL = (5, 60 * 2)  # min max seconds between pinging server
TIMEOUT = 24 * 60 * 60  # seconds before timeout
status_map = {
    'ACTIVE': ('QUEUED', 'ACTIVE', 'GC_NOT_CONNECTED', 'UNKNOWN'),
    'FAILED': ('ENDPOINT_ERROR', 'PERMISSION_DENIED', 'CONNECT_FAILED'),
    'INACTIVE': 'PAUSED_BY_ADMIN'
}

# Check path exists
try:
    globus.ls('flatiron-integration', '')
except TransferAPIError as ex:
    logger.error('Failed to query source endpoint path %s', globus.endpoints['flatiron-integration']['root_path'])
    raise ex

# Create the destination path if it does not exist
try:
    globus.ls('local', '')
except TransferAPIError as ex:
    if ex.http_status == 404:
        # Directory not found; create it
        try:
            globus.client.operation_mkdir(globus.endpoints['local'], globus.endpoints['local']['root_path'])
            logger.info('Created directory: %s', globus.endpoints['local']['root_path'])
        except TransferAPIError as tapie:
            logger.error(f'Failed to create directory: {tapie.message}')
            raise tapie
    else:
        raise ex

task_id = globus.transfer_data(
    '', 'flatiron-integration', 'local',
    recursive=True, verify_checksum=False, delete_destination_extra=True, sync_level='mtime',
    label='integration data', deadline=datetime.now() + timedelta(0, TIMEOUT))

# What for transfer to complete
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
    Nice statuses = (None, 'OK', 'Queued', 'PERMISSION_DENIED', 'UNKNOWN',
                     'ENDPOINT_ERROR', 'CONNECT_FAILED', 'PAUSED_BY_ADMIN')
    """
    tr = globus.client.get_task(task_id)
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
                globus.client.cancel_task(task_id)
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
    elif detail == 'UNKNOWN' and prev_detail != detail:
        logger.warning('Unknown error from client, this may be temporary')
        poll = poll[0]
    else:
        poll = min((poll * 2, POLL[1]))
    prev_detail = detail
    time.sleep(poll) if running else logger.info(f'Final status: {last_status}')

if logger.level == logging.DEBUG:
    """Sometimes Globus sets the status to SUCCEEDED but doesn't truly finish.
    The try/except handles an error thrown when querying task_successful_transfers too early"""
    try:
        for info in globus.client.task_successful_transfers(task_id):
            src_file = info['source_path'].replace(globus.endpoints['flatiron-integration']['root_path'] + '/', '')
            dst_file = info['destination_path'].replace(globus.endpoints['local']['root_path'] + '/', '')
            logger.debug(f'{src_file} -> {dst_file}')
    except TransferAPIError:
        logger.debug('Failed to query transferred files')
        logger.debug('Status: %s; nice status: %s', tr.data['status'], tr.data['nice_status'])

# Here we should exit
if __name__ == '__main__':
    exit(last_status != 'SUCCEEDED')
