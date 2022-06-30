import logging
import traceback
from pathlib import Path
from one.api import ONE

from ibllib.pipes.local_server import report_health
from ibllib.pipes.local_server import job_creator

_logger = logging.getLogger('ibllib')

subjects_path = Path('/mnt/s0/Data/Subjects/')
one = ONE(cache_rest=None)

# Label the lab endpoint json field with health indicators
try:
    one = ONE(cache_rest=None)
    report_health(one=one)
except BaseException:
    _logger.error(f"Error in report_health\n {traceback.format_exc()}")

#  Create sessions: for this server, finds the extract_me flags, identify the session type,
#  create the session on Alyx if it doesn't already exist, register the raw data and create
#  the tasks backlog
try:
    job_creator(subjects_path, one=one, dry=False)
except BaseException:
    _logger.error(f"Error in job_creator\n {traceback.format_exc()}")
