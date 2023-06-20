from process import consolidate_logs
import logging

logger = logging.getLogger('ibllib')
logger.setLevel(logging.INFO)

if __name__ == '__main__':

    import sys
    command = sys.argv[1]
    ipinfo_token = sys.argv[2]
    logger.warning(command)
    logger.warning(ipinfo_token)
    consolidate_logs(date=command, ipinfo_token=ipinfo_token, profile_name='ibl')
