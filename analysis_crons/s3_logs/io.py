"""Functions for accessing bucket logs, loading into a data frame and saving locally."""
import warnings
from pathlib import PurePosixPath, Path
import re
from io import BytesIO
from typing import Optional
from itertools import filterfalse
from tempfile import TemporaryDirectory

from tqdm import tqdm
import numpy as np
import pandas as pd
from one.remote import aws
from one.webclient import AlyxClient
from one.util import validate_date_range

COL_NAMES = [
    'Bucket_Owner', 'Bucket', 'Time', 'Time_Offset', 'Remote_IP', 'Requester_ARN/Canonical_ID',
    'Request_ID', 'Operation', 'Key', 'Request_URI', 'HTTP_status', 'Error_Code', 'Bytes_Sent',
    'Object_Size', 'Total_Time', 'Turn_Around_Time', 'Referrer', 'User_Agent', 'Version_Id',
    'Host_Id', 'Signature_Version', 'Cipher_Suite', 'Authentication_Type', 'Host_Header',
    'TLS_version', 'Access_Point_ARN', 'ACL_Required'
]
"""str: default remote log location, can be found in server access logging config page."""
REMOTE_LOG_LOCATION = 'info/ibl-brain-wide-map-public/logs/server-access-logs/'
LOCAL_LOG_LOCATION = Path.home().joinpath('s3_logs')


def get_log_directory(key=REMOTE_LOG_LOCATION):
    """
    Returns collection of S3 objects found within the server access logs location.

    Parameters
    ----------
    key : str
        The location of the server access log files on the private bucket.

    Returns
    -------
    s3.Bucket.objectsCollection
        The S3 objects within the log location.
    """
    s3, bucket_name = aws.get_s3_from_alyx(AlyxClient())
    return s3.Bucket(name=bucket_name).objects.filter(Prefix=key)


def get_log_table_by_month(date, location=None):
    """
    Return the S3 consolidated access log table for a given month.

    Parameters
    ----------
    date : datetime.datetime
        Retrieve log table object contains logs of this datetime.
    location : str
        The location of the server access log files on the private bucket.

    Returns
    -------
    s3.ObjectSummary
        The S3 object for the log table file.

    Notes
    -----
    - Logs are consolidated based on the log file timestamp, not the timestamp of the requests
     contained.  Logs for requests towards the end of the month (usually the last day) may only
     appear on next month's logs table.
    - Prior to March 2023 the log tables may contain duplicate rows as sometimes a request is
     logged more than once.
    - IP address information is also available in a separate file with the suffix '_IP-info'.
    - Logs are not sorted.
    """
    s3, bucket_name = aws.get_s3_from_alyx(AlyxClient())
    filename = date.strftime('%Y-%m') + '_ibl-brain-wide-map-public.pqt'
    if date.date() < pd.Timestamp(2023, 3, 1).date():
        warnings.warn('Note: duplicate rows were not removed prior to 2023-03-01')
    if not location:
        location = PurePosixPath(REMOTE_LOG_LOCATION, 'consolidated')
    location = PurePosixPath(location, filename)
    today = pd.Timestamp.today()
    if (date.year, date.month) == (today.year, today.month):
        location = location.with_name(location.stem + '_INCOMPLETE' + location.suffix)
    file_obj = s3.Object(bucket_name=bucket_name, key=location)
    return file_obj


def iter_log_tables(location=None):
    """
    Yield the consolidated log tables from a given bucket location.

    Parameters
    ----------
    location : str
        The location of the server access log files on the private bucket.

    Yields
    -------
    s3.ObjectSummary
        A consolidated log table S3 object.
    """
    if not location:
        location = PurePosixPath(REMOTE_LOG_LOCATION, 'consolidated').as_posix()
    s3, bucket_name = aws.get_s3_from_alyx(AlyxClient())
    file_objects = s3.Bucket(name=bucket_name).objects.filter(Prefix=location)
    for obj in filterfalse(aws.is_folder, file_objects):
        yield obj


def _timestamp(key: str) -> Optional[pd.Timestamp]:
    """
    Parse out datetime from a log filename with pattern
    TargetPrefixYYYY-mm-DD-HH-MM-SS-UniqueString.

    Parameters
    ----------
    key : str
        The filepath of an S3 server access log file.

    Returns
    -------
    pd.Timestamp
        The datetime of the log file if valid.  Returns None is key name is irregular.
    """
    if not (match := re.match(r'^\w*(\d{4}(?:-\d{2}){5})', PurePosixPath(key).name)):
        return
    timestamp, = match.groups()
    return pd.Timestamp(*map(int, timestamp.split('-')))


def _within_range(timestamp, date_range) -> bool:
    """
    Check if a given datetime is within the provided range.

    Parameters
    ----------
    timestamp : pd.Timestamp
        The datetime to evaluate
    date_range : (pd.Timestamp, pd.Timestamp), Optional
        The start and end timestamps.

    Returns
    -------
    bool
        True if date_range is not None and timestamp is within said range.
    """
    if not date_range:
        return True
    start, end = date_range
    return start < timestamp < end


def _iter_objects(log_location, date_range=None, s3_bucket=None):
    """
    Iterate over S3 objects in a collection, yield log file keys that fall within a given date
    range.

    Parameters
    ----------
    log_location : str
        The location of the server access log files on the private bucket.
    date_range : str, list, datetime.datetime, datetime.date, pd.timestamp, Optional
        An optional date range to filter logs by.

    Yields
    -------
    s3.ObjectSummary
        An S3 object within the log_dir, whose key datetime is within the provided date range.
    """
    date_range = validate_date_range(date_range)
    if date_range:
        # Filter on common key server-side (significantly improves performance)
        # e.g. for ['2022-01-01', '2022-08-15'], filter keys starting '2022-0'
        dt_str = list(map(lambda x: x.strftime('%Y-%m-%d-%H-%M-%S'), date_range))
        key_match = list(map(lambda x: x[0] != x[1], zip(*dt_str)))
        if any(key_match):
            log_location += dt_str[0][:key_match.index(True)]

    if s3_bucket:
        file_objects = s3_bucket.objects.filter(Prefix=log_location)
    else:
        file_objects = get_log_directory(log_location)

    for obj in filterfalse(aws.is_folder, file_objects):
        if (ts := _timestamp(obj.key)) and _within_range(ts, date_range):
            yield obj


def _iter_logs(log_location, date_range=None):
    """
    Iterate over S3 objects in a collection, yield logs that fall within a given date range.

    Parameters
    ----------
    log_location : str
        The location of the server access log files on the private bucket.
    date_range : str, list, datetime.datetime, datetime.date, pd.timestamp, Optional
        An optional date range to filter logs by.

    Yields
    -------
    io.BytesIO
        A byte string buffer of each log file.
    """
    for obj in _iter_objects(log_location, date_range):
        yield BytesIO(obj.get()['Body'].read())


def read_remote_logs(date_range=None, log_location=REMOTE_LOG_LOCATION) -> pd.DataFrame:
    """

    Parameters
    ----------
    date_range : str, list, datetime.datetime, datetime.date, pd.timestamp, Optional
        An optional date range to filter logs by.
    log_location : str
        The location of the server access log files on the private bucket.

    Returns
    -------
    pd.DataFrame
        A pandas data frame of remote server access logs.
    """
    log_files = tqdm(_iter_logs(log_location, date_range), unit=' files')
    logs = (pd.read_csv(log, sep=' ', header=None) for log in log_files)
    df = pd.concat(logs, ignore_index=True)
    df.columns = COL_NAMES
    return df


def read_remote_logs_robust(date_range=None, log_location=REMOTE_LOG_LOCATION) -> pd.DataFrame:
    """
    Sometimes logs contain an uneven number of columns (they keep adding columns). This function
    deals with those logs but it's slow and ugly. Additionally the raw log strings are pickled
    in the home directory before attempting to parse them.

    Parameters
    ----------
    date_range : str, list, datetime.datetime, datetime.date, pd.timestamp, Optional
        An optional date range to filter logs by.
    log_location : str
        The location of the server access log files on the private bucket.

    Returns
    -------
    pd.DataFrame
        A pandas data frame of remote server access logs.
    """
    import pickle
    from io import StringIO
    log_files = tqdm(_iter_logs(log_location, date_range), unit=' files')
    log_files = map(BytesIO.read, log_files)
    # Decode each log and save as list of strings in home dir in case we fail to parse them
    logstr = list(map(bytes.decode, log_files))
    with open(LOCAL_LOG_LOCATION.joinpath(date_range[0].strftime('%Y-%m') + '_logs_raw.pkl'), 'wb') as file:
        pickle.dump(logstr, file)

    # Parse each str individually. AWS occasionally introduces new columns and within one file
    # there may rows with different numbers of columns.
    dfs = []
    for i, log in enumerate(map(StringIO, logstr)):
        try:
            dfs.append(pd.read_csv(log, sep=' ', header=None))
        except pd.errors.ParserError as ex:
            # Parser expects N columns of nth row <= N columns of 1st row.
            # If there's a column mismatch, attempt to parse each row individually.
            # NB: To be extra safe we search for the expected column mismatch in the error str.
            n_fields = len(COL_NAMES)
            if re.search(rf'Expected {n_fields - 1} fields in line \d+, saw {n_fields}', str(ex)):
                rows = map(StringIO, logstr[i].strip().split('\n'))
                rows = (pd.read_csv(row, sep=' ', header=None) for row in rows)
                dfs.append(pd.concat(rows, ignore_index=True))  # concat handles missing columns
            else:
                raise ex
    df = pd.concat(dfs, ignore_index=True)
    df.columns = COL_NAMES
    return df


def get_remote_log_date_range(log_location=REMOTE_LOG_LOCATION):
    """
    For a given log location, return the datetime of earliest and most recent log file.
    NB: There are typically thousands, maybe millions, of log files.  This is not performant.

    Parameters
    ----------
    log_location : str
        The location of the server access log files on the private bucket.

    Returns
    -------
    tuple of pd.Timestamp
        The start and end timestamps.
    """
    log_files = sorted(map(lambda x: x.key, _iter_objects(log_location)))
    return _timestamp(log_files[0]), _timestamp(log_files[-1])


def prepare_for_parquet(df):
    """
    Ensures log table has correct column names and data types suitable for parquet.

    Parameters
    ----------
    df : pandas.DataFrame
        A data frame of AWS S3 access logs.

    Returns
    -------
    pandas.DataFrame
        A data frame comprising core data types suitable for saving to parquet
    """
    # If empty return black data frame with the expected column names
    if df.empty or len(df.columns) == 0:
        return pd.DataFrame(columns=COL_NAMES)

    # ACL_Required was recently added to log files
    if len(df.columns) == 26 and 'ACL_Required' not in df.columns:
        df['ACL_Required'] = pd.NA

    # Ensure correct data types for pyarrow
    df.loc[df['Bytes_Sent'] == '-', 'Bytes_Sent'] = 0
    df['Bytes_Sent'] = df['Bytes_Sent'].astype(int)
    for col in ('Object_Size', 'Turn_Around_Time', 'Total_Time'):
        df.loc[df[col] == '-', col] = np.nan
        df[col] = df[col].astype(float)
    df.loc[df.ACL_Required.isin(('-', np.nan)), 'ACL_Required'] = pd.NA
    return df


def download_public_logs(download_location=None, date_range=None) -> pd.DataFrame:
    """
    Download the remote logs and save as parquet table if local file does not already exist for the
    provided date range.

    Parameters
    ----------
    download_location : str, pathlib.Path
        The location of the local log parquet files to load.
    date_range : str, list, datetime.datetime, datetime.date, pd.timestamp, Optional
        An optional date range to filter logs by.

    Returns
    -------
    pd.DataFrame
        A (down)loaded data frame of logs.
    """
    date_range = validate_date_range(date_range)
    download_location = Path(download_location or LOCAL_LOG_LOCATION)
    if not date_range:
        # Get full range from remote directory.  If no new logs we may load from local files.
        date_range = get_remote_log_date_range()
    range_str = '{}_{}'.format(*map(pd.Timestamp.isoformat, date_range)).replace(':', '')
    if (local_path := next(download_location.glob(f'*{range_str}*.pqt'), False)):
        print(f'Loading {local_path}')
        return pd.read_parquet(local_path)
    else:
        print('Reading remote logs')
        df = read_remote_logs(date_range=date_range)
        bucket_name, = df['Bucket_Owner'].unique()
        local_path = download_location.joinpath(f'{range_str}_{bucket_name}.pqt')
        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f'Saving to {local_path}')
        prepare_for_parquet(df).to_parquet(local_path)
        return df


def load_all_local_logs(local_log_location=None, glob_pattern='*.pqt'):
    """

    Parameters
    ----------
    local_log_location : str, pathlib.Path
        The location of the local log parquet files to load.
    glob_pattern : str
        The pattern for globbing local log files.

    Returns
    -------
    pd.DataFrame
        A data frame of all local log.
    """
    log_dir = Path(local_log_location or LOCAL_LOG_LOCATION)
    local_logs = map(pd.read_parquet, log_dir.glob(glob_pattern))
    empty = pd.DataFrame(columns=COL_NAMES)
    all_logs = pd.concat((empty, *local_logs))
    if all_logs.empty:
        return all_logs
    # Get unique and sort
    all_logs = pd.unique(all_logs)
    if not all_logs.index.is_monotonic_increasing:
        all_logs.sort_index(inplace=True)
    return all_logs


def upload_table(df, key, bucket):
    """
    Upload a data frame to a given bucket location.

    Writes the table to a temporary directory, then downloads and reads the file after upload.

    Parameters
    ----------
    df : pd.DataFrame
        A data frame to save in parquet format.
    key : PurePosixPath, str
        The destination location within the bucket.
    bucket : s3.Bucket
        The S3 bucket object.
    """
    if isinstance(key, str):
        key = PurePosixPath(key)
    # df.to_parquet(f's3://{dst_bucket_name}/{s3_url}')
    with TemporaryDirectory() as tdir:
        # Save
        filepath = Path(tdir) / key.name
        df.to_parquet(filepath)
        # Upload
        bucket.upload_file(str(filepath), key.as_posix())

    # Read after write
    with TemporaryDirectory() as tdir:
        filepath = Path(tdir) / key.name
        # Download
        bucket.download_file(key.as_posix(), str(filepath))
        df2 = pd.read_parquet(filepath)
        assert (df2.size == df.size) and (df2.shape == df.shape)
