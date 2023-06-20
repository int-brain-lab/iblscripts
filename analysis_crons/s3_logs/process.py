"""Module for digesting and performing analytics on S3 public bucket access logs"""
from datetime import datetime, timedelta
from calendar import monthrange
from pathlib import PurePosixPath, Path
import warnings
from time import sleep
import pickle
from urllib.request import urlopen
from ipaddress import IPv4Address, IPv6Address
import json
import time

import pandas as pd
import numpy as np
import pyarrow as pa
import boto3
from botocore.exceptions import ClientError
from one.alf.files import get_session_path

from analysis_crons.s3_logs.io import REMOTE_LOG_LOCATION, LOCAL_LOG_LOCATION, _iter_objects, prepare_for_parquet, upload_table
from analysis_crons.s3_logs import io as s3io


def files_accessed_this_month():
    """Print the number of times each file was accessed this month"""
    # https://aws.amazon.com/blogs/storage/monitor-amazon-s3-activity-using-s3-server-access-logs-and-pandas-in-python/
    df = s3io.get_log_table_by_month(datetime.today())
    file_access = (df['Operation'] == 'REST.GET.OBJECT') & (df['Key'].str.startswith('data/'))
    print(df.loc[file_access, 'Key'].value_counts())
    return df


def week_of_month(dt):
    """
    Returns the week of the month for the specified date.

    Weeks are numbered from 1-5.  Also returns the date times of the first and last week days.

    Parameters
    ----------
    dt : datetime.datetime
        The datetime for which to determine the week number.

    Returns
    -------
    int
        The week number within the month of dt.
    datetime.datetime
        The datetime of the first day of the week within the month.
    datetime.datetime
        The datetime of the last day of the week within the month.
    """
    _, n_days = monthrange(dt.year, dt.month)
    first_day = dt.replace(day=1)
    adjusted_dom = dt.day + first_day.weekday()
    week_number = int(np.ceil(adjusted_dom / 7))

    first_day = dt.day - dt.weekday()
    if first_day < 1:
        first_day = 1
    last_day = dt.day + (6 - dt.weekday())
    if last_day > n_days:
        last_day = n_days

    start_date = dt.replace(day=first_day).date()
    start = datetime(*start_date.timetuple()[:3])
    end = (start.replace(day=last_day) +
           timedelta(hours=23, minutes=59, seconds=59, milliseconds=999))

    return week_number, (start, end)


def consolidate_logs(boto_session=None, date='last_month', ipinfo_token=None, profile_name='miles'):
    """
    Download last month's log files, upload as parquet table to S3 and delete individual log files.

    Logs are uploaded to the REMOTE_LOG_LOCATION with the following name pattern:
        consolidated/YYYY-MM_<BUCKET-NAME>.pqt
    If the logs are consolidated for the current month, the file will end with '_INCOMPLETE.pqt'.
    The IP address locations are stored in a file with the same name, ending with '_IP-info.pqt'.

    Individual log files are only deleted once the consolidated logs are

    Parameters
    ----------
    boto_session : boto3.Session
        An S3 Session with PUT and DELETE permissions.
    date : str, datetime.datetime, datetime.date
        If 'last_month', consolidate all of last month's logs; if 'this_month', consolidate the
        logs for the current month so far (still deletes the logs); if a date is provided, the logs
        for that date's month are consolidated.
    ipinfo_token : str
        An API token for using with the ipinfo API to query IP address location.
    profile_name: str
        The profile name of the boto s3 credentials

    Returns
    -------
    pandas.DataFrame
        The downloaded logs.
    str
        The URI of the uploaded parquet table.
    """
    today = datetime.utcnow()
    if date == 'this_month':
        partial = True
        start_date = today.replace(day=1).date()
        start = datetime(*start_date.timetuple()[:3])
        end = today
    elif date == 'last_month':
        partial = False
        # The date range for last month
        month = today.month
        start_date = today.replace(
            day=1, month=month - 1 or 12, year=today.year - int(not bool(month - 1))).date()
        start = datetime(*start_date.timetuple()[:3])
        end = (start.replace(day=monthrange(start.year, start.month)[1]) +
               timedelta(hours=23, minutes=59, seconds=59, milliseconds=999))
    else:
        if isinstance(date, str):
            date = datetime.fromisoformat(date)
        partial = (date.year, date.month) == (today.year, today.month)
        start_date = date.replace(day=1).date()
        start = datetime(*start_date.timetuple()[:3])
        end = (start.replace(day=monthrange(start.year, start.month)[1]) +
               timedelta(hours=23, minutes=59, seconds=59, milliseconds=999))

    # Check for parquet file on S3
    session = boto_session or boto3.Session(profile_name=profile_name)
    dst_bucket_name = 'ibl-brain-wide-map-private'
    s3 = session.resource('s3')
    bucket = s3.Bucket(name=dst_bucket_name)
    prefix = f'{REMOTE_LOG_LOCATION}consolidated/{start.strftime("%Y-%m")}'
    consolidated = (x.key for x in bucket.objects.filter(Prefix=prefix)
                    if not (x.key.endswith('INCOMPLETE.pqt') or x.key.endswith('IP-info.pqt')))
    assert next(iter(consolidated), False) is False, \
        'logs already consolidated for ' + start.strftime('%B')

    print(f'Reading remote logs for {start.strftime("%B")}' + (' so far' if partial else ''))
    try:
        df = s3io.read_remote_logs(date_range=(start, end), log_location=REMOTE_LOG_LOCATION)
    except pd.errors.ParserError as ex:
        # pandas.errors.ParserError: Error tokenizing data. C error: Expected 26 fields in line 2, saw 27
        warnings.warn(f'{ex}\n')
        df = s3io.read_remote_logs_robust(date_range=(start, end), log_location=REMOTE_LOG_LOCATION)

    # Process table
    df = prepare_for_parquet(df)
    # Check every row was parsed correctly
    assert all(isinstance(x, str) and len(x) == 64 for x in df.Bucket_Owner.unique())
    if df.empty:
        warnings.warn('No logs found!')
        return df, None

    bucket_name, = df['Bucket'].unique()
    filename = f'{start.strftime("%Y-%m")}_{bucket_name}.pqt'
    s3_url = PurePosixPath(REMOTE_LOG_LOCATION, 'consolidated', filename)

    # Attempt to download the incomplete logs table and merge
    partial_file = s3_url.with_name(s3_url.stem + '_INCOMPLETE.pqt')
    # Download partial month logs
    filepath = LOCAL_LOG_LOCATION / (partial_file.stem + f'.tmp{np.floor(time.time()):.0f}.pqt')
    try:
        # Check for partial log table
        s3.Object(bucket_name=dst_bucket_name, key=partial_file.as_posix()).load()
        print('Downloading partial log table')
        bucket.download_file(partial_file.as_posix(), str(filepath))
        df_ = pd.read_parquet(filepath)
        assert np.all(df['Bucket'].unique() == bucket_name), 'multiple bucket logs'
        print(f'Concatenating logs ({df_.size} + {df.size} rows)')
        df = pd.concat([df_, df], ignore_index=True)
    except ClientError as ex:
        if ex.response['Error']['Code'] != '404':
            raise ex

    print('Removing duplicate rows')
    df.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=True)

    print('Uploading table')
    upload_table(df, partial_file if partial else s3_url, bucket)

    print('Deleting log files')
    for obj in s3io._iter_objects(REMOTE_LOG_LOCATION, date_range=(start, end), s3_bucket=bucket):
        assert PurePosixPath(obj.key).name.startswith(start_date.strftime('%Y-%m'))
        print(f'deleting {obj.key}')
        obj.delete()

    if not partial:
        try:
            # Delete incomplete log table if exists
            bucket.objects.filter(Prefix=partial_file.as_posix()).delete()
            print(f'deleted {partial_file}')
        except ClientError as ex:
            if ex.response['Error']['Code'] != '404':
                raise ex

    print('Fetching IP info...')
    # Unique accesses
    unique_ips = df.loc[:, 'Remote_IP'].unique()

    print(f'Data was accessed from {len(unique_ips):,d} unique devices')
    # Attempt to download current IP info table, if exists
    ip_table_url = s3_url.with_name(s3_url.stem + '_IP-info.pqt')
    try:
        # Check for partial log table
        bucket.download_file(ip_table_url.as_posix(), str(LOCAL_LOG_LOCATION / ip_table_url.name))
        ip_details_ = pd.read_parquet(LOCAL_LOG_LOCATION / ip_table_url.name)
        unique_ips = np.setdiff1d(unique_ips, ip_details_.index, assume_unique=True)
    except ClientError as ex:
        if ex.response['Error']['Code'] != '404':
            raise ex
        ip_details_ = None

    if unique_ips.any():
        print(f'Querying location for {unique_ips.size} IPs')
        ip_details = ip_info(unique_ips, wait=None, token=ipinfo_token)
        ip_details = pd.DataFrame(ip_details).set_index('ip')
        if ip_details_ is not None:
            ip_details = pd.concat([ip_details_, ip_details], verify_integrity=True)
        print('Uploading IP table')
        upload_table(ip_details, ip_table_url, bucket)

    return df, f's3://{dst_bucket_name}/{partial_file if partial else s3_url}'


def key2date(key: str) -> datetime:
    """
    Convert a access log file key to a datetime object.

    Parameters
    ----------
    key : str
        The location of an S3 log file.

    Returns
    -------
    datetime.datetime
        The datetime parsed from the file name.

    Example
    -------
    >>> key2date('logs/server-access-logs/2023-03-01-00-09-07-9774A35EB7B3DBE3')
    datetime.datetime(2023, 3, 1, 0, 9, 7)
    """
    return datetime(*map(int, PurePosixPath(key).name.split('-')[:-1]))


def _first_log_datetime():
    """Get the datetime of the first S3 log file."""
    obj = next(_iter_objects(REMOTE_LOG_LOCATION))
    return key2date(obj.key)


def consolidate_logs_by_week(boto_session=None):
    """
    Download logs by week, consolidate and upload as parquet table to S3, then delete individual
    the individual log files.

    Parameters
    ----------
    boto_session : boto3.Session
        An S3 Session with PUT and DELETE permissions.

    Returns
    -------
    list of str
        The URIs of the uploaded parquet tables.
    """
    session = boto_session or boto3.Session(profile_name='miles')
    dst_bucket_name = 'ibl-brain-wide-map-private'
    s3 = session.resource('s3')
    bucket = s3.Bucket(name=dst_bucket_name)

    urls = []
    while datetime.utcnow() - (dt := _first_log_datetime()) > timedelta(days=7):
        week_number, (start, end) = week_of_month(dt)

        # Check for parquet file on S3
        consolidated = bucket.objects.filter(
            Prefix=f'{REMOTE_LOG_LOCATION}/consolidated/{start.strftime("%Y-%m")}_{week_number}')
        assert next(iter(consolidated), False) is False, \
            f'logs already consolidated for week {week_number} of ' + start.strftime('%B')

        print(f'Reading remote logs for week {week_number} of {start.strftime("%B")}')
        try:
            df = s3io.read_remote_logs(date_range=(start, end), log_location=REMOTE_LOG_LOCATION)
        except pd.errors.ParserError as ex:
            # pandas.errors.ParserError: Error tokenizing data. C error: Expected 26 fields in line 2, saw 27
            warnings.warn(f'{ex}\n')
            df = s3io.read_remote_logs_robust(date_range=(start, end), log_location=REMOTE_LOG_LOCATION)

        # Process table
        df = prepare_for_parquet(df)
        # Check every row was parsed correctly
        assert all(isinstance(x, str) and len(x) == 64 for x in df.Bucket_Owner.unique())
        bucket_name, = df['Bucket'].unique()
        filename = f'{start.strftime("%Y-%m")}_{week_number}_{bucket_name}.pqt'
        s3_url = PurePosixPath(REMOTE_LOG_LOCATION, 'consolidated', filename)

        print('Uploading table')
        upload_table(df, s3_url, bucket)

        print('Deleting log files')
        for obj in s3io._iter_objects(REMOTE_LOG_LOCATION, date_range=(start, end), s3_bucket=bucket):
            assert end > key2date(obj.key) > start
            print(f'deleting {obj.key}')
            obj.delete()

        urls.append(f's3://{dst_bucket_name}/{s3_url}')

        print('Fetching IP info...')
        # Unique accesses
        unique_ips = df.loc[:, 'Remote_IP'].unique()

        print(f'Data was accessed from {len(unique_ips):,d} unique devices')
        ip_details = ip_info(unique_ips, wait=.2)  # pause to avoid DoS
        ip_details = pd.DataFrame(ip_details).set_index('ip')
        upload_table(ip_details, s3_url.with_name(s3_url.stem + '_IP-info.pqt'), bucket)

    return urls


def parse_time_column(df):
    """Converts the Time and Time_Offset string columns to pandas Datetime objects"""
    return pd.to_datetime(df['Time'] + df['Time_Offset'], format='[%d/%b/%Y:%H:%M:%S%z]', utc=True)


def ip_info(ip_address, wait=None, token=None):
    """
    Fetch DNS information associated with the IP address(es).

    Uses the public API of ipinfo.io. Use the wait arg if concerned about reaching request limit
    for large IP lists.

    Parameters
    ----------
    ip_address : iterable, str, IPv4Address, IPv6Address
        One or more IP addresses to look up.
    wait : float, bool, optional
        Whether to wait between API queries to avoid DoS.
    token : str, optional
        An optional API token to use.

    Returns
    -------
    dict, list of dict
        The IP address lookup details.
    """
    if not isinstance(ip_address, (str, IPv4Address, IPv6Address)):
        return [ip_info(ip, wait, token) for ip in ip_address if wait is None or not sleep(wait)]

    url = f'https://ipinfo.io/{ip_address}' + (f'?token={token}' if token else '/json')
    res = urlopen(url)
    assert res
    info = json.load(res)
    info['accessed'] = datetime.utcnow().isoformat()
    return info


def SCGB_renewal():
    """Plots and stats for the SCGB renewal grant"""
    sfn_start_date = pd.Timestamp.fromisoformat('2022-11-12')
    # Download logs for this month
    file_obj = s3io.get_log_table_by_month(sfn_start_date)
    table = pa.BufferReader(file_obj.get()['Body'].read())
    df = pd.read_parquet(table)

    # Print number of files accessed
    file_access = (df['Operation'] == 'REST.GET.OBJECT') & (df['Key'].str.startswith('data/'))
    datasets = df.loc[file_access, 'Key'].unique()
    print(f'{sum(file_access):,} total downloads for the month of {sfn_start_date.strftime("%B")}')
    print(f'{len(datasets):,} different datasets downloaded')

    # Unique accesses
    unique_ips = df.loc[file_access, 'Remote_IP'].unique()
    print(f'Data was accessed from {len(unique_ips)} unique devices')
    _filename = f'{datetime.today().strftime("%Y-%m")}_log_ips.pkl'
    filename = Path.home().joinpath(_filename)
    if filename.exists():
        with open(filename, 'rb') as file:
            ip_info_map = pickle.load(file)
    else:
        ip_info_map = []
        for ip in unique_ips:
            sleep(.2)  # pause to avoid DoS
            ip_info_map.append(ip_info(ip))
        ip_info_map = {x.pop('ip'): x for x in ip_info_map}
        with open(filename, 'wb') as file:
            pickle.dump(ip_info_map, file)

    city = 'San Diego'  # Location of SfN 2022
    sfn_ips = [ip for ip, det in ip_info_map.items() if det['city'] == city]
    at_sfn = df['Remote_IP'].isin(sfn_ips)
    print(f'{len(df.loc[file_access & at_sfn, "Key"]):,} datasets downloaded at SfN alone')

    # N sessions accessed
    sessions = set(filter(None, map(get_session_path, datasets)))
    print(f'{len(sessions):,} unique sessions accessed')

    print(df.loc[file_access, 'Key'].value_counts())
    # ONE cache downloads
    cache_access = (df['Operation'] == 'REST.GET.OBJECT') & \
                   (df['Key'].str.match(r'caches/openalyx/[a-zA-Z0-9_/]*cache.zip'))
    # df.loc[cache_access, 'Key'].unique()
    print(f'Public ONE cache was downloaded a total of {sum(cache_access):,} times, of which '
          f'{sum(cache_access & at_sfn):,} downloads occured at SfN')

    # (for devs) Python versions used
    py_ver = df.loc[file_access, 'User_Agent'].str.extract(r'(?<=Python/)(\d+\.\d+)')
    py_ver[0].hist()


def plot_access():
    """Plots and stats for the SCGB renewal grant"""
    start_date = pd.Timestamp.fromisoformat('2022-06-01')
    columns = [
        'total_file_access', 'unique_file_access', 'date', 'unique_ips', 'unique_countries',
        'unique_cities', 'unique_sessions', 'cache_access']
    download_data = pd.DataFrame(columns=columns)
    # Download logs for each month
    for file_obj in s3io.iter_log_tables():
        table = pa.BufferReader(file_obj.get()['Body'].read())
        df = pd.read_parquet(table)

        data = {'date': PurePosixPath(file_obj.key).name[:7]}
        # Print number of files accessed
        file_access = (df['Operation'] == 'REST.GET.OBJECT') & (df['Key'].str.startswith('data/'))
        print(f'{sum(file_access):,} total downloads for the month of {start_date.strftime("%B")}')
        print(f'{len(df.loc[file_access, "Key"].unique()):,} different datasets downloaded')
        data['total_file_access'] = sum(file_access)
        data['unique_file_access'] = len(df.loc[file_access, "Key"].unique())

        # Unique accesses
        unique_ips = df.loc[file_access, 'Remote_IP'].unique()
        print(f'Data was accessed from {len(unique_ips)} unique devices')
        data['unique_ips'] = len(unique_ips)
        _filename = f'{data["date"]}_log_ips.pkl'
        filename = Path.home().joinpath(_filename)
        if filename.exists():
            with open(filename, 'rb') as file:
                ip_info_map = pickle.load(file)
        else:
            ip_info_map = []
            for ip in unique_ips:
                sleep(.2)  # pause to avoid DoS
                ip_info_map.append(ip_info(ip))
            ip_info_map = {x.pop('ip'): x for x in ip_info_map}
            with open(filename, 'wb') as file:
                pickle.dump(ip_info_map, file)

        unique_cities = set('/'.join((x['city'], x['region'], x['country'])) for x in ip_info_map.values())
        data['unique_cities'] = len(unique_cities)
        data['unique_countries'] = len(set(x['country'] for x in ip_info_map.values()))

        # N sessions accessed
        datasets = df.loc[file_access, 'Key'].unique()
        sessions = set(map(get_session_path, datasets))
        print(f'{len(sessions):,} unique sessions accessed')
        data['unique_sessions'] = len(sessions)

        print(df.loc[file_access, 'Key'].value_counts())
        # ONE cache downloads
        cache_access = (df['Operation'] == 'REST.GET.OBJECT') & \
                       (df['Key'].str.match(r'caches/openalyx/[a-zA-Z0-9_/]*cache.zip'))
        # df.loc[cache_access, 'Key'].unique()
        print(f'Public ONE cache was downloaded a total of {sum(cache_access):,} times')
        data['cache_access'] = sum(cache_access)

        download_data = pd.concat([download_data, pd.DataFrame(data, index=[0])])

    download_data.set_index('date').plot(subplots=True)
    return download_data.set_index('date')
