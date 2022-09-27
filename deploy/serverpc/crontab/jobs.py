import argparse
import logging
from pathlib import Path
import socket
import time

from one.api import ONE
from ibllib.pipes.local_server import job_creator, task_queue, tasks_runner, report_health

DEFINED_PORTS = {
    'run': 54320,
    'run_small': 54320,
    'create': 54321,
    'report': 54322,
}

_logger = logging.getLogger('ibllib')


def _parametrized(dec):
    def layer(*args, **kwargs):
        def repl(f):
            return dec(f, *args, **kwargs)
        return repl
    return layer


@_parametrized
def forever(func, port=None, sleep=600):
    """
    Runs function forever, with a sleep time between successive runs.
    Relies on socket and checks every 2 seconds between function calls for termination requests
    (if the socket receives 'STOP' , the loop exits gracefully)
    Can't run at a rate higher than 2 seconds as it.
    Allows only a single instance by binding to a unique port so subsequent attempts to run
    will indicate that the process is already running and result in an error
    :param port:
    :param sleep:
    :return:
    """
    def wrapper(*args, **kwargs):
        assert port
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.1)
        try:
            s.bind(('localhost', port))
        except OSError:
            print("One instance of the job is already running. Exiting now.")
            return
        t = 0
        # loop every 2 seconds to check the socket for termination request.
        # if the cumulative time is greater than sleep, run the function and re-init t
        while True:
            try:
                # check if a message has been received
                data, address = s.recvfrom(4096)
                if data == b'STOP':
                    print('ABORT !!')
                    s.sendto(b'ACK STOP', address)
                    return
                elif data == b'CHECK':
                    s.sendto(b"I'm fine, helicopter mum", address)
            except socket.timeout:
                pass
            if (time.time() - t) > sleep:
                t = time.time()
                func(*args, **kwargs)
            time.sleep(2)
    return wrapper


@forever(DEFINED_PORTS['run'], 600)
def run_tasks(subjects_path, dry=False, lab=None, count=20):
    """
    Runs task backlog from task records in Alyx for this server
    :param subjects_path: "/mnt/s0/Data/Subjects"
    :param dry:
    :return:
    """
    one = ONE(cache_rest=None)
    waiting_tasks = task_queue(mode='all', lab=lab, one=one)
    tasks_runner(subjects_path, waiting_tasks, one=one, count=count, time_out=3600, dry=dry)


@forever(DEFINED_PORTS['run_small'], 600)
def run_tasks_small(subjects_path, dry=False, lab=None, count=20):
    """
    Runs backlog of tasks excluding video compression, spike sorting and dlc from task records in Alyx for this server
    :param subjects_path: "/mnt/s0/Data/Subjects"
    :param dry:
    :return:
    """
    one = ONE(cache_rest=None)
    waiting_tasks = task_queue(mode='small', lab=lab, one=one)
    tasks_runner(subjects_path, waiting_tasks, one=one, count=count, time_out=3600, dry=dry)


@forever(DEFINED_PORTS['report'], 3600 * 2)
def report():
    """
    Labels the lab endpoint json field with health indicators every 2 hours
    """
    one = ONE(cache_rest=None)
    report_health(one=one)


@forever(DEFINED_PORTS['create'], 60 * 15)
def create_sessions(root_path, dry=False):
    """
    Create sessions: for this server, finds the extract_me flags, identify the session type,
    create the session on Alyx if it doesn't already exist, register the raw data and create
    the tasks backlog
    """
    job_creator(root_path, dry=dry)


@forever(DEFINED_PORTS['create'], 4)
def test_fcn():
    print('Toto')


def _send2job(name, bmessage):
    port = DEFINED_PORTS[name]
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('localhost', port))
        s.send(bmessage)
        print('status/kill request sent, waiting for job response')
        print(str(s.recv(4096)))
    except ConnectionRefusedError:
        print("Job doesn't seem to be running")
        return -1
    return 0


if __name__ == "__main__":
    """
    Create: creates session as they get copied on the server
    Run: run the tasks labeled as waiting on Alyx
    Report: label the corresponding lab json field with server health indicators
    Launch neverending jobs (only single instance allowed):
        python jobs.py create /mnt/s0/Data (--dry, --restart)
        python jobs.py run /mnt/s0/Data (--dry, --restart)
        python jobs.py report
    Check them:
        python jobs.py status create
        python jobs.py status run
        python jobs.py status report
    Kill them:
        python jobs.py kill create
        python jobs.py kill run
        python jobs.py kill report
    """
    JOBS = ['create', 'run', 'run_small', 'test', 'report']
    ALLOWED_ACTIONS = ['kill', 'status'] + JOBS

    parser = argparse.ArgumentParser(description='Creates jobs for new sessions')
    parser.add_argument('action', help='Action: ' + ','.join(ALLOWED_ACTIONS))
    parser.add_argument('folder', help='A Folder containing a session', nargs="?")
    parser.add_argument('--dry', help='Dry Run', required=False, action='store_true')
    parser.add_argument('--restart', help='Restart if running', required=False,
                        action='store_true')

    args = parser.parse_args()  # returns data from the options specified (echo)
    if args.action == 'create':
        assert (Path(args.folder).exists())
        if args.restart:
            _send2job(args.action, b"STOP")
        create_sessions(args.folder, args.dry)
    elif args.action == 'run':
        assert (Path(args.folder).exists())
        if args.restart:
            _send2job('run', b"STOP")
        run_tasks(args.folder, args.dry)
    elif args.action == 'run_small':
        assert (Path(args.folder).exists())
        if args.restart:
            _send2job('run_small', b"STOP")
        run_tasks_small(args.folder, args.dry)
    elif args.action == 'report':
        if args.restart:
            _send2job('report', b"STOP")
        report()
    elif args.action == 'test':
        if args.restart:
            _send2job('create', b"STOP")
        test_fcn()
    elif args.action == 'kill':
        if _send2job(args.folder, b"STOP") == 0:
            print('Job terminated successfully')
    elif args.action == 'status':
        if _send2job(args.folder, b"CHECK") == 0:
            print('Job seems to be alright')
    else:
        _logger.error(f'Action "{args.action}" not valid. Allowed actions are: '
                      f'{"., ".join(ALLOWED_ACTIONS)}')
