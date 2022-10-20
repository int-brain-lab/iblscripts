import logging

from iblutil.io import net
import iblutil.io.net.app
from iblutil.io.net.base import ExpMessage
from ibllib.io import session_params
from one.converters import ConversionMixin
import one.params
from one.api import ONE


_logger = logging.getLogger(__name__)


def install_alyx_token(base_url, token):
    """Save Alyx token sent from remote device.

    Saves an Alyx token into the ONE params for a given database instance.

    Parameters
    ----------
    base_url : str
        The Alyx database URL.
    token : dict[str, str]
        The token in the form {username: taken}.

    Returns
    -------
    bool
        True if the token was far a user not already cached.
    """
    par = one.params.get(base_url)
    is_new_user = next(iter(token)) not in par.TOKEN
    par.TOKEN.update(token)
    one.params.save(par, base_url)
    return is_new_user


def alyx_request_callback(*args):
    """Callback to return ONE logged in with token from another device.

    Parameters
    ----------
    args
        The event message containing the token, e.g. (3, [url, {username: token}]).

    Returns
    -------
    one.api.OneAlyx
        A logged-in instance of ONE.
    bool
        If True, the user was not previously logged in on this device and should therefore be
        logged out during cleanup.
    """
    _, (base_url, token) = args
    logout = install_alyx_token(base_url, token)
    username = next(iter(token[1]))
    one = ONE(base_url=base_url, username=username, silent=True)
    return one, logout


def fetch_remote_clients(session_path):
    """Load remote device clients for a given session.

    Parameters
    ----------
    session_path : str, pathlib.Path
        The session path containing an experiment description file.

    Returns
    -------
    list of iblutil.io.net.app.EchoProtocol
        A list of remote clients.
    """
    if exp_info := session_params.read_params(session_path):
        # FIXME Set of URIs
        remote_devices = {k: v['URI'] for k, v in exp_info.get('devices', {}) if 'URI' in v}
        remote_rigs = [await net.app.EchoProtocol.client(server_uri, name)
                       for server_uri, name in remote_devices.items()]
    else:
        remote_rigs = []
    return remote_rigs


async def prepare_remote_services(session_path):
    """Example routine to run before starting an experiment."""
    exp_ref = ConversionMixin.path2ref(session_path, as_dict=False)  # 2022-01-01_1_subject
    remote_rigs = fetch_remote_clients(session_path)
    services = net.app.Services(remote_rigs)

    # Set up callbacks
    callback = lambda x: print(map(str, x[0]))
    services.assign_callback(ExpMessage.EXPSTART, callback)

    # START EXPERIMENT
    responses = await services.start(exp_ref)

    # STOP EXPERIMENT
    for rig in remote_rigs.reverse():
        await rig.stop(exp_ref)


if __name__ == '__main__':
    import argparse
    import tempfile
    from datetime import date
    from pathlib import Path
    from iblutil.util import get_logger
    from iblutil.io.net.base import hostname2ip

    SESSION = ('subject', str(date.today()), '001')
    ip_address = hostname2ip()

    parser = argparse.ArgumentParser(description='Remote experiment playground.')
    parser.add_argument('role', choices=('server', 'client'),
                        help='communicator role i.e. server or client')

    # Set up logs
    get_logger(_logger.name).setLevel(logging.DEBUG)
    get_logger('iblutil.io.net.app').setLevel(logging.DEBUG)

    # Dummy experiment
    with tempfile.TemporaryDirectory() as tmpdir:
        # SERVER
        if parser.parse_args().role == 'server':
            session_path = Path(tmpdir, 'server', *SESSION)
            session_path.mkdir(parents=True, exist_ok=True)
            await net.app.EchoProtocol.server(ip_address, 'behaviour')
        # CLIENT
        else:
            session_path = Path(tmpdir, 'client', *SESSION)
            session_path.mkdir(parents=True, exist_ok=True)
