import argparse
from pprint import pformat

from one.webclient import AlyxClient
import one.alf.exceptions as alferr


def update_craniotomy_coordinates(subject, ml, ap, name='craniotomy_00', lambda_mlap=None, bregma_mlap=None, alyx=None):
    """
    Update the JSON field for a subject's most recent surgery with craniotomy coordinates.
    Optionally can the coordinates of lambda and bregma.

    Parameters
    ----------
    subject : str
        The name of a subject that underwent a surgery.
    ml : float
        The distance from the midline in mm of the craniotomy centre.
    ap : float
        The anterio-posterior distance from the lambda of the craniotomy centre, in mm.
    name : str
        The name of the craniotomy (default: 'craniotomy_00').
    lambda_mlap : (float, float), optional
        The ML and AP coordinates, respectively, of lambda (in mm).
    bregma_mlap : (float, float), optional
        The ML and AP coordinates, respectively, of bregma (in mm).
    alyx : one.webclient.AlyxClient, optional
        An instance of Alyx to update.

    Returns
    -------
    dict:
        The updated surgery JSON data.
    """
    alyx = alyx or AlyxClient()
    # Check subject exists
    if not any(alyx.rest('subjects', 'list', nickname=subject)):
        raise alferr.AlyxSubjectNotFound(subject)
    surgeries = alyx.rest('surgeries', 'list', subject=subject, procedure='craniotomy')
    # surgeries = sorted(surgeries, key=lambda x: x['start_time'], reverse=True)
    if not surgeries:
        raise alferr.ALFError(f'Surgery not found for subject "{subject}"')

    data = {name: (float(ml), float(ap))}
    if lambda_mlap:
        data['lambda'] = tuple(map(float, lambda_mlap))
        if len(data['lambda']) != 2:
            raise ValueError('Lamba coordinates must be a tuple of length 2.')
    if bregma_mlap:
        data['bregma'] = tuple(map(float, bregma_mlap))
        if len(data['bregma']) != 2:
            raise ValueError('Bregma coordinates must be a tuple of length 2.')

    surgery = surgeries[0]  # Update most recent surgery in list
    surgery['json'] = alyx.json_field_update('surgeries', surgery['id'], data=data)
    return surgery


if __name__ == '__main__':
    r"""Update craniotomy coordinates.

    Update the JSON field for a subject's most recent surgery with craniotomy coordinates.
    Optionally can the coordinates of lambda and bregma.
    NB: If you specify an Alyx username your login token will not be saved for the next time. 

    Examples:
      python update_craniotomy.py SP026 2.7 1 -D https://alyx.cortexlab.net -u samuel
      python update_craniotomy.py SP026 2.7 1 -n left_hemi

    """
    # Parse parameters
    parser = argparse.ArgumentParser(description='Update craniotomy coordinates.')
    parser.add_argument('subject', type=str, help='A subject name.')
    parser.add_argument('ml', type=float,
                        help='The distance from the midline in mm of the craniotomy centre.')
    parser.add_argument('ap', type=float,
                        help='The anterio-posterior distance from the lambda of the craniotomy centre, in mm.')
    parser.add_argument('--name', '-n', type=str, help='The name of the craniotomy.')
    parser.add_argument('--base-url', '-D', type=str, help='An Alyx database URL.')
    parser.add_argument('--user', '-u', type=str, help='The Alyx username to login as.')
    parser.add_argument('--lambda', dest='lmda', type=float, nargs='+',
                        help='The ML and AP coordinates, respectively, of lambda (in mm).')
    parser.add_argument('--bregma', type=float, nargs='+',
                        help='The ML and AP coordinates, respectively, of bregma (in mm).')
    args = parser.parse_args()  # returns data from the options specified (echo)

    alyx_args = {}
    if args.user:
        alyx_args['username'] = args.user
    if args.base_url:
        alyx_args['base_url'] = args.base_url

    alyx = AlyxClient(**alyx_args)
    try:
        record = update_craniotomy_coordinates(
            args.subject, args.ml, args.ap, name=args.name, alyx=alyx, bregma_mlap=args.bregma, lambda_mlap=args.lmda
        )
        print('Surgery ID: ' + record['id'])
        print('Start time: ' + record['start_time'])
        print('Subject: ' + record['subject'])
        print('JSON:\n' + pformat(record['json']))
        print('Narrative:')
        for line in record['narrative'].strip().split('\n'):
            print(line)
    finally:
        if args.user:
            alyx.logout()
