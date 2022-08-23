"""A module for running ibllib continuous integration tests
In order for this to work ibllib and iblscripts must be installed as python package from GitHub.
"""
import argparse
import unittest
import doctest
import json
import sys
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List, Union

import ibllib
from iblutil.util import flatten
import projects

logger = logging.getLogger('ibllib')

try:  # Import the test packages
    import brainbox.tests, ci.tests, ibllib.tests
except ModuleNotFoundError as ex:
    logger.warning(f'Failed to import test packages: {ex} encountered')


def list_tests(suite: Union[List, unittest.TestSuite, unittest.TestCase]) -> Union[List[str], str]:
    """
    Returns a full list of the tests run in the format 'TestClassName/test_method'
    :param suite: A TestCase or TestSuite instance, or list thereof
    :return: A list of tests
    """
    if isinstance(suite, list):
        return flatten([list_tests(x) for x in suite])
    elif not unittest.suite._isnotsuite(suite):
        return list_tests(suite._tests)
    elif isinstance(suite, (unittest.TestSuite, unittest.TestCase)):
        return f'{suite.__class__.__name__}/{suite._testMethodName}'


def load_doctests(test_dir, options) -> unittest.TestSuite:
    return doctest.DocFileSuite(*list(map(str, test_dir.rglob('*.py'))))


def run_tests(complete: bool = True,
              strict: bool = True,
              dry_run: bool = False,
              failfast: bool = False) -> (unittest.TestResult, str):
    """
    Run integration tests
    :param complete: When true ibllib unit tests are run in addition to the integration tests.
    :param strict: When true asserts that all gathered tests were successfully imported.  This
    means that a module not found error in any test module will raise an exception.
    :param dry_run: When true the tests are gathered but not run.
    :param failfast: Stop the test run on the first error or failure.
    :return Test results and test list (or test suite if dry-run).
    """
    # Gather tests
    test_dir = str(Path(ci.tests.__file__).parent)
    logger.info(f'Loading integration tests from {test_dir}')
    ci_tests = unittest.TestLoader().discover(test_dir, pattern='test_*')
    if complete:
        # include ibllib and brainbox unit tests, plus personal projects
        root = Path(ibllib.__file__).parents[1]  # Search relative to our imported ibllib package
        test_dirs = [root.joinpath(x) for x in ('brainbox', 'ibllib')]
        test_dirs.append(Path(projects.__file__).parent)  # this contains the personal projects tests
        for tdir in test_dirs:
            logger.info(f'Loading unit tests from folders: {tdir}')
            assert tdir.exists(), f'Failed to find unit test folders in {tdir}'
            unit_tests = unittest.TestLoader().discover(str(tdir), pattern='test_*', top_level_dir=root)
            logger.info(f"Found {unit_tests.countTestCases()}, appending to the test suite")
            ci_tests.addTests(unit_tests)


    logger.info(f'Complete suite contains {ci_tests.countTestCases()} tests')
    # Check all tests loaded successfully
    not_loaded = [x[12:] for x in list_tests(ci_tests) if x.startswith('_Failed')]
    if len(not_loaded) != 0:
        err_msg = 'Failed to import the following tests:\n\t' + '\n\t'.join(not_loaded)
        assert not strict, err_msg
        logger.warning(err_msg)

    if dry_run:
        return unittest.TestResult(), ci_tests

    # Make list of tests - once successfully run the tests are removed from the suite
    test_list = list_tests(ci_tests)

    # Run tests
    result = unittest.TextTestRunner(verbosity=2, stream=sys.stdout, failfast=failfast).run(ci_tests)

    return result, test_list


if __name__ == "__main__":
    r"""Run all the integration tests
    The commit id is used to identify the test report.  If none is provided no test record is saved

    python runAllTests.py --logdir <log directory> --commit <commit sha> --repo <repo path>

    Examples:
      python runAllTests.py -l C:\Users\User\AppData\Roaming\CI
      python runAllTests.py -l ~/.ci
    """
    timestamp = datetime.utcnow().isoformat()
    # Defaults
    root = Path(__file__).parent.absolute()  # Default root folder
    repo_dir = Path(ibllib.__file__).parent  # Default repository source for coverage

    version = getattr(ibllib, '__version__', timestamp)

    # Parse parameters
    parser = argparse.ArgumentParser(description='Integration tests for ibllib.')
    parser.add_argument('--commit', '-c', default=version,
                        help='commit id.  If none provided record isn''t saved')
    parser.add_argument('--logdir', '-l', help='the log path', default=root)
    parser.add_argument('--repo', '-r', help='repo directory', default=repo_dir)
    parser.add_argument('--dry-run', help='gather tests without running', action='store_true')
    parser.add_argument('--failfast', action='store_true',
                        help='stop the test run on the first error or failure')
    parser.add_argument('--failfast', action='store_true',
                        help='stop the test run on the first error or failure')
    parser.add_argument('--exit', action='store_true', help='return 1 if tests fail')
    args = parser.parse_args()  # returns data from the options specified (echo)

    # Paths
    report_dir = Path(args.logdir).joinpath('reports', args.commit)
    # Create the reports tree if it doesn't already exist
    report_dir.mkdir(parents=True, exist_ok=True)
    logfile = report_dir / 'test_output.log'
    db_file = Path(args.logdir, '.db.json')

    # Setup backup log (NB: the system output is also saved by the ci)
    fh = RotatingFileHandler(logfile, maxBytes=(1048576 * 5))
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)

    # Tests
    logger.info(Path(args.repo).joinpath('*'))
    result, test_list = run_tests(dry_run=args.dry_run, failfast=args.failfast)
    exit_code = int(not result.wasSuccessful()) if args.exit else 0

    # Generate report
    logger.info('Saving coverage report to %s', report_dir)
    duration = datetime.now().utcnow() - datetime.fromisoformat(timestamp)

    # When running tests without a specific commit, exit without saving the result
    if args.commit is parser.get_default('commit'):
        sys.exit(exit_code)

    # Summarize the results of the tests and write results to the JSON file
    logger.info('Saving outcome to %s', db_file)
    status = 'success' if result.wasSuccessful() else 'failure'
    n_failed = len(result.failures) + len(result.errors)
    fail_str = f'{n_failed}/{result.testsRun} tests failed'
    description = 'All passed' if result.wasSuccessful() else fail_str
    # Save all test names if all passed, otherwise save those that failed and their error stack
    if result.wasSuccessful():
        details = test_list
        logger.info('All tests pass...')
    else:
        details = [(list_tests(c), err) for c, err in result.failures + result.errors]
        logger.warning('Tests failing...')

    # A breakdown of the test numbers
    stats = {
        'total': len(list_tests(test_list)) if args.dry_run else result.testsRun,
        'failed': len(result.failures),
        'errored': len(result.errors),
        'skipped': len(result.skipped),
        'passed': result.testsRun - (len(result.skipped) + n_failed),
        'duration': duration.total_seconds()
    }

    report = {
        'commit': args.commit + ('_dry-run' if args.dry_run else ''),
        'datetime': timestamp,  # UTC
        'results': details,
        'status': status,
        'description': description,
        'statistics': stats,
        'coverage': None  # coverage updated from XML report by Node.js script
    }

    if db_file.exists():
        with open(db_file, 'r') as json_file:
            records = json.load(json_file)
        try:  # update existing
            idx = next(i for i, r in enumerate(records) if r['commit'] == args.commit)
            records[idx] = report
        except StopIteration:  # ...or append record
            records.append(report)
    else:
        records = [report]

    # Save record to file
    with open(db_file, 'w') as json_file:
        json.dump(records, json_file)

    sys.exit(exit_code)
