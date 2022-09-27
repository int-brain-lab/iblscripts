from pathlib import Path
from os import sep
import re
import argparse
from logging import DEBUG

from ibllib.misc.misc import _logger
_logger.setLevel(DEBUG)

if __name__ == "__main__":
    r"""Remove source paths from coverage HTML report files
    The coverage HTML and XML files contain the full paths to the code files.  This function
    replaces all full paths with paths relative to a source directory.  The `source` tag is
    properly updated in the coverage XML file.

    python renameHTML.py --directory <html directory> --source <code root directory>

    Examples:
      python renameHTML.py -d C:\Users\User\AppData\Roaming\CI\Reports\develop -s C:\Users\User

    TODO This function is not perfect: site-packages remains in text.
    """

    # Parse parameters
    parser = argparse.ArgumentParser(description='Rename coverage HTML files.')
    parser.add_argument('--directory', '-d', help='HTML directory')
    parser.add_argument('--source', '-s', help='root directory of code files', default='')
    args = parser.parse_args()  # returns data from the options specified (echo)

    if not args.source:
        import ibllib
        # Default repository source for coverage
        relative_to = Path(ibllib.__file__).parent.parent
    else:
        relative_to = Path(args.source)

    _logger.info('removing source directory from HTML report')
    _logger.debug(f'HTML source directory = {args.directory}')
    _logger.debug(f'code source directory = {relative_to}')

    # Rename the HTML files for readability and to obscure the server's directory structure
    pattern = re.sub(r'^[a-zA-Z]:[/\\]|[/\\]|\.', '_', str(relative_to)) + '_'  # / -> _
    n_changed = 0
    for file in Path(args.directory).glob('*.html'):  # Open each html report file
        with open(file, 'r') as f:
            data = f.read()
        data = data.replace(pattern, '')  # Remove long paths in filename links
        data = data.replace(str(relative_to) + sep, '')  # Remove from text
        with open(file, 'w') as f:
            f.write(data)  # Write back into file
        file.rename(str(file).replace(pattern, ''))  # Rename file
        n_changed += 1

    # Add the source path to the XML coverage report
    xml_file = Path(args.directory) / 'CoverageResults.xml'
    if xml_file.exists():
        with open(xml_file, 'r') as f:
            data = f.read()
        # Replace filename values
        data = data.replace(str(relative_to) + sep, '')
        # Replace name values
        pattern = re.sub(r'^[a-zA-Z]:[/\\]|[/\\]', '.', str(relative_to)) + '.'
        pattern = pattern[int(relative_to.anchor == '/'):]
        data = data.replace(pattern, '')
        # data = data.replace('<package name=".', '<package name="')  # Remove starting period
        # Inject source into tag
        src_tag = '<source>'
        i = data.index(src_tag) + len(src_tag)
        if data[i] == '<':
            data = data[:i] + str(relative_to) + data[i:]
        with open(xml_file, 'w') as f:
            f.write(data)  # Write back into file
        n_changed += 1

    _logger.debug(f'{n_changed} files changed')
