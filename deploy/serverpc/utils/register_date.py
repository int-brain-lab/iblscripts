# Per dataset type
import sys
import datetime
from pathlib import Path

from ibllib.io import flags

dry = True

# NB: make sure the path is relative to session root !
date_range = ['2019-06-14', '2019-06-16']
dtype = None

if sys.platform == 'linux':
    main_path = '/mnt/s0/Data/Subjects'
else:
    main_path = r'C:\globus_server\Subjects'

# compute the date range including both bounds
start = datetime.datetime.strptime(date_range[0], '%Y-%m-%d').date()
end = datetime.datetime.strptime(date_range[1], '%Y-%m-%d').date()
date_range = [start + datetime.timedelta(days=x) for x in range(0, (end - start).days + 1)]

for dat in date_range:
    for fol in Path(main_path).rglob(r"**/" + str(dat) + r"/*"):
        print(fol)
        if not dry:
            flags.create_register_flags(fol, force=True, file_list=dtype)

# /home/ibladmin/Documents/PYTHON/iblscripts/deploy/serverpc/crontab/02_register.sh
