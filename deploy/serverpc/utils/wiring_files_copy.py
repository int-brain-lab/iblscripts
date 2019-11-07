from pathlib import Path
import shutil

from ibllib.io import spikeglx

PROBE_TYPE = '3A'
DRY = True

ses_paths = ['/mnt/s0/Data/Subjects/CSHL050/2019-10-31/001',
             '/mnt/s0/Data/Subjects/CSHL050/2019-10-30/001',
             '/mnt/s0/Data/Subjects/CSHL050/2019-10-29/001',
             '/mnt/s0/Data/Subjects/CSHL021/2019-10-01/001']

wnidq = Path('/home/ibladmin/Documents/PYTHON/iblscripts/deploy/ephyspc/wirings/nidq.wiring.json')
w3b = Path('/home/ibladmin/Documents/PYTHON/iblscripts/deploy/ephyspc/wirings/3B.wiring.json')
w3a = Path('/home/ibladmin/Documents/PYTHON/iblscripts/deploy/ephyspc/wirings/3A.wiring.json')

# end of parameter section

if PROBE_TYPE == '3A':
    pfile = w3a
elif PROBE_TYPE == '3B':
    pfile = w3b

for ses_path in ses_paths:
    ses_path = Path(ses_path)
    efiles = spikeglx.glob_ephys_files(ses_path)
    for ef in efiles:
        if ef.get('ap'):
            fil = ef.get('ap')
            src = pfile
            dst = fil.name.replace('ap.bin', 'wiring.json')
        if ef.get('nidq'):
            fil = ef.get('nidq')
            src = wnidq
            dst = fil.name.replace('nidq.bin', 'wiring.json')
        print(src, fil.parent.joinpath(dst))
        if not DRY:
            shutil.copy(src, fil.parent.joinpath(dst))
