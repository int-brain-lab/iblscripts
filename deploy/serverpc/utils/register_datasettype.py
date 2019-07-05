# Per dataset type
import sys
from pathlib import Path
from ibllib.io import flags

dry = False

# NB: make sure the path is relative to session root !
dtype = 'raw_video_data/_iblrig_leftCamera.raw.mp4'

if sys.platform == 'linux':
    main_path = '/mnt/s1/Data/Subjects'

else:
    main_path = r'C:\globus_server\Subjects'


for fil in Path(main_path).rglob(dtype):
    print(fil)
    session_path = Path(str(fil).replace(dtype, ''))
    print(session_path)
    if not dry:
        flags.create_register_flags(session_path, force=True, file_list=dtype)
