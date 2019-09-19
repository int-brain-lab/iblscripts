::   first step is to update the script
cd C:\iblscripts\deploy\ephyspc
git fetch --all
git checkout -f master
git reset --hard
git pull
::   second step is to update ibllib
conda activate iblenv
pip install ibllib --upgrade
pip install PyQt5
:: at last run the script
python TTL_checklist.py
