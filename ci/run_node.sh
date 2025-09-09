cd /home/experiment/Documents/PYTHON/ci/LabCI
DEBUG=*ci* node -r dotenv/config main.js 2>&1 | tee /var/log/ibllib-ci.log
