#### Entry point
* Normal run: `python fiber_photometry_form.py`
* Test mode: `python fiber_photometry_form.py --test`

#### Parameters
Configurable default data and backup directories: `fp_params.yml`

#### Requirements
Requirement file for python environment: `fp_requirements.txt`
* `pip install fp_requirements.txt`

---
#### Development machine details:
- Ubuntu 22.04
- Anaconda 4.13.0
- Python 3.8
- opencv-python 4.3.0.36
- PyQt5 5.15.7
- ibllib v2.15

---
#### TODO:
- add FP3002Config.01.xml data to parquet metadata
- flesh out placeholder try/except blocks with additional messaging
- better implementation for testing and folder creation
- add ability to select number of patch cord end points