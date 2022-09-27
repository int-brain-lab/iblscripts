entry point: `fiber_photometry_form.py`

configurable default data directories defined in `fiber_photometry_form_util.py`

---
Development machine details:
- Ubuntu 22.04
- Anaconda 4.13.0
- Python 3.8
- opencv-python 4.3.0.36
- PyQt5 5.15.7
- ibllib v2.15

---
TODO:
- add FP3002Config.01.xml data to parquet metadata
- flesh out placeholder try/except blocks with additional messaging
- better implementation for testing and folder creation
- determine if number of available "FP3002Config.xml" files is something worth checking
- add ability to select number of patch cord end points