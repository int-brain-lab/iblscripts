# iblscripts

## Tests
The `ci/tests` folder contains the ibllib integration tests.  These require a dataset found on
 FlatIron, in `/integration`.  To run these tests...
 
 1. Run the `ci/setup.py` script and following the instructions to prepare Globus for downloading 
 the data
 2. Download the integration data by running the `ci/download_data.py` script.
 3. Run the tests with `python -m unittest discover -s "./ci/tests"` or
 `python ci/runAllTests.py -l path/to/log/output/dir`
 
 NB: `test_ephys_mtscomp/TestEphysCompression` and `test_ephys_pipeline/TestEphysPipeline` require
  admin privileges in order to create symbolic links on Windows.
