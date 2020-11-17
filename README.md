# iblscripts

## Tests
The `ci/tests` folder contains the ibllib integration tests.  These require a dataset found on
 FlatIron, in `/integration`.  To run these tests...
 
 1. Download the integration test dataset from FlatIron vis Globus.
 2. Set the path to the integration folder by running the following in python:
    ```python
    from ibllib.io import params
    data_path = r'path\to\integration'
    params.write('ibl_ci', {'data_root': data_path})
    ```
 3. Run the tests with `python -m unittest discover -s "./ci/tests"` or
 `python ci/runAllTests.py -l path/to/log/output/dir`
 
 NB: `test_ephys_mtscomp/TestEphysCompression` and `test_ephys_pipeline/TestEphysPipeline` require
  admin privileges in order to create symbolic links on Windows.
