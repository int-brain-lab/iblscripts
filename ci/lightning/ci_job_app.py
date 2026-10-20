import lightning as L
from lightning.app.components.python import PythonScript


class IntegrationTestJob(L.LightningWork):
    def __init__(self, python_version: str, s3_path: str, *args, **kwargs):
        # Define the container environment
        super().__init__(
            *args, 
            **kwargs,
            # 1. HARDWARE: Request GPU if needed, else CPU
            cloud_compute=L.CloudCompute("cpu", idle_timeout=60), # "gpu", "gpu-fast", "a100" etc.
            # 2. REQUIREMENTS: Install deps at build time (faster startup)
            requirements=[
                "pytest", "pytest-cov", "coverage", "s3fs", "fuse3", "boto3", 
                "my-package @ git+https://github.com/org/repo.git@branch" # Your code
            ],
            # 3. MOUNTS: Native S3 FUSE Mount (Best Performance)
            # This mounts BEFORE the script runs. No code changes needed in run_tests.py for mounting.
            mounts=[
                L.Mount(
                    source=s3_path,      # "s3://my-bucket/test-data/"
                    mount_path="/data",  # Accessible at /data in container
                    # auth handled by AWS env vars passed to Job
                )
            ],
        )
        self.python_version = python_version
        self.s3_path = s3_path

    def run(self):
        # The PythonScript component runs the file. 
        # Env vars are passed via `env` in the spec or set here.
        import os
        os.environ["PY_VERSION"] = self.python_version
        os.environ["S3_DATA_PATH"] = self.s3_path
        # LIGHTNING_ARTIFACTS_DIR is auto-injected by Lightning
        
        # Execute the script
        # Note: PythonScript handles the subprocess execution
        script = PythonScript(
            script_path="run_tests.py", 
            # args=["--arg1", "val"] # if needed
        )
        script.run()


# The App just holds the work
class CIApp(L.LightningFlow):
    def __init__(self, python_version, s3_path):
        super().__init__()
        self.job = IntegrationTestJob(python_version, s3_path)

    def run(self):
        self.job.run()


if __name__ == "__main__":
    # Entrypoint for `lightning run app ci_job_app.py --arg=val`
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--python_version", type=str, default="3.12")
    parser.add_argument("--s3_path", type=str, default="s3://my-bucket/test-data/")
    args = parser.parse_args()
    
    app = CIApp(args.python_version, args.s3_path)
    L.LightningApp(app)
