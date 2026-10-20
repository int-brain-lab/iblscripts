"""
2.  GitHub Secrets (Repo Settings → Secrets → Actions):
    *   LIGHTNING_CREDENTIALS: Content of ~/.lightning/credentials.yaml (base64 encoded is safer: cat ~/.lightning/credentials.yaml | base64 -w0).
    *   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY: For S3 access (if bucket isn't public/no-requester-pays).
    *   COVERALLS_REPO_TOKEN: From Coveralls.io repo settings.
    *   GITHUB_TOKEN: Auto-provided by Actions (needs contents: read, actions: read usually default).
"""

# run_tests.py
import os
import subprocess
import sys
import lightning as L
from lightning.app.storage import Drive # For persistent cache (optional)
from lightning.app.storage.path import Path as LPath

# 1. CONFIGURATION (Injected via Env Vars by Dispatcher)
S3_DATA_PATH = os.environ.get("S3_DATA_PATH", "s3://my-bucket/test-data/")
MOUNT_POINT = "/data" # Where dataset appears locally
PYTHON_VERSION = os.environ.get("PY_VERSION", "3.10") # For labeling artifacts
ARTIFACT_DIR = os.environ.get("LIGHTNING_ARTIFACTS_DIR", "/artifacts") # Lightning injects this

def main():
    print(f"🚀 Starting Integration Tests (Python {PYTHON_VERSION})")
    print(f"📦 Mounting S3: {S3_DATA_PATH} -> {MOUNT_POINT}")

    # 2. MOUNT S3 (Lightning handles fsspec/s3fs auth via AWS env vars automatically)
    # This creates a local directory /data backed by S3. 
    # Requires: pip install s3fs (add to requirements.txt)
    fs = L.cloud_open(S3_DATA_PATH, "rb").fs # Trigger auth check / init
    
    # Actually mount for subprocess access (pytest needs POSIX paths)
    # Lightning Studios/Jobs support `cloud_mount` in spec, but doing it in code gives control:
    import fsspec
    fuse_mount = fsspec.filesystem("s3").mount(S3_DATA_PATH, MOUNT_POINT, background=True)
    # Note: For Jobs, the `mounts` config in LightningWork/Spec is cleaner (see Step 2). 
    # If using raw `lightning run app`, use the `mounts` parameter below.

    # 3. RUN PYTEST WITH COVERAGE
    # Output .coverage file to the Lightning Artifact Directory
    coverage_file = os.path.join(ARTIFACT_DIR, f".coverage.{PYTHON_VERSION}.{os.environ.get('HOSTNAME', 'worker')}")
    
    cmd = [
        "pytest", 
        "tests/integration", 
        f"--cov=my_package", 
        f"--cov-data-file={coverage_file}",
        "-v", 
        "--tb=short",
        # Pass S3 path to tests if they need it explicitly
        f"--s3-data-path={MOUNT_POINT}" 
    ]
    
    print(f"▶️ Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False) # Stream logs to Lightning UI
    
    # 4. VERIFY ARTIFACT
    if os.path.exists(coverage_file):
        print(f"✅ Coverage data saved to Lightning Artifacts: {coverage_file}")
        print(f"   Size: {os.path.getsize(coverage_file)} bytes")
    else:
        print(f"❌ ERROR: Coverage file not found at {coverage_file}")
        # List dir for debugging
        print(os.listdir(ARTIFACT_DIR))

    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
