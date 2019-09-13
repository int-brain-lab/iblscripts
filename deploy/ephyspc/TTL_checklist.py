from pathlib import Path

from PyQt5.QtWidgets import QFileDialog
import ibllib.ephys.ephysqc as ephysqc

session_path = Path(QFileDialog.getExistingDirectory(None, "Select Directory"))

if session_path.exists():
    ephysqc.validate_ttl_test(session_path, display=True)
