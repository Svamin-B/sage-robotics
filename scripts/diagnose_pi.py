"""Print setup information without opening the camera or changing the system."""

import importlib.metadata
import importlib.util
import json
import platform
from pathlib import Path
import subprocess
import sys


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=10)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"unavailable": str(error)}


def main():
    packages = {}
    for name in ("picamera2", "cv2", "numpy", "psutil"):
        packages[name] = {"import_available": importlib.util.find_spec(name) is not None}
    report = {"python": sys.version, "architecture": platform.machine(),
              "os_release": Path("/etc/os-release").read_text() if Path("/etc/os-release").exists() else None,
              "packages": packages, "rpicam_version": command(["rpicam-hello", "--version"]),
              "throttled": command(["vcgencmd", "get_throttled"]),
              "temperature": command(["vcgencmd", "measure_temp"])}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
