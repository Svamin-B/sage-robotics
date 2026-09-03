"""Constant-memory timing summaries and optional Pi health telemetry."""

import importlib.metadata
import platform
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np
import psutil


def command_output(arguments):
    try:
        result = subprocess.run(arguments, capture_output=True, text=True, timeout=1, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def read_text(path):
    try:
        return Path(path).read_text().strip().strip("\x00")
    except OSError:
        return None


def environment():
    versions = {"opencv": cv2.__version__, "numpy": np.__version__, "psutil": psutil.__version__}
    try:
        versions["picamera2"] = importlib.metadata.version("picamera2")
    except importlib.metadata.PackageNotFoundError:
        versions["picamera2"] = None
    return {"platform": platform.platform(), "machine": platform.machine(),
            "python": platform.python_version(), "versions": versions,
            "os_release": read_text("/etc/os-release"),
            "board": read_text("/proc/device-tree/model"),
            "total_ram_bytes": psutil.virtual_memory().total,
            "git_revision": command_output(["git", "rev-parse", "HEAD"]),
            "git_status": command_output(["git", "status", "--porcelain"])}


class MeanMax:
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.maximum = None

    def add(self, value):
        if value is not None:
            self.count += 1
            self.total += value
            self.maximum = value if self.maximum is None else max(self.maximum, value)

    def result(self):
        return {"samples": self.count, "mean": self.total / self.count if self.count else None,
                "max": self.maximum}


class Health:
    def __init__(self):
        self.process = psutil.Process()
        self.vcgencmd = shutil.which("vcgencmd")
        self.process.cpu_percent()
        psutil.cpu_percent()

    def sample(self):
        temperature = read_text("/sys/class/thermal/thermal_zone0/temp")
        try:
            temperature = float(temperature) / 1000 if temperature is not None else None
        except ValueError:
            temperature = None
        return {"monotonic_ns": time.monotonic_ns(),
                "process_cpu_percent": self.process.cpu_percent(),
                "system_cpu_percent": psutil.cpu_percent(),
                "process_rss_bytes": self.process.memory_info().rss,
                "system_available_ram_bytes": psutil.virtual_memory().available,
                "temperature_c": temperature,
                "throttled": command_output([self.vcgencmd, "get_throttled"]) if self.vcgencmd else None}


def sensor_age_ms(sensor_ns):
    if sensor_ns is None or not hasattr(time, "CLOCK_BOOTTIME"):
        return None
    age = (time.clock_gettime_ns(time.CLOCK_BOOTTIME) - sensor_ns) / 1e6
    return age if age >= 0 else None

