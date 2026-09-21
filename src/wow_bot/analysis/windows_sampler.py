"""Windows resource sampler using stdlib ctypes and Win32 APIs for WoW-bot soak harness."""

from __future__ import annotations

import ctypes
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from wow_bot.analysis.lab_soak_v2 import ResourceSnapshot


class FILETIME(ctypes.Structure):
    """Win32 FILETIME structure containing 64-bit timestamp in 100-nanosecond intervals."""

    _fields_ = [
        ("dwLowDateTime", ctypes.c_ulong),
        ("dwHighDateTime", ctypes.c_ulong),
    ]


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    """Win32 PROCESS_MEMORY_COUNTERS structure for process memory statistics."""

    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


class WindowsResourceSampler:
    """ResourceSampler implementation capturing current process RSS, CPU, and log size on Windows."""

    _get_current_process: Callable[[], Any]
    _get_process_times: Callable[..., Any]
    _get_process_memory_info: Callable[..., Any]

    def __init__(self, *, log_path: Path | None = None) -> None:
        if sys.platform != "win32" or not hasattr(ctypes, "WinDLL"):
            raise RuntimeError("WindowsResourceSampler is only available on Windows")

        self._log_path: Path | None = log_path
        self._prev_cpu_time_s: float | None = None
        self._prev_wall_time_s: float | None = None

        try:
            win_dll = getattr(ctypes, "WinDLL")  # noqa: B009
            kernel32 = win_dll("kernel32", use_last_error=True)
            psapi = win_dll("psapi", use_last_error=True)

            self._get_current_process = kernel32.GetCurrentProcess
            self._get_current_process.restype = ctypes.c_void_p
            self._get_current_process.argtypes = []

            self._get_process_times = kernel32.GetProcessTimes
            self._get_process_times.restype = ctypes.c_int
            self._get_process_times.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(FILETIME),
                ctypes.POINTER(FILETIME),
                ctypes.POINTER(FILETIME),
                ctypes.POINTER(FILETIME),
            ]

            self._get_process_memory_info = psapi.GetProcessMemoryInfo
            self._get_process_memory_info.restype = ctypes.c_int
            self._get_process_memory_info.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
                ctypes.c_ulong,
            ]
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize Windows API functions: {exc}") from exc

        # Verify GetProcessMemoryInfo and GetProcessTimes are functional on construction
        h_process = self._get_current_process()
        pmc = PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        if not self._get_process_memory_info(h_process, ctypes.byref(pmc), pmc.cb):
            raise RuntimeError("GetProcessMemoryInfo is unavailable or failed during initialization")

        creation = FILETIME()
        exit_time = FILETIME()
        kernel = FILETIME()
        user = FILETIME()
        if not self._get_process_times(
            h_process,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            raise RuntimeError("GetProcessTimes is unavailable or failed during initialization")

    def sample(self, now: float) -> ResourceSnapshot:
        """Sample current process resource usage at given timestamp `now`."""
        h_process = self._get_current_process()

        pmc = PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        if not self._get_process_memory_info(h_process, ctypes.byref(pmc), pmc.cb):
            raise RuntimeError("GetProcessMemoryInfo failed during sample")

        rss_bytes = int(pmc.WorkingSetSize)

        creation = FILETIME()
        exit_time = FILETIME()
        kernel = FILETIME()
        user = FILETIME()
        if not self._get_process_times(
            h_process,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            raise RuntimeError("GetProcessTimes failed during sample")

        kernel_100ns = (kernel.dwHighDateTime << 32) | kernel.dwLowDateTime
        user_100ns = (user.dwHighDateTime << 32) | user.dwLowDateTime
        cpu_time_s = (kernel_100ns + user_100ns) * 1e-7

        if self._prev_cpu_time_s is None or self._prev_wall_time_s is None:
            cpu_percent = 0.0
        else:
            delta_cpu = cpu_time_s - self._prev_cpu_time_s
            delta_wall = now - self._prev_wall_time_s
            if delta_wall > 0.0:
                cpu_percent = (delta_cpu / delta_wall) * 100.0
            else:
                cpu_percent = 0.0

        self._prev_cpu_time_s = cpu_time_s
        self._prev_wall_time_s = now

        cpu_count = os.cpu_count()
        max_cpu_percent = 100.0 * float(cpu_count) if cpu_count is not None else 100.0
        cpu_percent = max(0.0, min(cpu_percent, max_cpu_percent))

        log_size_bytes = 0
        if self._log_path is not None and self._log_path.exists():
            log_size_bytes = self._log_path.stat().st_size

        return ResourceSnapshot(
            ts=now,
            cpu_percent=cpu_percent,
            rss_bytes=rss_bytes,
            log_size_bytes=log_size_bytes,
        )


__all__ = ["WindowsResourceSampler"]
