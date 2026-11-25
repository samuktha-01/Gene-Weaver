"""System profiling and memory measurement utilities for GeneWeaver CPU benchmarks.

Provides cross-platform memory tracking (distinguishing Python heap allocation via
tracemalloc from OS-level process RSS memory), hardware CPU detection, and dependency
version discovery. Gracefully handles environments where optional dependencies like
psutil are absent.
"""

import ctypes
import os
import platform
import sys
import tracemalloc
from typing import Optional, Tuple

from benchmarks.models import EnvironmentInfo, MemoryMetrics


class SystemProfiler:
    """Hardware environment detection and process memory profiling."""

    _has_psutil: Optional[bool] = None

    @classmethod
    def check_psutil(cls) -> bool:
        """Determines if psutil is available in the runtime environment."""
        if cls._has_psutil is None:
            try:
                import psutil  # type: ignore # noqa: F401
                cls._has_psutil = True
            except ImportError:
                cls._has_psutil = False
        return cls._has_psutil

    @classmethod
    def get_process_memory_mb(cls) -> Tuple[float, float, str]:
        """Returns current process (RSS_mb, VMS_mb, method_name).

        Attempts psutil first. If unavailable, falls back to Windows Win32 API
        GetProcessMemoryInfo, Unix resource module, or returns zeros if unavailable.
        """
        if cls.check_psutil():
            try:
                import psutil  # type: ignore
                proc = psutil.Process()
                mem = proc.memory_info()
                rss_mb = mem.rss / (1024.0 * 1024.0)
                vms_mb = mem.vms / (1024.0 * 1024.0)
                return rss_mb, vms_mb, "psutil"
            except Exception:
                pass

        # Windows ctypes fallback if psutil is unavailable
        if platform.system() == "Windows":
            try:
                class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
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

                kernel32 = ctypes.windll.kernel32
                kernel32.GetCurrentProcess.restype = ctypes.c_void_p
                handle = kernel32.GetCurrentProcess()

                psapi = ctypes.windll.psapi
                psapi.GetProcessMemoryInfo.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
                    ctypes.c_ulong,
                ]
                psapi.GetProcessMemoryInfo.restype = ctypes.c_bool

                counters = PROCESS_MEMORY_COUNTERS()
                counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
                if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                    rss_mb = counters.WorkingSetSize / (1024.0 * 1024.0)
                    vms_mb = counters.PagefileUsage / (1024.0 * 1024.0)
                    return rss_mb, vms_mb, "win32_ctypes"
            except Exception:
                pass

        # Unix resource module fallback
        try:
            import resource  # type: ignore
            # maxrss is in kilobytes on Linux, bytes on macOS
            rusage = resource.getrusage(resource.RUSAGE_SELF)
            max_rss = rusage.ru_maxrss
            if platform.system() == "Darwin":
                rss_mb = max_rss / (1024.0 * 1024.0)
            else:
                rss_mb = max_rss / 1024.0
            return rss_mb, rss_mb, "resource"
        except Exception:
            pass

        return 0.0, 0.0, "unavailable"

    @classmethod
    def get_environment_info(cls) -> EnvironmentInfo:
        """Discovers host hardware and software environment details."""
        logical_cores = os.cpu_count() or 1
        physical_cores = None
        total_ram_gb = 0.0

        if cls.check_psutil():
            try:
                import psutil  # type: ignore
                physical_cores = psutil.cpu_count(logical=False)
                total_ram_gb = psutil.virtual_memory().total / (1024.0 ** 3)
            except Exception:
                pass

        # Windows total RAM fallback if psutil is unavailable
        if total_ram_gb <= 0.0 and platform.system() == "Windows":
            try:
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                    total_ram_gb = stat.ullTotalPhys / (1024.0 ** 3)
            except Exception:
                pass

        # NumPy version
        numpy_ver = ""
        try:
            import numpy as np
            numpy_ver = np.__version__
        except Exception:
            numpy_ver = "unknown"

        # Processor description
        processor_str = platform.processor() or "Unknown Processor"
        if platform.system() == "Windows":
            try:
                import winreg
                key = winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
                )
                val, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                if val:
                    processor_str = str(val).strip()
            except Exception:
                pass

        return EnvironmentInfo(
            python_version=sys.version.split()[0],
            python_compiler=platform.python_compiler(),
            os_system=platform.system(),
            os_release=platform.release(),
            os_machine=platform.machine(),
            processor=processor_str,
            cpu_count_logical=logical_cores,
            cpu_count_physical=physical_cores,
            total_ram_gb=total_ram_gb,
            numpy_version=numpy_ver,
            geneweaver_version="0.1.0",
        )


class MemoryTracker:
    """Context manager for tracing Python heap allocation and OS process memory."""

    def __init__(self) -> None:
        self.python_peak_mb: float = 0.0
        self.process_rss_mb: float = 0.0
        self.process_vms_mb: float = 0.0
        self.method_name: str = "tracemalloc"

    def __enter__(self) -> "MemoryTracker":
        tracemalloc.start()
        tracemalloc.reset_peak()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.python_peak_mb = peak_bytes / (1024.0 * 1024.0)

        rss_mb, vms_mb, method = SystemProfiler.get_process_memory_mb()
        self.process_rss_mb = rss_mb
        self.process_vms_mb = vms_mb
        self.method_name = f"tracemalloc+{method}" if method != "unavailable" else "tracemalloc_only"

    def get_metrics(self) -> MemoryMetrics:
        """Returns populated MemoryMetrics container."""
        return MemoryMetrics(
            python_peak_memory_mb=self.python_peak_mb,
            process_memory_rss_mb=self.process_rss_mb,
            process_memory_vms_mb=self.process_vms_mb,
            measurement_method=self.method_name,
        )
