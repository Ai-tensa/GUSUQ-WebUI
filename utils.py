import gc
import platform
import ctypes
import psutil
import torch


def round32(x):
    return int(round(x / 32.0) * 32)


def release_memory_resources():
    """memory release helper"""
    # collect garbage
    gc.collect()

    # CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # heap shrink
    sysname = platform.system()
    try:
        if sysname == "Linux":
            ctypes.CDLL("libc.so.6").malloc_trim(0)

        elif sysname == "Darwin":
            ctypes.CDLL("libc.dylib").malloc_zone_pressure_relief(0, 0)

        elif sysname == "Windows":
            # Not needed for Windows?
            # psapi = ctypes.WinDLL("psapi")
            # psapi.EmptyWorkingSet(ctypes.windll.kernel32.GetCurrentProcess())
            pass
    except OSError:
        pass


def rss_mb():
    return psutil.Process().memory_info().rss // 1024 // 1024
