from __future__ import annotations
import gc, os
try:
    import torch
except ImportError:
    torch = None
try:
    import psutil
except ImportError:
    psutil = None

def process_rss_mb():
    if psutil is None:
        return None
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

def system_ram():
    info = {"rss_mb": process_rss_mb()}
    if psutil is not None:
        vm = psutil.virtual_memory()
        info.update({"total_mb": vm.total / (1024 * 1024), "available_mb": vm.available / (1024 * 1024), "percent": vm.percent})
    return info

def apply_ram_safe_mode(enabled: bool, cpu_threads: int = 2) -> dict:
    if enabled:
        threads = max(1, min(cpu_threads, 2))
        os.environ["OMP_NUM_THREADS"] = str(threads)
        if torch is not None:
            torch.set_num_threads(threads)
        gc.collect()
        return {"enabled": True, "cpu_threads": threads}
    return {"enabled": False, "cpu_threads": cpu_threads}
