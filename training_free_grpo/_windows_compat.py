"""
Windows multiprocessing compatibility shim.

This module MUST be imported before any other imports that might use multiprocessing.
It completely disables multiprocessing on Windows to prevent handle errors.
"""
import os
import sys

if os.name == 'nt':  # Windows only
    print("[Windows Compatibility] Disabling multiprocessing to prevent handle errors...")

    # Monkey-patch multiprocessing module before it's used
    import multiprocessing
    import multiprocessing.spawn

    # Set spawn method
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # Override Process class to prevent creation
    _original_Process = multiprocessing.Process

    class DisabledProcess:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "Multiprocessing.Process is disabled on Windows to prevent handle errors. "
                "Please use threading instead, or set UTU_ALLOW_MULTIPROCESSING=1 if you "
                "really need it (not recommended)."
            )

    # Only disable if not explicitly allowed
    if os.getenv('UTU_ALLOW_MULTIPROCESSING', '0') != '1':
        multiprocessing.Process = DisabledProcess
        # Also patch spawn module
        multiprocessing.spawn.Process = DisabledProcess

    # Disable ProcessPoolExecutor
    from concurrent.futures import ProcessPoolExecutor as _OriginalProcessPoolExecutor
    from concurrent.futures import ThreadPoolExecutor

    class SafeProcessPoolExecutor(ThreadPoolExecutor):
        """On Windows, redirect ProcessPoolExecutor to ThreadPoolExecutor"""
        def __init__(self, *args, **kwargs):
            print("[Windows Compatibility] ProcessPoolExecutor redirected to ThreadPoolExecutor")
            # Map max_workers parameter
            if 'max_workers' not in kwargs and len(args) > 0:
                kwargs['max_workers'] = args[0]
                args = args[1:]
            super().__init__(**kwargs)

    # Replace ProcessPoolExecutor globally
    import concurrent.futures
    concurrent.futures.ProcessPoolExecutor = SafeProcessPoolExecutor

    # Also patch in sys.modules for already-imported modules
    sys.modules['concurrent.futures'].ProcessPoolExecutor = SafeProcessPoolExecutor

    print("[Windows Compatibility] Multiprocessing disabled successfully")
