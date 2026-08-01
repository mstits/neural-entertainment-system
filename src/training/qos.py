"""macOS QoS helpers — push housekeeping threads onto the E-cores.

XNU has no thread-to-core affinity; core selection follows Quality of
Service classes. QOS_CLASS_BACKGROUND schedules the calling thread on
the 4 Efficiency cores exclusively, keeping P-core caches and pipelines
for solver workers (receipted driver: the 610 MB archive-flush pickles
that starved workers from 1563 to 322 sps until flush cadence was
capped). Call `demote_current_thread()` as the FIRST statement of any
housekeeping thread: archive flush/pickle, video encode, heatmap
render, spectator tick.

No-op (returns False) on non-Darwin platforms or lookup failure.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import sys

QOS_CLASS_BACKGROUND = 0x09
QOS_CLASS_UTILITY = 0x11


def demote_current_thread(qos_class: int = QOS_CLASS_BACKGROUND) -> bool:
    """Set the calling thread's QoS class. Returns True on success."""
    if sys.platform != "darwin":
        return False
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        fn = libc.pthread_set_qos_class_self_np
        fn.argtypes = [ctypes.c_int, ctypes.c_int]
        fn.restype = ctypes.c_int
        return fn(qos_class, 0) == 0
    except Exception:
        return False
