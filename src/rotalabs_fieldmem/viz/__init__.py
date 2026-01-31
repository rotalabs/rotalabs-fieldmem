"""Visualization utilities for FTMS.

Provides field visualization (optional - requires matplotlib/seaborn).
"""

# Optional: visualization (requires matplotlib)
try:
    from rotalabs_fieldmem.viz.plotting import FieldVisualizer
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False

__all__ = [
    "HAS_VIZ",
]

if HAS_VIZ:
    __all__.append("FieldVisualizer")
