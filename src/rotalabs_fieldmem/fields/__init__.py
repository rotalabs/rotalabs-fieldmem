"""Field implementations for FTMS.

Provides PDE-based memory fields with diffusion, decay, and thermal noise.
"""

from rotalabs_fieldmem.fields.memory_field import MemoryField, FieldConfig

__all__ = [
    "MemoryField",
    "FieldConfig",
]

# Optional: sparse field (may require additional dependencies)
try:
    from rotalabs_fieldmem.fields.sparse import SparseMemoryField
    __all__.append("SparseMemoryField")
except ImportError:
    pass
