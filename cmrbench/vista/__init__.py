"""Native CMRBench VISTA mask generation.

The sampling registry exposes this generator through ``vista/cmrbench`` while
the generator itself remains independent of sampling-framework concerns.
"""

from cmrbench.vista.generator import generate_vista_mask

__all__ = ["generate_vista_mask"]
