"""
Core module containing shared utilities and base classes
"""

from .constants import *
from .base import *
from .utils import *

__all__ = [
    'CONSTANTS',
    'BaseClass',
    'ArrayType',
    'validate_array',
    'validate_bounds'
] 