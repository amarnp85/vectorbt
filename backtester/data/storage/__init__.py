"""
Storage Module

This module contains data storage functionality for the backtester system:

VBT Data Storage:
- data_storage: VBT-native storage using VectorBT Pro's built-in pickle persistence
- DataStorage: Class for VBT data storage instances

Uses VectorBT Pro's native pickle storage for complete metadata preservation
and optimal VBT functionality retention.

For information on the VBT-native approach, see README_VBT_NATIVE.md
"""

# VBT Data storage
from .data_storage import data_storage, DataStorage

__all__ = ["data_storage", "DataStorage"]
