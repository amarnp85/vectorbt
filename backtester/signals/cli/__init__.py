"""CLI tools for signal operations.

This module provides command-line interfaces for:
- Signal diagnostics and troubleshooting
- Signal analysis and validation
"""

from .diagnose import main as diagnose_main

__all__ = [
    "diagnose_main",
] 