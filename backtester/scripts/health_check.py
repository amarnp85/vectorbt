#!/usr/bin/env python3
"""Convenience script for running data health check.

This is a simple wrapper that imports and runs the data health check module.
Users can run this directly or use the module form:

Direct script:
    python backtester/scripts/health_check.py --auto-fix

Module form:
    python -m backtester.data.health_check.data_healthcheck --auto-fix
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    # Import and run the main health check module
    from backtester.data.health_check.data_healthcheck import main
    exit(main()) 