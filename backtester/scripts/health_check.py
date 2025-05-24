#!/usr/bin/env python3
"""Streamlined Health Check Script Wrapper

Simple wrapper for the streamlined data health check module.
This script provides easy access to the health checker from the scripts directory.

Examples:
    # Quick health check
    python backtester/scripts/health_check.py
    
    # Auto-fix critical issues
    python backtester/scripts/health_check.py --auto-fix
    
    # Check specific exchange with details
    python backtester/scripts/health_check.py --exchange binance --detailed
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    # Import and run the streamlined health check module
    from backtester.data.health_check.data_healthcheck import main
    exit(main()) 