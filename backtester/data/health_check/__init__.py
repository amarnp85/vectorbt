"""Streamlined Data Health Check Module

Provides focused data quality analysis and automated fixing capabilities
for VBT cached cryptocurrency data.

Key Features:
- Critical gap detection with realistic thresholds
- Data freshness monitoring with timeframe-aware staleness
- Cache integrity validation
- Actionable fix recommendations using available CLI tools
- Auto-fix capabilities for critical issues
"""

from .data_healthcheck import StreamlinedHealthChecker

__all__ = ['StreamlinedHealthChecker'] 