"""Data Health Check Module

Provides comprehensive data quality analysis and automated fixing capabilities
for VBT cached cryptocurrency data.

Key Features:
- Cached inception date integration (3,256+ symbols across exchanges)
- Comprehensive health checks (gaps, freshness, completeness, consistency)
- Auto-fix capabilities for critical issues
- Detailed reporting with actionable insights
"""

from .data_healthcheck import DataHealthChecker

__all__ = ['DataHealthChecker'] 