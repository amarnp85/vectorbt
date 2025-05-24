#!/usr/bin/env python3
# setup.py
"""
Setup script for the backtester package.
"""

from setuptools import setup, find_packages

setup(
    name="backtester",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "vectorbtpro",
        "ccxt",
        "pandas",
        "numpy",
    ],
    entry_points={
        'console_scripts': [
            'exchange-info=backtester.utilities.exchange_info:main',
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="Backtesting framework using VectorBT Pro",
    keywords="backtesting, trading, cryptocurrency, vectorbt",
    python_requires=">=3.8",
)
