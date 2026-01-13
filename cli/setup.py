#!/usr/bin/env python3
"""
KERNELIZE CLI Setup and Installation
Setup script for CLI tool installation and configuration
"""

from setuptools import setup, find_packages
import os
import sys

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    requirements = []
    with open("requirements.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                requirements.append(line)
    return requirements

# CLI-specific requirements
cli_requirements = [
    "click>=8.0.0",
    "rich>=12.0.0",
    "requests>=2.28.0",
    "aiohttp>=3.8.0",
    "pydantic>=1.10.0",
    "structlog>=22.0.0",
    "python-dateutil>=2.8.0",
    "urllib3>=1.26.0",
]

setup(
    name="kernelize-cli",
    version="1.0.0",
    author="KERNELIZE Team",
    author_email="dev@kernelize.com",
    description="KERNELIZE Command-Line Interface for compression and kernel operations",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/kernelize/cli",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=cli_requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "myst-parser>=0.18.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "kernelize=kernelize_cli.cli:cli",
            "kz=kernelize_cli.cli:cli",  # Short alias
        ],
    },
    include_package_data=True,
    package_data={
        "kernelize_cli": [
            "data/*",
            "templates/*",
            "examples/*",
        ],
    },
    keywords="compression kernel machine-learning ai data-processing",
    project_urls={
        "Bug Reports": "https://github.com/kernelize/cli/issues",
        "Source": "https://github.com/kernelize/cli",
        "Documentation": "https://docs.kernelize.com/cli",
    },
)