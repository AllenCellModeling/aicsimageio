#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup
from typing import Dict, List

with open("README.md") as readme_file:
    readme = readme_file.read()

format_libs: Dict[str, List[str]] = {
    "base-imageio": ["imageio[ffmpeg]~=2.9.0", "Pillow~=8.2.0,!=8.3.0"],
    "lif": ["readlif~=0.6.1"],
    "czi": ["aicspylibczi~=3.0.2"],
}

all_formats: List[str] = []
for deps in format_libs.values():
    for dep in deps:
        all_formats.append(dep)

setup_requirements = [
    "pytest-runner>=5.2",
]

test_requirements = [
    *all_formats,
    "black>=19.10b0",
    "codecov>=2.1.4",
    "distributed>=2021.4.1",
    "docutils>=0.10,<0.16",
    "flake8>=3.8.3",
    "flake8-debugger>=3.2.1",
    "isort>=5.7.0",
    "mypy>=0.800",
    "psutil>=5.7.0",
    "pytest>=5.4.3",
    "pytest-cov>=2.9.0",
    "pytest-raises>=0.11",
    "quilt3",  # no pin to avoid pip cycling (boto is really hard to manage)
    "s3fs[boto3]>=0.4.2",
]

dev_requirements = [
    *setup_requirements,
    *test_requirements,
    "asv>=0.4.2",
    "bump2version>=1.0.1",
    "coverage>=5.1",
    "gitchangelog>=3.0.4",
    "ipython>=7.15.0",
    "m2r2>=0.2.7",
    "pytest-runner>=5.2",
    "Sphinx>=3.4.3",
    "sphinx_rtd_theme>=0.5.1",
    "tox>=3.15.2",
    "twine>=3.1.1",
    "wheel>=0.34.2",
]

benchmark_requirements = [
    *dev_requirements,
    "dask-image~=0.6.0",
]

requirements = [
    "dask[array]>=2021.4.1",
    "fsspec>=2021.4.0",
    "imagecodecs>=2020.5.30",
    "numpy~=1.16",
    "ome-types~=0.2.4",
    "tifffile>=2021.2.1",
    "toolz~=0.11.0",
    "xarray~=0.16.1",
    "xmlschema",  # no pin because it's pulled in from OME types
    "zarr~=2.6.1",
]

extra_requirements = {
    "setup": setup_requirements,
    "test": test_requirements,
    "dev": dev_requirements,
    "benchmark": benchmark_requirements,
    **format_libs,
    "all": all_formats,
}

setup(
    author="Jackson Maxfield Brown, Allen Institute for Cell Science",
    author_email="jmaxfieldbrown@gmail.com, jamie.sherman@gmail.com, bowdenm@spu.edu",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    description=(
        "Image Reading, Metadata Conversion, and Image Writing for Microscopy Images "
        "in Pure Python"
    ),
    entry_points={},
    install_requires=requirements,
    license="BSD-3-Clause",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="imageio, image reading, image writing, metadata, microscopy, allen cell",
    name="aicsimageio",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*"]),
    python_requires=">=3.7",
    setup_requires=setup_requirements,
    test_suite="aicsimageio/tests",
    tests_require=test_requirements,
    extras_require=extra_requirements,
    url="https://github.com/AllenCellModeling/aicsimageio",
    # Do not edit this string manually, always use bumpversion
    # Details in CONTRIBUTING.md
    version="4.0.4",
    zip_safe=False,
)
