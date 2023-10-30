#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py
from pathlib import Path
from typing import Dict, List


class BuildPyCommand(build_py):
    """Check for existence of XSLT before building."""

    def run(self):
        xslt = (
            Path(__file__).parent
            / "aicsimageio/metadata/czi-to-ome-xslt/xslt/czi-to-ome.xsl"
        )
        if not xslt.is_file():
            raise FileNotFoundError("XSLT not found. Is the submodule checked out?")
        build_py.run(self)


with open("README.md") as readme_file:
    readme = readme_file.read()


# If you need to update a version pin for one of these supporting format libs,
# be sure to also check if you need to update these versions in the
# "READER_TO_INSTALL" lookup table from aicsimageio/formats.py.
format_libs: Dict[str, List[str]] = {
    "base-imageio": [
        "imageio[ffmpeg]>=2.11.0,<2.28.0",
        "Pillow>=9.3.0",
    ],
    "nd2": ["nd2[legacy]>=0.6.0"],
    "dv": ["mrc>=0.2.0"],
    "bfio": ["bfio==2.3.0", "tifffile<2022.4.22"],
    # "czi": [  # excluded for licensing reasons
    #     "fsspec>=2022.8.0",
    #     "aicspylibczi>=3.1.1",
    # ],
    # "bioformats": ["bioformats_jar"],  # excluded for licensing reasons
    # "lif": ["readlif>=0.6.4"],  # excluded for licensing reasons
}

all_formats: List[str] = []
for deps in format_libs.values():
    for dep in deps:
        all_formats.append(dep)

setup_requirements = [
    "pytest-runner>=5.2",
]

test_requirements = [
    "dask[array,distributed]>=2021.4.1,!=2022.5.1",
    "docutils>=0.10,<0.16",
    "psutil>=5.7.0",
    "pytest>=5.4.3",
    "pytest-cov>=2.9.0",
    "pytest-raises>=0.11",
    "quilt3",  # no pin to avoid pip cycling (boto is really hard to manage)
    "s3fs[boto3]>=2022.11.0",
    "tox==3.27.1",
]

dev_requirements = [
    *setup_requirements,
    *test_requirements,
    "asv>=0.4.2",
    "black>=22.3.0",
    "bump2version>=1.0.1",
    "coverage>=5.1",
    "flake8>=3.8.3",
    "flake8-debugger>=3.2.1",
    "gitchangelog>=3.0.4",
    "ipython>=7.15.0",
    "isort>=5.11.5",
    "m2r2>=0.2.7",
    "mypy>=0.800",
    "pytest-runner>=5.2",
    "Sphinx>=3.4.3",
    "sphinx_rtd_theme>=0.5.1",
    "twine>=3.1.1",
    "types-PyYAML>=6.0.12.9",
    "wheel>=0.34.2",
    # reader deps
    *all_formats,
    "bioformats_jar",  # to test bioformats
    "bfio>=2.3.0",
    "readlif>=0.6.4",  # to test lif
    "aicspylibczi>=3.1.1",  # to test czi
]

benchmark_requirements = [
    *dev_requirements,
    "dask-image>=0.6.0",
]

requirements = [
    "dask[array]>=2021.4.1",
    # fssspec restricted due to glob issue tracked here, when fixed remove ceiling
    # https://github.com/fsspec/filesystem_spec/issues/1380
    "fsspec>=2022.8.0,<2023.9.0",
    "imagecodecs>=2020.5.30",
    "lxml>=4.6,<5",
    "numpy>=1.21.0",
    "ome-types>=0.3.4",
    "ome-zarr>=0.6.1",
    "PyYAML>=6.0",
    "wrapt>=1.12",
    "resource-backed-dask-array>=0.1.0",
    "tifffile>=2021.8.30,<2023.3.15",
    "xarray>=0.16.1",
    "xmlschema",  # no pin because it's pulled in from OME types
    "zarr>=2.6,<2.16.0",
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
    author="Eva Maxfield Brown, Allen Institute for Cell Science",
    author_email="evamaxfieldbrown@gmail.com, jamie.sherman@gmail.com, bowdenm@spu.edu",
    cmdclass={"build_py": BuildPyCommand},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
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
    packages=find_packages(
        exclude=[
            "tests",
            "*.tests",
            "*.tests.*",
            "benchmarks",
            "*.benchmarks",
            "*.benchmarks.*",
        ]
    ),
    python_requires=">=3.9",
    setup_requires=setup_requirements,
    test_suite="aicsimageio/tests",
    tests_require=test_requirements,
    extras_require=extra_requirements,
    url="https://github.com/AllenCellModeling/aicsimageio",
    # Do not edit this string manually, always use bumpversion
    # Details in CONTRIBUTING.md
    version="4.14.0",
    zip_safe=False,
)
