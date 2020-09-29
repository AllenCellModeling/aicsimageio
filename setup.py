#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

requirements = [
    "aicspylibczi>=2.5.0",
    "dask>=2.9.0",
    "distributed>=2.9.3",
    "numpy>=1.16",
    "imagecodecs>=2020.2.18",
    "imageio[ffmpeg]>=2.3.0",
    "readlif>=0.2.1",
    "lxml>=4.4.2",
    "tifffile>=2019.7.26.2",
    "toolz>=0.10.0",
]

setup_requirements = [
    "pytest-runner>=5.2",
]

test_requirements = [
    "black>=19.10b0",
    "codecov>=2.1.4",
    "docutils>=0.10,<0.16",
    "flake8>=3.8.3",
    "flake8-debugger>=3.2.1",
    "psutil>=5.7.0",
    "pytest>=5.4.3",
    "pytest-cov>=2.9.0",
    "pytest-raises>=0.11",
    "quilt3>=3.1.12",
]

dev_requirements = [
    *setup_requirements,
    *test_requirements,
    "bumpversion>=0.6.0",
    "coverage>=5.1",
    "gitchangelog>=3.0.4",
    "ipython>=7.15.0",
    "m2r>=0.2.1",
    "pytest-runner>=5.2",
    "Sphinx>=2.0.0b1,<3",
    "sphinx_rtd_theme>=0.4.3",
    "tox>=3.15.2",
    "twine>=3.1.1",
    "wheel>=0.34.2",
]

benchmark_requirements = [
    "altair",
    "altair_saver",
    "czifile==2019.7.2",
    "dask_jobqueue==0.7.0",
    "imageio==2.8.0",
    "quilt3>=3.1.12",
    "tifffile==2020.2.16",
    "tqdm",
]

extra_requirements = {
    "setup": setup_requirements,
    "test": test_requirements,
    "dev": dev_requirements,
    "benchmark": benchmark_requirements,
    "all": [
        *requirements,
        *dev_requirements,
    ]
}

setup(
    author="Allen Institute for Cell Science",
    author_email=("jacksonb@alleninstitute.org, " "bowdenm@spu.edu"),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description=(
        "Delayed Image Reading, Metadata Conversion, and Image Writing for Microscopy "
        "Images in Pure Python"
    ),
    entry_points={},
    install_requires=requirements,
    license="BSD-3-Clause",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="aicsimageio, allen cell, imaging, computational biology",
    name="aicsimageio",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*"]),
    python_requires=">=3.6",
    setup_requires=setup_requirements,
    test_suite="aicsimageio/tests",
    tests_require=test_requirements,
    extras_require=extra_requirements,
    url="https://github.com/AllenCellModeling/aicsimageio",
    # Do not edit this string manually, always use bumpversion
    # Details in CONTRIBUTING.md
    version="3.3.0",
    zip_safe=False,
)
