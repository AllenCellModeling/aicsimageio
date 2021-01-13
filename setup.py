#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

requirements = [
    "aicspylibczi>=2.7.5,<3.0",
    "dask>=2.9.0",
    "distributed>=2.9.3",
    "numpy>=1.16",
    "imagecodecs>=2020.5.30",
    "imageio>=2.3.0",
    "readlif>=0.2.1",
    "lxml>=4.4.2",
    "tifffile>=2020.9.22",
    "toolz>=0.10.0",
]

test_requirements = [
    "black>=19.10b0",
    "codecov>=2.0.22",
    "docutils>=0.10,<0.16",
    "flake8>=3.7.7",
    "napari[pyqt5]>=0.2.10",
    "psutil>=5.7.0",
    "pytest>=4.3.0",
    "pytest-cov==2.6.1",
    "pytest-raises>=0.10",
    "quilt3>=3.1.12",
]

dev_requirements = [
    "black>=19.10b0",
    "bumpversion>=0.5.3",
    "coverage>=5.0a4",
    "docutils>=0.10,<0.16",
    "flake8>=3.7.7",
    "gitchangelog>=3.0.4",
    "ipython>=7.5.0",
    "m2r>=0.2.1",
    "pytest>=4.3.0",
    "pytest-cov==2.6.1",
    "pytest-raises>=0.10",
    "pytest-runner>=4.4",
    "quilt3>=3.1.12",
    "Sphinx>=2.0.0b1,<3",
    "sphinx_rtd_theme>=0.1.2",
    "tox>=3.5.2",
    "twine>=1.13.0",
    "wheel>=0.33.1",
]

setup_requirements = [
    "pytest-runner",
]

interactive_requirements = [
    "altair",
    "bokeh",
    "jupyterlab",
    "matplotlib",
    "napari[pyqt5]>=0.2.10",
    "pillow",
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
    "test": test_requirements,
    "dev": dev_requirements,
    "setup": setup_requirements,
    "interactive": interactive_requirements,
    "benchmark": benchmark_requirements,
    "all": [
        *requirements,
        *test_requirements,
        *setup_requirements,
        *dev_requirements,
        *interactive_requirements,
    ],
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
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    description=(
        "Delayed parallel image reading, metadata parsing, and image writing for "
        "microscopy formats in pure Python from the Allen Institute for Cell Science."
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
    python_requires=">=3.7",
    setup_requires=setup_requirements,
    test_suite="aicsimageio/tests",
    tests_require=test_requirements,
    extras_require=extra_requirements,
    url="https://github.com/AllenCellModeling/aicsimageio",
    # Do not edit this string manually, always use bumpversion
    # Details in CONTRIBUTING.md
    version="3.3.4",
    zip_safe=False,
)
