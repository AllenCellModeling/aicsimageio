#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

requirements = [
    "numpy>=1.16",
    "imageio>=2.3.0",
    "scipy>=1.1.0",
    "tifffile==0.15.1",
]

test_requirements = [
    "codecov",
    "flake8",
    "pytest",
    "pytest-cov",
    "pytest-raises",
]

dev_requirements = [
    "bumpversion>=0.5.3",
    "coverage>=5.0a4",
    "flake8>=3.7.7",
    "ipython>=7.5.0",
    "pytest>=4.3.0",
    "pytest-cov==2.6.1",
    "pytest-raises>=0.10",
    "pytest-runner>=4.4",
    "Sphinx>=2.0.0b1",
    "tox>=3.5.2",
    "twine>=1.13.0",
    "wheel>=0.33.1",
]

setup_requirements = [
    "pytest-runner",
]

interactive_requirements = [
    "altair",
    "jupyterlab",
    "matplotlib",
    "PIL",
]

extra_requirements = {
    "test": test_requirements,
    "dev": dev_requirements,
    "setup": setup_requirements,
    "interactive": interactive_requirements,
    "all": [
        *requirements,
        *test_requirements,
        *setup_requirements,
        *dev_requirements,
        *interactive_requirements
    ]
}

setup(
    author=(
        "Dan Toloudis, "
        "Matthew Bowden, "
        "Jamie Sherman, "
        "Jackson Maxfield Brown, "
        "Zach Crabtree"
    ),
    author_email=(
        "danielt@alleninstitute.org, "
        "bowdenm@spu.edu, "
        "jamies@alleninstitute.org, "
        "jacksonb@alleninstitute.org, "
    ),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "License :: Allen Institute Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    description=(
        "Python library for reading and writing image data with special handlers for bio-formats "
        "from Allen Institute for Cell Science."
    ),
    entry_points={},
    install_requires=requirements,
    license="Allen Institute Software License",
    long_description=readme,
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
    version="3.0.0",
    zip_safe=False,
)
