#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Benchmarks for general library operations.
"""


class LibSuite:
    def time_base_import(self):
        """
        Benchmark how long it takes to import the library as a whole.
        """
        import aicsimageio  # noqa: F401
