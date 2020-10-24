#!/usr/bin/env python
# -*- coding: utf-8 -*-


class ConflictingArgumentsError(Exception):
    """
    This exception is returned when 2 arguments to the same function are in conflict.
    """

    pass


class UnsupportedFileFormatError(Exception):
    """
    This exception is intended to communicate that the file extension is not one of
    the supported file types and cannot be parsed with AICSImage.
    """

    def __init__(self, extension, **kwargs):
        super().__init__(**kwargs)
        self.extension = extension

    def __str__(self):
        return f"AICSImageIO does not support the image file type: '{self.extension}'."
