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

    def __init__(self, reader_name: str, extension: str, **kwargs):
        super().__init__(**kwargs)
        self.reader_name = reader_name
        self.extension = extension

    def __str__(self):
        return (
            f"{self.reader_name} does not support the "
            f"image file type: '{self.extension}'."
        )
