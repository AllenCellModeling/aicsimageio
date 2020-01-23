#!/usr/bin/env python
# -*- coding: utf-8 -*-


class UnsupportedFileFormatError(Exception):
    """
    This exception is intended to communicate that the file extension is not one of
    the supported file types and cannot be parsed with AICSImage.
    """

    def __init__(self, data, **kwargs):
        super().__init__(**kwargs)
        self.data = data

    def __str__(self):
        return f"AICSImage module does not support this image file type: '{self.data}'."


class InvalidDimensionOrderingError(Exception):
    """
    A general exception that can be thrown when handling dimension ordering or validation. Should be provided a message
    for the user to be given more context.
    """

    def __init__(self, message: str, **kwargs):
        super().__init__(**kwargs)
        self.message = message

    def __str__(self):
        return self.message


class ConflictingArgumentsError(Exception):
    """
    This exception is returned when 2 arguments to the same function are in conflict.
    """
    pass


class InconsistentShapeError(Exception):
    """
    A general function to use when the shape returned or requested from an array operation is invalid.
    """
    pass
