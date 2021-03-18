#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional


class ConflictingArgumentsError(Exception):
    """
    This exception is returned when 2 arguments to the same function are in conflict.
    """

    pass


class InvalidDimensionOrderingError(Exception):
    """
    A general exception that can be thrown when handling dimension ordering or
    validation. Should be provided with a message for the user to be given more context.
    """

    pass


class UnexpectedShapeError(Exception):
    """
    A general exception that can be thrown when handling shape validation.
    Should be provided with a message for the user to be given more context.
    """

    pass


class UnsupportedFileFormatError(Exception):
    """
    This exception is intended to communicate that the file extension is not one of
    the supported file types and cannot be parsed with AICSImage.
    """

    def __init__(self, reader_name: str, path: str, msg_extra: Optional[str] = None):
        super().__init__()
        self.reader_name = reader_name
        self.path = path
        self.msg_extra = msg_extra

    def __str__(self) -> str:
        msg = f"{self.reader_name} does not support the image: '{self.path}'."

        if self.msg_extra is not None:
            msg = f"{msg} {self.msg_extra}"

        return msg
