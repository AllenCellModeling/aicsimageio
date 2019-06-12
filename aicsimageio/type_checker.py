import logging
import re
import sys


from . import types


class TypeChecker:
    """Cheaply check to see if a given file is a recognized type.
    Currently recognized types are TIFF, OME TIFF, and CZI.
    If the file is a TIFF, then the description (OME XML if it is OME TIFF) can be retrieved via read_description.
    Similarly, if the file is a CZI, then the metadata XML can be retrieved via read_description.
    """
    pass


class ByteReader:

    INTEL_ENDIAN = b'II'
    MOTOROLA_ENDIAN = b'MM'

    def __init__(self, byte_io: types.FileLike):
        self.byte_io = byte_io
        self.previous_position = self.byte_io.tell()
        self.description_length = 0
        self.description_offset = 0
        self.endianness = None

    def __enter__(self):
        self.byte_io.seek(0)
        self.endianness = bytearray(self.byte_io.read(2))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reset()

    def reset(self):
        self.byte_io.seek(self.previous_position)

    # All of these read_uint* routines obey the endianness, with 'II' being little-endian
    # and 'MM' being big-endian (per TIFF-6)
    def read_uint16(self):
        value = bytearray(self.byte_io.read(2))
        return (value[0] + (value[1] << 8)) if self.endianness == b'II' else (value[1] + (value[0] << 8))

    def read_uint32(self):
        value = bytearray(self.byte_io.read(4))
        return (value[0] + (value[1] << 8) + (value[2] << 16) + (value[3] << 24)) if self.endianness == b'II' else (value[3] + (value[2] << 8) + (value[1] << 16) + (value[0] << 24))

    def read_uint64(self):
        return (self.read_uint32() + (self.read_uint32() << 32)) if self.endianness == b'II' else ((self.read_uint32() << 32) + self.read_uint32())

