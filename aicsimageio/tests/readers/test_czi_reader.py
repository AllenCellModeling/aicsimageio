import pytest
from aicsimageio.readers.czi_reader import CziReader
from aicsimageio.vendor.czifile import CziFile


def test_czifile():
    czi = CziFile("/Users/jamies/20180907_M01_001.czi")
    print(f"axes:= {czi.axes}")

