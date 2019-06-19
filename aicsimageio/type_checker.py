
from . import types
from .readers.czi_reader import CziReader
from .readers.ome_tiff_reader import OmeTiffReader
from .readers.tiff_reader import TiffReader


class TypeChecker:
    """Cheaply check to see if a given file is a recognized type.
    Currently recognized types are TIFF, OME TIFF, and CZI.
    If the file is a TIFF, then the description (OME XML if it is OME TIFF) can be retrieved via read_description.
    Similarly, if the file is a CZI, then the metadata XML can be retrieved via read_description.
    """
    def __init__(self, file: types.FileLike):
        self.is_czi = CziReader.is_this_type(file)
        self.is_tiff = TiffReader.is_this_type(file)
        self.is_ome = OmeTiffReader.is_this_type(file)
