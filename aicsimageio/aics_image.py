import logging
import typing
from typing import Optional, Type

import numpy as np

from . import constants, transforms, types
from .exceptions import InvalidDimensionOrderingError, UnsupportedFileFormatError
from .readers import CziReader, DefaultReader, NdArrayReader, OmeTiffReader, TiffReader
from .readers.reader import Reader

log = logging.getLogger(__name__)


class AICSImage:
    """
    AICSImage takes microscopy image data types (files / bytestreams) of varying dimensions ("ZYX", "TCZYX", "CYX") and
    puts them into a consistent 6D "STCZYX" ordered numpy.ndarray. The data, metadata are lazy loaded and can be
    accessed as needed. Note the dims are assumed to match "STCZYX" from right to left meaning if 4 dimensional
    data is provided then the dimensions are assigned to be "CZYX", 2 dimensional would be "YX". This guessed assignment
    is only for file types without dimension metadata (ie not .ome.tiff or .czi).

    Note: if you absolutely know the dims and they are not as guessed then you can override them by immediately setting
    them after initialization see example below.

    Simple Example
    --------------
    with open("filename.czi", 'rb') as fp:
        img = AICSImage(fp)
        data = img.data  # data is a 6D "STCZYX" object
        metadata = img.metadata  # metadata from the file, an xml.etree
        zstack_t8 = img.get_image_data("ZYX", S=0, T=8, C=0)  # returns a 3D "ZYX" numpy.ndarray

    zstack_t10 = data[0, 10, 0, :, :, :]  # access the S=0, T=10, C=0 "ZYX" cube


    File Examples
    -------------
    OmeTif
        img = AICSImage("filename.ome.tif")
    CZI (Zeiss)
        img = AICSImage("filename.czi") or AICSImage("filename.czi", max_workers=8)
    Tiff
        img = AICSImage("filename.tif")
    Png/Gif/...
        img = AICSImage("filename.png")
        img = AICSImage("filename.gif")

    Bytestream Examples
    -------------------
    OmeTif
        with open("filename.ome.tif", 'rb') as fp:
            img = AICSImage(fp)
    CZI
        with open("filename.czi", 'rb') as fp:
            img = AICSImage(fp, max_workers=7)
    Tiff/Png/Gif
        with open("filename.png", 'rb') as fp:
            img = AICSImage(fp)

    Numpy.ndarray Example
    ---------------------
    blank = numpy.zeros((2, 600, 900))
    img = AICSImage(blank)

    Example with known dimensions different than guessed
    ----------------------------------------------------
    img = AICSImage('TCXimage.gif')
    img.reader.dims = 'TCX'
    data = img.data  # get a 6D ndarray back in "STCZYX" order
    """

    SUPPORTED_READERS = [CziReader, OmeTiffReader, TiffReader, DefaultReader]

    def __init__(
        self,
        data: typing.Union[types.FileLike, np.ndarray],
        known_dims: Optional[str] = None,
        **kwargs,
    ):
        """
        Constructor for AICSImage class intended for providing a unified interface for dealing with
        microscopy images. To extend support to a new reader simply add a new reader child class of
        Reader ([readers/reader.py]) and add the class to SUPPORTED_READERS in AICSImage.

        Parameters
        ----------
        data: String with path to ometif/czi/tif/png/gif file, or ndarray with up to 6 dimensions
        known_dims: Optional string with the known dimension order. If None, the reader will attempt to parse dim order.
        kwargs: Parameters to be passed through to the reader class
                       max_workers (optional Czi) specifies the number of worker threads for the backend library
        """

        # Hold onto known dims until data is requested
        self._known_dims = known_dims

        # Dims should nearly always be default dim order unless explictly overridden
        self.dims = constants.DEFAULT_DIMENSION_ORDER

        # Lazy load later
        self._data = None
        self._metadata = None

        # Determine reader class and load data
        reader_class = self.determine_reader(data)
        self._reader = reader_class(data, **kwargs)

    @staticmethod
    def determine_reader(data: types.ImageLike) -> Type[Reader]:
        """Cheaply check to see if a given file is a recognized type.
        Currently recognized types are TIFF, OME TIFF, and CZI.
        If the file is a TIFF, then the description (OME XML if it is OME TIFF) can be retrieved via read_description.
        Similarly, if the file is a CZI, then the metadata XML can be retrieved via read_description.
        """
        # The order of the readers in this list is important.
        # Example: if TiffReader was placed before OmeTiffReader, we would never use the OmeTiffReader.
        for reader_class in [
            NdArrayReader,
            CziReader,
            OmeTiffReader,
            TiffReader,
            DefaultReader,
        ]:
            if reader_class.is_this_type(data):
                return reader_class

        raise UnsupportedFileFormatError(type(data))

    @property
    def data(self):
        """
        Returns
        -------
        Returns a numpy.ndarray with dimension ordering "STCZYX"
        """
        if self._data is None:
            reader_data = self._reader.data

            # Read and reshape and handle delayed known dims reshape
            self._data = transforms.reshape_data(
                data=reader_data,
                given_dims=self._known_dims or self.reader.dims,
                return_dims=self.dims,
            )
        return self._data

    def size(self, dims: str = "STCZYX"):
        """
        Parameters
        ----------
        dims: A string containing a list of dimensions being requested.  The default is to return the 6 standard dims

        Returns
        -------
        Returns a tuple with the requested dimensions filled in
        """
        dims = dims.upper()
        if not (all(d in "STCZYX" for d in dims)):
            raise InvalidDimensionOrderingError(f"Invalid dimensions requested: {dims}")
        if not (all(d in self.dims for d in dims)):
            raise InvalidDimensionOrderingError(f"Invalid dimensions requested: {dims}")
        return tuple([self.data.shape[self.dims.index(dim)] for dim in dims])

    @property
    def size_x(self):
        """
        Returns
        -------
        Returns the x size
        """
        return self.size("X")[0]

    @property
    def size_y(self):
        """
        Returns
        -------
        Returns the y size
        """
        return self.size("Y")[0]

    @property
    def size_z(self):
        """
        Returns
        -------
        Returns the z size
        """
        return self.size("Z")[0]

    @property
    def size_c(self):
        """
        Returns
        -------
        Returns the c size
        """
        return self.size("C")[0]

    @property
    def size_t(self):
        """
        Returns
        -------
        Returns the t size
        """
        return self.size("T")[0]

    @property
    def size_s(self):
        """
        Returns
        -------
        Returns the s size
        """
        return self.size("S")[0]

    @property
    def metadata(self):
        """
        Returns
        -------
        The Metadata from the Czi, or Ome.Tiff file, or other base class type with metadata.
        For pure image files and empty string or None is returned.

        """
        if self._metadata is None:
            self._metadata = self._reader.metadata
        return self._metadata

    @property
    def reader(self) -> Reader:
        """
        This property returns the class created to read the image file type.
        The intent is that if the AICSImage class doesn't provide a raw enough
        interface then the base class can be used directly.

        Returns
        -------
        A child of Reader, CziReader OmeTiffReader, TiffReader, DefaultReader, etc.

        """
        return self._reader

    def get_image_data(
        self, out_orientation: str = None, copy: bool = False, **kwargs
    ) -> np.ndarray:
        """

        Parameters
        ----------
        out_orientation: A string containing the dimension ordering desired for the returned ndarray
        copy: boolean value to get image data by reference or by value [True, False]

        kwargs:
            C=1: specifies Channel 1
            T=3: specifies the fourth index in T
            D=n: D is Dimension letter and n is the index desired D should not be present in the out_orientation

        Returns
        -------
        ndarray with dimension ordering that was specified with out_orientation
        Note: if a requested dimension is not present in the data the dimension is added with
        a depth of 1. The default return dimensions are "STCZYX".
        """
        out_orientation = self.dims if out_orientation is None else out_orientation
        if out_orientation == self.dims:
            return self.data
        return transforms.reshape_data(
            data=self.data,
            given_dims=self.dims,
            return_dims=out_orientation,
            copy=copy,
            **kwargs,
        )

    def __repr__(self) -> str:
        return f"<AICSImage [{type(self.reader).__name__}]>"


def imread(data: types.ImageLike, **kwargs):
    return AICSImage(data, **kwargs).data
