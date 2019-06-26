import logging
import typing
from typing import Type

import numpy as np

from . import constants, transforms, types
from .exceptions import UnsupportedFileFormatError
from .readers import (CziReader, DefaultReader, NdArrayReader, OmeTiffReader,
                      TiffReader)
from .readers.reader import Reader

log = logging.getLogger(__name__)


class AICSImage:
    """
    AICSImage takes microscopy image data types (files / bytestreams) of varying dimensions ("ZYX", "TCZYX", "CYX") and
    puts them into a consistent 6D "STCZYX" ordered numpy.ndarray. The data, metadata are lazy loaded and can be
    accessed as needed.

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
        Tif/Png/Gif
            img = AICSImage("filename.png", known_dims="ZYX")
            img = AICSImage("filename.tif", known_dims="CZYX")

        Bytestream Examples
        -------------------
        OmeTif
            with open("filename.ome.tif", 'rb') as fp:
                img = AICSImage(fp)
        CZI
            with open("filename.czi", 'rb') as fp:
                img = AICSImage(fp, max_workers=7)
        Tif/Png/Gif
            with open("filename.png", 'rb') as fp:
                img = AICSImage(fp, known_dims="YX")

        Numpy.ndarray Example
        ---------------------
        blank = numpy.zeros((2, 600, 900))
        img = AICSImage(blank, known_dims="CYX")
    """
    SUPPORTED_READERS = [CziReader, OmeTiffReader, TiffReader, DefaultReader]

    def __init__(self, data: typing.Union[types.FileLike, types.SixDArray], **kwargs):
        """
        Constructor for AICSImage class intended for providing a unified interface for dealing with
        microscopy images. To extend support to a new reader simply add a new reader child class of
        Reader ([readers/reader.py]) and add the class to SUPPORTED_READERS in AICSImage.

        Parameters
        ----------
        data: String with path to ometif/czi/tif/png/gif file, or ndarray with up to 6 dimensions
        kwargs: Parameters to be passed through to the reader class
                       known_dims (required Tif/Png/Gif/numpy.ndarray) known Dimensions of input file, ie "TCZYX"
                       max_workers (optional Czi) specifies the number of worker threads for the backend library
        """
        self.dims = constants.DEFAULT_DIMENSION_ORDER
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
        for reader_class in [NdArrayReader, CziReader, OmeTiffReader, TiffReader, DefaultReader]:
            if reader_class.is_this_type(data):
                return reader_class

        raise UnsupportedFileFormatError(type(data))

    @property
    def data(self):
        """
        Returns
        -------
        returns a numpy.ndarray with dimension ordering "STCZYX"
        """
        if self._data is None:
            reader_data = self._reader.data
            self._data = transforms.reshape_data(data=reader_data,
                                                 given_dims=self._reader.dims,
                                                 return_dims=self.dims)
        return self._data

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

    def get_image_data(self, out_orientation=None, **kwargs) -> np.ndarray:
        """

        Parameters
        ----------
        out_orientation: A string containing the dimension ordering desired for the returned ndarray
        kwargs:
            copy: boolean value to get image data by reference or by value [True, False]
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
        return transforms.reshape_data(self.data, given_dims=self.dims, return_dims=out_orientation, **kwargs)

    # Do We want to add this functionality back in?
    # def get_channel_names(self):
    #     if self.metadata is not None:
    #         if hasattr(self.metadata, "image"):
    #             return [
    #                 self.metadata.image().Pixels.Channel(i).Name
    #                 for i in range(self.size_c)
    #             ]
    #         else:
    #             # consider this to be CZI metadata!
    #             chelem = self.metadata.findall(
    #                 "./Metadata/Information/Image/Dimensions/Channels/Channel"
    #             )
    #             return [ch.get("Name") for ch in chelem]
    #     else:
    #         return None
    #
    # def _getmetadataxmltext(self, findpath, default=None):
    #     ref = self.metadata.find(findpath)
    #     if ref is None:
    #         return default
    #     return ref.text
    #
    # def get_physical_pixel_size(self):
    #     if self.metadata is not None:
    #         if hasattr(self.metadata, "image"):
    #             p = self.metadata.image().Pixels
    #             return [
    #                 p.get_PhysicalSizeX(),
    #                 p.get_PhysicalSizeY(),
    #                 p.get_PhysicalSizeZ(),
    #             ]
    #         else:
    #             # consider this to be CZI metadata!
    #             px = float(
    #                 self._getmetadataxmltext(
    #                     "./Metadata/Scaling/Items/Distance[@Id='X']/Value", "1.0"
    #                 )
    #             )
    #             py = float(
    #                 self._getmetadataxmltext(
    #                     "./Metadata/Scaling/Items/Distance[@Id='Y']/Value", "1.0"
    #                 )
    #             )
    #             pz = float(
    #                 self._getmetadataxmltext(
    #                     "./Metadata/Scaling/Items/Distance[@Id='Z']/Value", "1.0"
    #                 )
    #             )
    #             return [px, py, pz]
    #     else:
    #         return None
