from collections import Counter
import logging
import typing
from io import BufferedIOBase
from pathlib import Path
from typing import Type
import re

import numpy as np

from . import types
from .exceptions import UnsupportedFileFormatError, InvalidDimensionOrderingError, ConflictingArgsError
from .readers import CziReader, OmeTiffReader, TiffReader, DefaultReader
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
    DEFAULT_DIMS = "BTCZYX"
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
        self.dims = AICSImage.DEFAULT_DIMS
        self._data = None
        self._metadata = None

        # Determine reader class and load data
        reader_class = self.determine_reader(data)
        self.reader = reader_class(data, **kwargs)

    @staticmethod
    def determine_reader(data: types.ImageLike) -> Type[Reader]:
        """Cheaply check to see if a given file is a recognized type.
        Currently recognized types are TIFF, OME TIFF, and CZI.
        If the file is a TIFF, then the description (OME XML if it is OME TIFF) can be retrieved via read_description.
        Similarly, if the file is a CZI, then the metadata XML can be retrieved via read_description.
        """
        # The order of the readers in this list is important.
        # Example: if TiffReader was placed before OmeTiffReader, we would never use the OmeTiffReader.
        for reader_class in [CziReader, OmeTiffReader, TiffReader, DefaultReader]:
            if reader_class.is_this_type(data):
                return reader_class
        else:
            raise UnsupportedFileFormatError(type(data))

    def is_valid_dimension(self, dimensions):
        if dimensions is None:
            raise ValueError("Dimensions can not be None")
        if dimensions.strip(self.dims):
            # dims contains more than the standard 5 dims we're used to
            raise ValueError("{} contains invalid dimensions!".format(dimensions))

        count = {}
        for char in dimensions:
            if char in count:
                raise ValueError("{} contains duplicate dimensions!".format(dimensions))
            else:
                count[char] = 1

        return True

    @property
    def data(self):
        if self._data is None:
            self._data = AICSImage._reshape_data(data=self.reader.data,
                                                 given_dims=self.reader.dims,
                                                 return_dims=self.DEFAULT_DIMS
                                                 )
        return self._data

    @property
    def metadata(self):
        if self._metadata is None:
            self._metadata = self.reader.metadata
        return self._metadata

    @staticmethod
    def _reshape_data(data: np.ndarray, given_dims: str, return_dims: str, **kwargs) -> np.ndarray:
        """
        Reshape the data into return_dims, pad missing dimensions, and prune extra dimensions.
        Warns the user to use the base reader if the depth of the Dimension being removed is not 1.

        Parameters
        ----------
        data: a numpy.ndarray of arbitrary shape but with the dimensions specified in given_dims
        given_dims: the dimension ordering of data, "CZYX", "VBTCXZY" etc
        return_dims: the dimension ordering of the return data
        kwargs:
            C=1 => desired specific channel, if C in the input data has depth 3 then C=1 takes index 1
            copy=True => copy the data object passed in and return a new object
        Returns
        -------
        a numpy.ndarray in return_dims order, if return_dims=DEFAULT_DIMS then the return would have order "STCZYX"

        """
        # copy the data object if copy=True is in kwargs
        data = data.copy() if kwargs.get('copy', False) else data
        # check for conflicting return_dims and fixed channels 'C=1'
        for dim in return_dims:
            if kwargs.get(dim, None) is not None:
                msg = f"argument return_dims={return_dims} and argument {dim}={kwargs.get(dim)} conflict. Check usage."
                raise ConflictingArgsError(msg)
        # add each dimension not included in original data
        new_dims = given_dims
        excluded_dims = re.sub('|'.join(given_dims), '', return_dims)
        for dim in excluded_dims:
            data = np.expand_dims(data, axis=0)
            new_dims = dim + new_dims  # add the missing Dimension to the front
        # if given dims contains a Dimension not in DEFAULT_DIMS and its depth is 1 remove it
        # if it's larger than 1 give a warning and suggest interfacing with the Reader object
        extra_dims = re.sub('|'.join(return_dims), '', given_dims)
        for dim in extra_dims:
            index = new_dims.find(dim)
            if data.shape[index] > 1:
                index_depth = kwargs.get(dim, None)
                if index_depth is None:
                    msg = (f'data has dimension {dim} with depth {data.shape[index]}, assuming {dim}=0 is  '
                           f'the desired value, if not the case specify {dim}=x where '
                           f'x is an integer in [0, {data.shape[index]}).')
                    log.warning(msg)
                    index_depth = 0
                if index_depth >= data.shape[index]:
                    raise IndexError(f'Dimension specified with {dim}={index_depth} '
                                     f'but Dimension shape is {data.shape[index]}.')
                planes = np.split(data, data.shape[index], axis=index)  # split dim into list of arrays
                data = planes[index_depth]  # take the specified value of the dim
            data = np.squeeze(data, axis=index)  # remove the dim from ndarray
            new_dims = new_dims[0:index:] + new_dims[index + 1::]  # clip out the Dimension from new_dims
        # any extra dimensions have been removed, only a problem if the depth is > 1
        return AICSImage._transpose_to_dims(data, given_dims=new_dims, return_dims=return_dims)

    @staticmethod
    def _transpose_to_dims(data: np.ndarray, given_dims: str, return_dims: str) -> np.ndarray:
        """
        This shuffles the data dimensions from know_dims to return_dims, return_dims can be and subset
        of known_dims in any order.

        Parameters
        ----------
        data: the input data with dimensions known_dims
        given_dims: the dimensions of data
        return_dims: the subset of known_dims to return

        Returns
        -------
        a numpy.ndarray with known_dims

        """
        if Counter(given_dims) != Counter(return_dims):
            print(Counter(given_dims), Counter(return_dims))
            raise ConflictingArgsError(f"given_dims={given_dims} and return_dims={return_dims} are incompatible.")
        # resort the data into return_dims order
        match_map = {dim: given_dims.find(dim) for dim in given_dims}
        transposer = []
        for dim in return_dims:
            if match_map[dim] == -1:
                msg = f'Dimension {dim} requested but not present in given_dims={given_dims}.'
                raise ConflictingArgsError(msg)
            transposer.append(match_map[dim])
        data = data.transpose(transposer)
        return data


    def get_channel_names(self):
        if self.metadata is not None:
            if hasattr(self.metadata, "image"):
                return [
                    self.metadata.image().Pixels.Channel(i).Name
                    for i in range(self.size_c)
                ]
            else:
                # consider this to be CZI metadata!
                chelem = self.metadata.findall(
                    "./Metadata/Information/Image/Dimensions/Channels/Channel"
                )
                return [ch.get("Name") for ch in chelem]
        else:
            return None

    def _getmetadataxmltext(self, findpath, default=None):
        ref = self.metadata.find(findpath)
        if ref is None:
            return default
        return ref.text

    def get_physical_pixel_size(self):
        if self.metadata is not None:
            if hasattr(self.metadata, "image"):
                p = self.metadata.image().Pixels
                return [
                    p.get_PhysicalSizeX(),
                    p.get_PhysicalSizeY(),
                    p.get_PhysicalSizeZ(),
                ]
            else:
                # consider this to be CZI metadata!
                px = float(
                    self._getmetadataxmltext(
                        "./Metadata/Scaling/Items/Distance[@Id='X']/Value", "1.0"
                    )
                )
                py = float(
                    self._getmetadataxmltext(
                        "./Metadata/Scaling/Items/Distance[@Id='Y']/Value", "1.0"
                    )
                )
                pz = float(
                    self._getmetadataxmltext(
                        "./Metadata/Scaling/Items/Distance[@Id='Z']/Value", "1.0"
                    )
                )
                return [px, py, pz]
        else:
            return None

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
        Note: if you constructed AICSImage with a datablock with less than 5D you must still include the omitted
        dimensions in your out_orientation if you want the same < 5D block back
        """
        out_orientation = self.DEFAULT_DIMS if out_orientation is None else out_orientation
        return AICSImage._reshape_data(self.data, given_dims=self.dims, return_dims=out_orientation, **kwargs)
