# author: Zach Crabtree zacharyc@alleninstitute.org
from pathlib import Path

import numpy as np
from . import omeTifReader, cziReader, tifReader, typeChecker
from .exceptions import UnsupportedFileFormatError


def enum(**named_values):
    return type('Enum', (), named_values)


FileType = enum(OMETIF=1, TIF=2, CZI=3)


# TODO I am not sure what the behavior should be in the case where img = AICSimage(file, dims="CTX")
# TODO and then img.get_image_data requests a sub-block. I kind of expect images we deal with to have 5 channels (TCZYX)


class AICSImage:
    """
    A wrapper class for ndarrays.

    Example:
        # valid ndarrays are between 1 and 5 dimensions
        >>> data4d = numpy.zeros((2, 25, 1024, 1024))
        >>> data2d = numpy.zeros((99, 100))
        # any dimension ordering of T, C, Z, Y, X is valid
        >>> image_from_4d = AICSImage(data4d, dims="TZYX")
        >>> image_from_2d = AICSImage(data2d, dims="YX")
        # now both images are expanded to contain all 5 dims
        # you can access it in any dimension ordering, no matter how nonsensical
        >>> image_to_5d_from_4d = image_from_4d.get_image_data("XYCZT")
        >>> image_to_5d_from_2d = image_from_2d.get_image_data("YCZTX")
        # you can also access specific slices from each dimension you leave out
        >>> image_to_1d_from_2d = image_from_2d.get_image_data("X", Y=12)

        # finally, AICSImage objects can be generated from ometifs and czis (could be removed in later revisions)
        >>> image_from_file = AICSImage("image_data.ome.tif")
        >>> image_from_file = AICSImage("image_data.czi")
    NOTE: If you construct with a data/image block less than 5D the class upscales the data to be 5 D
    """
    default_dims = "TCZYX"

    def __init__(self, data, **kwargs):
        """
        Constructor for AICSImage class
        :param data: String with path to ometif/czi file, or ndarray with up to 5 dimensions
        :param kwargs: If ndarray is used for data, then you can specify the dim ordering
                       with dims arg (ie dims="TZCYX").
        """
        self.metadata = None
        self.dims = AICSImage.default_dims
        if isinstance(data, (str, Path)):
            # check input is a filepath
            check_file_path = Path(data).resolve(strict=True)
            if not check_file_path.is_file():
                raise IsADirectoryError(check_file_path)

            # assign proven existing file to member variable (as string for compatibility with readers)
            self.file_path = str(check_file_path)

            # check for compatible data types
            checker = typeChecker.TypeChecker(self.file_path)
            if checker.is_czi:
                self.reader = cziReader.CziReader(self.file_path, max_workers=kwargs.get('max_workers', None))
            elif checker.is_ome:
                self.reader = omeTifReader.OmeTifReader(self.file_path)
            elif checker.is_tiff:
                self.reader = tifReader.TifReader(self.file_path)
            else:
                raise UnsupportedFileFormatError("CellImage can only accept OME-TIFF, TIFF, and CZI file formats!")
            # TODO make this lazy, so we don't have to read all the pixels if all we want is metadata
            self.data = self.reader.load()
            # TODO remove this transpose call once reader output is changed
            # this line assumes that all the above readers return TZCYX order, and converts to TCZYX
            self.data = self.data.transpose(0, 2, 1, 3, 4)
            self.metadata = self.reader.get_metadata()
            self.shape = self.data.shape
            # we are not using the reader anymore
            self.reader.close()

        elif isinstance(data, np.ndarray):
            # input is a data array
            self.data = data
            if self.is_valid_dimension(kwargs["dims"]):
                self.dims = kwargs["dims"]

            if len(self.dims) != len(self.data.shape):
                raise ValueError("Number of dimensions must match dimensions of array provided!")
            self._reshape_data()
            self.shape = self.data.shape

        else:
            raise TypeError("Unable to process item of type {}".format(type(data)))

        self.size_t, self.size_c, self.size_z, self.size_y, self.size_x = tuple(self.shape)

    def is_valid_dimension(self, dimensions):
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

    def _transpose_to_defaults(self):
        match_map = {dim: self.default_dims.find(dim) for dim in self.dims}
        transposer = []
        for dim in self.dims:
            if not match_map[dim] == -1:
                transposer.append(match_map[dim])
        self.data = self.data.transpose(transposer)
        self.dims = self.default_dims

    def _reshape_data(self):
        # this function will add in the missing dimensions in order to make a complete 5d array
        # get each dimension not included in original data
        excluded_dims = self.default_dims.strip(self.dims)
        for dim in excluded_dims:
            self.data = np.expand_dims(self.data, axis=0)
            self.dims = dim + self.dims
        self._transpose_to_defaults()

    def get_channel_names(self):
        if self.metadata is not None:
            if hasattr(self.metadata, 'image'):
                return [self.metadata.image().Pixels.Channel(i).Name for i in range(self.size_c)]
            else:
                # consider this to be CZI metadata!
                chelem = self.metadata.findall("./Metadata/Information/Image/Dimensions/Channels/Channel")
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
            if hasattr(self.metadata, 'image'):
                p = self.metadata.image().Pixels
                return [p.get_PhysicalSizeX(), p.get_PhysicalSizeY(), p.get_PhysicalSizeZ()]
            else:
                # consider this to be CZI metadata!
                px = float(self._getmetadataxmltext("./Metadata/Scaling/Items/Distance[@Id='X']/Value", "1.0"))
                py = float(self._getmetadataxmltext("./Metadata/Scaling/Items/Distance[@Id='Y']/Value", "1.0"))
                pz = float(self._getmetadataxmltext("./Metadata/Scaling/Items/Distance[@Id='Z']/Value", "1.0"))
                return [px, py, pz]
        else:
            return None

    # TODO get_reference_data if user is not going to manipulate data
    # TODO (minor) allow uppercase and lowercase kwargs
    def get_image_data(self, out_orientation="TCZYX", **kwargs):
        """
        :param out_orientation: A string containing the dimension ordering desired for the returned ndarray
        :param kwargs: These can contain the dims you exclude from out_orientation (out of the set "TCZYX").
                       If you want all slices of ZYX, but only one from T and C, you can enter:
                       >>> image.get_image_data("ZYX", T=1, C=3)
                       Unspecified dimensions that are left of out the out_orientation default to 0.
                       :param reference: boolean value to get image data by reference or by value
        :return: ndarray with dimension ordering that was specified with out_orientation
        Note: if you constructed AICSImage with a datablock with less than 5D you must still include the omitted
        dimensions in your out_orientation if you want the same < 5D block back
        """
        if kwargs.get("reference", False):
            # get data by reference
            image_data = self.data
        else:
            # make a copy of the data
            image_data = self.data.copy()

        out_order, slice_dict = self.__process_args(out_orientation, **kwargs)
        image_data = self.__transpose(image_data, self.dims, out_order)
        return self.__get_slice(image_data, out_order, slice_dict)

    def __process_args(self, out_order, **kwargs):
        """
        take the arguments and convert them from say out_order="ZYX", kwargs{'T':3, 'C':0} and
        return out_order="TCZYX" and slice_dict = {'T':3, 'C':0, 'Z':slice(None,None), 'Y':slice(None,None),
        'X':slice(None,None)}
        :param out_order: The desired output order substring
        :param kwargs: any specific slices T=3, C=0
        :return: a 5 channel out_order, and a dictionary of specified slices
        """
        # use sets to get the channels that aren't in out_order
        out_order_set = set(out_order)
        ref_order_set = set(self.dims)
        slice_set = ref_order_set - out_order_set
        slice_dict = {}
        for channel in slice_set:
            specified_channel = kwargs.get(channel, 0)
            # check that the specified channel is within the defined domain
            if specified_channel >= self.shape[self.dims.find(channel)] or specified_channel < 0:
                raise ValueError("{} is not a valid index for the {} dimension".format(specified_channel, channel))
            slice_dict[channel] = specified_channel

        # add the slice equivalent of : for the other channels
        for channel in out_order_set:
            slice_dict[channel] = slice(None, None)

        # Add user-specified slices to the beginning of the returned order so subblock indexing is more efficient
        indices = {c: i for i, c in enumerate(self.dims)}  # constructs ordering dictionary apply to the set
        new_out_order = sorted(list(slice_set), key=indices.get)
        new_out_order = "".join(new_out_order) + out_order
        return new_out_order, slice_dict

    @staticmethod
    def __transpose(image_data, sdims, output_dims):
        """
        Takes an image data block and an ordered set of all channels
        :param image_data: block of image date in a 5D matrix
        :param sdims: the dims of the image data likely self.dims
        :param output_dims: the dims ordered the way the user wants
        :return: the image data block ordered as prescribed
        """
        match_map = {dim: sdims.find(dim) for dim in output_dims}
        transposer = [match_map[dim] for dim in output_dims]  # compose the order mapping
        transposed_image_data = image_data.transpose(transposer)
        # this changes the numpy wrapper around the data not the actual underlying data
        # thus even if the user has requested a reference the internal object isn't changed
        return transposed_image_data

    @staticmethod
    def __get_slice(image_data, out_order, slice_dict):
        """
        once the image data is sorted into out_order this function's purpose is to align the slice_dict to
        the same order and return the relevant slice/subblock of the data
        :param image_data:
        :param out_order:
        :param slice_dict:
        :return:
        """
        slice_list = [slice_dict[channel] for channel in out_order]
        return image_data[tuple(slice_list)]
