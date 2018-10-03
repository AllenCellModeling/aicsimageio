# author: Zach Crabtree zacharyc@alleninstitute.org

import numpy as np

from . import omeTifReader, cziReader, tifReader, typeChecker


def enum(**named_values):
    return type('Enum', (), named_values)
FileType = enum(OMETIF=1, TIF=2, CZI=3)


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

    """
    default_dims = "TCZYX"

    def __init__(self, data, **kwargs):
        """
        Constructor for AICSImage class
        :param data: String with path to ometif/czi file, or ndarray with up to 5 dimensions
        :param kwargs: If ndarray is used for data, then you can specify the dim ordering
                       with dims arg (ie dims="TZCYX").
        """
        self.dims = AICSImage.default_dims
        if isinstance(data, str):
            # input is a filepath
            self.file_path = data

            # check for compatible data types
            checker = typeChecker.TypeChecker(self.file_path)
            if checker.is_czi:
                self.reader = cziReader.CziReader(self.file_path)
            elif checker.is_ome:
                self.reader = omeTifReader.OmeTifReader(self.file_path)
            elif checker.is_tiff:
                self.reader = tifReader.TifReader(self.file_path)
            else:
                raise TypeError("CellImage can only accept OME-TIFF, TIFF, and CZI file formats!")
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
        dim_list = list(self.dims)
        for dim in self.dims:
            if not match_map[dim] == -1:
                transposer.append(match_map[dim])
        # self.dims = str(np.transpose(dim_list, transposer))
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
        """
        if kwargs.get("reference", False):
            # get data by reference
            image_data = self.data
        else:
            # make a copy of the data
            image_data = self.data.copy()
        if out_orientation != self.dims and self.is_valid_dimension(out_orientation):
            # map each dimension (TCZYX) to its index in out_orientation
            match_map = {dim: out_orientation.find(dim) for dim in self.dims}
            slicer, transposer = [], []
            for dim in self.dims:
                if match_map[dim] == -1:
                    # only get the bottom slice of this dimension, unless the user specified another in the args
                    slice_value = kwargs.get(dim, 0)
                    if slice_value >= self.shape[self.dims.find(dim)] or slice_value < 0:
                        raise ValueError("{} is not a valid index for the {} dimension".format(slice_value, dim))
                    slicer.append(slice_value)
                else:
                    # append all slices of this dimension
                    slicer.append(slice(None, None))
                    transposer.append(match_map[dim])
            image_data = image_data[tuple(slicer)].transpose(transposer)

        return image_data
