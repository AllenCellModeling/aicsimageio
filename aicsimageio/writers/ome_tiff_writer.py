import dask.array as da
from ome_types import from_xml, to_xml
from ome_types.model.ome import OME, Image, Pixels, PixelType, TiffData, Channel
import tifffile
from typing import List, Tuple, Union

from .. import types
from ..readers import DefaultReader
from ..utils import io_utils
from .writer import Writer

# This is the threshold to use BigTiff, if it's the 4GB boundary it should be 2**22 but
# the libtiff writer was unable to handle a 2GB numpy array.
# It would be great if we better understood exactly what this threshold is and how to
# calculate it but for now this is a stopgap working value
BYTE_BOUNDARY = 2 ** 21


class OmeTiffWriter(Writer):
    @staticmethod
    def save(
        data: types.ArrayLike,
        uri: types.PathLike,
        dim_order: str = None,
        ome_xml: Union[str, OME, None] = None,
        channel_names: List[str] = None,
        image_name: str = None,
        pixels_physical_size: List[float] = [1, 1, 1],
        channel_colors: List[float] = None,
        **kwargs,
    ):
        """
        Write a data array to a file.

        Parameters
        ----------
        data: types.ArrayLike
            The array of data to store. Data must have 2 to 5 dimensions
        uri: types.PathLike
            The URI or local path for where to save the data.
        dim_order: str
            The dimension order of the provided data.
            Default: None. Based off the number of dimensions, will assume
            the dimensions similar to how aicsimageio.readers.DefaultReader reads in
            data. That is, two dimensions: YX and three dimensions: YXS.
        ome_xml: Union[str, OME, None]
            Provided OME metadata. The metadata can be an xml string or an OME object
            from ome-types.
            Default: None
            The passed-in metadata will be validated against current OME_XML schema and
            raise exception if invalid.
            The ome_xml will also be compared against the dimensions if the input data.
            If None is given, then OME-XML metadata will be generated from the data
            array and any of the following metadata arguments.
        channel_names: List[str]
            List of strings representing the names of the data channels
            Default: None
            If None is given, the list will be created as a 0-indexed list of strings
            of the form "Channel:0"
        image_names: List[str]
            List of strings representing the names of the images
            Default: None
            If None is given, the list will be created as a 0-indexed list of strings
            of the form "Image:0"
        pixels_physical_size: List[float]
            List of numbers representing the physical pixel sizes in x,y,z in microns
            Default: [1,1,1]
        channel_colors: List[float]
            List of rgb color values per channel
            Default: None

        Examples
        --------
        >>> image = numpy.ndarray([1, 10, 3, 1024, 2048])
        ... OmeTiffWriter.save(image, "file.ome.tif")

        Convert a CZI file into OME-Tiff

        >>> reader = cziReader.CziReader("file3.czi")
        ... OmeTiffWriter.save(reader.load(), "file3.ome.tif")
        """
        # Check unpack uri and extension
        fs, path = io_utils.pathlike_to_fs(uri)
        (
            extension,
            imageio_mode,
        ) = DefaultReader._get_extension_and_mode(path)

        # Assumption: if provided a dask array to save, it can fit into memory
        if isinstance(data, da.core.Array):
            data = data.compute()

        dim_order, data = OmeTiffWriter._resolve_dimension_order(data, dim_order)

        # xml = OmeTiffWriter._resolve_ome_xml(ome_xml)
        xml = ""
        if ome_xml is None:
            ome_xml = OmeTiffWriter.build_ome(
                data,
                channel_names=channel_names,
                image_name=image_name,
                pixels_physical_size=pixels_physical_size,
                channel_colors=channel_colors,
                dimension_order=dim_order,
            )
            xml = to_xml(ome_xml).encode()
        elif isinstance(ome_xml, str):
            # if the xml passed in is a string,
            # then just pass it straight through to the writer.
            # do not validate or anything.(?)
            xml = ome_xml.encode()
        elif isinstance(ome_xml, OME):
            xml = to_xml(ome_xml).encode()
        else:
            raise (
                "Unknown OME-XML metadata passed in. Use OME object, or xml string or None"
            )

        # Save image to tiff!
        tif = tifffile.TiffWriter(
            path, bigtiff=OmeTiffWriter._size_of_ndarray(data=data) > BYTE_BOUNDARY
        )

        # check data shape for TZCYX or ZCYX or ZYX
        # minisblack instructs TiffWriter to not try to infer rgb color within the
        # data array metadata param fixes the double image description bug
        tif.save(
            data,
            compress=9,
            description=xml,
            photometric="minisblack",
            metadata=None,
        )

        tif.close()

    @staticmethod
    def _resolve_dimension_order(
        data: types.ArrayLike, dim_order: str
    ) -> Tuple[str, types.ArrayLike]:
        return dim_order, data
        # ndims = len(data.shape)

        # assert (
        #     ndims == 5 or ndims == 4 or ndims == 3
        # ), "Expected 3, 4, or 5 dimensions in data array"

        # # assert valid characters in dimension_order
        # if not (all(d in "STCZYX" for d in dimension_order)):
        #     raise InvalidDimensionOrderingError(
        #         f"Invalid dimension_order {dimension_order}"
        #     )
        # if dimension_order[-2:] != "YX":
        #     raise InvalidDimensionOrderingError(
        #         f"Last two characters of dimension_order {dimension_order} expected to \
        #         be YX.  Please transpose your data."
        #     )
        # if len(dimension_order) < ndims:
        #     raise InvalidDimensionOrderingError(
        #         f"dimension_order {dimension_order} must have at least as many \
        #         dimensions as data shape {shape}"
        #     )
        # if dimension_order.find("S") > 0:
        #     raise InvalidDimensionOrderingError(
        #         f"S must be the leading dim in dimension_order {dimension_order}"
        #     )
        # # todo ensure no letter appears more than once?

        # # ensure dimension_order is same len as shape
        # if len(dimension_order) > ndims:
        #     dimension_order = dimension_order[-ndims:]

        # # if this is 3d data, then expand to 5D and add appropriate dimensions
        # if ndims == 3:
        #     data = np.expand_dims(data, axis=0)
        #     data = np.expand_dims(data, axis=0)
        #     # prepend either TC, TZ or CZ
        #     if dimension_order[0] == "T":
        #         dimension_order = "CZ" + dimension_order
        #     elif dimension_order[0] == "C":
        #         dimension_order = "TZ" + dimension_order
        #     elif dimension_order[0] == "Z":
        #         dimension_order = "TC" + dimension_order

        # # if this is 4d data, then expand to 5D and add appropriate dimensions
        # elif ndims == 4:
        #     data = np.expand_dims(data, axis=0)
        #     # prepend either T, C, or Z
        #     first2 = dimension_order[:2]
        #     if first2 == "TC" or first2 == "CT":
        #         dimension_order = "Z" + dimension_order
        #     elif first2 == "TZ" or first2 == "ZT":
        #         dimension_order = "C" + dimension_order
        #     elif first2 == "CZ" or first2 == "ZC":
        #         dimension_order = "T" + dimension_order

    # def save(
    #     self,
    #     data,
    #     ome_xml=None,
    #     channel_names=None,
    #     image_name="IMAGE0",
    #     pixels_physical_size=None,
    #     channel_colors=None,
    #     dimension_order="STZCYX",
    # ):
    #     """
    #     Save an image with the proper OME XML metadata.

    #     Parameters
    #     ----------
    #     data: An array of 3, 4, or 5 dimensions to be written out to a file.
    #     ome_xml: A premade omexml.OMEXML object to use for metadata.
    #     channel_names: The names for each channel to be put into the OME metadata
    #     image_name: The name of the image to be put into the OME metadata
    #     pixels_physical_size: The physical size of each pixel in the image
    #     channel_colors: The channel colors to be put into the OME metadata
    #     dimension_order: The dimension ordering in the data array. Will be assumed
    #     STZCYX if not specified
    #     """
    #     if self.silent_pass:
    #         return

    #     shape = data.shape
    #     ndims = len(shape)

    #     assert (
    #         ndims == 5 or ndims == 4 or ndims == 3
    #     ), "Expected 3, 4, or 5 dimensions in data array"

    #     # assert valid characters in dimension_order
    #     if not (all(d in "STCZYX" for d in dimension_order)):
    #         raise InvalidDimensionOrderingError(
    #             f"Invalid dimension_order {dimension_order}"
    #         )
    #     if dimension_order[-2:] != "YX":
    #         raise InvalidDimensionOrderingError(
    #             f"Last two characters of dimension_order {dimension_order} expected to \
    #             be YX.  Please transpose your data."
    #         )
    #     if len(dimension_order) < ndims:
    #         raise InvalidDimensionOrderingError(
    #             f"dimension_order {dimension_order} must have at least as many \
    #             dimensions as data shape {shape}"
    #         )
    #     if dimension_order.find("S") > 0:
    #         raise InvalidDimensionOrderingError(
    #             f"S must be the leading dim in dimension_order {dimension_order}"
    #         )
    #     # todo ensure no letter appears more than once?

    #     # ensure dimension_order is same len as shape
    #     if len(dimension_order) > ndims:
    #         dimension_order = dimension_order[-ndims:]

    #     # if this is 3d data, then expand to 5D and add appropriate dimensions
    #     if ndims == 3:
    #         data = np.expand_dims(data, axis=0)
    #         data = np.expand_dims(data, axis=0)
    #         # prepend either TC, TZ or CZ
    #         if dimension_order[0] == "T":
    #             dimension_order = "CZ" + dimension_order
    #         elif dimension_order[0] == "C":
    #             dimension_order = "TZ" + dimension_order
    #         elif dimension_order[0] == "Z":
    #             dimension_order = "TC" + dimension_order

    #     # if this is 4d data, then expand to 5D and add appropriate dimensions
    #     elif ndims == 4:
    #         data = np.expand_dims(data, axis=0)
    #         # prepend either T, C, or Z
    #         first2 = dimension_order[:2]
    #         if first2 == "TC" or first2 == "CT":
    #             dimension_order = "Z" + dimension_order
    #         elif first2 == "TZ" or first2 == "ZT":
    #             dimension_order = "C" + dimension_order
    #         elif first2 == "CZ" or first2 == "ZC":
    #             dimension_order = "T" + dimension_order

    #     if ome_xml is None:
    #         self._make_meta(
    #             data,
    #             channel_names=channel_names,
    #             image_name=image_name,
    #             pixels_physical_size=pixels_physical_size,
    #             channel_colors=channel_colors,
    #             dimension_order=dimension_order,
    #         )
    #         xml = self.omeMetadata.to_xml().encode()
    #     elif isinstance(ome_xml, str):
    #         # if the xml passed in is a string,
    #         # then just pass it straight through to the writer.
    #         self.omeMetadata = omexml.OMEXML(ome_xml)
    #         xml = ome_xml.encode()
    #     else:
    #         pixels = ome_xml.image().Pixels
    #         pixels.populate_TiffData()
    #         self.omeMetadata = ome_xml
    #         xml = self.omeMetadata.to_xml().encode()

    #     tif = tifffile.TiffWriter(
    #         self.file_path, bigtiff=self._size_of_ndarray(data=data) > BYTE_BOUNDARY
    #     )

    #     # check data shape for TZCYX or ZCYX or ZYX
    #     if ndims == 5 or ndims == 4 or ndims == 3:
    #         # minisblack instructs TiffWriter to not try to infer rgb color within the
    #         # data array metadata param fixes the double image description bug
    #         tif.save(
    #             data,
    #             compress=9,
    #             description=xml,
    #             photometric="minisblack",
    #             metadata=None,
    #         )

    #     tif.close()

    @staticmethod
    def _size_of_ndarray(data: types.ArrayLike) -> int:
        """
        Calculate the size of data to determine if we require bigtiff
        Returns
        -------
        the size of data in bytes
        """
        if data is None:
            return 0
        size = data.size * data.itemsize
        return size

    # set up some sensible defaults from provided info
    @staticmethod
    def build_ome(
        data: types.ArrayLike,
        dim_order: str = None,
        channel_names: List[str] = None,
        image_name: str = None,
        pixels_physical_size: Tuple[float, float, float] = (1, 1, 1),
        channel_colors: List[int] = None,
    ) -> OME:
        """Creates the necessary metadata for an OME tiff image
        :param data: An array to be written out to a file.
        :param channel_names: The names for each channel to be put into the OME metadata
        :param image_name: The name of the image to be put into the OME metadata
        :param pixels_physical_size: The physical size of each pixel in the image
        :param channel_colors: The channel colors to be put into the OME metadata
        :param dimension_order: The order of dimensions in the data array, using
        T,C,Z,Y and X
        """
        pixels = Pixels(id="0")
        if pixels_physical_size is not None:
            pixels.physical_size_x = pixels_physical_size[0]
            pixels.physical_size_y = pixels_physical_size[1]
            pixels.physical_size_z = pixels_physical_size[2]
        shape = data.shape

        def dim_or_1(dim):
            idx = dim_order.find(dim)
            return 1 if idx == -1 else shape[idx]

        # pixels.channel_count = dim_or_1("C")
        pixels.size_t = dim_or_1("T")
        pixels.size_c = dim_or_1("C")
        pixels.size_z = dim_or_1("Z")
        pixels.size_y = dim_or_1("Y")
        pixels.size_x = dim_or_1("X")

        # this must be set to the *reverse* of what dimensionality
        # the ome tif file is saved as
        pixels.dimension_order = dim_order[::-1]

        # convert our numpy dtype to a ome compatible pixeltype
        pixels.type = PixelType(data.dtype)

        # one single tiffdata indicating sequential tiff IFDs based on dimension_order
        pixels.tiff_data_blocks = [
            TiffData(plane_count=pixels.size_t * pixels.size_c * pixels.size_z)
        ]

        pixels.channels = [Channel() for i in range(pixels.size_c)]
        if channel_names is None:
            for i in range(pixels.size_c):
                pixels.channels[i].id = "Channel:0:" + str(i)
                pixels.channels[i].name = "C:" + str(i)
        else:
            for i in range(pixels.size_c):
                name = channel_names[i]
                pixels.channels[i].id = "Channel:0:" + str(i)
                pixels.channels[i].name = name

        if channel_colors is not None:
            assert len(channel_colors) >= pixels.size_c
            for i in range(pixels.size_c):
                pixels.channels[i].color = channel_colors[i]

        # assume 1 sample per channel
        for i in range(pixels.size_c):
            pixels.channels[i].samples_per_pixel = 1

        img = Image(name=image_name, id="Image:0", pixels=pixels)

        # TODO get aics version string here
        ox = OME(creator="aicsimageio 4.x", images=[img])

        # validate????
        test = to_xml(ox)
        tested = from_xml(test)

        return ox
