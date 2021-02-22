import dask.array as da
import numpy as np
from ome_types import from_xml, to_xml
from ome_types.model import OME, Image, Channel, Pixels, TiffData
import tifffile
from tifffile import TIFF
from typing import List, Tuple, Union

from .. import exceptions, types, get_module_version
from ..dimensions import (
    DEFAULT_DIMENSION_ORDER,
    DEFAULT_DIMENSION_ORDER_WITH_SAMPLES,
    DEFAULT_DIMENSION_ORDER_LIST_WITH_SAMPLES,
)
from ..exceptions import InvalidDimensionOrderingError
from ..metadata import utils
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
        data: Union[List[types.ArrayLike], types.ArrayLike],
        uri: types.PathLike,
        dimension_order: Union[str, List[str], None] = None,
        ome_xml: Union[str, OME, None] = None,
        channel_names: Union[List[str], List[List[str]], None] = None,
        image_name: Union[str, List[str], None] = None,
        pixels_physical_size: Union[
            Tuple[float, float, float], List[Tuple[float, float, float]]
        ] = (1.0, 1.0, 1.0),
        channel_colors: Union[List[int], List[List[int]], None] = None,
        **kwargs,
    ):
        """
        Write a data array to a file.

        Parameters
        ----------
        data: Union[List[types.ArrayLike], types.ArrayLike]
            The array of data to store. Data must have 2 to 5 dimensions.  If a list is
            provided, then it is understood to be multiple images written to the
            ome-tiff file. All following metadata parameters will be expanded to the
            length of this list.
        uri: types.PathLike
            The URI or local path for where to save the data.
        dimension_order: Union[str, list[str], None]
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
            The ome_xml will also be compared against the dimensions of the input data.
            If None is given, then OME-XML metadata will be generated from the data
            array and any of the following metadata arguments.
        channel_names: Union[List[str], List[List[str]], None]
            List of strings representing the names of the data channels
            Default: None
            If None is given, the list will be generated as a 0-indexed list of strings
            of the form "Channel:image_index:channel_index"
        image_names: Union[str, List[str], None]
            List of strings representing the names of the images
            Default: None
            If None is given, the list will be generated as a 0-indexed list of strings
            of the form "Image:image_index"
        pixels_physical_size: Union[Tuple[float, float, float],
                List[Tuple[float, float, float]]]
            List of numbers representing the physical pixel sizes in x,y,z in microns
            Default: (1,1,1)
        channel_colors: Union[List[int], List[List[int]], None]
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

        # If metadata is attached as lists
        # Enforcing matching shape
        if isinstance(data, list):
            num_scenes = len(data)
            if isinstance(dimension_order, list):
                if len(dimension_order) != len(data):
                    raise exceptions.ConflictingArgumentsError(
                        f"OmeTiffWriter received a list of arrays to use as scenes "
                        f"but the provided list of dimension_order is of different "
                        f"length. "
                        f"Number of provided scenes: {len(data)}, "
                        f"Number of provided known dimension strings: "
                        f"{len(dimension_order)}"
                    )
            if isinstance(image_name, list):
                if len(image_name) != len(data):
                    raise exceptions.ConflictingArgumentsError(
                        f"OmeTiffWriter received a list of arrays to use as scenes "
                        f"but the provided list of image_names is of different "
                        f"length. "
                        f"Number of provided scenes: {len(data)}, "
                        f"Number of provided known dimension strings: {len(image_name)}"
                    )
            if isinstance(pixels_physical_size, list):
                if len(pixels_physical_size) != len(data):
                    raise exceptions.ConflictingArgumentsError(
                        f"OmeTiffWriter received a list of arrays to use as scenes "
                        f"but the provided list of image_names is of different "
                        f"length. "
                        f"Number of provided scenes: {len(data)}, "
                        f"Number of provided known dimension strings: "
                        f"{len(pixels_physical_size)}"
                    )

            if channel_names is not None:
                if isinstance(channel_names[0], list):
                    if len(channel_names) != len(data):
                        raise exceptions.ConflictingArgumentsError(
                            f"OmeTiffWriter received a list of arrays to use as scenes "
                            f"but the provided list of channel_names is of different "
                            f"length. "
                            f"Number of provided scenes: {len(data)}, "
                            f"Number of provided known dimension strings: "
                            f"{len(channel_names)}"
                        )
            if channel_colors is not None:
                if isinstance(channel_colors[0], list):
                    if len(channel_colors) != len(data):
                        raise exceptions.ConflictingArgumentsError(
                            f"OmeTiffWriter received a list of arrays to use as scenes "
                            f"but the provided list of channel_colors is of different "
                            f"length. "
                            f"Number of provided scenes: {len(data)}, "
                            f"Number of provided known dimension strings: "
                            f"{len(channel_colors)}"
                        )

        if not isinstance(data, list):
            data = [data]
        num_scenes = len(data)

        # If metadata is attached as singles, expand to lists to match data
        if dimension_order is None or isinstance(dimension_order, str):
            dimension_order = [dimension_order] * num_scenes
        if image_name is None or isinstance(image_name, str):
            image_name = [image_name] * num_scenes
        if pixels_physical_size is None or isinstance(pixels_physical_size, tuple):
            pixels_physical_size = [pixels_physical_size] * num_scenes
        if channel_names is None or isinstance(channel_names[0], str):
            channel_names = [channel_names] * num_scenes
        if channel_colors is None or isinstance(channel_colors[0], int):
            channel_colors = [channel_colors] * num_scenes

        # once this is done, we can loop thru all items in data list...
        scene_index = 0

        # Assumption: if provided a dask array to save, it can fit into memory
        if isinstance(data[scene_index], da.core.Array):
            image_data = data[scene_index].compute()
        else:
            image_data = data[scene_index]

        # make sure we are writing 5D data to ome-tiff
        ome_dimension_order, image_data, is_rgb = OmeTiffWriter._resolve_dimensionality(
            image_data, dimension_order[scene_index]
        )

        xml = ""
        if ome_xml is None:
            ome_xml = OmeTiffWriter.build_ome(
                image_data.shape,
                image_data.dtype,
                channel_names=channel_names[scene_index],
                image_name=image_name[scene_index],
                pixels_physical_size=pixels_physical_size[scene_index],
                channel_colors=channel_colors[scene_index],
                dimension_order=ome_dimension_order,
                is_rgb=is_rgb,
            )
            xml = to_xml(ome_xml).encode()
        elif isinstance(ome_xml, str):
            # if the xml passed in is a string,
            # then just pass it straight through to the writer.
            # But first, validate it.
            valid_ome = from_xml(ome_xml)
            OmeTiffWriter._check_ome_dims(
                valid_ome, scene_index, image_data.shape, image_data.dtype
            )
            xml = to_xml(valid_ome).encode()
        elif isinstance(ome_xml, OME):
            # do some simple consistency check against the passed in OME dimensions
            OmeTiffWriter._check_ome_dims(
                ome_xml, scene_index, image_data.shape, image_data.dtype
            )
            xml = to_xml(ome_xml).encode()
        else:
            raise ValueError(
                "Unknown OME-XML metadata passed in. Use OME object, or xml string or \
                None"
            )

        # Save image to tiff!
        tif = tifffile.TiffWriter(
            path,
            bigtiff=OmeTiffWriter._size_of_ndarray(data=image_data) > BYTE_BOUNDARY,
        )

        tif.write(
            image_data,
            description=xml,
            photometric=TIFF.PHOTOMETRIC.RGB if is_rgb else TIFF.PHOTOMETRIC.MINISBLACK,
            planarconfig=TIFF.PLANARCONFIG.CONTIG,
            metadata=None,
            compression=TIFF.COMPRESSION.ADOBE_DEFLATE,
        )

        tif.close()

    @staticmethod
    def _resolve_dimensionality(
        data: types.ArrayLike,
        dimension_order: str,
    ) -> Tuple[str, types.ArrayLike, bool]:
        ndims = len(data.shape)

        # data is rgb if last dimension is S and its size is 3 or 4
        is_rgb = False
        if dimension_order is None:
            # we will only guess rgb here if ndims > 5
            # I could make a better guess if I look at any ome-xml passed in
            is_rgb = ndims > 5 and data.shape[-1] == 3 or data.shape[-1] == 4
            dimension_order = (
                DEFAULT_DIMENSION_ORDER_WITH_SAMPLES
                if is_rgb
                else DEFAULT_DIMENSION_ORDER
            )
        else:
            is_rgb = dimension_order[-1] == "S" and (
                data.shape[-1] == 3 or data.shape[-1] == 4
            )

        # select last 5 (or 6 if rgb) dims
        max_dims = 6 if is_rgb else 5
        if ndims > max_dims:
            slc = [0] * (ndims - max_dims)
            slc += [slice(None)] * max_dims
            data = data[slc]

        ndims = len(data.shape)
        assert (
            ndims == 6 or ndims == 5 or ndims == 4 or ndims == 3 or ndims == 2
        ), "Expected no more than 6 dimensions in data array"

        # assert valid characters in dimension_order
        if not (
            all(d in DEFAULT_DIMENSION_ORDER_LIST_WITH_SAMPLES for d in dimension_order)
        ):
            raise InvalidDimensionOrderingError(
                f"Invalid dimension_order {dimension_order}"
            )
        if dimension_order.find("S") > -1 and not is_rgb:
            raise InvalidDimensionOrderingError(
                "Samples must be last dimension if present, and only S=3 or 4 is \
                supported."
            )
        if dimension_order[-2:] != "YX" and dimension_order[-3:] != "YXS":
            raise InvalidDimensionOrderingError(
                f"Last characters of dimension_order {dimension_order} expected to \
                be YX or YXS.  Please transpose your data."
            )
        if len(dimension_order) < ndims:
            raise InvalidDimensionOrderingError(
                f"dimension_order {dimension_order} must have at least as many \
                dimensions as data shape {data.shape}"
            )

        # ensure dimension_order is same len as shape
        if len(dimension_order) > ndims:
            dimension_order = dimension_order[-ndims:]

        # remember whether S was a dim or not, and remove it for now
        if is_rgb:
            ndims = ndims - 1
            dimension_order = dimension_order[:-1]

        # expand to 5D and add appropriate dimensions
        if ndims == 2:
            data = np.expand_dims(data, axis=0)
            data = np.expand_dims(data, axis=0)
            data = np.expand_dims(data, axis=0)
            dimension_order = "TCZ" + dimension_order

        # expand to 5D and add appropriate dimensions
        elif ndims == 3:
            data = np.expand_dims(data, axis=0)
            data = np.expand_dims(data, axis=0)
            # prepend either TC, TZ or CZ
            if dimension_order[0] == "T":
                dimension_order = "CZ" + dimension_order
            elif dimension_order[0] == "C":
                dimension_order = "TZ" + dimension_order
            elif dimension_order[0] == "Z":
                dimension_order = "TC" + dimension_order

        # expand to 5D and add appropriate dimensions
        elif ndims == 4:
            data = np.expand_dims(data, axis=0)
            # prepend either T, C, or Z
            first2 = dimension_order[:2]
            if first2 == "TC" or first2 == "CT":
                dimension_order = "Z" + dimension_order
            elif first2 == "TZ" or first2 == "ZT":
                dimension_order = "C" + dimension_order
            elif first2 == "CZ" or first2 == "ZC":
                dimension_order = "T" + dimension_order

        return dimension_order, data, is_rgb

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
        data_shape: Tuple,
        data_dtype: np.dtype,
        dimension_order: str = DEFAULT_DIMENSION_ORDER,
        channel_names: List[str] = None,
        image_name: str = None,
        pixels_physical_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        channel_colors: List[int] = None,
        is_rgb: bool = False,
    ) -> OME:
        """Creates the necessary metadata for an OME tiff image
        :param data_shape: A 5- or 6-d tuple of TCZYX(S) dimensions
        :param data_dtype: a numpy dtype of the data array
        :param dimension_order: The order of dimensions in the data array, using
        T,C,Z,Y,X and optionally S
        :param channel_names: The names for each channel to be put into the OME metadata
        :param image_name: The name of the image to be put into the OME metadata
        :param pixels_physical_size: X,Y, and Z physical dimensions of each pixel,
        defaulting to microns
        :param channel_colors: The channel colors to be put into the OME metadata
        :param is_rgb: is a S dimension present?  S is expected to be the last dim in
        the data shape
        """
        image_index = 0
        shape = data_shape

        if len(shape) != 5 and not (is_rgb and len(shape) == 6):
            raise ValueError(
                "OmeTiffWriter.build_ome only accepts 5d arrays or 6d with RGB"
            )
        if len(dimension_order) != len(DEFAULT_DIMENSION_ORDER):
            raise ValueError("OmeTiffWriter.build_ome only accepts 5d dimension_order")
        for c in DEFAULT_DIMENSION_ORDER:
            if c not in dimension_order:
                raise ValueError(f"Unrecognized OME TIFF dimension {c}")

        def dim_or_1(dim):
            idx = dimension_order.find(dim)
            return 1 if idx == -1 else shape[idx]

        channel_count = dim_or_1("C")

        # should only ever be 1, 3 or 4
        samples = shape[-1] if is_rgb else 1

        # dimension_order must be set to the *reverse* of what dimensionality
        # the ome tif file is saved as
        pixels = Pixels(
            id="Pixels:0",
            dimension_order=dimension_order[::-1],
            type=utils.dtype_to_ome_type(data_dtype),
            size_t=dim_or_1("T"),
            size_c=channel_count * samples,
            size_z=dim_or_1("Z"),
            size_y=dim_or_1("Y"),
            size_x=dim_or_1("X"),
            interleaved=True if is_rgb else None,
        )

        if pixels_physical_size is not None:
            pixels.physical_size_x = pixels_physical_size[0]
            pixels.physical_size_y = pixels_physical_size[1]
            pixels.physical_size_z = pixels_physical_size[2]

        # one single tiffdata indicating sequential tiff IFDs based on dimension_order
        pixels.tiff_data_blocks = [
            TiffData(plane_count=pixels.size_t * channel_count * pixels.size_z)
        ]

        pixels.channels = [
            Channel(samples_per_pixel=samples) for i in range(channel_count)
        ]
        if channel_names is None:
            for i in range(channel_count):
                pixels.channels[i].id = utils.generate_ome_channel_id(
                    str(image_index), i
                )
                pixels.channels[i].name = "C:" + str(i)
        else:
            for i in range(channel_count):
                name = channel_names[i]
                pixels.channels[i].id = utils.generate_ome_channel_id(
                    str(image_index), i
                )
                pixels.channels[i].name = name

        if channel_colors is not None:
            assert len(channel_colors) >= pixels.size_c
            for i in range(channel_count):
                pixels.channels[i].color = channel_colors[i]

        img = Image(
            name=image_name,
            id=utils.generate_ome_image_id(str(image_index)),
            pixels=pixels,
        )

        # TODO get aics version string here
        ox = OME(creator=f"aicsimageio {get_module_version()}", images=[img])

        # validate????
        test = to_xml(ox)
        from_xml(test)

        return ox

    @staticmethod
    def _check_ome_dims(
        ome_xml: OME, image_index: int, data_shape: Tuple, data_dtype: np.dtype
    ):
        if len(ome_xml.images) < 1:
            raise ValueError("OME has no images")

        # look at number of samples from first channel only (possible bad assumption)
        samples = ome_xml.images[image_index].pixels.channels[0].samples_per_pixel

        # reverse the OME dimension order to compare against numpy shape
        dimension_order = ome_xml.images[image_index].pixels.dimension_order.value[::-1]

        dims = {
            "T": ome_xml.images[image_index].pixels.size_t,
            "C": ome_xml.images[image_index].pixels.size_c,
            "Z": ome_xml.images[image_index].pixels.size_z,
            "Y": ome_xml.images[image_index].pixels.size_y,
            "X": ome_xml.images[image_index].pixels.size_x,
        }
        if samples > 1:
            dims["C"] = len(ome_xml.images[image_index].pixels.channels)
            dims["S"] = samples
            dimension_order += "S"

        expected_shape = tuple(dims[i] for i in dimension_order)
        if expected_shape != data_shape:
            raise ValueError(
                f"OME shape {expected_shape} is not the same as data array shape: \
                {data_shape}"
            )

        expected_type = utils.ome_to_numpy_dtype(
            ome_xml.images[image_index].pixels.type
        )
        if expected_type != data_dtype:
            raise ValueError(
                f"OME pixel type {expected_type.name} is not the same as data array type: \
                {data_dtype.name}"
            )
