from typing import Any, List, Optional, Tuple, Union

import dask.array as da
import numpy as np
import tifffile
from fsspec.implementations.local import LocalFileSystem
from ome_types import from_xml, to_xml
from ome_types.model import OME, Channel, Image, Pixels, TiffData
from tifffile import TIFF

from .. import exceptions, get_module_version, types
from ..dimensions import (
    DEFAULT_DIMENSION_ORDER,
    DEFAULT_DIMENSION_ORDER_LIST_WITH_SAMPLES,
    DEFAULT_DIMENSION_ORDER_WITH_SAMPLES,
    DimensionNames,
)
from ..metadata import utils
from ..utils import io_utils
from .writer import Writer

# This is the threshold to use BigTiff, if it's the 4GB boundary it should be 2**22 but
# the libtiff writer was unable to handle a 2GB numpy array.
# It would be great if we better understood exactly what this threshold is and how to
# calculate it but for now this is a stopgap working value
BIGTIFF_BYTE_LIMIT = 2 ** 21


class OmeTiffWriter(Writer):
    @staticmethod
    def save(
        data: Union[List[types.ArrayLike], types.ArrayLike],
        uri: types.PathLike,
        dim_order: Optional[Union[str, List[Union[str, None]]]] = None,
        ome_xml: Optional[Union[str, OME]] = None,
        channel_names: Optional[Union[List[str], List[Optional[List[str]]]]] = None,
        image_name: Optional[Union[str, List[Union[str, None]]]] = None,
        physical_pixel_sizes: Optional[
            Union[
                types.PhysicalPixelSizes,
                List[types.PhysicalPixelSizes],
            ]
        ] = None,
        channel_colors: Optional[Union[List[int], List[Optional[List[int]]]]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Write a data array to a file.

        Parameters
        ----------
        data: Union[List[types.ArrayLike], types.ArrayLike]
            The array of data to store. Data arrays must have 2 to 6 dimensions. If a
            list is provided, then it is understood to be multiple images written to the
            ome-tiff file. All following metadata parameters will be expanded to the
            length of this list.
        uri: types.PathLike
            The URI or local path for where to save the data.
            Note: OmeTiffWriter can only write to local file systems.
        dim_order: Optional[Union[str, List[Union[str, None]]]]
            The dimension order of the provided data.
            Dimensions must be a list of T, C, Z, Y, Z, and S (S=samples for rgb data).
            Dimension strings must be same length as number of dimensions in the data.
            If S is present it must be last and its data count must be 3 or 4.
            Default: None.
            If None is provided for any data array, we will guess dimensions based on a
            TCZYX ordering.
            In the None case, data will be assumed to be scalar, not RGB.
        ome_xml: Optional[Union[str, OME]]
            Provided OME metadata. The metadata can be an xml string or an OME object
            from ome-types. A provided ome_xml will override any other provided
            metadata arguments.
            Default: None
            The passed-in metadata will be validated against current OME_XML schema and
            raise exception if invalid.
            The ome_xml will also be compared against the dimensions of the input data.
            If None is given, then OME-XML metadata will be generated from the data
            array and any of the following metadata arguments.
        channel_names: Optional[Union[List[str], List[Optional[List[str]]]]]
            Lists of strings representing the names of the data channels
            Default: None
            If None is given, the list will be generated as a 0-indexed list of strings
            of the form "Channel:image_index:channel_index"
        image_names: Optional[Union[str, List[Union[str, None]]]]
            List of strings representing the names of the images
            Default: None
            If None is given, the list will be generated as a 0-indexed list of strings
            of the form "Image:image_index"
        physical_pixel_sizes: Optional[Union[types.PhysicalPixelSizes,
                List[types.PhysicalPixelSizes]]]
            List of numbers representing the physical pixel sizes in Z, Y, X in microns
            Default: None
            If None is given, pixel size will be (1.0, 1.0, 1.0) for all images
        channel_colors: Optional[Union[List[int], List[Optional[List[int]]]]]
            List of rgb color values per channel. These must be values compatible with
            the OME spec.
            Default: None

        Raises
        ------
        ValueError:
            Non-local file system URI provided.

        Examples
        --------
        Write a TCZYX data set to OME-Tiff

        >>> image = numpy.ndarray([1, 10, 3, 1024, 2048])
        ... OmeTiffWriter.save(image, "file.ome.tif")

        Write data with a dimension order into OME-Tiff

        >>> image = numpy.ndarray([10, 3, 1024, 2048])
        ... OmeTiffWriter.save(image, "file.ome.tif", dim_order="ZCYX")

        Write multi-scene data to OME-Tiff, specifying channel names

        >>> image0 = numpy.ndarray([3, 10, 1024, 2048])
        ... image1 = numpy.ndarray([3, 10, 512, 512])
        ... OmeTiffWriter.save(
        ...     [image0, image1],
        ...     "file.ome.tif",
        ...     dim_order="CZYX",  # this single value will be repeated to each image
        ...     channel_names=[["C00","C01","C02"],["C10","C11","C12"]]
        ... )
        """
        # Resolve final destination
        fs, path = io_utils.pathlike_to_fs(uri)

        # Catch non-local file system
        if not isinstance(fs, LocalFileSystem):
            raise ValueError(
                f"Cannot write to non-local file system. "
                f"Received URI: {uri}, which points to {type(fs)}."
            )

        # If metadata is attached as lists, enforce matching shape
        if isinstance(data, list):
            num_images = len(data)
            if isinstance(dim_order, list):
                if len(dim_order) != num_images:
                    raise exceptions.ConflictingArgumentsError(
                        f"OmeTiffWriter received a list of arrays to use as scenes "
                        f"but the provided list of dimension_order is of different "
                        f"length. "
                        f"Number of provided scenes: {num_images}, "
                        f"Number of provided dimension strings: "
                        f"{len(dim_order)}"
                    )
            if isinstance(image_name, list):
                if len(image_name) != num_images:
                    raise exceptions.ConflictingArgumentsError(
                        f"OmeTiffWriter received a list of arrays to use as scenes "
                        f"but the provided list of image_names is of different "
                        f"length. "
                        f"Number of provided scenes: {num_images}, "
                        f"Number of provided dimension strings: {len(image_name)}"
                    )
            if isinstance(physical_pixel_sizes, list):
                if len(physical_pixel_sizes) != num_images:
                    raise exceptions.ConflictingArgumentsError(
                        f"OmeTiffWriter received a list of arrays to use as scenes "
                        f"but the provided list of image_names is of different "
                        f"length. "
                        f"Number of provided scenes: {num_images}, "
                        f"Number of provided dimension strings: "
                        f"{len(physical_pixel_sizes)}"
                    )

            if channel_names is not None:
                if isinstance(channel_names[0], list):
                    if len(channel_names) != num_images:
                        raise exceptions.ConflictingArgumentsError(
                            f"OmeTiffWriter received a list of arrays to use as scenes "
                            f"but the provided list of channel_names is of different "
                            f"length. "
                            f"Number of provided scenes: {num_images}, "
                            f"Number of provided dimension strings: "
                            f"{len(channel_names)}"
                        )
            if channel_colors is not None:
                if isinstance(channel_colors[0], list):
                    if len(channel_colors) != num_images:
                        raise exceptions.ConflictingArgumentsError(
                            f"OmeTiffWriter received a list of arrays to use as scenes "
                            f"but the provided list of channel_colors is of different "
                            f"length. "
                            f"Number of provided scenes: {num_images}, "
                            f"Number of provided dimension strings: "
                            f"{len(channel_colors)}"
                        )

        # make sure data is a list
        if not isinstance(data, list):
            data = [data]
        num_images = len(data)

        # If metadata is attached as singles, expand to lists to match data
        if dim_order is None or isinstance(dim_order, str):
            dim_order = [dim_order] * num_images
        if image_name is None or isinstance(image_name, str):
            image_name = [image_name] * num_images
        if isinstance(physical_pixel_sizes, tuple):
            physical_pixel_sizes = [physical_pixel_sizes] * num_images
        elif physical_pixel_sizes is None:
            physical_pixel_sizes = [
                types.PhysicalPixelSizes(1.0, 1.0, 1.0)
            ] * num_images
        if channel_names is None or isinstance(channel_names[0], int):
            channel_names = [channel_names] * num_images  # type: ignore
        if channel_colors is None or isinstance(channel_colors[0], int):
            channel_colors = [channel_colors] * num_images  # type: ignore

        xml = ""
        # try to construct OME from params
        if ome_xml is None:
            ome_xml = OmeTiffWriter.build_ome(
                [i.shape for i in data],
                [i.dtype for i in data],
                channel_names=channel_names,  # type: ignore
                image_name=image_name,
                physical_pixel_sizes=physical_pixel_sizes,
                channel_colors=channel_colors,  # type: ignore
                dimension_order=dim_order,
            )
        # else if string, then construct OME from string
        elif isinstance(ome_xml, str):
            ome_xml = from_xml(ome_xml)

        # if we do not have an OME object now, something is wrong
        if not isinstance(ome_xml, OME):
            raise TypeError(
                "Unknown OME-XML metadata passed in. Use OME object, or xml string or \
                None"
            )

        # vaidate ome
        for scene_index in range(num_images):
            OmeTiffWriter._check_ome_dims(
                ome_xml, scene_index, data[scene_index].shape, data[scene_index].dtype
            )

        # convert to string for writing
        xml = to_xml(ome_xml).encode()

        # Save image to tiff!
        with fs.open(path, "wb") as open_resource:
            tif = tifffile.TiffWriter(
                open_resource,
                bigtiff=OmeTiffWriter._size_of_ndarray(data=data) > BIGTIFF_BYTE_LIMIT,
            )

            # now the heavy lifting. assemble the raw data and write it
            for scene_index in range(num_images):
                image_data = data[scene_index]
                # Assumption: if provided a dask array to save, it can fit into memory
                if isinstance(image_data, da.core.Array):
                    image_data = data[scene_index].compute()  # type: ignore

                description = xml if scene_index == 0 else None
                # assume if first channel is rgb then all of it is
                is_rgb = (
                    ome_xml.images[scene_index].pixels.channels[0].samples_per_pixel > 1
                )
                photometric = (
                    TIFF.PHOTOMETRIC.RGB if is_rgb else TIFF.PHOTOMETRIC.MINISBLACK
                )
                planarconfig = TIFF.PLANARCONFIG.CONTIG if is_rgb else None
                tif.write(
                    image_data,
                    description=description,
                    photometric=photometric,
                    metadata=None,
                    planarconfig=planarconfig,
                    compression=TIFF.COMPRESSION.ADOBE_DEFLATE,
                )

            tif.close()

    @staticmethod
    def _resolve_OME_dimension_order(
        shape: Tuple[int, ...], dimension_order: Union[str, None]
    ) -> Tuple[str, bool]:
        """
        Do some dimension validation and return an ome-compatible 5D dimension order
        and whether the data is rgb multisample

        Parameters
        ----------
        shape: Tuple[int, ...]
            A data array shape
        dimension_order: Union[str, None]
            A dimension order string, composed of some subset of TCZYXS

        Returns
        -------
        Tuple[str, bool]
            An OME-compatible 5D dimension_order string and a boolean for whether the
            data shape had rgb samples
        """
        ndims = len(shape)

        if ndims > 5 and (shape[-1] != 3 and shape[-1] != 4):
            raise ValueError(
                f"Passed in greater than 5D data but last dimension is not 3 or 4: "
                f"{shape[-1]}"
            )

        if dimension_order is not None and len(dimension_order) != ndims:
            raise exceptions.InvalidDimensionOrderingError(
                f"Dimension order string has {len(dimension_order)} dims but data "
                f"shape has {ndims} dims"
            )

        # data is rgb if last dimension is S and its size is 3 or 4
        is_rgb = False
        if dimension_order is None:
            # we will only guess rgb here if ndims > 5
            # I could make a better guess if I look at any ome-xml passed in
            is_rgb = ndims > 5 and (shape[-1] == 3 or shape[-1] == 4)
            dimension_order = (
                DEFAULT_DIMENSION_ORDER_WITH_SAMPLES
                if is_rgb
                else DEFAULT_DIMENSION_ORDER
            )
        else:
            is_rgb = dimension_order[-1] == DimensionNames.Samples and (
                shape[-1] == 3 or shape[-1] == 4
            )

        if (ndims > 5 and not is_rgb) or ndims > 6 or ndims < 2:
            raise ValueError(
                f"Data array has unexpected number of dimensions: is_rgb = {is_rgb} "
                f"and shape is {shape}"
            )

        # assert valid characters in dimension_order
        if not (
            all(d in DEFAULT_DIMENSION_ORDER_LIST_WITH_SAMPLES for d in dimension_order)
        ):
            raise exceptions.InvalidDimensionOrderingError(
                f"Invalid dimension_order {dimension_order}"
            )
        if dimension_order.find(DimensionNames.Samples) > -1 and not is_rgb:
            raise exceptions.InvalidDimensionOrderingError(
                "Samples must be last dimension if present, and only S=3 or 4 is \
                supported."
            )
        if dimension_order[-2:] != "YX" and dimension_order[-3:] != "YXS":
            raise exceptions.InvalidDimensionOrderingError(
                f"Last characters of dimension_order {dimension_order} expected to \
                be YX or YXS.  Please transpose your data."
            )

        # remember whether S was a dim or not, and remove it for now
        if is_rgb:
            ndims = ndims - 1
            dimension_order = dimension_order[:-1]

        # expand to 5D and add appropriate dimensions
        if len(dimension_order) == 2:
            dimension_order = "TCZ" + dimension_order

        # expand to 5D and add appropriate dimensions
        elif len(dimension_order) == 3:
            # prepend either TC, TZ or CZ
            if dimension_order[0] == DimensionNames.Time:
                dimension_order = "CZ" + dimension_order
            elif dimension_order[0] == DimensionNames.Channel:
                dimension_order = "TZ" + dimension_order
            elif dimension_order[0] == DimensionNames.SpatialZ:
                dimension_order = "TC" + dimension_order

        # expand to 5D and add appropriate dimensions
        elif len(dimension_order) == 4:
            # prepend either T, C, or Z
            first2 = dimension_order[:2]
            if first2 == "TC" or first2 == "CT":
                dimension_order = DimensionNames.SpatialZ + dimension_order
            elif first2 == "TZ" or first2 == "ZT":
                dimension_order = DimensionNames.Channel + dimension_order
            elif first2 == "CZ" or first2 == "ZC":
                dimension_order = DimensionNames.Time + dimension_order

        return dimension_order, is_rgb

    @staticmethod
    def _size_of_ndarray(data: List[types.ArrayLike]) -> int:
        """
        Calculate the size of data to determine if we require bigtiff

        Parameters
        ----------
        data: list of data arrays, one per image to be saved to tiff

        Returns
        -------
        the total size of data in bytes
        """
        size = 0
        for i in range(len(data)):
            size += data[i].size * data[i].itemsize
        return size

    @staticmethod
    def _extend_data_shape(shape: Tuple[int, ...], num_dims: int) -> Tuple[int, ...]:
        # extend data shape to be same len as dimension_order
        if len(shape) < num_dims:
            shape = tuple([1] * (num_dims - len(shape))) + shape
        return shape

    @staticmethod
    def _build_ome_image(
        image_index: int = 0,
        tiff_plane_offset: int = 0,
        data_shape: Tuple[int, ...] = (1, 1, 1, 1, 1),
        data_dtype: np.dtype = np.dtype(np.uint8),
        is_rgb: bool = False,
        dimension_order: str = DEFAULT_DIMENSION_ORDER,
        image_name: Optional[str] = "I0",
        physical_pixel_sizes: types.PhysicalPixelSizes = types.PhysicalPixelSizes(
            1.0, 1.0, 1.0
        ),
        channel_names: List[str] = None,
        channel_colors: List[int] = None,
    ) -> Image:
        if len(data_shape) < 2 or len(data_shape) > 6:
            raise ValueError(f"Bad OME image shape length: {data_shape}")

        # extend data shape to be same len as dimension_order, accounting for rgb
        if is_rgb:
            data_shape = OmeTiffWriter._extend_data_shape(
                data_shape, len(dimension_order) + 1
            )
        else:
            data_shape = OmeTiffWriter._extend_data_shape(
                data_shape, len(dimension_order)
            )

        def dim_or_1(dim: str) -> int:
            idx = dimension_order.find(dim)
            return 1 if idx == -1 else data_shape[idx]

        channel_count = dim_or_1(DimensionNames.Channel)

        if len(dimension_order) != 5:
            raise ValueError(f"Unrecognized OME TIFF dimension order {dimension_order}")
        for c in dimension_order:
            if c not in DEFAULT_DIMENSION_ORDER:
                raise ValueError(f"Unrecognized OME TIFF dimension {c}")
        if isinstance(channel_names, list) and len(channel_names) != channel_count:
            raise ValueError(f"Wrong number of channel names {len(channel_names)}")
        if isinstance(channel_colors, list) and len(channel_colors) != channel_count:
            raise ValueError(f"Wrong number of channel colors {len(channel_colors)}")

        samples_per_pixel = 1
        if is_rgb:
            samples_per_pixel = data_shape[-1]

        # dimension_order must be set to the *reverse* of what dimensionality
        # the ome tif file is saved as
        pixels = Pixels(
            id=f"Pixels:{image_index}:0",
            dimension_order=dimension_order[::-1],
            type=utils.dtype_to_ome_type(data_dtype),
            size_t=dim_or_1(DimensionNames.Time),
            size_c=channel_count * samples_per_pixel,
            size_z=dim_or_1(DimensionNames.SpatialZ),
            size_y=dim_or_1(DimensionNames.SpatialY),
            size_x=dim_or_1(DimensionNames.SpatialX),
            interleaved=True if samples_per_pixel > 1 else None,
        )
        # expected in ZYX order
        pixels.physical_size_z = physical_pixel_sizes.Z
        pixels.physical_size_y = physical_pixel_sizes.Y
        pixels.physical_size_x = physical_pixel_sizes.X

        # one single tiffdata indicating sequential tiff IFDs based on dimension_order
        pixels.tiff_data_blocks = [
            TiffData(
                plane_count=pixels.size_t * channel_count * pixels.size_z,
                ifd=tiff_plane_offset,
            )
        ]

        pixels.channels = [
            Channel(samples_per_pixel=samples_per_pixel) for i in range(channel_count)
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
        return img

    @staticmethod
    def build_ome(
        data_shapes: List[Tuple[int, ...]],
        data_types: List[np.dtype],
        dimension_order: Optional[List[Optional[str]]] = None,
        channel_names: Optional[List[Optional[List[str]]]] = None,
        image_name: List[Optional[str]] = None,
        physical_pixel_sizes: List[types.PhysicalPixelSizes] = None,
        channel_colors: List[Optional[List[int]]] = None,
    ) -> OME:
        """

        Create the necessary metadata for an OME tiff image

        Parameters
        ----------
        data_shapes:
            A list of 5- or 6-d tuples
        data_types:
            A list of data types
        dimension_order:
            The order of dimensions in the data array, using
            T,C,Z,Y,X and optionally S
        channel_names:
            The names for each channel to be put into the OME metadata
        image_name:
            The name of the image to be put into the OME metadata
        physical_pixel_sizes:
            Z,Y, and X physical dimensions of each pixel,
            defaulting to microns
        channel_colors:
            The channel colors to be put into the OME metadata
        is_rgb:
            is a S dimension present?  S is expected to be the last dim in
            the data shape

        Returns
        -------
        OME
            An OME object that can be converted to a valid OME-XML string
        """
        num_images = len(data_shapes)
        # resolve defaults that are None
        if dimension_order is None:
            dimension_order = [None] * num_images
        if channel_names is None:
            channel_names = [None] * num_images
        if image_name is None:
            image_name = [None] * num_images
        if physical_pixel_sizes is None:
            physical_pixel_sizes = [
                types.PhysicalPixelSizes(1.0, 1.0, 1.0)
            ] * num_images
        if channel_colors is None:
            channel_colors = [None] * num_images

        # assert all lists are same length
        if (
            num_images != len(data_types)
            or num_images != len(dimension_order)
            or num_images != len(channel_names)
            or num_images != len(image_name)
            or num_images != len(physical_pixel_sizes)
            or num_images != len(channel_colors)
        ):
            raise ValueError("Mismatched array counts in parameters to build_ome")

        images = []
        tiff_plane_offset = 0
        for image_index in range(len(data_shapes)):
            # correct the dimension_order for ome
            ome_dimension_order, is_rgb = OmeTiffWriter._resolve_OME_dimension_order(
                data_shapes[image_index], dimension_order[image_index]
            )
            img = OmeTiffWriter._build_ome_image(
                image_index,
                tiff_plane_offset,
                data_shapes[image_index],
                data_types[image_index],
                is_rgb,
                ome_dimension_order,
                image_name[image_index],
                physical_pixel_sizes[image_index],
                channel_names[image_index],
                channel_colors[image_index],
            )
            # increment tiff_plane_offset for next image
            tiff_plane_offset += (
                img.pixels.size_z * img.pixels.size_t * len(img.pixels.channels)
            )
            images.append(img)

        ome_object = OME(creator=f"aicsimageio {get_module_version()}", images=images)

        # validate! (TODO: Is there a better api in ome-types for this?)
        test = to_xml(ome_object)
        from_xml(test)

        return ome_object

    @staticmethod
    def _check_ome_dims(
        ome_xml: OME, image_index: int, data_shape: Tuple, data_dtype: np.dtype
    ) -> None:
        if len(ome_xml.images) < 1:
            raise ValueError("OME has no images")

        # look at number of samples from first channel only (possible bad assumption)
        samples = ome_xml.images[image_index].pixels.channels[0].samples_per_pixel

        # reverse the OME dimension order to compare against numpy shape
        dimension_order = ome_xml.images[image_index].pixels.dimension_order.value[::-1]
        dims = {
            DimensionNames.Time: ome_xml.images[image_index].pixels.size_t,
            DimensionNames.Channel: ome_xml.images[image_index].pixels.size_c,
            DimensionNames.SpatialZ: ome_xml.images[image_index].pixels.size_z,
            DimensionNames.SpatialY: ome_xml.images[image_index].pixels.size_y,
            DimensionNames.SpatialX: ome_xml.images[image_index].pixels.size_x,
        }
        if samples > 1:
            dims[DimensionNames.Channel] = len(
                ome_xml.images[image_index].pixels.channels
            )
            dims[DimensionNames.Samples] = samples
            dimension_order += DimensionNames.Samples

        expected_shape = tuple(dims[i] for i in dimension_order)
        data_shape = OmeTiffWriter._extend_data_shape(data_shape, len(dimension_order))
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
