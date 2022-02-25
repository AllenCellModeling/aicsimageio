from typing import Any, Dict, List, Optional, Tuple, Union

import dask.array as da
from fsspec.implementations.local import LocalFileSystem

# import shutil
import numpy
import zarr
import pathlib

from ome_zarr.writer import write_image
from ome_zarr.scale import Scaler
from ome_zarr.io import parse_url

from .. import exceptions, types
from ..utils import io_utils
from .writer import Writer


class OmeZarrWriter(Writer):
    @staticmethod
    def save(
        data: Union[List[types.ArrayLike], types.ArrayLike],
        uri: types.PathLike,
        dim_order: Optional[Union[str, List[Union[str, None]]]] = None,
        channel_names: Optional[Union[List[str], List[Optional[List[str]]]]] = None,
        image_name: Optional[Union[str, List[Union[str, None]]]] = None,
        physical_pixel_sizes: Optional[
            Union[types.PhysicalPixelSizes, List[types.PhysicalPixelSizes]]
        ] = None,
        channel_colors: Optional[
            Union[List[List[int]], List[Optional[List[List[int]]]]]
        ] = None,
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
            Note: OmeZarrWriter can only write to local file systems.
        dim_order: Optional[Union[str, List[Union[str, None]]]]
            The dimension order of the provided data.
            Dimensions must be a list of T, C, Z, Y, Z, and S (S=samples for rgb data).
            Dimension strings must be same length as number of dimensions in the data.
            If S is present it must be last and its data count must be 3 or 4.
            Default: None.
            If None is provided for any data array, we will guess dimensions based on a
            TCZYX ordering.
            In the None case, data will be assumed to be scalar, not RGB.
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
        channel_colors: Optional[Union[List[List[int]], List[Optional[List[List[int]]]]]
            List of rgb color values per channel or a list of lists for each image.
            These must be values compatible with the OME spec.
            Default: None

        Raises
        ------
        ValueError:
            Non-local file system URI provided.

        Examples
        --------
        Write a TCZYX data set to OME-Tiff

        >>> image = numpy.ndarray([1, 10, 3, 1024, 2048])
        ... OmeZarrWriter.save(image, "file.ome.tif")

        Write data with a dimension order into OME-Tiff

        >>> image = numpy.ndarray([10, 3, 1024, 2048])
        ... OmeZarrWriter.save(image, "file.ome.tif", dim_order="ZCYX")

        Write multi-scene data to OME-Tiff, specifying channel names

        >>> image0 = numpy.ndarray([3, 10, 1024, 2048])
        ... image1 = numpy.ndarray([3, 10, 512, 512])
        ... OmeZarrWriter.save(
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
                        f"OmeZarrWriter received a list of arrays to use as scenes "
                        f"but the provided list of dimension_order is of different "
                        f"length. "
                        f"Number of provided scenes: {num_images}, "
                        f"Number of provided dimension strings: "
                        f"{len(dim_order)}"
                    )
            if isinstance(image_name, list):
                if len(image_name) != num_images:
                    raise exceptions.ConflictingArgumentsError(
                        f"OmeZarrWriter received a list of arrays to use as scenes "
                        f"but the provided list of image_names is of different "
                        f"length. "
                        f"Number of provided scenes: {num_images}, "
                        f"Number of provided dimension strings: {len(image_name)}"
                    )
            if isinstance(physical_pixel_sizes, list):
                if len(physical_pixel_sizes) != num_images:
                    raise exceptions.ConflictingArgumentsError(
                        f"OmeZarrWriter received a list of arrays to use as scenes "
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
                            f"OmeZarrWriter received a list of arrays to use as scenes "
                            f"but the provided list of channel_names is of different "
                            f"length. "
                            f"Number of provided scenes: {num_images}, "
                            f"Number of provided dimension strings: "
                            f"{len(channel_names)}"
                        )
            if channel_colors is not None:
                if isinstance(channel_colors[0], list):
                    if not isinstance(channel_colors[0][0], int):
                        if len(channel_colors) != num_images:
                            raise exceptions.ConflictingArgumentsError(
                                f"OmeZarrWriter received a list of arrays to use as "
                                f"scenes but the provided list of channel_colors is of "
                                f"different length. "
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
                types.PhysicalPixelSizes(None, None, None)
            ] * num_images
        if channel_names is None or isinstance(channel_names[0], str):
            channel_names = [channel_names] * num_images  # type: ignore

        if channel_colors is not None:
            if all(
                [
                    (
                        channel_colors[img_idx] is None
                        or isinstance(channel_colors[img_idx], list)
                    )
                    for img_idx in range(num_images)
                ]
            ):
                single_image_channel_colors_provided = False
            else:
                single_image_channel_colors_provided = True

            if (
                channel_colors[0] is not None
                and isinstance(channel_colors[0], list)
                and isinstance(channel_colors[0][0], int)
            ):
                single_image_channel_colors_provided = True

        if channel_colors is None or single_image_channel_colors_provided:
            channel_colors = [channel_colors] * num_images  # type: ignore

        # Save image to zarr store!
        mypath = pathlib.Path(uri)
        # TODO handle overwrite scenario?
        # shutil.rmtree(mypath)
        # print(mypath)
        store = parse_url(mypath, mode="w").store
        # print(store)
        root = zarr.group(store=store)
        # print(root)
        for scene_index in range(num_images):
            image_data = data[scene_index]
            pixelsize = physical_pixel_sizes[scene_index]
            # TODO can this be deferred to inside of write_image?
            # Assumption: if provided a dask array to save, it can fit into memory
            if isinstance(image_data, da.core.Array):
                image_data = data[scene_index].compute()  # type: ignore

            # TODO image names must be unique!!!!!!
            group = root.create_group(image_name[scene_index])
            # TODO scaler might want to use different method for segmentations than raw
            # TODO control how many levels of zarr are created
            scaler = Scaler()
            scaler.method = "nearest"
            # TODO calculate these for desired downscaling factors
            scaler.max_layer = 2
            scaler.downscale = 2

            # try to construct per-image metadata
            ome_json = OmeZarrWriter.build_ome(
                image_data.shape,
                channel_names=channel_names[scene_index],  # type: ignore
                channel_colors=channel_colors[scene_index],  # type: ignore
                # this can be slow if going over all T values,
                # might be better if user supplies the min/max?
                channel_minmax=[
                    (numpy.min(image_data[:, i, :]), numpy.max(image_data[:, i, :]))
                    for i in range(image_data.shape[1])
                ],
            )

            write_image(
                image_data,
                group,
                chunks=(60, 256, 256),
                scaler=scaler,
                omero=ome_json,
                axes=[
                    {"name": "t", "type": "time", "unit": "millisecond"},
                    {"name": "c", "type": "channel"},
                    {"name": "z", "type": "space", "unit": "micrometer"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                # For each resolution, we have a List of transformation Dicts (not
                # validated). Each list of dicts are added to each datasets in order.
                coordinate_transformations=[
                    [
                        # the voxel size for the first scale level (0.5 micrometer)
                        {
                            "type": "scale",
                            "scale": [1.0, 1.0, pixelsize.Z, pixelsize.Y, pixelsize.X],
                        }
                    ],
                    [
                        # the voxel size for the second scale level
                        # (downscaled by a factor of 2 -> 1 micrometer)
                        {
                            "type": "scale",
                            "scale": [
                                1.0,
                                1.0,
                                pixelsize.Z * scaler.downscale
                                if pixelsize.Z is not None
                                else scaler.downscale,
                                pixelsize.Y * scaler.downscale
                                if pixelsize.Y is not None
                                else scaler.downscale,
                                pixelsize.X * scaler.downscale
                                if pixelsize.X is not None
                                else scaler.downscale,
                            ],
                        }
                    ],
                    [
                        # the voxel size for the second scale level
                        # (downscaled by a factor of 4 -> 2 micrometer)
                        {
                            "type": "scale",
                            "scale": [
                                1.0,
                                1.0,
                                pixelsize.Z * scaler.downscale * scaler.downscale
                                if pixelsize.Z is not None
                                else scaler.downscale * scaler.downscale,
                                pixelsize.Y * scaler.downscale * scaler.downscale
                                if pixelsize.Y is not None
                                else scaler.downscale * scaler.downscale,
                                pixelsize.X * scaler.downscale * scaler.downscale
                                if pixelsize.X is not None
                                else scaler.downscale * scaler.downscale,
                            ],
                        }
                    ],
                ],
                # Options to be passed on to the storage backend. A list would need to
                # match the number of datasets in a multiresolution pyramid. One can
                # provide different chunk size for each level of a pyramind using this
                # option.
                storage_options=[],
            )
            # print(os.listdir(mypath))
            # print(os.listdir(mypath / "image0"))

    @staticmethod
    def build_ome(
        data_shape: Tuple[int, ...],
        channel_names: List[str],
        channel_colors: List[int],
        channel_minmax: List[Tuple[float, float]],
    ) -> Dict:
        """
        Create the necessary metadata for an OME tiff image

        Parameters
        ----------
        data_shape:
            A 5-d tuple, assumed to be TCZYX order
        channel_names:
            The names for each channel
        channel_colors:
            List of all channel colors
        channel_minmax:
            List of all (min, max) pairs of channel intensities

        Returns
        -------
        Dict
            An "omero" metadata object suitable for writing to ome-zarr
        """
        ch = []
        for i in range(data_shape[1]):
            ch.append(
                {
                    "active": True,
                    "coefficient": 1,
                    "color": f"{channel_colors[i]:06x}",
                    "family": "linear",
                    "inverted": False,
                    "label": channel_names[i],
                    "window": {
                        "end": float(channel_minmax[i][1]),
                        "max": float(channel_minmax[i][1]),
                        "min": float(channel_minmax[i][0]),
                        "start": float(channel_minmax[i][0]),
                    },
                }
            )

        omero = {
            "id": 1,  # ID in OMERO
            "name": "image0",  # Name as shown in the UI
            "version": "0.4",  # Current version
            "channels": ch,
            "rdefs": {
                "defaultT": 0,  # First timepoint to show the user
                "defaultZ": data_shape[2] // 2,  # First Z section to show the user
                "model": "color",  # "color" or "greyscale"
            },
            # # from here down this is all extra and not part of the ome-zarr spec
            # "meta": {
            #     "projectDescription": "20+ lines of gene edited cells etc",
            #     "datasetName": "aics_hipsc_v2020.1",
            #     "projectId": 2,
            #     "imageDescription": "foo bar",
            #     "imageTimestamp": 1277977808.0,
            #     "imageId": 12,
            #     "imageAuthor": "danielt",
            #     "imageName": "AICS-12_143.ome.tif",
            #     "datasetDescription": "variance dataset after QC",
            #     "projectName": "aics cell variance project",
            #     "datasetId": 3
            # },
            # no longer needed as this is captured elsewhere?
            # or is this still a convenience for the 3d viewer?
            # "size": {
            #     "width": shape[4],
            #     "c": shape[1],
            #     "z": shape[2],
            #     "t": shape[0],
            #     "height": shape[3]
            # },
        }
        return omero
