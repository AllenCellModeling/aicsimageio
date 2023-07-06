import math
from typing import Dict, List, Optional, Tuple

import zarr
from ome_zarr.io import parse_url
from ome_zarr.scale import Scaler
from ome_zarr.writer import write_image
from zarr.storage import default_compressor

from .. import exceptions, types
from ..metadata import utils
from ..utils import io_utils


class OmeZarrWriter:
    def __init__(self, uri: types.PathLike):
        """
        Constructor.

        Parameters
        ----------
        uri: types.PathLike
            The URI or local path for where to save the data.
        """
        # Resolve final destination
        fs, path = io_utils.pathlike_to_fs(uri)

        # Save image to zarr store!
        self.store = parse_url(uri, mode="w").store
        self.root_group = zarr.group(store=self.store)

    @staticmethod
    def build_ome(
        size_z: int,
        image_name: str,
        channel_names: List[str],
        channel_colors: List[int],
        channel_minmax: List[Tuple[float, float]],
    ) -> Dict:
        """
        Create the omero metadata for an OME zarr image

        Parameters
        ----------
        size_z:
            Number of z planes
        image_name:
            The name of the image
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
        for i in range(len(channel_names)):
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
            "name": image_name,  # Name as shown in the UI
            "version": "0.4",  # Current version
            "channels": ch,
            "rdefs": {
                "defaultT": 0,  # First timepoint to show the user
                "defaultZ": size_z // 2,  # First Z section to show the user
                "model": "color",  # "color" or "greyscale"
            },
            # TODO: can we add more metadata here?
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
        }
        return omero

    def write_image(
        self,
        # TODO how to pass in precomputed multiscales?
        image_data: types.ArrayLike,  # must be 3D, 4D or 5D
        image_name: str,
        physical_pixel_sizes: Optional[types.PhysicalPixelSizes],
        channel_names: Optional[List[str]],
        channel_colors: Optional[List[int]],
        scale_num_levels: int = 1,
        scale_factor: float = 2.0,
        dimension_order: Optional[str] = None,
    ) -> None:
        """
        Write a data array to a file.
        NOTE that this API is not yet finalized and will change in the future.

        Parameters
        ----------
        image_data: types.ArrayLike
            The array of data to store. Data arrays must have 2 to 6 dimensions. If a
            list is provided, then it is understood to be multiple images written to the
            ome-tiff file. All following metadata parameters will be expanded to the
            length of this list.
        image_name: str
            string representing the name of the image
        physical_pixel_sizes: Optional[types.PhysicalPixelSizes]
            PhysicalPixelSizes object representing the physical pixel sizes in Z, Y, X
            in microns.
            Default: None
        channel_names: Optional[List[str]]
            Lists of strings representing the names of the data channels
            Default: None
            If None is given, the list will be generated as a 0-indexed list of strings
            of the form "Channel:image_index:channel_index"
        channel_colors: Optional[List[int]]
            List of rgb color values per channel or a list of lists for each image.
            These must be values compatible with the OME spec.
            Default: None
        scale_num_levels: Optional[int]
            Number of pyramid levels to use for the image.
            Default: 1 (represents no downsampled levels)
        scale_factor: Optional[float]
            The scale factor to use for the image. Only active if scale_num_levels > 1.
            Default: 2.0
        dimension_order: Optional[str]
            The dimension order of the data. If None is given, the dimension order will
            be guessed from the number of dimensions in the data according to TCZYX
            order.

        Examples
        --------
        Write a TCZYX data set to OME-Zarr

        >>> image = numpy.ndarray([1, 10, 3, 1024, 2048])
        ... writer = OmeZarrWriter("/path/to/file.ome.zarr")
        ... writer.write_image(image)

        Write multi-scene data to OME-Zarr, specifying channel names

        >>> image0 = numpy.ndarray([3, 10, 1024, 2048])
        ... image1 = numpy.ndarray([3, 10, 512, 512])
        ... writer = OmeZarrWriter("/path/to/file.ome.zarr")
        ... writer.write_image(image0, "Image:0", ["C00","C01","C02"])
        ... writer.write_image(image1, "Image:1", ["C10","C11","C12"])
        """
        ndims = len(image_data.shape)
        if ndims < 2 or ndims > 5:
            raise exceptions.InvalidDimensionOrderingError(
                f"Image data must have 2, 3, 4, or 5 dimensions. "
                f"Received image data with shape: {image_data.shape}"
            )
        if dimension_order is None:
            dimension_order = "TCZYX"[-ndims:]
        if len(dimension_order) != ndims:
            raise exceptions.InvalidDimensionOrderingError(
                f"Dimension order {dimension_order} does not match data "
                f"shape: {image_data.shape}"
            )
        if (len(set(dimension_order) - set("TCZYX")) > 0) or len(
            dimension_order
        ) != len(set(dimension_order)):
            raise exceptions.InvalidDimensionOrderingError(
                f"Dimension order {dimension_order} is invalid or "
                "contains unexpected dimensions. Only TCZYX currently supported."
            )
        xdimindex = dimension_order.find("X")
        ydimindex = dimension_order.find("Y")
        zdimindex = dimension_order.find("Z")
        cdimindex = dimension_order.find("C")
        if cdimindex > min(i for i in [xdimindex, ydimindex, zdimindex] if i > -1):
            raise exceptions.InvalidDimensionOrderingError(
                f"Dimension order {dimension_order} is invalid. Channel dimension "
                "must be before X, Y, and Z."
            )

        if physical_pixel_sizes is None:
            pixelsizes = (1.0, 1.0, 1.0)
        else:
            pixelsizes = (
                physical_pixel_sizes.Z if physical_pixel_sizes.Z is not None else 1.0,
                physical_pixel_sizes.Y if physical_pixel_sizes.Y is not None else 1.0,
                physical_pixel_sizes.X if physical_pixel_sizes.X is not None else 1.0,
            )
        if channel_names is None:
            # TODO this isn't generating a very pretty looking name but it will be
            # unique
            channel_names = (
                [
                    utils.generate_ome_channel_id(image_id=image_name, channel_id=i)
                    for i in range(image_data.shape[cdimindex])
                ]
                if cdimindex > -1
                else [utils.generate_ome_channel_id(image_id=image_name, channel_id=0)]
            )
        if channel_colors is None:
            # TODO generate proper colors or confirm that the underlying lib can handle
            # None
            channel_colors = (
                [i for i in range(image_data.shape[cdimindex])]
                if cdimindex > -1
                else [0]
            )
        scale_dim_map = {
            "T": 1.0,
            "C": 1.0,
            "Z": pixelsizes[0],
            "Y": pixelsizes[1],
            "X": pixelsizes[2],
        }
        transforms = [
            [
                # the voxel size for the first scale level
                {
                    "type": "scale",
                    "scale": [scale_dim_map[d] for d in dimension_order],
                }
            ]
        ]
        # TODO precompute sizes for downsampled also.
        plane_size = (
            image_data.shape[xdimindex]
            * image_data.shape[ydimindex]
            * image_data.itemsize
        )
        target_chunk_size = 16 * (1024 * 1024)  # 16 MB
        # this is making an assumption of chunking whole XY planes.
        # TODO allow callers to configure chunk dims?
        nplanes_per_chunk = int(math.ceil(target_chunk_size / plane_size))
        nplanes_per_chunk = (
            min(nplanes_per_chunk, image_data.shape[zdimindex]) if zdimindex > -1 else 1
        )
        chunk_dim_map = {
            "T": 1,
            "C": 1,
            "Z": nplanes_per_chunk,
            "Y": image_data.shape[ydimindex],
            "X": image_data.shape[xdimindex],
        }
        chunk_dims = [
            dict(
                chunks=tuple(chunk_dim_map[d] for d in dimension_order),
                compressor=default_compressor,
            )
        ]
        lasty = image_data.shape[ydimindex]
        lastx = image_data.shape[xdimindex]
        # TODO scaler might want to use different method for segmentations than raw
        # TODO allow custom scaler or pre-computed multiresolution levels
        if scale_num_levels > 1:
            # TODO As of this writing, this Scaler is not the most general
            # implementation (it does things by xy plane) but it's code already
            # written that also works with dask, so it's a good starting point.
            scaler = Scaler()
            scaler.method = "nearest"
            scaler.max_layer = scale_num_levels - 1
            scaler.downscale = scale_factor if scale_factor is not None else 2
            for i in range(scale_num_levels - 1):
                scale_dim_map["Y"] *= scaler.downscale
                scale_dim_map["X"] *= scaler.downscale
                transforms.append(
                    [
                        {
                            "type": "scale",
                            "scale": [scale_dim_map[d] for d in dimension_order],
                        }
                    ]
                )
                lasty = int(math.ceil(lasty / scaler.downscale))
                lastx = int(math.ceil(lastx / scaler.downscale))
                plane_size = lasty * lastx * image_data.itemsize
                nplanes_per_chunk = int(math.ceil(target_chunk_size / plane_size))
                nplanes_per_chunk = (
                    min(nplanes_per_chunk, image_data.shape[zdimindex])
                    if zdimindex > -1
                    else 1
                )
                chunk_dim_map["Z"] = nplanes_per_chunk
                chunk_dim_map["Y"] = lasty
                chunk_dim_map["X"] = lastx
                chunk_dims.append(
                    dict(
                        chunks=tuple(chunk_dim_map[d] for d in dimension_order),
                        compressor=default_compressor,
                    )
                )
        else:
            scaler = None

        # try to construct per-image metadata
        ome_json = OmeZarrWriter.build_ome(
            image_data.shape[zdimindex] if zdimindex > -1 else 1,
            image_name,
            channel_names=channel_names,  # type: ignore
            channel_colors=channel_colors,  # type: ignore
            # This can be slow if computed here.
            # TODO: Rely on user to supply the per-channel min/max.
            channel_minmax=[
                (0.0, 1.0)
                for i in range(image_data.shape[cdimindex] if cdimindex > -1 else 1)
            ],
        )
        # TODO user supplies units?
        dim_to_axis = {
            "T": {"name": "t", "type": "time", "unit": "millisecond"},
            "C": {"name": "c", "type": "channel"},
            "Z": {"name": "z", "type": "space", "unit": "micrometer"},
            "Y": {"name": "y", "type": "space", "unit": "micrometer"},
            "X": {"name": "x", "type": "space", "unit": "micrometer"},
        }

        axes = [dim_to_axis[d] for d in dimension_order]

        # TODO image name must be unique within this root group
        group = self.root_group  # .create_group(image_name, overwrite=True)
        group.attrs["omero"] = ome_json

        write_image(
            image=image_data,
            group=group,
            scaler=scaler,
            axes=axes,
            # For each resolution, we have a List of transformation Dicts (not
            # validated). Each list of dicts are added to each datasets in order.
            coordinate_transformations=transforms,
            # Options to be passed on to the storage backend. A list would need to
            # match the number of datasets in a multiresolution pyramid. One can
            # provide different chunk size for each level of a pyramid using this
            # option.
            storage_options=chunk_dims,
        )
