#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
import logging
from typing import Dict, List, Tuple, Union
import re
import xml.etree.ElementTree as ET

from fsspec.spec import AbstractFileSystem
import numpy as np
from ome_types import from_xml
from ome_types.model.ome import OME
from tifffile import TiffFile, TiffFileError, TiffTag
import xarray as xr

from .. import constants, exceptions, types
from ..dimensions import DimensionNames
from ..utils import io_utils
from .tiff_reader import TiffReader

###############################################################################

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)4s: %(module)s:%(lineno)4s %(asctime)s] %(message)s",
)
log = logging.getLogger(__name__)

###############################################################################

# This is a known issue that could have been caused by prior versions of aicsimageio
# due to our old OMEXML.py file.
#
# You can see the PR that updated this exact line here:
# https://github.com/AllenCellModeling/aicsimageio/pull/116/commits/e3f9cde7f680edeef3ef3586a67fd8106e746167#diff-46a483e94af833f7eaa1106921191fed5e7c77f33a5c0c47a8f5a2d35ad3ba96L47
#
# Notably why this is invalid is that the 2012-03 schema _doesn't exist_
#
# Don't know how this wasn't ever caught before that PR but to ensure that we don't
# error in reading the OME in aicsimageio>=4.0.0, we manually find and replace this
# line in OME xml prior to creating the OME object.
KNOWN_INVALID_OME_XSD_REFERENCES = [
    "www.openmicroscopy.org/Schemas/ome/2013-06",
    "www.openmicroscopy.org/Schemas/OME/2012-03",
]
REPLACEMENT_OME_XSD_REFERENCE = "www.openmicroscopy.org/Schemas/OME/2016-06"

###############################################################################


class OmeTiffReader(TiffReader):
    @staticmethod
    def _clean_ome_xml_for_known_issues(xml: str) -> str:
        # Store list of changes to print out with warning
        metadata_changes = []

        # Fix xsd reference
        # This is from OMEXML object just having invalid reference
        for known_invalid_ref in KNOWN_INVALID_OME_XSD_REFERENCES:
            if known_invalid_ref in xml:
                xml = xml.replace(
                    known_invalid_ref,
                    REPLACEMENT_OME_XSD_REFERENCE,
                )
                metadata_changes.append(
                    f"Replaced '{known_invalid_ref}' with "
                    f"'{REPLACEMENT_OME_XSD_REFERENCE}'."
                )

        # Read in XML
        root = ET.fromstring(xml)

        # Get the namespace
        # In XML etree this looks like
        # "{http://www.openmicroscopy.org/Schemas/OME/2016-06}"
        # and must prepend any etree finds
        namespace_matches = re.match(r"\{.*\}", root.tag)
        if namespace_matches is not None:
            namespace = namespace_matches.group(0)
        else:
            raise ValueError("XML does not contain a namespace")

        # Find all Image elements and fix IDs
        # This is for certain for test files of ours and ACTK files
        for image_index, image in enumerate(root.findall(f"{namespace}Image")):
            image_id = image.get("ID")
            if not image_id.startswith("Image"):
                image.set("ID", f"Image:{image_id}")
                metadata_changes.append(
                    f"Updated attribute 'ID' from '{image_id}' to 'Image:{image_id}' "
                    f"on Image element at position {image_index}."
                )

            # Find all Pixels elements and fix IDs
            for pixels_index, pixels in enumerate(image.findall(f"{namespace}Pixels")):
                pixels_id = pixels.get("ID")
                if not pixels_id.startswith("Pixels"):
                    pixels.set("ID", f"Pixels:{pixels_id}")
                    metadata_changes.append(
                        f"Updated attribute 'ID' from '{pixels_id}' to "
                        f"Pixels:{pixels_id}' on Pixels element at "
                        f"position {pixels_index}."
                    )

                # Determine if there is an out-of-order channel / plane elem
                # This is due to OMEXML "add channel" function
                # That added Channels and appropriate Planes to the XML
                # But, placed them in:
                # Channel
                # Plane
                # Plane
                # ...
                # Channel
                # Plane
                # Plane
                #
                # Instead of grouped together:
                # Channel
                # Channel
                # ...
                # Plane
                # Plane
                # ...
                #
                # This effects all CFE files (new and old) but for different reasons
                pixels_children_out_of_order = False
                encountered_something_besides_channel = False
                for child in pixels:
                    if child.tag != f"{namespace}Channel":
                        encountered_something_besides_channel = True
                    if (
                        encountered_something_besides_channel
                        and child.tag == f"{namespace}Channel"
                    ):
                        pixels_children_out_of_order = True
                        break

                # Ensure order of:
                # channels -> bindata | tiffdata | metadataonly -> planes
                if pixels_children_out_of_order:
                    # Get all relevant elems
                    channels = [
                        deepcopy(c) for c in pixels.findall(f"{namespace}Channel")
                    ]
                    bin_data = [
                        deepcopy(b) for b in pixels.findall(f"{namespace}BinData")
                    ]
                    tiff_data = [
                        deepcopy(t) for t in pixels.findall(f"{namespace}TiffData")
                    ]
                    # There should only be one metadata only element but to standardize
                    # list comprehensions later we findall
                    metadata_only = [
                        deepcopy(m) for m in pixels.findall(f"{namespace}MetadataOnly")
                    ]
                    planes = [deepcopy(p) for p in pixels.findall(f"{namespace}Plane")]

                    # Old (2018 ish) cell feature explorer files sometimes contain both
                    # an empty metadata only element and filled tiffdata elements
                    # Since the metadata only elements are empty we can check this and
                    # choose the tiff data elements instead
                    #
                    # First check if there are any metadata only elements
                    if len(metadata_only) == 1:
                        # Now check if _one of_ of the other two choices are filled
                        # ^ in Python is XOR
                        if (len(bin_data) > 0) ^ (len(tiff_data) > 0):
                            metadata_children = list(metadata_only[0])
                            # Now check if the metadata only elem has no children
                            if len(metadata_children) == 0:
                                # If so, just "purge" by creating empty list
                                metadata_only = []

                            # If there are children elements
                            # Return XML and let XMLSchema Validation show error
                            else:
                                return xml

                    # After cleaning metadata only, validate the normal behaviors of
                    # OME schema
                    #
                    # Validate that there is only one of bindata, tiffdata, or metadata
                    if len(bin_data) > 0:
                        if len(tiff_data) == 0 and len(metadata_only) == 0:
                            selected_choice = bin_data
                        else:
                            # Return XML and let XMLSchema Validation show error
                            return xml
                    elif len(tiff_data) > 0:
                        if len(bin_data) == 0 and len(metadata_only) == 0:
                            selected_choice = tiff_data
                        else:
                            # Return XML and let XMLSchema Validation show error
                            return xml
                    elif len(metadata_only) == 1:
                        if len(bin_data) == 0 and len(tiff_data) == 0:
                            selected_choice = metadata_only
                        else:
                            # Return XML and let XMLSchema Validation show error
                            return xml
                    else:
                        # Return XML and let XMLSchema Validation show error
                        return xml

                    # Remove all children from element to be replaced
                    # with ordered elements
                    for elem in list(pixels):
                        pixels.remove(elem)

                    # Re-attach elements
                    for channel in channels:
                        pixels.append(channel)
                    for elem in selected_choice:
                        pixels.append(elem)
                    for plane in planes:
                        pixels.append(plane)

                    metadata_changes.append(
                        f"Reordered children of Pixels element at "
                        f"position {pixels_index}."
                    )

        # This is a result of dumping basically all experiement metadata
        # into "StructuredAnnotation" blocks
        #
        # This affects new (2020) Cell Feature Explorer files
        #
        # Because these are structured annotations we don't want to mess with anyones
        # besides the AICS generated bad structured annotations
        aics_anno_removed_count = 0
        sa = root.find(f"{namespace}StructuredAnnotations")
        if sa is not None:
            for xml_anno in sa.findall(f"{namespace}XMLAnnotation"):
                # At least these are namespaced
                if xml_anno.get("Namespace") == "alleninstitute.org/CZIMetadata":
                    # Get ID because some elements have annotation refs
                    # in both the base Image element and all plane elements
                    aics_anno_id = xml_anno.get("ID")
                    for image in root.findall(f"{namespace}Image"):
                        for anno_ref in image.findall(f"{namespace}AnnotationRef"):
                            if anno_ref.get("ID") == aics_anno_id:
                                image.remove(anno_ref)

                        # Clean planes
                        pixels = image.find(f"{namespace}Pixels")
                        for plane in pixels.findall(f"{namespace}Plane"):
                            for anno_ref in plane.findall(f"{namespace}AnnotationRef"):
                                if anno_ref.get("ID") == aics_anno_id:
                                    plane.remove(anno_ref)

                    # Remove the whole etree
                    sa.remove(xml_anno)
                    aics_anno_removed_count += 1

        # Log changes
        if aics_anno_removed_count > 0:
            metadata_changes.append(
                f"Removed {aics_anno_removed_count} AICS generated XMLAnnotations."
            )

        # If there are no annotations in StructuredAnnotations, remove it
        if sa is not None:
            if len(list(sa)) == 0:
                root.remove(sa)

        # If any piece of metadata was changed alert and rewrite
        if len(metadata_changes) > 0:
            log.debug(
                "OME metadata was cleaned for known AICSImageIO 3.x OMEXML errors."
            )
            log.debug(f"Full list of OME cleaning changes: {metadata_changes}")

            # Register namespace
            ET.register_namespace("", f"http://{REPLACEMENT_OME_XSD_REFERENCE}")

            # Write out cleaned XML to string
            xml = ET.tostring(
                root,
                encoding="unicode",
                method="xml",
            )

        return xml

    @staticmethod
    def _get_ome(ome_xml: str, clean_metadata: bool = True) -> OME:
        # To clean or not to clean, that is the question
        if clean_metadata:
            ome_xml = OmeTiffReader._clean_ome_xml_for_known_issues(ome_xml)

        return from_xml(ome_xml)

    @staticmethod
    def _is_supported_image(
        fs: AbstractFileSystem, path: str, clean_metadata: bool = True
    ) -> bool:
        try:
            with fs.open(path) as open_resource:
                with TiffFile(open_resource) as tiff:
                    # Get first page description (aka the description tag in general)
                    xml = tiff.pages[0].description
                    return OmeTiffReader._get_ome(xml, clean_metadata)

        # tifffile exception, tifffile exception, ome-types / etree exception
        except (TiffFileError, TypeError, ET.ParseError):
            return False

    def __init__(self, image: types.PathLike, clean_metadata: bool = True):
        """
        Wraps the tifffile and ome-types APIs to provide the same aicsimageio Reader
        API but for volumetric OME-TIFF images.

        Parameters
        ----------
        image: types.PathLike
            Path to image file to construct Reader for.

        clean_metadata: bool
            Should the OME XML metadata found in the file be cleaned for known
            AICSImageIO 3.x created errors.
            Default: True (Do not clean, any error will result in failure to read)

        Notes
        -----
        If the OME metadata in your file isn't OME schema compilant or does not validate
        this will fail to read you file and raise an exception.

        If the OME metadata in your file doesn't use the latest OME schema (2016-06),
        this reader will make a request to the referenced remote OME schema to validate.
        """
        # Expand details of provided image
        self.fs, self.path = io_utils.pathlike_to_fs(image, enforce_exists=True)
        self.extension = ".".join(self.path.split(".")[1:])
        self.clean_metadata = clean_metadata

        # Enforce valid image
        if not self._is_supported_image(self.fs, self.path, clean_metadata):
            raise exceptions.UnsupportedFileFormatError(
                self.__class__.__name__, self.extension
            )

    @property
    def scenes(self) -> Tuple[str]:
        if self._scenes is None:
            with self.fs.open(self.path) as open_resource:
                with TiffFile(open_resource) as tiff:
                    ome = self._get_ome(tiff.pages[0].description, self.clean_metadata)
                    self._scenes = tuple(image_meta.id for image_meta in ome.images)

        return self._scenes

    @staticmethod
    def _process_ome_metadata(
        ome: TiffTag,
        scene_index: int,
    ) -> Tuple[List[str], Dict[str, Union[List, types.ArrayLike]]]:
        """
        Process the OME metadata to retrieve the dimension names and coordinate planes.

        Parameters
        ----------
        ome: OME
            A constructed OME object to retrieve data from.
        scene_index: int
            The current operating scene index to pull metadata from.

        Returns
        -------
        dims: List[str]
            The dimension names pulled from the OME metadata.
        coords: Dict[str, Union[List, types.ArrayLike]]
            The coordinate planes / data for each dimension.
        """
        # Select scene
        scene_meta = ome.images[scene_index]

        # Create dimension order by getting the current scene's dimension order
        # and reversing it because OME store order vs use order is :shrug:
        dims = [d for d in scene_meta.pixels.dimension_order.value[::-1]]

        # Get coordinate planes
        coords = {}

        # Channels
        coords[DimensionNames.Channel] = [
            channel.name for channel in scene_meta.pixels.channels
        ]

        # Time
        # If global linear timescale we can np.linspace with metadata
        if scene_meta.pixels.time_increment is not None:
            coords[DimensionNames.Time] = np.linspace(
                0,
                scene_meta.pixels.time_increment_quantity,
                scene_meta.pixels.time_increment,
            )
        # If non global linear timescale, we need to create an array of every plane
        # time value
        elif scene_meta.pixels.size_t > 1:
            t_index_to_delta_map = {
                p.the_t: p.delta_t for p in scene_meta.pixels.planes
            }
            coords[DimensionNames.Time] = list(t_index_to_delta_map.values())

        # Handle Spatial Dimensions
        if scene_meta.pixels.physical_size_z is not None:
            coords[DimensionNames.SpatialZ] = np.arange(
                0,
                scene_meta.pixels.size_z * scene_meta.pixels.physical_size_z,
                scene_meta.pixels.physical_size_z,
            )

        if scene_meta.pixels.physical_size_y is not None:
            coords[DimensionNames.SpatialY] = np.arange(
                0,
                scene_meta.pixels.size_y * scene_meta.pixels.physical_size_y,
                scene_meta.pixels.physical_size_y,
            )

        if scene_meta.pixels.physical_size_x is not None:
            coords[DimensionNames.SpatialX] = np.arange(
                0,
                scene_meta.pixels.size_x * scene_meta.pixels.physical_size_x,
                scene_meta.pixels.physical_size_x,
            )

        return dims, coords

    @staticmethod
    def _expand_dims_to_match_ome(
        image_data: types.ArrayLike,
        ome: OME,
        dims: List[str],
        scene_index: int,
    ) -> types.ArrayLike:
        # Expand image_data for empty dimensions
        ome_shape = []
        for d in dims:
            ome_shape.append(
                getattr(ome.images[scene_index].pixels, f"size_{d.lower()}")
            )

        # Check for num samples and expand dims if greater than 1
        n_samples = ome.images[scene_index].pixels.channels[0].samples_per_pixel
        if n_samples > 1:
            # Append to the end, i.e. the last dimension
            dims.append("S")
            ome_shape.append(n_samples)

        # The file may not have all the data but OME requires certain dimensions
        # expand to fill
        expand_dim_ops = []
        for d_size in ome_shape:
            # Add empty dimension where OME requires dimension but no data exists
            if d_size == 1:
                expand_dim_ops.append(None)
            # Add full slice where data exists
            else:
                expand_dim_ops.append(slice(None, None, None))

        # Apply operators to dask array
        return image_data[tuple(expand_dim_ops)]

    def _read_delayed(self) -> xr.DataArray:
        """
        Construct the delayed xarray DataArray object for the image.

        Returns
        -------
        image: xr.DataArray
            The fully constructed and fully delayed image as a DataArray object.
            Metadata is attached in some cases as coords, dims, and attrs contains
            unprocessed tags and processed OME object.

        Raises
        ------
        exceptions.UnsupportedFileFormatError: The file could not be read or is not
            supported.
        """
        # Create the delayed dask array
        image_data = self._create_dask_array()

        # Get unprocessed metadata from tags
        tiff_tags = self._get_tiff_tags()

        # Create OME
        with self.fs.open(self.path) as open_resource:
            with TiffFile(open_resource) as tiff:
                ome = self._get_ome(tiff.pages[0].description, self.clean_metadata)

        # Unpack dims and coords from OME
        dims, coords = self._process_ome_metadata(
            ome=ome,
            scene_index=self.current_scene_index,
        )

        # Expand the image data to match the OME empty dimensions
        image_data = self._expand_dims_to_match_ome(
            image_data=image_data,
            ome=ome,
            dims=dims,
            scene_index=self.current_scene_index,
        )

        return xr.DataArray(
            image_data,
            dims=dims,
            coords=coords,
            attrs={
                constants.METADATA_UNPROCESSED: tiff_tags,
                constants.METADATA_PROCESSED: ome,
            },
        )

    def _read_immediate(self) -> xr.DataArray:
        """
        Construct the in-memory xarray DataArray object for the image.

        Returns
        -------
        image: xr.DataArray
            The fully constructed and fully read into memory image as a DataArray
            object. Metadata is attached in some cases as coords, dims, and attrs
            contains unprocessed tags and processed OME object.

        Raises
        ------
        exceptions.UnsupportedFileFormatError: The file could not be read or is not
            supported.
        """
        with self.fs.open(self.path) as open_resource:
            with TiffFile(open_resource) as tiff:
                # Read image into memory
                image_data = tiff.series[self.current_scene_index].asarray()

                # Get unprocessed metadata from tags
                tiff_tags = self._get_tiff_tags()

                # Create OME
                with self.fs.open(self.path) as open_resource:
                    with TiffFile(open_resource) as tiff:
                        ome = self._get_ome(
                            tiff.pages[0].description, self.clean_metadata
                        )

                # Unpack dims and coords from OME
                dims, coords = self._process_ome_metadata(
                    ome=ome,
                    scene_index=self.current_scene_index,
                )

                # Expand the image data to match the OME empty dimensions
                image_data = self._expand_dims_to_match_ome(
                    image_data=image_data,
                    ome=ome,
                    dims=dims,
                    scene_index=self.current_scene_index,
                )

                return xr.DataArray(
                    image_data,
                    dims=dims,
                    coords=coords,
                    attrs={
                        constants.METADATA_UNPROCESSED: tiff_tags,
                        constants.METADATA_PROCESSED: ome,
                    },
                )
