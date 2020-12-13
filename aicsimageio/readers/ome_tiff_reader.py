#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
from typing import Dict, List, Tuple, Union
import re
import warnings
import xml.etree.ElementTree as ET

from fsspec.spec import AbstractFileSystem
from ome_types import from_xml
from ome_types.model.ome import OME
import requests
from tifffile import TiffFile, TiffFileError, TiffTag
import xarray as xr

from .. import constants, exceptions, types
from ..dimensions import DimensionNames
from ..utils import io_utils
from .tiff_reader import TiffReader

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
        with open("pre-fix.xml", "w", encoding="utf-8") as open_resource:
            open_resource.write(xml)

        # Store bool to ultimately alert user that we changed the metadata
        # with known common errors
        metadata_was_changed = False

        # Fix old aicsimageio / vendor code mistakes
        # Fix xsd reference
        for known_invalid_ref in KNOWN_INVALID_OME_XSD_REFERENCES:
            if known_invalid_ref in xml:
                xml = xml.replace(
                    known_invalid_ref,
                    REPLACEMENT_OME_XSD_REFERENCE,
                )
                metadata_was_changed = True

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
        for image in root.findall(f"{namespace}Image"):
            image_id = image.attrib["ID"]
            if not image_id.startswith("Image"):
                image.set("ID", f"Image:{image_id}")
                metadata_was_changed = True

            # Find all Pixels elements and fix IDs
            for pixels in image.findall(f"{namespace}Pixels"):
                pixels_id = pixels.attrib["ID"]
                if not pixels_id.startswith("Pixels"):
                    pixels.set("ID", f"Pixels:{pixels_id}")
                    metadata_was_changed = True

                # Determine if there is an out-of-order channel / plane elem
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
                    metadata_only = [
                        deepcopy(m) for m in pixels.findall(f"{namespace}MetadataOnly")
                    ]
                    planes = [deepcopy(p) for p in pixels.findall(f"{namespace}Plane")]

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

                    metadata_was_changed = True
        # If any piece of metadata was changed alert and rewrite
        if metadata_was_changed:
            warnings.warn(
                "OME metadata was cleaned and fixed for known AICSImageIO 3.x OMEXML "
                "errors. It is recommended to rewrite image data with 4.x "
                "OmeTiffWriter. To see exactly how the metadata is cleaned, view the "
                "`_clean_ome_xml_for_known_issues` function on `OmeTiffReader`."
            )

            # Register namespace
            ET.register_namespace("", f"http://{REPLACEMENT_OME_XSD_REFERENCE}")

            # Write out cleaned XML to string
            xml = ET.tostring(
                root,
                encoding="unicode",
                method="xml",
                xml_declaration=True,
            )

            with open("post-fix.xml", "w", encoding="utf-8") as open_resource:
                open_resource.write(xml)

        return xml

    @staticmethod
    def _get_ome(ome_xml: str, clean_metadata: bool = True) -> OME:
        # To clean or not to clean, that is the question
        if clean_metadata:
            return from_xml(OmeTiffReader._clean_ome_xml_for_known_issues(ome_xml))

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

        # In the case where the OME metadata is not the latest version it will attempt
        # make a request to the remote XSD if this fails we need to catch the error.
        # TODO
        # Figure out why this is failing to catch
        except requests.exceptions.HTTPError:
            raise exceptions.MalformedMetadataError(
                path=path,
                msg="Failed in retrieving the referenced remote OME schema",
            )

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
            Default: True

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
            raise exceptions.UnsupportedFileFormatError(self.extension)

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
        coords[DimensionNames.Channel] = [
            channel.name for channel in scene_meta.pixels.channels
        ]
        # TODO ADD COORDS
        # coords[DimensionNames.SpatialZ] = [
        #     np.linspace(0, )
        # ]

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

        # TODO handle Samples dimension attachment for RGB OME TIFF

        expand_dim_ops = []
        for d_size in ome_shape:
            # Add empty dimension where OME requires dimension but not data exists
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
