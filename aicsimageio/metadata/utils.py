#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import re
import xml.etree.ElementTree as ET
from copy import deepcopy
from typing import Dict, Optional, Union

import numpy as np
from ome_types.model.simple_types import PixelType

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


def generate_ome_image_id(image_id: Union[str, int]) -> str:
    """
    Naively generates the standard OME image ID using a provided ID.

    Parameters
    ----------
    image_id: Union[str, int]
        A string or int representing the ID for an image.
        In the context of the usage of this function, this is usually used with the
        index of the scene / image.

    Returns
    -------
    ome_image_id: str
        The OME standard for image IDs.
    """
    return f"Image:{image_id}"


def generate_ome_channel_id(image_id: str, channel_id: Union[str, int]) -> str:
    """
    Naively generates the standard OME channel ID using a provided ID.

    Parameters
    ----------
    image_id: str
        An image id to pull the image specific index from.
        See: `generate_ome_image_id` for more details.
    channel_id: Union[str, int]
        A string or int representing the ID for a channel.
        In the context of the usage of this function, this is usually used with the
        index of the channel.

    Returns
    -------
    ome_channel_id: str
        The OME standard for channel IDs.

    Notes
    -----
    ImageIds are usually: "Image:0", "Image:1", or "Image:N",
    ChannelIds are usually the combination of image index + channel index --
    "Channel:0:0" for the first channel of the first image for example.
    """
    # Remove the prefix 'Image:' to get just the index
    image_index = image_id.replace("Image:", "")
    return f"Channel:{image_index}:{channel_id}"


def generate_ome_instrument_id(instrument_id: Union[str, int]) -> str:
    """
    Naively generates the standard OME instrument ID using a provided ID.

    Parameters
    ----------
    instrument_id: Union[str, int]
        A string or int representing the ID for an instrument.

    Returns
    -------
    ome_instrument_id: str
        The OME standard for instrument IDs.
    """
    return f"Instrument:{instrument_id}"


def generate_ome_detector_id(detector_id: Union[str, int]) -> str:
    """
    Naively generates the standard OME detector ID using a provided ID.

    Parameters
    ----------
    detector_id: Union[str, int]
        A string or int representing the ID for a detector.

    Returns
    -------
    ome_detector_id: str
        The OME standard for detector IDs.
    """
    return f"Detector:{detector_id}"


def clean_ome_xml_for_known_issues(xml: str) -> str:
    """
    Clean an OME XML string for known issues created by AICS or MicroManager
    systems and tools.

    Commonly this is used for cleaning a file produced by AICS prior to noticing the
    issue (2021), or for other users of aicsimageio as a whole prior to 4.x series of
    releases.

    The result of this function should be an OME XML string that is relatively the
    same (no major pieces missing) but that validates against the reference OME
    XSD.

    Parameters
    ----------
    xml: str
        The OME XML string to clean for errors.

    Returns
    -------
    cleaned_xml: str
        The cleaned OME XML string.

    Raises
    ------
    ValueError
        Provided XML does not contain a namespace.
    """
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

    # Fix MicroManager Instrument and Detector
    instrument = root.find(f"{namespace}Instrument")
    if instrument is not None:
        instrument_id = instrument.get("ID")
        if instrument_id == "Microscope":
            ome_instrument_id = generate_ome_instrument_id(0)
            instrument.set("ID", ome_instrument_id)
            metadata_changes.append(
                f"Updated attribute 'ID' from '{instrument_id}' to "
                f"'{ome_instrument_id}' on Instrument element."
            )

            for detector_index, detector in enumerate(
                instrument.findall(f"{namespace}Detector")
            ):
                detector_id = detector.get("ID")
                if detector_id is not None:
                    # Create ome detector id if needed
                    ome_detector_id = None
                    if detector_id == "Camera":
                        ome_detector_id = generate_ome_detector_id(detector_index)
                    elif not detector_id.startswith("Detector:"):
                        ome_detector_id = generate_ome_detector_id(detector_id)

                    # Apply ome detector id if replaced
                    if ome_detector_id is not None:
                        detector.set("ID", ome_detector_id)
                        metadata_changes.append(
                            f"Updated attribute 'ID' from '{detector_id}' to "
                            f"'{ome_detector_id}' on Detector element at "
                            f"position {detector_index}."
                        )

    # Find all Image elements and fix IDs and refs to fixed instruments
    # This is for certain for test files of o.urs and ACTK files
    for image_index, image in enumerate(root.findall(f"{namespace}Image")):
        image_id = image.get("ID")
        if image_id is not None:
            found_image_id = image_id

            if not found_image_id.startswith("Image"):
                ome_image_id = generate_ome_image_id(found_image_id)
                image.set("ID", ome_image_id)
                metadata_changes.append(
                    f"Updated attribute 'ID' from '{image_id}' to '{ome_image_id}' "
                    f"on Image element at position {image_index}."
                )

        # Fix MicroManager bad instrument refs
        instrument_ref = image.find(f"{namespace}InstrumentRef")
        if instrument_ref is not None:
            instrument_ref_id = instrument_ref.get("ID")
            if instrument_ref_id == "Microscope":
                instrument_ref.set("ID", ome_instrument_id)

        # Find all Pixels elements and fix IDs
        for pixels_index, pixels in enumerate(image.findall(f"{namespace}Pixels")):
            pixels_id = pixels.get("ID")
            if pixels_id is not None:
                found_pixels_id = pixels_id

                if not found_pixels_id.startswith("Pixels"):
                    pixels.set("ID", f"Pixels:{found_pixels_id}")
                    metadata_changes.append(
                        f"Updated attribute 'ID' from '{found_pixels_id}' to "
                        f"Pixels:{found_pixels_id}' on Pixels element at "
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
                channels = [deepcopy(c) for c in pixels.findall(f"{namespace}Channel")]
                bin_data = [deepcopy(b) for b in pixels.findall(f"{namespace}BinData")]
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
                    if image is not None:
                        found_image = image

                        pixels_planes: Optional[ET.Element] = found_image.find(
                            f"{namespace}Pixels"
                        )
                        if pixels_planes is not None:
                            for plane in pixels_planes.findall(f"{namespace}Plane"):
                                for anno_ref in plane.findall(
                                    f"{namespace}AnnotationRef"
                                ):
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
        log.debug("OME metadata was cleaned for known AICSImageIO 3.x OMEXML errors.")
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


def dtype_to_ome_type(npdtype: np.dtype) -> PixelType:
    """
    Convert numpy dtype to OME PixelType

    Parameters
    ----------
    npdtype: numpy.dtype
        A numpy datatype.

    Returns
    -------
    ome_type: PixelType
        One of the supported OME Pixels types

    Raises
    ------
    ValueError
        No matching pixel type for provided numpy type.
    """
    ometypedict = {
        np.dtype(np.int8): PixelType.INT8,
        np.dtype(np.int16): PixelType.INT16,
        np.dtype(np.int32): PixelType.INT32,
        np.dtype(np.uint8): PixelType.UINT8,
        np.dtype(np.uint16): PixelType.UINT16,
        np.dtype(np.uint32): PixelType.UINT32,
        np.dtype(np.float32): PixelType.FLOAT,
        np.dtype(np.float64): PixelType.DOUBLE,
        np.dtype(np.complex64): PixelType.COMPLEXFLOAT,
        np.dtype(np.complex128): PixelType.COMPLEXDOUBLE,
    }
    ptype = ometypedict.get(npdtype)
    if ptype is None:
        raise ValueError(f"Ome utils can't resolve pixel type: {npdtype.name}")
    return ptype


def ome_to_numpy_dtype(ome_type: PixelType) -> np.dtype:
    """
    Convert OME PixelType to numpy dtype

    Parameters
    ----------
    ome_type: PixelType
        One of the supported OME Pixels types

    Returns
    -------
    npdtype: numpy.dtype
        A numpy datatype.

    Raises
    ------
    ValueError
        No matching numpy type for the provided pixel type.
    """
    ometypedict: Dict[PixelType, np.dtype] = {
        PixelType.INT8: np.dtype(np.int8),
        PixelType.INT16: np.dtype(np.int16),
        PixelType.INT32: np.dtype(np.int32),
        PixelType.UINT8: np.dtype(np.uint8),
        PixelType.UINT16: np.dtype(np.uint16),
        PixelType.UINT32: np.dtype(np.uint32),
        PixelType.FLOAT: np.dtype(np.float32),
        PixelType.DOUBLE: np.dtype(np.float64),
        PixelType.COMPLEXFLOAT: np.dtype(np.complex64),
        PixelType.COMPLEXDOUBLE: np.dtype(np.complex128),
    }
    nptype = ometypedict.get(ome_type)
    if nptype is None:
        raise ValueError(f"Ome utils can't resolve pixel type: {ome_type.value}")
    return nptype
