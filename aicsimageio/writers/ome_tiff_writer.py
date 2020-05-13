from __future__ import print_function

import os
from pathlib import Path
from typing import Union

import lxml.etree as ET
import numpy as np
import tifffile

from ..aics_image import AICSImage
from ..exceptions import InvalidDimensionOrderingError
from ..vendor import omexml

BYTE_BOUNDARY = 2 ** 21

# This is the threshold to use BigTiff, if it's the 4GB boundary it should be 2**22 but
# the libtiff writer was unable to handle a 2GB numpy array.
# It would be great if we better understood exactly what this threshold is and how to
# calculate it but for now this is a stopgap working value


class OmeTiffWriter:
    def __init__(self, file_path, overwrite_file=None):
        """
        This class can take arrays of pixel values and do the necessary metadata
        creation to write them properly in OME xml format.

        Parameters
        ----------
        file_path
            Path to image output location
        overwrite_file
            Flag to overwrite image or pass over image if it already exists.
            None: (default) throw IOError if file exists
            True: overwrite existing file if file exists
            False: silently perform no write actions if file exists

        Example
        -------
        >>> image = numpy.ndarray([1, 10, 3, 1024, 2048])
        ... writer = ome_tiff_writer.OmeTiffWriter("file.ome.tif")
        ... writer.save(image)

        Using the context manager to close the write once done.

        >>> image2 = numpy.ndarray([5, 486, 210])
        ... with ome_tiff_writer.OmeTiffWriter("file2.ome.tif") as writer2:
        ...     writer2.save(image2)

        Convert a CZI file into OME-Tiff

        >>> reader = cziReader.CziReader("file3.czi")
        ... with ome_tiff_writer.OmeTiffWriter("file3.ome.tif") as writer3:
        ...     writer.save(reader.load())
        """
        self.file_path = file_path
        self.omeMetadata = omexml.OMEXML()
        self.silent_pass = False
        if os.path.isfile(self.file_path):
            if overwrite_file:
                os.remove(self.file_path)
            elif overwrite_file is None:
                raise IOError(
                    "File {} exists but user has chosen not to overwrite it".format(
                        self.file_path
                    )
                )
            elif overwrite_file is False:
                self.silent_pass = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        pass

    def _build_ome(
        self,
        data,
        channel_names,
        image_name,
        pixels_physical_size,
        channel_colors,
        dimension_order,
    ) -> str:
        """
        When no `ome_xml` object is passed to the main `save` function,
        this function runs. It builds the OME metadata using the provided parameters
        using the `vendor.omexml.OMEXML` object.

        Because of this, this OME generator only supports up to 5D data.
        """
        shape = data.shape
        ndims = len(shape)

        assert (
            ndims == 5 or ndims == 4 or ndims == 3
        ), "Expected 3, 4, or 5 dimensions in data array"

        # assert valid characters in dimension_order
        if not (all(d in "STCZYX" for d in dimension_order)):
            raise InvalidDimensionOrderingError(
                f"Invalid dimension_order {dimension_order}"
            )
        if dimension_order[-2:] != "YX":
            raise InvalidDimensionOrderingError(
                f"Last two characters of dimension_order {dimension_order} expected to \
                be YX.  Please transpose your data."
            )
        if len(dimension_order) < ndims:
            raise InvalidDimensionOrderingError(
                f"dimension_order {dimension_order} must have at least as many \
                dimensions as data shape {shape}"
            )
        if dimension_order.find("S") > 0:
            raise InvalidDimensionOrderingError(
                f"S must be the leading dim in dimension_order {dimension_order}"
            )
        # todo ensure no letter appears more than once?

        # ensure dimension_order is same len as shape
        if len(dimension_order) > ndims:
            dimension_order = dimension_order[-ndims:]

        # if this is 3d data, then expand to 5D and add appropriate dimensions
        if ndims == 3:
            data = np.expand_dims(data, axis=0)
            data = np.expand_dims(data, axis=0)
            # prepend either TC, TZ or CZ
            if dimension_order[0] == "T":
                dimension_order = "CZ" + dimension_order
            elif dimension_order[0] == "C":
                dimension_order = "TZ" + dimension_order
            elif dimension_order[0] == "Z":
                dimension_order = "TC" + dimension_order

        # if this is 4d data, then expand to 5D and add appropriate dimensions
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

        # Make the OME metadata
        self._make_meta(
            data,
            channel_names=channel_names,
            image_name=image_name,
            pixels_physical_size=pixels_physical_size,
            channel_colors=channel_colors,
            dimension_order=dimension_order,
        )
        xml_str = self.omeMetadata.to_xml().encode()

        return xml_str

    def save(
        self,
        data,
        ome_xml=None,
        channel_names=None,
        image_name="IMAGE0",
        pixels_physical_size=None,
        channel_colors=None,
        dimension_order="STZCYX",
    ):
        """
        Save an image with the proper OME XML metadata.

        Parameters
        ----------
        data: An array of 3, 4, 5, or 6 dimensions to be written out to a file.

        ome_xml: A premade omexml.OMEXML object to use for metadata or the result of a
        metadata transform. If this object is provided, all other parameters are
        ignored.

        channel_names: The names for each channel to be put into the OME metadata

        image_name: The name of the image to be put into the OME metadata

        pixels_physical_size: The physical size of each pixel in the image

        channel_colors: The channel colors to be put into the OME metadata

        dimension_order: The dimension ordering in the data array. Will be assumed
        STZCYX if not specified
        """
        if self.silent_pass:
            return

        # If ome_xml is None, construct the OME from the provided parameters
        # As this function uses the `vendor.omexml.OMEXML` object we do some
        # data validation during the `_build_ome` function to ensure that the
        # data is at most 5 dimensions.
        if ome_xml is None:
            xml_str = self._build_ome(
                data,
                channel_names,
                image_name,
                pixels_physical_size,
                channel_colors,
                dimension_order,
            )

        # Otherwise a prebuilt xml tree or OMEXML object were provided. Trust them.
        else:
            # XML produced from a transform
            if isinstance(ome_xml, ET._XSLTResultTree):
                self.omeMetadata = ome_xml
                xml_str = str(self.omeMetadata).encode("utf-8")
            # OMEXML object provided
            elif isinstance(ome_xml, omexml.OMEXML):
                pixels = ome_xml.image().Pixels
                pixels.populate_TiffData()
                self.omeMetadata = ome_xml
                xml_str = self.omeMetadata.to_xml().encode()
            else:
                raise TypeError(
                    f"'ome_xml' parameter must be of type: "
                    f"`omexml.OMEXML` or `lxml.etree._XSLTResultTree`"
                )

        # Save tiff
        with tifffile.TiffWriter(
            self.file_path, bigtiff=self._size_of_ndarray(data=data) > BYTE_BOUNDARY,
        ) as writer:
            writer.save(
                data,
                compress=9,
                description=xml_str,
                photometric="minisblack",
                metadata=None,
            )

    def save_slice(self, data, z=0, c=0, t=0):
        """ this doesn't do the necessary functionality at this point

        TODO:
            * make this insert a YX slice in between two other slices inside a full
              image
            * data should be a 5 dim array

        :param data:
        :param z:
        :param c:
        :param t:
        :return:
        """
        if self.silent_pass:
            return

        assert len(data.shape) == 2
        assert data.shape[0] == self.size_y()
        assert data.shape[1] == self.size_x()
        tif = tifffile.TiffWriter(self.file_path)
        tif.save(data, compress=9)
        tif.close()

    def set_metadata(self, ome_metadata):
        self.omeMetadata = ome_metadata

    def size_z(self):
        return self.omeMetadata.image().Pixels.SizeZ

    def size_c(self):
        return self.omeMetadata.image().Pixels.SizeC

    def size_t(self):
        return self.omeMetadata.image().Pixels.SizeT

    def size_x(self):
        return self.omeMetadata.image().Pixels.SizeX

    def size_y(self):
        return self.omeMetadata.image().Pixels.SizeY

    @staticmethod
    def _size_of_ndarray(data: np.ndarray) -> int:
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
    def _make_meta(
        self,
        data,
        channel_names=None,
        image_name="IMAGE0",
        pixels_physical_size=None,
        channel_colors=None,
        dimension_order="TCZYX",
    ):
        """Creates the necessary metadata for an OME tiff image

        :param data: An array of dimensions TZCYX, ZCYX, or ZYX to be written out to a
        file.
        :param channel_names: The names for each channel to be put into the OME metadata
        :param image_name: The name of the image to be put into the OME metadata
        :param pixels_physical_size: The physical size of each pixel in the image
        :param channel_colors: The channel colors to be put into the OME metadata
        :param dimension_order: The order of dimensions in the data array, using
        T,C,Z,Y and X
        """
        ox = self.omeMetadata

        ox.image().set_Name(image_name)
        ox.image().set_ID("0")
        pixels = ox.image().Pixels
        pixels.ome_uuid = ox.uuidStr
        pixels.set_ID("0")
        if pixels_physical_size is not None:
            pixels.set_PhysicalSizeX(pixels_physical_size[0])
            pixels.set_PhysicalSizeY(pixels_physical_size[1])
            pixels.set_PhysicalSizeZ(pixels_physical_size[2])

        shape = data.shape

        def dim_or_1(dim):
            idx = dimension_order.find(dim)
            return 1 if idx == -1 else shape[idx]

        pixels.channel_count = dim_or_1("C")
        pixels.set_SizeT(dim_or_1("T"))
        pixels.set_SizeC(dim_or_1("C"))
        pixels.set_SizeZ(dim_or_1("Z"))
        pixels.set_SizeY(dim_or_1("Y"))
        pixels.set_SizeX(dim_or_1("X"))

        # this must be set to the *reverse* of what dimensionality
        # the ome tif file is saved as
        pixels.set_DimensionOrder(dimension_order[::-1])

        # convert our numpy dtype to a ome compatible pixeltype
        pixels.set_PixelType(omexml.get_pixel_type(data.dtype))

        if channel_names is None:
            for i in range(pixels.SizeC):
                pixels.Channel(i).set_ID("Channel:0:" + str(i))
                pixels.Channel(i).set_Name("C:" + str(i))
        else:
            for i in range(pixels.SizeC):
                name = channel_names[i]
                pixels.Channel(i).set_ID("Channel:0:" + str(i))
                pixels.Channel(i).set_Name(name)

        if channel_colors is not None:
            assert len(channel_colors) >= pixels.get_SizeC()
            for i in range(pixels.SizeC):
                pixels.Channel(i).set_Color(channel_colors[i])

        # assume 1 sample per channel
        for i in range(pixels.SizeC):
            pixels.Channel(i).set_SamplesPerPixel(1)

        # many assumptions in here: one file per image, one plane per tiffdata, etc.
        pixels.populate_TiffData()

        return ox


def convert_to_ome_tiff(
    original_file: Union[str, Path, AICSImage], save_path: Union[str, Path],
) -> Path:
    """
    Given a filepath or AICSImage object, convert and save as an OME-TIFF.

    Parameters
    ----------
    original_file: Union[str, Path, AICSImage]
        The original file as a path or already initialized AICSImage

    save_path: Union[str, Path]
        The target path to save the produced OME-TIFF to.

    Returns
    -------
    save_path: Path
        The location to the saved OME-TIFF.
    """
    # Read as AICSImage
    if not isinstance(original_file, AICSImage):
        original_file = AICSImage(original_file)

    # Fully resolve save_path
    save_path = Path(save_path).resolve()

    # Get metadata
    ome_metadata = original_file.get_ome_metadata()

    # Get per-scene dimension order
    scene_dim_order = ome_metadata.find("Image").get("DimensionOrder")

    # Get image data reshaped to metadata dimension order
    # OME metadata stores dimension order in reverse of the order it is saved in
    # This requests the image data in "S" + the reverse of the per-scene dimension order
    # 🤷 image formats 🤷
    ome_image_data = original_file.get_image_data(f"S{scene_dim_order[::-1]}")

    # Write OME-TIFF
    with OmeTiffWriter(save_path) as writer:
        writer.save(ome_image_data, ome_xml=ome_metadata)

    return save_path
