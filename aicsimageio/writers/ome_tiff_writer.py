from ome_types import from_xml, to_xml
from ome_types.model.ome import OME, Image, Pixels, PixelType, TiffData, Channel
from tifffile import TiffFile, TiffFileError, TiffTag
from typing import Dict, List, Tuple, Union

from .. import constants, exceptions, types

# set up some sensible defaults from provided info
def make_meta(
    data: types.ArrayLike,
    channel_names: List[str] = None,
    image_name: str = "IMAGE0",
    pixels_physical_size: Tuple[float, float, float] = None,
    channel_colors: List[int] = None,
    dimension_order: str = "TCZYX",
):
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
        idx = dimension_order.find(dim)
        return 1 if idx == -1 else shape[idx]

    # pixels.channel_count = dim_or_1("C")
    pixels.size_t = dim_or_1("T")
    pixels.size_c = dim_or_1("C")
    pixels.size_z = dim_or_1("Z")
    pixels.size_y = dim_or_1("Y")
    pixels.size_x = dim_or_1("X")

    # this must be set to the *reverse* of what dimensionality
    # the ome tif file is saved as
    pixels.dimension_order = dimension_order[::-1]

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
