# AICS Image library
The oldaicsimageio package is designed to provide an easy interface with CZI, OME-TIFF, and PNG file formats.

## Level of Support
We are not currently supporting this code for external use, but simply releasing it 
to the community AS IS. It is used for within our organization. We are not able to 
provide any guarantees of support. The community is welcome to submit issues, but 
you should not expect an active response.

## Development
See [BUILD.md](BUILD.md) for information operations related to developing the code.

## Usage

```
from oldaicsimageio import AICSImage

img = AICSImage("my_ome.tiff_or_tiff_or_czi")

# Get the image data as TCZYX
img.data

# Get channel information if you have an OME tiff
pixels = img.metadata.image().Pixels
channels = [pixels.Channel(i) for i in range(pixels.get_channel_count())]
channels = [{"name": c.get_Name(), "index": c.get_ID()} for c in channels]


# Note on channel id differences between oldaicsimageio.OMEXML and lxml.etree._Element:
        # Under lxml.etree._Element, Channel Id looks like the following: `'Channel:0'`
        # Where the single integer corresponds to the channel dimension index in image data.
        # Under oldaicsimageio, the same Channel Id looks like the following: `'Channel:0:0'`
        # Where it is the second of the two integers that corresponds to the channel dimension index in image data.
        # Regardless of structure, these can both be parsed as integers with the following:
        # `int(channel_id.split(":")[-1])`

```