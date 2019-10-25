import numpy as np
import tifffile

from . import omexml


class OmeTifReader:
    """This class is used primarily for opening and processing the contents of an OME Tiff file

    Example:
        reader = omeTifReader.OmeTifReader(path="file.ome.tif")
        file_image = reader.load()
        file_slice = reader.load_slice(t=1, z=2, c=3)

        with omeTifReader.OmeTifReader(path="file2.ome.tif") as reader:
            file2_image = reader.load()
            file2_slice = reader.load_slice(t=1, z=2, c=3)

    The load() function gathers all the slices into a single 5d array with dimensions TZCYX.
    This should be used when the entire image needs to be processed or transformed in some way.

    The load_slice() function takes a single 2D slice with dimensions YX out of the 5D image.
    This should be used when only a few select slices need to be processed
    (e.g. printing out the middle slice for a thumbnail image)

    This class has a similar interface to CziReader.
    """

    def __init__(self, file_path):
        """
        :param file_path(str): The path for the file that is to be opened.
        """
        self.file_path = file_path
        try:
            self.tif = tifffile.TiffFile(self.file_path)
        except ValueError:
            raise AssertionError("File is not a valid file type")
        except IOError:
            raise AssertionError("File is empty or does not exist")
        if self.tif.is_ome:
            d = self.tif.pages[0].description.strip()
            assert d.startswith('<?xml version=') and d.endswith('</OME>')
            self.omeMetadata = omexml.OMEXML(d)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self.tif.close()

    def load(self):
        """Retrieves an array for all z-slices and channels.

        :return: 5D array with dimensions TZCYX.
        """
        dimension_order = self.omeMetadata.image().Pixels.DimensionOrder
        # reverse the string
        dimension_order = dimension_order[::-1]
        # get the permutation of dimensionOrder from 'TZCYX', our preferred dimension order.
        transposition = tuple('TZCYX'.find(c) for c in dimension_order)

        # load the data
        data = self.tif.asarray()

        # fixups to get a 5D array
        if len(data.shape) == 1:
            # add dimensions T,Z,C,Y
            data = np.expand_dims(data, axis=0)
            data = np.expand_dims(data, axis=0)
            data = np.expand_dims(data, axis=0)
            data = np.expand_dims(data, axis=0)
        elif len(data.shape) == 2:
            # ASSUMPTION: both X and Y are > 1
            # add dimensions T,Z,C
            data = np.expand_dims(data, axis=0)
            data = np.expand_dims(data, axis=0)
            data = np.expand_dims(data, axis=0)
        elif len(data.shape) == 3:
            # ASSUMPTION: both X and Y are > 1
            # only one of z,c,t is > 1.  no transposing needed.
            if self.size_z() > 1:
                # insert C
                data = np.expand_dims(data, axis=1)
                # insert T
                data = np.expand_dims(data, axis=0)
            elif self.size_c() > 1:
                # insert T and Z at the beginning
                data = np.expand_dims(data, axis=0)
                data = np.expand_dims(data, axis=0)
            elif self.size_t() > 1:
                # insert C and Z after T
                data = np.expand_dims(data, axis=1)
                data = np.expand_dims(data, axis=1)
        elif len(data.shape) == 4:
            # ASSUMPTION: both X and Y are > 1
            # only one of z,c,t is dimension 1.
            if self.size_z() == 1:
                data = np.expand_dims(data, axis=dimension_order.find('Z'))
            elif self.size_c() == 1:
                data = np.expand_dims(data, axis=dimension_order.find('C'))
            elif self.size_t() == 1:
                data = np.expand_dims(data, axis=dimension_order.find('T'))
            else:
                data = np.expand_dims(data, axis=0)
            data = np.transpose(data, transposition)
        elif len(data.shape) == 5:
            data = np.transpose(data, transposition)

        if not len(data.shape) == 5:
            raise ValueError("Unexpected number of dimensions in ome.tif file")
        return data

    def load_slice(self, z=0, c=0, t=0):
        """Retrieves the 2D YX slice from the image

        :param z: The z index that will be accessed
        :param c: The channel that will be accessed
        :param t: The time index that will be accessed
        :return: 2D array with dimensions YX
        """
        index = c + (self.size_c() * z) + (self.size_c() * self.size_z() * t)
        data = self.tif.asarray(key=index)
        return data

    def get_metadata(self):
        return self.omeMetadata

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

    def dtype(self):
        return self.tif.pages[0].dtype

    def is_ome(self):
        """ This checks to make sure the metadata of the file to assure it is an OME Tiff file.

        TODO:
            * This function is not versatile and could certainly be tricked if somebody desired to do so.

        :return: True if file is OMETiff, False otherwise.
        """
        return self.file_path[-7:] == 'ome.tif' or self.file_path[-8:] == 'ome.tiff'
