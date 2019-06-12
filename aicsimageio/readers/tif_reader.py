import tifffile
import numpy as np


class TifReader:
    """This class is used to open and process the contents of a tif file.

    Examples:
        reader = tifReader.TifReader(path="file.tif")
        file_image = reader.load()

        with tifReader.TifReader(path="file2.tif") as reader:
            file2_image = reader.load()

    The load function will get a 3D ZYX array from a tif file.
    """

    def __init__(self, file_path):
        # nothing yet!
        self.filePath = file_path
        self.tif = tifffile.TiffFile(self.filePath)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self.tif.close()

    def load(self):
        """This will get an entire z stack from a tif file.

        :return: A 5D TZCYX array from the tif file.
        """
        im = self.tif.asarray()
        # convert to 5D TZCYX
        # 2 dims assumes YX (?)
        if len(im.shape) == 2:
            # insert C
            im = np.expand_dims(im, 0)
            # insert Z
            im = np.expand_dims(im, 0)
            # insert T
            im = np.expand_dims(im, 0)
        # 3 dims
        elif len(im.shape) == 3:
            # if shape is y,x,3 with only 1 page then assume this is an RGB image and treat im as a 1,1,3,y,x 5D tiff.
            if len(self.tif.pages) == 1 and im.shape[2] == 3:
                # transpose Y-X-3 to 3-Y-X
                im = im.transpose(2, 0, 1)
                # insert Z
                im = np.expand_dims(im, 0)
                # insert T
                im = np.expand_dims(im, 0)
            else:  # assume ZYX shape
                # insert C
                im = np.expand_dims(im, 1)
                # insert T
                im = np.expand_dims(im, 0)
        elif len(im.shape) == 4:
            # assume ZCYX if this image has imagej metadata.
            # Otherwise, well we don't really have a good guess so insert T dimension at the front anyway.
            # if self.tif.is_imagej:
            # insert T
            im = np.expand_dims(im, 0)
        return im

    def load_slice(self, z=0, c=0, t=0):
        """This will get a single slice out of the z stack of a tif file.

        :param z: The z index within the tiff stack
        :param c: An arbitrary c index that does nothing
        :param t: An arbitrary t index that does nothing
        :return: A 2D YX slice from the tiff file.
        """
        index = z
        data = self.tif.asarray(key=index)
        return data

    def get_metadata(self):
        return None

    def size_z(self):
        return len(self.tif.pages)

    def size_c(self):
        return 1

    def size_t(self):
        return 1

    def size_x(self):
        return self.tif.pages[0].shape[1]

    def size_y(self):
        return self.tif.pages[0].shape[0]

    def dtype(self):
        return self.tif.pages[0].dtype
