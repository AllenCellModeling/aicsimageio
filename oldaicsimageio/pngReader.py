from PIL import Image
import numpy as np


class PngReader:
    """This class is used to open and process the contents of a png file.

    Examples:
        reader = pngReader.PngReader(path="file.png")
        file_image = reader.load()

        with pngReader.PngReader(path="file2.png") as reader:
            file2_image = reader.load()

    The load function will get a 3D (RGB)YX array from the png file.
    """

    def __init__(self, file_path):
        # nothing yet!
        self.filePath = file_path

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        pass

    def load(self):
        """
        :return: A 3D array of CYX, where C is the RBG channel.
        """
        # this is dumb but this is the way to make the file close correctly with Py3.5 :(
        # sorry future programmer
        with open(self.filePath, 'rb') as image_file:
            with Image.open(image_file) as image:
                data = np.asarray(image)
                if len(data.shape) == 3:
                    # returns cyx where c is rgb, rgba, or r
                    data = np.transpose(data, (2, 0, 1))
                return data
