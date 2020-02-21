import os

import imageio
import numpy as np


class PngWriter:
    def __init__(self, file_path, overwrite_file=None):
        """
        This class can take 3D arrays of CYX pixel values and writes them to a PNG.

        Parameters
        ----------
        file_path
            Path to image output location
        overwrite_file
            Flag to overwrite image or pass over image if it already exists.
            None : (default) throw IOError if file exists.
            True : overwrite existing file if file exists.
            False: silently perform no write actions if file exists.

        Examples
        --------
        Construct and use as object

        >>> image = numpy.ndarray([3, 1024, 2048])
        ... writer = png_writer.PngWriter("file.png")
        ... writer.save(image)

        Construct within a context manager

        >>> image2 = numpy.ndarray([3, 1024, 2048])
        ... with png_writer.PngWriter("file2.png") as writer2:
        ...     writer2.save(image2)
        """
        self.file_path = file_path
        self.silent_pass = False
        if os.path.isfile(self.file_path):
            if overwrite_file:
                os.remove(self.file_path)
            elif overwrite_file is None:
                raise IOError("File {} exists but user has chosen not to overwrite it".format(self.file_path))
            elif overwrite_file is False:
                self.silent_pass = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        pass

    def save(self, data):
        """
        Takes in an array of CYX pixel values and writes them to a PNG.

        Parameters
        ----------
        data
            A CYX or YX array with C being the rgb channels for each pixel value.
        """
        if self.silent_pass:
            return

        # check for rgb, rgba, or r
        if len(data.shape) == 3:
            assert data.shape[0] in [4, 3, 2, 1]
            # if three dimensions, transpose to YXC (imsave() needs it in these axes)
            data = np.transpose(data, (2, 1, 0))
            # if there's only one channel, repeat across the next two channels
            if data.shape[2] == 1:
                data = np.repeat(data, repeats=3, axis=2)
            elif data.shape[2] == 2:
                data = np.pad(data, ((0, 0), (0, 0), (0, 1)), 'constant')
        elif len(data.shape) != 2:
            raise ValueError("Data was not of dimensions CYX or YX")

        imageio.imwrite(self.file_path, data, format="png")

    def save_slice(self, data, z=0, c=0, t=0):
        """
        Exactly the same functionality as save() but allows the interface to be the same as OmeTiffWriter.

        Parameters
        ----------
        data
            A CYX or YX array with C being the rgb channels for each pixel value.
        z
            An arbitrary Z index that does nothing.
        c
            An arbitrary C index that does nothing.
        t
            An arbitrary T index that does nothing.
        """
        if self.silent_pass:
            return

        self.save(data)
