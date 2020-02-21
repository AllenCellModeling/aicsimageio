from abc import ABC, abstractmethod

from .. import constants, types


class Writer(ABC):
    def __init__(self, file_path: types.PathLike):
        """
        Write STCZYX data arrays to file, with accompanying metadata
        Will overwrite existing files of same name.

        Parameters
        ----------
        file_path: types.PathLike
            Path to image output location

        Examples
        --------
        Construct and use as object

        >>> image = numpy.ndarray([1, 10, 3, 1024, 2048])
        ... writer = DerivedWriter("file.ome.tif")
        ... writer.save(image)
        ... writer.close()

        Construct with a context manager

        >>> image2 = numpy.ndarray([5, 486, 210])
        ... with DerivedWriter("file2.ome.tif") as writer2:
        ...     writer2.set_metadata(myMetaData)
        ...     writer2.save(image2)
        """
        self.file_path = file_path

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @abstractmethod
    def close(self) -> None:
        """Close file objects"""
        pass

    @abstractmethod
    def save(self, data, dims=constants.Dimensions.DefaultOrder, **kwargs) -> None:
        """Write to open file"""
        pass
