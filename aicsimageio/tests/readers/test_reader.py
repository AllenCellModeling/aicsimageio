from io import BytesIO
from pathlib import Path
import pytest
from tempfile import TemporaryFile
from aicsimageio.readers.reader import Reader


def test_reader_constructor_file_str():
    """
    test constructor takes a filename
    if filename doesn't exist expect FileNotFoundError
    """
    with pytest.raises(FileNotFoundError):
        Reader("non_existent_file.random")


def test_reader_constructor_pathlib():
    """
    test constructor takes a filepath object
    if filepath/file doesn't exist expect FileNotFound
    """
    with pytest.raises(FileNotFoundError):
        Reader(Path("/non/existent/file/path/non_existent_file.random"))


def test_reader_constructor_bytestream():
    """
    test constructor takes a bytestream
    this code should not raise
    """
    b = BytesIO(b"abcdef")
    Reader(b)


def test_reader_constructor_filepointer():
    """
    test constructor with a filepointer
    this code should not raise
    """
    with TemporaryFile() as fp:
        fp.write(b"aoetnuhasoenthuasnoethuasntoehunateou")
        fp.seek(0)
        Reader(fp)


def test_instantiation_loader():
    """
    test ABC.method _load_from_bytes
    """
    with pytest.raises(TypeError):
        with TemporaryFile() as fp:
            fp.write(b"aoetnuhasoenthuasnoethuasntoehunateou")
            fp.seek(0)
            r = Reader(fp)
            r._load_from_bytes()


def test_instantiation_check_type():
    """
    test ABC.method check_type()
    """
    with pytest.raises(TypeError):
        with TemporaryFile() as fp:
            fp.write(b"aoetnuhasoenthuasnoethuasntoehunateou")
            fp.seek(0)
            r = Reader(fp)
            r.check_type()
