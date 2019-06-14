from io import BytesIO
from pathlib import Path
import pytest
from aicsimageio.readers.reader import Reader


@pytest.mark.parametrize("file", [
    pytest.param("non_existent_file.random", marks=pytest.mark.raises(exception=FileNotFoundError)),
    pytest.param(
        Path("/non/existent/file/path/non_existent_file.random"),
        marks=pytest.mark.raises(exception=FileNotFoundError)
    ),
    BytesIO(b"abcdef"),
    b"abcdef"
    ]
)
def test_reader_constructor(file):
    """
    Testing the arguments to the static member function on the ABC
    Parameters
    ----------
    file The various objects [str(filename), pathlib.Path, BytesIO, bytestring]

    """
    Reader.convert_to_buffer(file)

