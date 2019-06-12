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
    (BytesIO(b"abcdef")),
    (b"abcdef")
])
def test_reader_constructor(file):
    """
    test constructor takes a filename
    if filename doesn't exist expect FileNotFoundError
    """
    Reader.convert_to_bytes_io(file)

