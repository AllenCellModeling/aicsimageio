import pytest
from aicsimageio import AICSImage, readers


# TODO: Move to AICSImage tests
@pytest.mark.parametrize('filename, expected_reader', [
    ('img40_1.png', readers.DefaultReader),
    ('img40_1_dna.tif', readers.TiffReader),
    ('img40_1.ome.tif', readers.OmeTiffReader),
    ('T=5_Z=3_CH=2_CZT_All_CH_per_Slice.czi', readers.CziReader),
    pytest.param('not/a/file.czi', None, marks=pytest.mark.raises(exception=FileNotFoundError))
])
def test_typing(filename, expected_reader, image_dir):
    actual_reader = AICSImage.determine_reader(image_dir / filename)
    assert actual_reader == expected_reader
