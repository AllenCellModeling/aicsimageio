from typing import TYPE_CHECKING
from aicsimageio.readers import _loci
loci = _loci.loci()
formats = loci.formats
if TYPE_CHECKING:
    from jpype import JClass

class Prefs:
    @classmethod
    def is_enabled(cls, c: JClass) -> bool:
        return True


# Creates an image reader according to the current configuration settings
def make_image_reader():
    defaultClasses = formats.ImageReader.getDefaultReaderClasses()
    enabled = formats.ClassList(formats.IFormatReader)
    for c in defaultClasses.getClasses():
        if Prefs.is_enabled(c):
            enabled.addClass(c)
    reader = formats.ImageReader(enabled)
