from aicsimageio.version import MODULE_VERSION

from .aics_image import AICSImage  # noqa: F401
from .aics_image import imread  # noqa: F401


def get_version():
    return MODULE_VERSION
