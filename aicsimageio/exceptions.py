

class UnsupportedFileFormatError(Exception):
    """
    This exception is intended to communicate that the file extension is not one of
    the supported file types and cannot be parsed with AICSImage.
    """
    def __init__(self, target, **kwargs):  # removed Argument Typing to allow py2.7 (W, Micro)
        super().__init__(**kwargs)
        self.target = target

    def __str__(self):
        return f"AICSImage module does not support this image file type: \t{self.target}"
