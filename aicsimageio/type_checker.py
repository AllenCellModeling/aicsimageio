

class TypeChecker:
    """Cheaply check to see if a given file is a recognized type.
    Currently recognized types are TIFF, OME TIFF, and CZI.
    If the file is a TIFF, then the description (OME XML if it is OME TIFF) can be retrieved via read_description.
    Similarly, if the file is a CZI, then the metadata XML can be retrieved via read_description.
    """
    pass
