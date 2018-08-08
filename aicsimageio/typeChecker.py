import logging
import re
import sys


class TypeChecker:
    """Cheaply check to see if a given file is a recognized type.
    Currently recognized types are TIFF, OME TIFF, and CZI.
    If the file is a TIFF, then the description (OME XML if it is OME TIFF) can be retrieved via read_description.
    Similarly, if the file is a CZI, then the metadata XML can be retrieved via read_description.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.description_length = 0
        self.description_offset = 0
        self.is_czi = False
        self.is_tiff = False
        self.is_ome = False

        with open(file_path, 'rb') as file:
            self.endianness = bytearray(file.read(2))
            # Per the TIFF-6 spec (https://www.itu.int/itudoc/itu-t/com16/tiff-fx/docs/tiff6.pdf),
            # 'II' is little-endian (Intel format) and 'MM' is big-endian (Motorola format)
            if self.endianness == b'II' or self.endianness == b'MM':
                magic = self.read_uint16(file)
                # Per TIFF-6, magic is 42.  Per BigTIFF (https://www.awaresystems.be/imaging/tiff/bigtiff.html), magic is 43.
                self.is_ome = self.check_tiff(file) if magic == 42 else self.check_bigtiff(file) if magic == 43 else False
            elif self.endianness == b'ZI':
                moremagic = self.endianness + bytearray(file.read(8))
                self.is_czi = moremagic == b'ZISRAWFILE'
                if self.is_czi:
                    # CZI files are little-endian
                    self.endianness = b'II'
                    self.find_czi_metadata(file)

    def read_description(self):
        """Retrive the image description as one large string.
        If this is an OME TIFF, then the description is OME XML.
        If this is a CZI, then the description is metadata XML from the ZISRAWMETADATA segment.
        """
        if self.description_offset > 0:
            with open(self.file_path, 'rb') as file:
                file.seek(self.description_offset, 0)
                return file.read(self.description_length)
        else:
            return ""

    # All of these read_uint* routines obey the endianness, with 'II' being little-endian and 'MM' being big-endian (per TIFF-6)
    def read_uint16(self, file):
        value = bytearray(file.read(2))
        return (value[0] + (value[1] << 8)) if self.endianness == b'II' else (value[1] + (value[0] << 8))

    def read_uint32(self, file):
        value = bytearray(file.read(4))
        return (value[0] + (value[1] << 8) + (value[2] << 16) + (value[3] << 24)) if self.endianness == b'II' else (value[3] + (value[2] << 8) + (value[1] << 16) + (value[0] << 24))

    def read_uint64(self, file):
        return (self.read_uint32(file) + (self.read_uint32(file) << 32)) if self.endianness == b'II' else ((self.read_uint32(file) << 32) + self.read_uint32(file))

    def check_tiff(self, file):
        ifd_offset = self.read_uint32(file)
        if ifd_offset == 0:
            return False
        self.is_tiff = True
        file.seek(ifd_offset, 0)
        entries = self.read_uint16(file)
        for n in range(0, entries):
            tag = self.read_uint16(file)
            type = self.read_uint16(file)
            count = self.read_uint32(file)
            offset = self.read_uint32(file)
            if tag == 270:
                return self.check_description(file, count, offset)
        # chaining IFDs - the description might not be in the first IFD
        return self.check_tiff(file)

    def check_bigtiff(self, file):
        if self.read_uint16(file) != 8:
            return False
        if self.read_uint16(file) != 0:
            return False
        return self.check_bigtiff_ifd(file)

    def check_bigtiff_ifd(self, file):
        ifd_offset = self.read_uint64(file)
        if ifd_offset == 0:
            return False
        self.is_tiff = True
        file.seek(ifd_offset, 0)
        entries = self.read_uint64(file)
        for n in range(0, entries):
            tag = self.read_uint16(file)
            type = self.read_uint16(file)
            count = self.read_uint64(file)
            offset = self.read_uint64(file)
            if tag == 270:
                return self.check_description(file, count, offset)
        # chaining IFDs - the description might not be in the first IFD
        return self.check_bigtiff_ifd(file)

    def check_description(self, file, count, offset):
        self.description_length = count - 1  # drop the NUL from the end
        self.description_offset = offset
        if count < 1024:
            return False
        file.seek(offset, 0)
        buf = bytearray(file.read(1024))
        if buf[0:5] != b"<?xml":
            return False
        match = re.search(b'<(\\w*)(:?)OME [^>]*xmlns\\2\\1="http://www.openmicroscopy.org/Schemas/[Oo][Mm][Ee]/', buf)
        if match is None:
            return False
        return True

    def find_czi_metadata(self, file):
        file.seek(92, 0)  # See the CZI format specification; this is the absolute offset of the MetadataPosition element of the file header
        segment_offset = self.read_uint64(file)
        if segment_offset > 0:
            file.seek(segment_offset, 0)
            segment_name = bytearray(file.read(16))
            if segment_name[0:14] == b'ZISRAWMETADATA':  # This should always be true, but best to be sure
                allocated_size = self.read_uint64(file)
                used_size = self.read_uint64(file)
                self.description_length = self.read_uint32(file)
                self.description_offset = segment_offset + 32 + 256

if __name__ == '__main__':
    log = logging.getLogger()
    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)4s:%(lineno)4s %(asctime)s] %(message)s')

    try:
        checker = TypeChecker(sys.argv[1])
        if checker.is_ome:
            print("OME-TIFF")
        elif checker.is_tiff:
            print("TIFF")
        elif checker.is_czi:
            print("CZI")
        else:
            print("unknown")
        sys.exit(0 if checker.is_ome else 1)
    except Exception as e:
        log.error("{}".format(e))
        log.error("=====================================================")
        log.error("\n" + traceback.format_exc())
        log.error("=====================================================")
        sys.exit(1)
    # How did we get here?
    sys.exit(2)
