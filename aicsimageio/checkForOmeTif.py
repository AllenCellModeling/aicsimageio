# This file is preserved only for backwards compatibility

import logging
import sys
from aicsimageio.typeChecker import TypeChecker as CheckForOmeTif

if __name__ == '__main__':
    log = logging.getLogger()
    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)4s:%(lineno)4s %(asctime)s] %(message)s')

    try:
        checker = CheckForOmeTif(sys.argv[1])
        print("OME-TIFF" if checker.is_ome else "not")
        sys.exit(0 if checker.is_ome else 1)
    except Exception as e:
        log.error("{}".format(e))
        log.error("=====================================================")
        log.error("\n" + traceback.format_exc())
        log.error("=====================================================")
        sys.exit(1)
    # How did we get here?
    sys.exit(2)
