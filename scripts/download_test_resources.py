#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import sys
import traceback
from pathlib import Path

from quilt3 import Package

###############################################################################

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)4s: %(module)s:%(lineno)4s %(asctime)s] %(message)s",
)
log = logging.getLogger(__name__)

###############################################################################
# Args


class Args(argparse.Namespace):
    def __init__(self):
        self.__parse()

    def __parse(self):
        # Setup parser
        p = argparse.ArgumentParser(
            prog="download_test_resources",
            description=(
                "Download files used for testing this project. This will download "
                "all the required test resources and place them in the "
                "`tests/resources` directory."
            ),
        )

        # Arguments
        p.add_argument(
            "--top-hash",
            # Generated package hash from upload_test_resources
            default=None,
            help=(
                "A specific version of the package to retrieve. "
                "If none, will read from the TEST_RESOURCES_HASH.txt file."
            ),
        )
        p.add_argument(
            "--debug",
            action="store_true",
            help="Show traceback if the script were to fail.",
        )

        # Parse
        p.parse_args(namespace=self)


###############################################################################
# Build package


def download_test_resources(args: Args):
    # Try running the download pipeline
    try:
        # Get test resources dir
        resources_dir = (
            Path(__file__).parent.parent / "aicsimageio" / "tests" / "resources"
        ).resolve()
        resources_dir.mkdir(exist_ok=True)

        # Use or read top hash
        if args.top_hash is None:
            with open(Path(__file__).parent / "TEST_RESOURCES_HASH.txt", "r") as f:
                top_hash = f.readline().rstrip()
        else:
            top_hash = args.top_hash

        log.info(f"Downloading test resources using top hash: {top_hash}")

        # Get quilt package
        package = Package.browse(
            "aicsimageio/test_resources",
            "s3://aics-modeling-packages-test-resources",
            top_hash=top_hash,
        )

        # Download
        package["resources"].fetch(resources_dir)

        log.info(f"Completed package download.")

    # Catch any exception
    except Exception as e:
        log.error("=============================================")
        if args.debug:
            log.error("\n\n" + traceback.format_exc())
            log.error("=============================================")
        log.error("\n\n" + str(e) + "\n")
        log.error("=============================================")
        sys.exit(1)


###############################################################################
# Runner


def main():
    args = Args()
    download_test_resources(args)


###############################################################################
# Allow caller to directly run this module (usually in development scenarios)

if __name__ == "__main__":
    main()
