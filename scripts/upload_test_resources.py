#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import sys
import traceback
from pathlib import Path

from quilt3 import Package

from aicsimageio import __version__

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
            prog="upload_test_resources",
            description=(
                "Upload files used for testing this project. This will upload "
                "whatever files are currently found in the `tests/resources` directory."
                "To add more test files, simply add them to the `tests/resources` "
                "directory and rerun this script."
            ),
        )

        # Arguments
        p.add_argument(
            "--dry-run",
            action="store_true",
            help=(
                "Conduct dry run of the package generation. Will create a JSON "
                "manifest file of that package instead of uploading."
            ),
        )
        p.add_argument(
            "-y", "--yes",
            action="store_true",
            dest="preappoved",
            help="Auto-accept upload of files."
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


def upload_test_resources(args: Args):
    # Try running the download pipeline
    try:
        # Get test resources dir
        resources_dir = (
            Path(__file__).parent.parent / "aicsimageio" / "tests" / "resources"
        ).resolve(strict=True)

        # Report with directory will be used for upload
        log.info(f"Using contents of directory: {resources_dir}")

        # Create quilt package
        package = Package()
        package.set_dir("resources", resources_dir)

        # Report package contents
        log.info(f"Package contents: {package}")

        # Construct package name
        package_name = "aicsimageio/test_resources"

        # Check for dry run
        if args.dry_run:
            # Attempt to build the package
            built = package.build(package_name)

            # Get resolved save path
            manifest_save_path = Path("upload_manifest.jsonl").resolve()
            with open(manifest_save_path, "w") as manifest_write:
                package.dump(manifest_write)

            # Report where manifest was saved
            log.info(f"Dry run generated manifest stored to: {manifest_save_path}")
            log.info(f"Completed package dry run. Result hash: {built.top_hash}")

        # Upload
        else:
            # Check pre-approved push
            if args.preapproved:
                confirmation = True
            else:
                # Get upload confirmation
                confirmation = None
                while confirmation is None:
                    # Get user input
                    user_input = input("Upload [y]/n? ")

                    # If the user simply pressed enter assume yes
                    if len(user_input) == 0:
                        user_input = "y"
                    # Get first character and lowercase
                    else:
                        user_input = user_input[0].lower()

                        # Set confirmation from None to a value
                        if user_input == "y":
                            confirmation = True
                        elif user_input == "n":
                            confirmation = False

            # Check confirmation
            if confirmation:
                pushed = package.push(
                    package_name,
                    "s3://aics-modeling-packages-test-resources",
                    message=f"Test resources for `aicsimageio` version: {__version__}."
                )

                log.info(f"Completed package push. Result hash: {pushed.top_hash}")
            else:
                log.info(f"Upload canceled.")

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
    upload_test_resources(args)


###############################################################################
# Allow caller to directly run this module (usually in development scenarios)

if __name__ == "__main__":
    main()
