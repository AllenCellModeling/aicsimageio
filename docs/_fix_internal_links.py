#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fixes internal cross-documentation links by switching out their
suffixes. I.E. we want the docs to work both as normal files on
GitHub (using markdown), and, when converted and rendered as HTML.
"""


import argparse
import logging
import re
from pathlib import Path

###############################################################################

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)4s: %(module)s:%(lineno)4s %(asctime)s] %(message)s",
)
log = logging.getLogger(__name__)

###############################################################################


class Args(argparse.Namespace):
    def __init__(self):
        self.__parse()

    def __parse(self):
        # Setup parser
        p = argparse.ArgumentParser(
            prog="fix_internal_links",
            description=(
                "Fixes internal cross-documentation links by switching out their "
                "suffixes. I.E. we want the docs to work both as normal files on "
                "GitHub (using markdown), and, when converted and rendered as HTML."
            ),
        )

        # Arguments
        p.add_argument(
            "--current-suffix",
            default=".md",
            dest="current_suffix",
            help="The suffix to switch internal cross-documentation references from.",
        )
        p.add_argument(
            "--target-suffix",
            default=".html",
            dest="target_suffix",
            help="The suffix to switch internal cross-documentation references to.",
        )

        # Parse
        p.parse_args(namespace=self)


###############################################################################


def fix_file(f: Path, current_suffix: str = ".md", target_suffix: str = ".html"):
    # We could use formatted strings here but {} are valid characters in regex
    # instead just use string appending
    #
    # Look for exact characters "./" followed by at least one, upper or lower A-Z
    # and allow hyphen and underscore characters
    # followed by the dev suffix
    # Group 1 is the file_name
    # Group 2 is the suffix
    RE_SUB_CURRENT = r"(\.\/[a-zA-Z_-]+)(" + current_suffix + r")"

    # Keep group 1
    # attach the new suffix
    RE_SUB_TARGET = r"\1" + target_suffix

    # Read in text
    with open(f, "r") as open_resource:
        txt = open_resource.read()

    # Fix suffixes
    cleaned = re.sub(RE_SUB_CURRENT, RE_SUB_TARGET, txt)

    with open(f, "w") as open_resource:
        open_resource.write(cleaned)


###############################################################################


def main():
    args = Args()

    # Get docs dir
    docs = Path(__file__).parent.resolve()

    # Get files in dir
    for f in docs.glob("*.md"):
        fix_file(f, args.current_suffix, args.target_suffix)
        log.info(f"Cleaned file: {f}")


###############################################################################
# Allow caller to directly run this module (usually in development scenarios)

if __name__ == "__main__":
    main()
