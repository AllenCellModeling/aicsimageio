#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import pytest


@pytest.fixture
def czi_to_ome_dir() -> Path:
    return Path(__file__).parent.parent.parent


@pytest.fixture
def OMEXSD(czi_to_ome_dir: Path) -> Path:
    return (czi_to_ome_dir / "ome" / "ome.xsd").resolve(strict=True)


@pytest.fixture
def CZIXML(czi_to_ome_dir: Path) -> Path:
    return (czi_to_ome_dir / "resources" / "example-czi.xml").resolve(strict=True)


@pytest.fixture
def xslt_template(czi_to_ome_dir: Path) -> Path:
    return (czi_to_ome_dir / "xslt" / "czi-to-ome.xsl").resolve(strict=True)
