#!/usr/bin/env python
# -*- coding: utf-8 -*-

from lxml import etree


def test_generate_and_validate(OMEXSD, CZIXML, xslt_template):
    # Parse template
    template = etree.parse(str(xslt_template))
    transform = etree.XSLT(template)

    # Parse CZI XML
    czixml = etree.parse(str(CZIXML))

    # Attempt to run transform
    produced = transform(czixml)

    # Parse OME XSD
    omexsd = etree.parse(str(OMEXSD))
    omexsd = etree.XMLSchema(omexsd)

    # Validate
    assert omexsd.validate(produced)
