<?xml version="1.0" encoding="UTF-8"?>
<!--
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Comments in this template will generally be pointers from spec to spec
# Example:
#   Get Instrument Info
#   zisraw/Instrument.xsd: 45
#   ome/ome.xsd: 979
#
# This means that for more details on how this section of the template was created
# view line 45 of the zisraw/Instrument.xsd file and view line 979 of the ome/ome.xsd file.
#
# We are doing some interesting data passing between templates
# because we can pass entire trees from template to template.
# To make components reusable we make base templates that return
# complete XML objects based off the data provided. An example of this is the
# Instrument "object" (template) which can be reused multiple times
# based off whatever `{type}_data` tree is passed in.
#
# This can be written similarly in Python like so:
# ```python
# class Microscope():
#   def __init__(self, microscope_data):
#       self.type = microscope_data["Type"]
#       ...
#
# class Instrument():
#   def __init__(self, instrument_data):
#       self.id = instrument_data[Id]
#       self.microscope = Microscope(instrument_data["Microscopes"][0])
#
# obj = Instrument("/xpath/...")
# ```
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
-->

<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <!-- Output format -->
    <xsl:output method="xml" version="1.0" encoding="UTF-8"/>

    <!-- Includes -->
    <xsl:include href="instrument.xsl"/>

    <!-- Begin Template -->
    <xsl:template match="/">
        <OME>
            <!-- Attach Instrument -->
            <xsl:call-template name="Instrument">
              <xsl:with-param name="instrument_data" select="/ImageDocument/Metadata/Information/Instrument"/>
            </xsl:call-template>
        </OME>
    </xsl:template>

</xsl:stylesheet>
