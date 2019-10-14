<?xml version="1.0" encoding="UTF-8"?>
<!--
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Comments in this template will generally be pointers from spec to spec
# Example:
#   Get Instrument Info
#   zisraw/Instrument.xsd: 0
#   ome/ome.xsd: 979
#
# This means that for more details on how this section of the template was created
# view line 0 of the zisraw/Instrument.xsd file and view line 979 of the ome/ome.xsd file.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
-->

<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

    <xsl:output method="xml" version="1.0" encoding="UTF-8"/>

    <xsl:template match="/">
        <OME>
            <!-- Get Instrument Info -->
            <!-- zisraw/Instrument.xsd: 0 -->
            <!-- ome/ome.xsd: 979 -->
            <Instrument>
                <!-- Attempt to get the instrument id but it may not be there, default to 0 -->
                <xsl:attribute name="ID">
                    <xsl:choose>
                        <xsl:when test="/ImageDocument/Metadata/Information/Instrument/@Id">
                            <xsl:value-of select="/ImageDocument/Metadata/Information/Instrument/@Id"/>
                        </xsl:when>
                        <xsl:otherwise>Instrument:0</xsl:otherwise>
                    </xsl:choose>
                </xsl:attribute>

                <!-- Go to the first Microscope element and get the Type -->
                <xsl:for-each select="/ImageDocument/Metadata/Information/Instrument/Microscopes">
                    <xsl:if test="position()=1">
                        <Microscope>
                            <xsl:attribute name="Type">
                                <xsl:value-of
                                    select="/ImageDocument/Metadata/Information/Instrument/Microscopes/Microscope/Type"
                                />
                            </xsl:attribute>
                        </Microscope>
                    </xsl:if>
                </xsl:for-each>
                <!-- <xsl:if test="/ImageDocument/Metadata/Information/Instrument/Manufacturer/Manufacturer">
                    <xsl:attribute name="Manufacturer">

                    </xsl:attribute>
                </xsl:if> -->
            </Instrument>

            <!-- Basic Deep Copy Instrument -->
            <!-- <xsl:copy-of select="/ImageDocument/Metadata/Information/Instrument"/> -->

        </OME>
    </xsl:template>

</xsl:stylesheet>
