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
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
-->

<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

    <xsl:output method="xml" version="1.0" encoding="UTF-8"/>

    <xsl:template match="/">
        <OME>
            <!-- Get Instrument Info -->
            <!-- zisraw/Instrument.xsd: 45 -->
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

                <!-- Microscope Info -->
                <!-- zisraw/Instrument.xsd: 50 -->
                <!-- ome/ome.xsd: 1016 -->
                <Microscope>
                    <!-- Go to the first Microscope element and attempt to get the Type -->
                    <!-- Also attempt to get the Manufacturer, Model, SerialNumber, and LotNumber -->
                    <xsl:for-each select="/ImageDocument/Metadata/Information/Instrument/Microscopes">
                        <xsl:if test="position()=1">
                            <xsl:if test="Microscope/Type">
                                <xsl:attribute name="Type">
                                    <xsl:value-of select="Microscope/Type"/>
                                </xsl:attribute>
                            </xsl:if>

                            <!-- Manufacturer Info -->
                            <!-- zisraw/Instrument.xsd: 11 -->
                            <!-- (referenced at zisraw/Instrument.xsd: 157) -->
                            <!-- ome/ome.xsd: 1429 -->
                            <xsl:if test="Microscope/Manufacturer">
                                <xsl:attribute name="Manufacturer">
                                    <xsl:value-of select="Microscope/Manufacturer/Manufacturer"/>
                                </xsl:attribute>
                                <xsl:attribute name="Model">
                                    <xsl:value-of select="Microscope/Manufacturer/Model"/>
                                </xsl:attribute>
                                <xsl:attribute name="SerialNumber">
                                    <xsl:value-of select="Microscope/Manufacturer/SerialNumber"/>
                                </xsl:attribute>
                                <xsl:attribute name="LotNumber">
                                    <xsl:value-of select="Microscope/Manufacturer/LotNumber"/>
                                </xsl:attribute>
                            </xsl:if>
                        </xsl:if>
                    </xsl:for-each>
                </Microscope>
            </Instrument>

            <!-- Basic Deep Copy Instrument -->
            <!-- <xsl:copy-of select="/ImageDocument/Metadata/Information/Instrument"/> -->

        </OME>
    </xsl:template>

</xsl:stylesheet>
