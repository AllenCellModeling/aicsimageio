<?xml version="1.0" encoding="UTF-8"?>
<!-- #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # # Comments in this template will generally be pointers from spec to spec # Example: # Get Instrument Info # zisraw/Instrument.xsd: 45 #
ome/ome.xsd: 979 # # This means that for more details on how this section of the template was created # view line 45 of the zisraw/Instrument.xsd file and view line 979 of the ome/ome.xsd file.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->

<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

    <!-- Manufacturer -->
    <!-- zisraw/Instrument.xsd: 11 -->
    <!-- (referenced at zisraw/Instrument.xsd: 157) -->
    <!-- ome/ome.xsd: 1429 -->
    <xsl:template name="Manufacturer">
        <xsl:param name="manufacturer_data"/>

        <xsl:if test="$manufacturer_data/Manufacturer">
            <xsl:attribute name="Manufacturer">
                <xsl:value-of select="$manufacturer_data/Manufacturer"/>
            </xsl:attribute>
        </xsl:if>

        <xsl:if test="$manufacturer_data/Model">
            <xsl:attribute name="Model">
                <xsl:value-of select="$manufacturer_data/Model"/>
            </xsl:attribute>
        </xsl:if>

        <xsl:if test="$manufacturer_data/SerialNumber">
            <xsl:attribute name="SerialNumber">
                <xsl:value-of select="$manufacturer_data/SerialNumber"/>
            </xsl:attribute>
        </xsl:if>

        <xsl:if test="$manufacturer_data/LotNumber">
            <xsl:attribute name="LotNumber">
                <xsl:value-of select="$manufacturer_data/LotNumber"/>
            </xsl:attribute>
        </xsl:if>

    </xsl:template>

    <!-- Microscope -->
    <!-- zisraw/Instrument.xsd: 50 -->
    <!-- ome/ome.xsd: 1016 -->
    <xsl:template name="Microscope">
        <xsl:param name="microscope_data"/>
        <Microscope>

            <xsl:attribute name="Type">
                <xsl:value-of select="$microscope_data/Type"/>
            </xsl:attribute>

            <xsl:if test="$microscope_data/Manufacturer">
                <xsl:call-template name="Manufacturer">
                    <xsl:with-param name="manufacturer_data" select="$microscope_data/Manufacturer"/>
                </xsl:call-template>
            </xsl:if>

        </Microscope>
    </xsl:template>

    <!-- Instrument -->
    <!-- zisraw/Instrument.xsd: 45 -->
    <!-- ome/ome.xsd: 979 -->
    <xsl:template name="Instrument">
        <xsl:param name="instrument_data"/>
        <Instrument>

            <xsl:attribute name="ID">
                <xsl:choose>
                    <xsl:when test="@Id">
                        <xsl:value-of select="$instrument_data/@Id"/>
                    </xsl:when>
                    <xsl:otherwise>Instrument:0</xsl:otherwise>
                </xsl:choose>
            </xsl:attribute>

            <!-- Attach Microscope -->
            <xsl:call-template name="Microscope">
                <xsl:with-param name="microscope_data" select="$instrument_data/Microscopes/Microscope[1]"/>
            </xsl:call-template>

        </Instrument>
    </xsl:template>

</xsl:stylesheet>
