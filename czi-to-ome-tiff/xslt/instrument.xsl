<?xml version="1.0" encoding="UTF-8"?>
<!-- #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # # Comments in this template will generally be pointers from spec to spec # Example: # Get Instrument Info # zisraw/Instrument.xsd: 45 #
ome/ome.xsd: 979 # # This means that for more details on how this section of the template was created # view line 45 of the zisraw/Instrument.xsd file and view line 979 of the ome/ome.xsd file.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->

<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

    <!-- Microscope -->
    <!-- zisraw/Instrument.xsd: 50 -->
    <!-- ome/ome.xsd: 1016 -->
    <xsl:template match="Microscope">
        <Microscope>
            <xsl:apply-templates select="@Id"/>
            <xsl:apply-templates select="@Name"/>
            <xsl:apply-templates select="Type"/>
            <xsl:apply-templates select="Manufacturer"/>
        </Microscope>
    </xsl:template>

    <xsl:template match="Type" name="Instrument_type">
        <Type>
            <xsl:value-of select="."/>
        </Type>
    </xsl:template>

    <xsl:template match="@Name">
        <xsl:attribute name="Name"> <xsl:value-of select="."/> </xsl:attribute>
    </xsl:template>



    <xsl:template match="Model">
        <xsl:value-of select="."/>
    </xsl:template>


    <!-- Manufacturer -->
    <!-- zisraw/Instrument.xsd: 11 -->
    <!-- (referenced at zisraw/Instrument.xsd: 157) -->
    <!-- ome/ome.xsd: 1429 -->
    <xsl:template match="Manufacturer">

            <xsl:attribute name="Manufacturer">
                <xsl:value-of select="."/>
            </xsl:attribute>

            <xsl:apply-templates select="Model"/>

            <xsl:attribute name="SerialNumber">
                <xsl:value-of select="SerialNumber"/>
            </xsl:attribute>

            <xsl:attribute name="LotNumber">
                <xsl:value-of select="LotNumber"/>
            </xsl:attribute>

    </xsl:template>



    <!-- Instrument -->
    <!-- zisraw/Instrument.xsd: 45 -->
    <!-- ome/ome.xsd: 979 -->
    <xsl:template match="Instrument" name="Instrument">
        <Instrument>
            <Microscopes>
                <xsl:apply-templates select="Microscopes/Microscope"  />
            </Microscopes>
        </Instrument>
    </xsl:template>

    <xsl:template match="@Id">
        <xsl:attribute name="ID">
            <xsl:value-of select="."/>
        </xsl:attribute>
    </xsl:template>


</xsl:stylesheet>
