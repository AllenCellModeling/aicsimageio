<?xml version="1.0" encoding="UTF-8"?>
<!-- #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # # Comments in this template will generally be pointers from spec to spec # Example: # Get Instrument Info # zisraw/Instrument.xsd: 45 #
ome/ome.xsd: 979 # # This means that for more details on how this section of the template was created # view line 45 of the zisraw/Instrument.xsd file and view line 979 of the ome/ome.xsd file.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->

<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

    <!-- Includes -->
    <xsl:include href="commontypes.xsl"/>

    <!-- ManufacturerSpec/Model -->
    <!-- zisraw/Instrument.xsd: 26 -->
    <!-- ome/ome.xsd: 6389 -->
    <xsl:template match="Model">
        <xsl:attribute name="Model">
            <xsl:value-of select="."/>
        </xsl:attribute>
    </xsl:template>

    <!-- ManufacturerSpec/SerialNumber -->
    <!-- zisraw/Instrument.xsd: 31 -->
    <!-- ome/ome.xsd: 6395 -->
    <xsl:template match="SerialNumber">
        <xsl:attribute name="SerialNumber">
            <xsl:value-of select="."/>
        </xsl:attribute>
    </xsl:template>

    <!-- ManufacturerSpec/LotNumber -->
    <!-- zisraw/Instrument.xsd: 36 -->
    <!-- ome/ome.xsd: 6401 -->
    <xsl:template match="LotNumber">
        <xsl:attribute name="LotNumber">
            <xsl:value-of select="."/>
        </xsl:attribute>
    </xsl:template>

    <!-- ManufacturerSpec/SpecsFile -->
    <!-- zisraw: No valid zisraw spec -->
    <!-- ome/ome.xsd: 6407 -->
    <!-- Note: This is required by the OME-4DN spec but not provided by ZISRAW -->
    <xsl:template match="SpecsFile">
        <xsl:attribute name="SpecsFile">
            <xsl:value-of select="."/>
        </xsl:attribute>
    </xsl:template>

    <!-- Manufacturer -->
    <!-- zisraw/Instrument.xsd: 11 -->
    <!-- ome/ome.xsd: 6378 -->
    <xsl:template match="Manufacturer">
        <xsl:attribute name="Manufacturer">
            <xsl:value-of select="."/>
        </xsl:attribute>

        <xsl:apply-templates select="Model"/>
        <xsl:apply-templates select="SerialNumber"/>
        <xsl:apply-templates select="LotNumber"/>
        <xsl:apply-templates select="SpecsFile"/>
    </xsl:template>

    <!-- Type -->
    <!-- zisraw/Instrument.xsd: 166 -->
    <!-- ome/ome.xsd: 7996 -->
    <xsl:template match="Type">
        <xsl:attribute name="Type">
            <xsl:value-of select="."/>
        </xsl:attribute>
    </xsl:template>

    <!-- Microscope -->
    <!-- zisraw/Instrument.xsd: 50 -->
    <!-- ome/ome.xsd: 2039 -->
    <xsl:template match="Microscope">
        <MicroscopeBody>

            <xsl:apply-templates select="@Id"/>
            <xsl:apply-templates select="@Name"/>
            <xsl:apply-templates select="Type"/>
            <xsl:apply-templates select="Manufacturer"/>

        </MicroscopeBody>
    </xsl:template>

    <!-- Instrument -->
    <!-- zisraw/Instrument.xsd: 45 -->
    <!-- ome/ome.xsd: 1235 -->
    <xsl:template match="Instrument">
        <Instrument>
            <xsl:apply-templates select="@Id"/>
            <xsl:apply-templates select="@Name"/>

            <!-- Plural pulled from ome/ome.xsd: 2042 -->
            <Microscopes>
                <xsl:apply-templates select="Microscopes"/>
            </Microscopes>
        </Instrument>
    </xsl:template>

</xsl:stylesheet>
