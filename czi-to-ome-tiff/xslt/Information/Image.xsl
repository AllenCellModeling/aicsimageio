<?xml version="1.0" encoding="UTF-8"?>

<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xmlns:xs="http://www.w3.org/2001/XMLSchema">

        <!-- Microscope -->
    <!-- zisraw/Information.xsd: 34 -->
    <!-- ome/ome.xsd: 209 -->

    <xsl:template match="Image">
        <Image>

            <xsl:attribute name="ID">
                <xsl:text>urn:lsid:allencell.org:Image:</xsl:text>
                <xsl:value-of select="position()" />
            </xsl:attribute>
            <xsl:attribute name="Name">
                <!-- get the filename from the first line in the file-->
            </xsl:attribute>
            <xsl:apply-templates select="AcquisitionDateAndTime" />

        </Image>
    </xsl:template>

    <xsl:template match="SizeX">
        <!-- what to map it to -->
    </xsl:template>

    <xsl:template match="SizeY">
        <!-- what to map it to -->
    </xsl:template>

    <xsl:template match="SizeC">
        <!-- what to map it to -->
    </xsl:template>

    <xsl:template match="SizeZ">
        <!-- what to map it to -->
    </xsl:template>

    <xsl:template match="SizeT">
        <!-- what to map it to -->
    </xsl:template>

    <xsl:template match="SizeH">
        <!-- what to map it to -->
    </xsl:template>

    <xsl:template match="SizeR">
        <!-- what to map it to -->
    </xsl:template>

    <xsl:template match="SizeV">
        <!-- what to map it to -->
    </xsl:template>

    <xsl:template match="SizeS">
        <!-- what to map it to -->
    </xsl:template>

    <xsl:template match="SizeI">
        <!-- what to map it to -->
    </xsl:template>

    <xsl:template match="SizeM">
        <!-- what to map it to -->
    </xsl:template>

    <xsl:template match="SizeB">
        <!-- what to map it to -->
    </xsl:template>

    <xsl:template match="AcquisitionDateAndTime">
        <AcquisitionDate>
            <xsl:value-of select="."/>
        </AcquisitionDate>
    </xsl:template>


</xsl:stylesheet>
