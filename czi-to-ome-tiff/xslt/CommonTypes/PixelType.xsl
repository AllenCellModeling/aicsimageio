<?xml version="1.0" encoding="UTF-8"?>

<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

    <!-- PixelType -->
    <!-- zisraw/CommonTypes.xsd: 224 -->
    <!-- ome/ome.xsd: 1689 -->

    <xsl:template match="PixelType[text() = 'Gray8'] | PixelType[text() = 'Bgr24'] | PixelType[text() = 'Brga32']">
        <PixelType>
            <xsl:text>uint8</xsl:text>
        </PixelType>
    </xsl:template>

    <xsl:template match="PixelType[text() = 'Gray16'] | PixelType[text() = 'Bgr48']">
        <PixelType>
            <xsl:text>uint16</xsl:text>
        </PixelType>
    </xsl:template>

    <xsl:template match="PixelType[text() = 'Gray32'] | PixelType[text() = 'Gray32Float'] | PixelType[text() = 'Bgr96Float']">
        <PixelType>
            <xsl:text>float</xsl:text>
        </PixelType>
    </xsl:template>

    <xsl:template match="PixelType[text() = 'Gray64']">
        <PixelType>
            <xsl:text>double</xsl:text>
        </PixelType>
    </xsl:template>

    <xsl:template match="PixelType[text() = 'Gray64ComplexFloat'] | PixelType[text() = 'Bgr192ComplexFloat']">
        <PixelType>
            <xsl:text>complex</xsl:text>
        </PixelType>
    </xsl:template>

</xsl:stylesheet>
