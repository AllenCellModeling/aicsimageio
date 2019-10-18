<?xml version="1.0" encoding="UTF-8"?>

<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

    <!-- PixelType -->
    <!-- zisraw/CommonTypes.xsd: 224 -->
    <!-- ome/ome.xsd: 1689 -->
    <xsl:template name="PixelType">
        <xsl:param name="pixel_data"/>
        <PixelType>
            <!-- Attempt to get to map the ZISRAW pixeltype to OME pixeltype -->
            <xsl:attribute name="ID">
                <xsl:choose>
                    <xsl:when test="$pixel_data==Gray8">
                        <xsl:text>uint8</xsl:text>
                    </xsl:when>
                    <xsl:when test="pixel_data==Gray16">
                        <xsl:text>uint16</xsl:text>
                    </xsl:when>
                    <xsl:when test="pixel_data==Gray32">
                        <!-- Zeiss Docs: planned-->
                        <xsl:text>float</xsl:text>
                    </xsl:when>
                    <xsl:when test="pixel_data==Gray64">
                        <!-- Zeiss Docs: planned-->
                        <xsl:text>double</xsl:text>
                    </xsl:when>
                    <xsl:when test="pixel_data==Bgr24">
                        <xsl:text>uint8</xsl:text>
                    </xsl:when>
                    <xsl:when test="pixel_data==Bgr48">
                        <xsl:text>uint16</xsl:text>
                    </xsl:when>
                    <xsl:when test="pixel_data==Gray32Float">
                        <!-- float, specifically an IEEE 4 byte float-->
                        <xsl:text>float</xsl:text>
                    </xsl:when>
                    <xsl:when test="pixel_data==Bgr96Float">
                        <!-- float, specifically an IEEE 4 byte float-->
                        <xsl:text>float</xsl:text>
                    </xsl:when>
                    <xsl:when test="pixel_data==Gray64ComplexFloat">
                        <!-- 2 x float, specifically an IEEE 4 byte float-->
                        <xsl:text>complex</xsl:text>
                    </xsl:when>
                    <xsl:when test="pixel_data==Bgr192ComplexFloat">
                        <!-- a BGR triplet of (2 x float), specifically an IEEE 4 byte float-->
                        <xsl:text>complex</xsl:text>
                    </xsl:when>
                    <xsl:when test="pixel_data==Bgra32">
                        <!-- Bgra32 = 3 uint8 followed by a 8 bit transparency value-->
                        <!-- From other sources (non-Zeiss) the a value is a uint8-->
                        <xsl:text>uint8</xsl:text>
                    </xsl:when>
                </xsl:choose>
            </xsl:attribute>
        </PixelType>
    </xsl:template>
</xsl:stylesheet>
