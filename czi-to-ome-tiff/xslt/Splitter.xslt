<xsl:stylesheet version="2.0"
                xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

    <xsl:output method="xml" />
    <xsl:template match="/">
        <xsl:copy-of select="*[not(self::TiffData)]"/>
    </xsl:template>

    <xsl:template match="//TiffData">
    </xsl:template>

    <xsl:template match="../TiffData">
    </xsl:template>


</xsl:stylesheet>