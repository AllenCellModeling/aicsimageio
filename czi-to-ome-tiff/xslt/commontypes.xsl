<?xml version="1.0" encoding="UTF-8"?>

<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

    <!-- Includes -->
    <xsl:include href="constants.xsl"/>

    <!-- Name -->
    <!-- zisraw/Instrument.xsd: Undocumented -->
    <!-- ome/ome.xsd: 2059 -->
    <xsl:template match="@Name">
        <xsl:attribute name="Name">
            <xsl:value-of select="."/>
        </xsl:attribute>
    </xsl:template>

    <!-- Generalizable ID generator -->
    <xsl:template match="@Id">
        <xsl:attribute name="ID">
            <xsl:text>urn:lsid:</xsl:text>
            <xsl:value-of select="$AUTHORITY"/>
            <xsl:text>:</xsl:text>
            <xsl:value-of select="name(parent::*)"/>
            <xsl:text>:</xsl:text>
            <xsl:value-of select="position()"/>
        </xsl:attribute>
    </xsl:template>



</xsl:stylesheet>
