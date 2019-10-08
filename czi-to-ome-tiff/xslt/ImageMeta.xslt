<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="3.0"
xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

<xsl:template match="/ImageDocument/MetaData/Information/Instrument/Microscopes/Microscope">
  <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">
    <xsl:template match="/ImageDocument/MetaData/Information/Instrument/Microscopes/Microscope">
    <Instrument ID="@ID">
      <Microscope Model="@Name" Type=""/>
    </Instrument>
      <h2>My CD Collection</h2>
      <table border="1">
        <tr bgcolor="#9acd32">
          <th>Title</th>
          <th>Artist</th>
        </tr>
        <tr>
          <td>
            <xsl:value-of select="catalog/cd/title"/>
          </td>
          <td>
            <xsl:value-of select="catalog/cd/artist"/>
          </td>
        </tr>
      </table>
    </Instrument>
  </OME>
</xsl:template>

</xsl:stylesheet>

