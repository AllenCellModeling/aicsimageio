from pathlib import Path

import lxml.etree as ET

###############################################################################

resources = Path("../resources").resolve(strict=True)
czixml = str((resources / "example-czi.xml").resolve(strict=True))
template = str(Path("czi-to-ome.xslt").resolve(strict=True))
output = Path("produced.ome.xml").resolve()

###############################################################################

# Parse template and generate transform function
template = ET.parse(template)
transform = ET.XSLT(template)

# Parse CZI XML
czixml = ET.parse(czixml)

# Run transform
ome = transform(czixml)

# Write file
with open(output, "w") as write_out:
    write_out.write(str(ome))
