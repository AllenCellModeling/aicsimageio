from pathlib import Path

import lxml.etree as ET

###############################################################################

resources = Path("../resources").resolve(strict=True)
czixml = str((resources / "example-czi.xml").resolve(strict=True))
template = str(Path("czi-to-ome.xsl").resolve(strict=True))
output = Path("produced.ome.xml").resolve()

###############################################################################

# Parse template and generate transform function
template = ET.parse(template)
transform = ET.XSLT(template)

# Parse CZI XML
czixml = ET.parse(czixml)

# Run transform
try:
    ome = transform(czixml)
except Exception as e:
    print(f"Error: {e}")
    print("-" * 80)
    print("Full Log:")
    for entry in transform.error_log:
        print(f"<{entry.filename}: {entry.line}, {entry.column}> {entry.message}")

# Write file
with open(output, "w") as write_out:
    write_out.write(str(ome))
