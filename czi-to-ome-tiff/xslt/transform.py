from pathlib import Path

import lxml.etree as ET

###############################################################################

resources = Path("../resources").resolve(strict=True)
omexsd = str(Path("../ome/ome.xsd").resolve(strict=True))
czixml = str((resources / "example-czi.xml").resolve(strict=True))
template = str(Path("czi-to-ome.xsl").resolve(strict=True))
output = Path("produced.ome.xml").resolve()

###############################################################################

# Parse template and generate transform function
template = ET.parse(template)
transform = ET.XSLT(template)

# Parse CZI XML
czixml = ET.parse(czixml)

# Parse OME XSD
omexsd = ET.parse(omexsd)
omexsd = ET.XMLSchema(omexsd)

# Attempt to run transform
try:
    ome = transform(czixml)

    # Write file
    with open(output, "w") as write_out:
        write_out.write(str(ome))

    # Validate file
    passing = omexsd.validate(ome)

    if passing:
        print(f"Produced XML passes OME XSD: {passing}")
    else:
        raise ValueError(f"Produced XML fails validation")


# Catch any exception
except Exception as e:
    print(f"Error: {e}")
    print("-" * 80)
    print("Full Log:")
    for entry in transform.error_log:
        print(f"{entry.filename}: {entry.line}, {entry.column}> {entry.message}>")
