from pathlib import Path
import saxonc
import lxml.etree as ET

###############################################################################

resources = Path("/Users/jamies/Sandbox/Python/aicsimageio/czi-to-ome-tiff/resources").resolve(strict=True)
xslt_path = Path("/Users/jamies/Sandbox/Python/aicsimageio/czi-to-ome-tiff/xslt").resolve(strict=True)
czixml = str((resources / "example-czi.xml").resolve(strict=True))
template = str((Path("czi-to-ome.xsl")).resolve(strict=True))
output = Path("produced.ome.xml").resolve()

###############################################################################

# Parse template and generate transform function
template = ET.parse(template)
transform = ET.XSLT(template)

# Parse CZI XML
czixml = ET.parse(czixml)

# Attempt to run transform
try:

#     with saxonc.PySaxonProcessor(license=False) as proc:
#         xdmAtomicval = proc.make_boolean_value(False)
#         xslt_process = proc.new_xslt30_processor()
#         #xslt_process.set_source(czixml)
#         ome = xslt_process.transform_to_string(source_file=czixml, stylesheet_file=template)

    ome = transform(czixml)

    # Write file
    with open(output, "w") as write_out:
        write_out.write(str(ome))

# Catch any exception
except Exception as e:
    print(f"Error: {e}")
    print("-" * 80)
    print("Full Log:")
    for entry in transform.error_log:
        print(f"{entry.filename}: {entry.line}, {entry.column}> {entry.message}>")
