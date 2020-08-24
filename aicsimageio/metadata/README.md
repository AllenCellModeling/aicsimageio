# AICSImage Metadata Module

Undecided how to organize this module. This likely isn't included in 4.0 but more
likely 4.1 or similar but good to start thinking about.

I am currently thinking:
```
aicsimageio/
    metadata/
        omexml.py  # our remade and updated omexml object generated from source XSD
        transform.py  # the transform functions to convert to ome
        xslt/
            czi.xslt  # from czi to ome
            3i.xslt  # from 3i to ome
            ...
```

This results in:
```python
from aicsimageio.metadata import OMEXML, transform_czi_to_ome
```
