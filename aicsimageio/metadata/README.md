# AICSImage Metadata Module

Undecided how to organize this module. This likely isn't included in 4.0 but more
likely 4.1 or similar but good to start thinking about.

I am currently thinking:
```
aicsimageio/
    metadata/
        ome/
            omexml.py  # our remade and updated omexml object generated from source XSD
            transform.py  # the transform functions to convert to ome
            xslt/
                czi.xslt  # from czi to ome
                3i.xslt  # from 3i to ome
                ...
        {some_other_format}/
            some_other_format.py  # our python equivalent object of their metadata model
            transform.py  # the transform functions to convert to their model
            xslt/
                ome.xslt  # from ome to some format
                czi.xslt  # from czi to some format
```

This results in:
```python
from aicsimageio.metadata.ome import OMEXML, transform_czi_to_ome
from aicsimageio.metadata.czi import CZIXML, transform_ome_to_czi
```

and etc.
