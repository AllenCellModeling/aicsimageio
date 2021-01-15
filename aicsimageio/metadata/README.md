# AICSImage Metadata Module

Planned layout of the `metadata` submodule

```
aicsimageio/
    metadata/
        utils.py  # common utilities and cleaning
        transform.py  # the transform functions to convert to ome
        xslt/
            czi.xslt  # from czi to ome
            3i.xslt  # from 3i to ome
            ...
```

This results in:

```python
from aicsimageio.metadata import transform_czi_to_ome
from aicsimageio.metadata import utils as metadata_utils
```
