# CZI to OME-TIFF Metadata Mapping

Using [XSLT](https://en.wikipedia.org/wiki/XSLT) to map CZI metadata schemas to OME-TIFF metadata specifications.

### Index
1. [Basics of XSLT](#basics)
2. [General Notes](#notes)
3. [Open Questions](#questions)

## Basics
In short: XSLT is used to transform XML to other formats, primarily, other XML formats or HTML.

_I would recommend watching the videos on 1.5x speed._

* [Simple XSLT Tutorial](https://www.youtube.com/watch?v=BujLy71JY1k)
_an eight minute video detailing how to convert xml to html_
* [XSLT Reference](https://developer.mozilla.org/en-US/docs/Web/XSLT)
_mozilla reference for all possible templating elements_
* [XSLT Deep Reference](https://developer.mozilla.org/en-US/docs/Web/XSLT/Transforming_XML_with_XSLT)
_mozilla reference for all possible attributes, elements, and functions_
* [A Bit More In Depth XSLT Tutorial](https://www.youtube.com/watch?v=Rn1bvTYYsCY)
_a thirty minute video with more details on nested for-each, etc_

## Testing
To test template changes run:

```bash
pip install -e .[dev]
cd czi-to-ome-tiff/xslt/
python transform.py
```

## Questions
* How will we handle schema version to schema version? CZI metadata schemas change over time and so does OME. On first
thought we could have templates for the most common CZI versions to the most recent OME. But, there also exist's `if`,
`choose`, `when`, and `otherwise` template tags in the `xslt` reference so we could just keep everything in one big
template file? Not my favourite option though.
