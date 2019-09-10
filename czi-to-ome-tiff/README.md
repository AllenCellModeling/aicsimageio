# CZI to OME-TIFF Metadata Mapping

Using [XSLT](https://en.wikipedia.org/wiki/XSLT) to map CZI metadata schemas to OME-TIFF metadata specifications.

### Index
1. [Basics of XSLT](#basics)
2. [General Notes](#notes)
3. [Open Questions](#questions)

## Basics
In short: XSLT is used to transform XML to other formats, primarily, other XML formats or HTML. We are fortunate that
Python has a great library to do this transformation, [`lxml`](https://lxml.de). An example of `lxml` being used to
transform one XML document into another can be found [here](https://stackoverflow.com/questions/16698935/how-to-transform-an-xml-file-using-xslt-in-python#answer-16699042).

Simple enough right? Well that is assuming we already have the template file written. Below are a bunch of links that
are relevant to writing an XSLT file:

_I would recommend watching the videos on 1.5x speed._

* [Simple XSLT Tutorial](https://www.youtube.com/watch?v=BujLy71JY1k)
_an eight minute video detailing how to convert xml to html_
* [XSLT Reference](https://www.w3schools.com/xml/xsl_elementref.asp)
_w3schools reference for all possible templating elements_
* [A Bit More In Depth XSLT Tutorial](https://www.youtube.com/watch?v=Rn1bvTYYsCY)
_a thirty minute video with more details on nested for-each, etc_


## Notes
To be added to as we write templates.

## Questions
* How will we handle schema version to schema version? CZI metadata schemas change over time and so does OME. On first
thought we could have templates for the most common CZI versions to the most recent OME. But, there also exist's `if`,
`choose`, `when`, and `otherwise` template tags in the `xslt` reference so we could just keep everything in one big
template file? Not my favourite option though.
