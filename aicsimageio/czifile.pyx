# -*- coding: utf-8 -*-
# czifile.pyx

# Copyright (c) 2013-2017, Christoph Gohlke
# Copyright (c) 2013-2017, The Regents of the University of California
# Produced at the Laboratory for Fluorescence Dynamics.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Decode JpegXrFile and JpgFile images in Carl Zeiss(r) ZISRAW (CZI) files.

:Author:
  `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics, University of California, Irvine

:Version: 2017.07.20

Requirements
------------
* `CPython 3.6 <http://www.python.org>`_
* `Numpy 1.13 <http://www.numpy.org>`_
* `Cython 0.25 <http://cython.org/>`_  (build)
* `jpeglib v9 <http://www.ijg.org/>`_  (build)
* `jxrlib 0.2.0 <https://github.com/glencoesoftware/jxrlib/>`_  (build)
* `jxrlib patch <https://www.lfd.uci.edu/~gohlke/code/
   jxrlib_CreateDecoderFromBytes.diff.html>`_  (build)
* A Python distutils compatible C compiler  (build)

Install
-------
Use this Cython distutils setup script to build the extension module::

  # setup.py
  # Usage: ``python setup.py build_ext --inplace``
  import sys
  import os
  from distutils.core import setup, Extension
  from Cython.Distutils import build_ext
  jxrlib_dir = 'jxrlib'  # directory where jxrlib was built
  jpeg_dir = 'jpeg-9'
  win32 = sys.platform == 'win32'
  include_dirs = [jpeg_dir]
  include_dirs += [os.path.join(jxrlib_dir, *d.split('/'))
                   for d in ('jxrgluelib', 'common/include', 'image/sys')]
  define_macros = [('WIN32', None)] if win32 else [('INITGUID', None)]
  ext = Extension('_czifile', sources=['_czifile.pyx'],
                  include_dirs=include_dirs, define_macros=define_macros,
                  library_dirs=[jxrlib_dir, jpeg_dir],
                  libraries=['jxrlib' if win32 else 'libjpegxr',
                             'jpeg'],)
  setup(name='_czifile', cmdclass={'build_ext': build_ext}, ext_modules=[ext])

"""

__version__ = "2017.07.20"

cimport cython
from cython.operator cimport dereference as deref
from libc.setjmp cimport setjmp, longjmp, jmp_buf
cimport numpy
import numpy

# jxrlib

cdef extern from "windowsmediaphoto.h":
    int WMP_errSuccess
    int WMP_errFail
    int WMP_errNotYetImplemented
    int WMP_errAbstractMethod
    int WMP_errOutOfMemory
    int WMP_errFileIO
    int WMP_errBufferOverflow
    int WMP_errInvalidParameter
    int WMP_errInvalidArgument
    int WMP_errUnsupportedFormat
    int WMP_errIncorrectCodecVersion
    int WMP_errIndexNotFound
    int WMP_errOutOfSequence
    int WMP_errNotInitialized
    int WMP_errMustBeMultipleOf16LinesUntilLastCall
    int WMP_errPlanarAlphaBandedEncRequiresTempFile
    int WMP_errAlphaModeCannotBeTranscoded
    int WMP_errIncorrectCodecSubVersion

    ctypedef long ERR
    ctypedef int I32
    ctypedef int PixelI
    ctypedef unsigned char U8
    ctypedef unsigned int U32


cdef extern from "guiddef.h":
    ctypedef struct GUID:
        pass

    int IsEqualGUID(GUID*, GUID*)


cdef extern from "JXRGlue.h":
    ctypedef GUID PKPixelFormatGUID

    ctypedef struct PKFactory:
        pass
    ctypedef struct PKCodecFactory:
        pass
    ctypedef struct PKImageDecode:
        pass
    ctypedef struct PKImageEncode:
        pass
    ctypedef struct PKFormatConverter:
        pass
    ctypedef struct PKRect:
        I32 X, Y, Width, Height

    cdef ERR PKCreateCodecFactory(PKCodecFactory**, U32) nogil
    cdef ERR PKCreateCodecFactory_Release(PKCodecFactory**) nogil
    cdef ERR PKCodecFactory_CreateFormatConverter(PKFormatConverter**) nogil
    cdef ERR PKCodecFactory_CreateDecoderFromBytes(void*, size_t,
                                                   PKImageDecode**) nogil
    cdef ERR PKImageDecode_GetSize(PKImageDecode*, I32*, I32*) nogil
    cdef ERR PKImageDecode_Release(PKImageDecode**) nogil
    cdef ERR PKImageDecode_GetPixelFormat(PKImageDecode*,
                                          PKPixelFormatGUID*) nogil
    cdef ERR PKFormatConverter_Release(PKFormatConverter**) nogil
    cdef ERR PKFormatConverter_Initialize(PKFormatConverter*, PKImageDecode*,
                                          char*, PKPixelFormatGUID) nogil
    cdef ERR PKFormatConverter_Copy(PKFormatConverter*, const PKRect*,
                                    U8*, U32) nogil
    cdef ERR PKFormatConverter_Convert(PKFormatConverter*, const PKRect*,
                                       U8*, U32) nogil

    GUID GUID_PKPixelFormat8bppGray
    GUID GUID_PKPixelFormat16bppGray
    GUID GUID_PKPixelFormat32bppGrayFloat
    GUID GUID_PKPixelFormat24bppBGR
    GUID GUID_PKPixelFormat24bppRGB
    GUID GUID_PKPixelFormat48bppRGB
    GUID GUID_PKPixelFormat128bppRGBFloat
    GUID GUID_PKPixelFormat32bppRGBA
    GUID GUID_PKPixelFormat32bppBGRA
    GUID GUID_PKPixelFormat64bppRGBA
    GUID GUID_PKPixelFormat128bppRGBAFloat


WMP_ERR = {
    WMP_errFail: "Fail",
    WMP_errNotYetImplemented: "NotYetImplemented",
    WMP_errAbstractMethod: "AbstractMethod",
    WMP_errOutOfMemory: "OutOfMemory",
    WMP_errFileIO: "FileIO",
    WMP_errBufferOverflow: "BufferOverflow",
    WMP_errInvalidParameter: "InvalidParameter",
    WMP_errInvalidArgument: "InvalidArgument",
    WMP_errUnsupportedFormat: "UnsupportedFormat",
    WMP_errIncorrectCodecVersion: "IncorrectCodecVersion",
    WMP_errIndexNotFound: "IndexNotFound",
    WMP_errOutOfSequence: "OutOfSequence",
    WMP_errNotInitialized: "NotInitialized",
    WMP_errAlphaModeCannotBeTranscoded: "AlphaModeCannotBeTranscoded",
    WMP_errIncorrectCodecSubVersion: "IncorrectCodecSubVersion",
    WMP_errMustBeMultipleOf16LinesUntilLastCall:
        "MustBeMultipleOf16LinesUntilLastCall",
    WMP_errPlanarAlphaBandedEncRequiresTempFile:
        "PlanarAlphaBandedEncRequiresTempFile",
    }


class WmpError(Exception):
    def __init__(self, msg, err):
        msg = "%s failed with %s error" % (msg, WMP_ERR.get(err, "Unknown"))
        Exception.__init__(self, msg)


def decode_jxr(data):
    """Return image data from JXR bytes as numpy array."""
    cdef unsigned char* cdata = data
    cdef numpy.ndarray out
    cdef PKImageDecode* decoder = NULL
    cdef PKFormatConverter* converter = NULL
    cdef PKPixelFormatGUID pixel_format
    cdef PKRect rect
    cdef I32 width
    cdef I32 height
    cdef U32 stride
    cdef ERR err

    try:
        err = PKCodecFactory_CreateDecoderFromBytes(cdata, len(data), &decoder)
        if err:
            raise WmpError("PKCodecFactory_CreateDecoderFromBytes", err)

        err = PKImageDecode_GetSize(decoder, &width, &height)
        if err:
            raise WmpError("PKImageDecode_GetSize", err)

        err = PKImageDecode_GetPixelFormat(decoder, &pixel_format)
        if err:
            raise WmpError("PKImageDecode_GetPixelFormat", err)

        if IsEqualGUID(&pixel_format, &GUID_PKPixelFormat8bppGray):
            dtype = numpy.uint8
            samples = 1
        elif IsEqualGUID(&pixel_format, &GUID_PKPixelFormat16bppGray):
            dtype = numpy.uint16
            samples = 1
        elif IsEqualGUID(&pixel_format, &GUID_PKPixelFormat32bppGrayFloat):
            dtype = numpy.float32
            samples = 1
        elif IsEqualGUID(&pixel_format, &GUID_PKPixelFormat24bppBGR):
            dtype = numpy.uint8
            samples = 3
            pixel_format = GUID_PKPixelFormat24bppRGB
        elif IsEqualGUID(&pixel_format, &GUID_PKPixelFormat24bppRGB):
            dtype = numpy.uint8
            samples = 3
        elif IsEqualGUID(&pixel_format, &GUID_PKPixelFormat48bppRGB):
            dtype = numpy.uint16
            samples = 3
        elif IsEqualGUID(&pixel_format, &GUID_PKPixelFormat128bppRGBFloat):
            dtype = numpy.float32
            samples = 3
        elif IsEqualGUID(&pixel_format, &GUID_PKPixelFormat32bppBGRA):
            dtype = numpy.uint8
            samples = 4
            pixel_format = GUID_PKPixelFormat32bppRGBA
        elif IsEqualGUID(&pixel_format, &GUID_PKPixelFormat32bppRGBA):
            dtype = numpy.uint8
            samples = 4
        elif IsEqualGUID(&pixel_format, &GUID_PKPixelFormat64bppRGBA):
            dtype = numpy.uint16
            samples = 4
        elif IsEqualGUID(&pixel_format, &GUID_PKPixelFormat128bppRGBAFloat):
            dtype = numpy.float32
            samples = 4
        else:
            raise ValueError("unknown pixel format")

        err = PKCodecFactory_CreateFormatConverter(&converter)
        if err:
            raise WmpError("PKCodecFactory_CreateFormatConverter", err)

        err = PKFormatConverter_Initialize(converter, decoder, NULL,
                                           pixel_format)
        if err:
            raise WmpError("PKFormatConverter_Initialize", err)

        shape = height, width
        if samples > 1:
            shape += samples,
        out = numpy.empty(shape, dtype)

        stride = out.strides[0]
        rect.X = 0
        rect.Y = 0
        rect.Width = out.shape[1]
        rect.Height = out.shape[0]

        # TODO: check alignment issues
        with nogil:
            err = PKFormatConverter_Copy(converter, &rect, <U8*>out.data,
                                         stride)
        if err:
            raise WmpError("PKFormatConverter_Copy", err)

        return out

    finally:
        if converter != NULL:
            PKFormatConverter_Release(&converter)
        if decoder != NULL:
            PKImageDecode_Release(&decoder)

# jpeglib

cdef extern from "jpeglib.h":
    ctypedef void noreturn_t
    ctypedef int boolean
    ctypedef unsigned int JDIMENSION
    ctypedef unsigned char JSAMPLE
    ctypedef JSAMPLE* JSAMPROW
    ctypedef JSAMPROW* JSAMPARRAY

    cdef enum J_COLOR_SPACE:
        JCS_UNKNOWN,
        JCS_GRAYSCALE,
        JCS_RGB,
        JCS_YCbCr,
        JCS_CMYK,
        JCS_YCCK

    cdef enum J_DITHER_MODE:
        JDITHER_NONE,
        JDITHER_ORDERED,
        JDITHER_FS

    cdef enum J_DCT_METHOD:
        JDCT_ISLOW,
        JDCT_IFAST,
        JDCT_FLOAT

    cdef struct jpeg_source_mgr:
        pass

    cdef struct jpeg_common_struct:
        jpeg_error_mgr* err

    cdef struct jpeg_error_mgr:
        int msg_code
        char** jpeg_message_table
        noreturn_t error_exit(jpeg_common_struct*)
        void output_message(jpeg_common_struct*)

    cdef struct jpeg_decompress_struct:
        jpeg_error_mgr* err
        void* client_data
        jpeg_source_mgr* src
        JDIMENSION image_width
        JDIMENSION image_height
        JDIMENSION output_width
        JDIMENSION output_height
        JDIMENSION output_scanline
        J_COLOR_SPACE jpeg_color_space
        J_COLOR_SPACE out_color_space
        J_DCT_METHOD dct_method
        J_DITHER_MODE dither_mode
        boolean buffered_image
        boolean raw_data_out
        boolean do_fancy_upsampling
        boolean do_block_smoothing
        boolean quantize_colors
        boolean two_pass_quantize
        unsigned int scale_num
        unsigned int scale_denom
        int num_components
        int out_color_components
        int output_components
        int rec_outbuf_height
        int desired_number_of_colors
        int actual_number_of_colors
        int data_precision
        double output_gamma

    cdef jpeg_error_mgr* jpeg_std_error(jpeg_error_mgr*) nogil
    cdef void jpeg_create_decompress(jpeg_decompress_struct*) nogil
    cdef void jpeg_destroy_decompress(jpeg_decompress_struct*) nogil
    cdef void jpeg_mem_src(jpeg_decompress_struct*,
                           unsigned char*, unsigned long) nogil
    cdef int jpeg_read_header(jpeg_decompress_struct*, boolean) nogil
    cdef boolean jpeg_start_decompress(jpeg_decompress_struct*) nogil
    cdef boolean jpeg_finish_decompress (jpeg_decompress_struct*) nogil
    cdef JDIMENSION jpeg_read_scanlines(jpeg_decompress_struct*,
                                        JSAMPARRAY, JDIMENSION) nogil


ctypedef struct my_error_mgr:
    jpeg_error_mgr pub
    jmp_buf setjmp_buffer


cdef void my_error_exit(jpeg_common_struct* cinfo):
    cdef my_error_mgr* error = <my_error_mgr*> deref(cinfo).err
    longjmp(deref(error).setjmp_buffer, 1)


cdef void my_output_message(jpeg_common_struct* cinfo):
    pass


class JpgError(Exception):
    pass


def decode_jpeg(data, tables=b''):
    """Return image data from in memory JPG file as numpy array."""
    cdef numpy.ndarray out
    cdef int width
    cdef int height
    cdef int numsamples
    cdef my_error_mgr err
    cdef jpeg_decompress_struct cinfo
    cdef JSAMPROW samples
    cdef unsigned char* cdata = data
    cdef unsigned char* ctables = tables

    cinfo.err = jpeg_std_error(&err.pub)
    err.pub.error_exit = my_error_exit
    err.pub.output_message = my_output_message
    if setjmp(err.setjmp_buffer):
        jpeg_destroy_decompress(&cinfo)
        raise JpgError(err.pub.jpeg_message_table[err.pub.msg_code])

    jpeg_create_decompress(&cinfo)
    cinfo.do_fancy_upsampling = True

    if len(tables) > 0:
        jpeg_mem_src(&cinfo, ctables, len(tables))
        jpeg_read_header(&cinfo, 0)

    jpeg_mem_src(&cinfo, cdata, len(data))
    jpeg_read_header(&cinfo, 1)

    shape = cinfo.image_height, cinfo.image_width
    if cinfo.num_components > 1:
        shape += cinfo.num_components,
    out = numpy.empty(shape, numpy.uint8)

    with nogil:
        jpeg_start_decompress(&cinfo)

        samples = <JSAMPROW>out.data
        while cinfo.output_scanline < cinfo.output_height:
            numsamples = jpeg_read_scanlines(&cinfo, <JSAMPARRAY> &samples, 1)
            samples += numsamples * cinfo.image_width * cinfo.num_components

        jpeg_finish_decompress(&cinfo)
        jpeg_destroy_decompress(&cinfo)

    return out
