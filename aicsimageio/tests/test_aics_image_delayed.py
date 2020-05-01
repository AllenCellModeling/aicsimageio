#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from dask.diagnostics import Profiler
from psutil import Process

from aicsimageio import dask_utils, imread

# Example files
BMP_FILE = "example.bmp"
JPG_FILE = "example.jpg"
PNG_FILE = "example.png"
GIF_FILE = "example.gif"
TIF_FILE = "s_1_t_1_c_1_z_1.tiff"
CZI_FILE = "s_1_t_1_c_1_z_1.czi"
LIF_FILE = "s_1_t_1_c_2_z_1.lif"
OME_FILE = "s_1_t_1_c_1_z_1.ome.tiff"
MED_TIF_FILE = "s_1_t_10_c_3_z_1.tiff"
BIG_OME_FILE = "s_3_t_1_c_3_z_5.ome.tiff"
BIG_CZI_FILE = "s_3_t_1_c_3_z_5.czi"
BIG_LIF_FILE = "s_1_t_4_c_2_z_1.lif"
TXT_FILE = "example.txt"


with dask_utils.cluster_and_client(processes=False) as (cluster, client):

    @pytest.mark.parametrize(
        "filename, expected_shape, expected_tasks",
        [
            (PNG_FILE, (1, 1, 4, 1, 800, 537), 2),
            (TIF_FILE, (1, 1, 1, 1, 325, 475), 2),
            (OME_FILE, (1, 1, 1, 1, 325, 475), 2),
            (CZI_FILE, (1, 1, 1, 1, 325, 475), 2),
            (LIF_FILE, (1, 1, 2, 1, 2048, 2048), 4),
            (MED_TIF_FILE, (1, 10, 3, 1, 325, 475), 60),
            (BIG_OME_FILE, (3, 1, 3, 5, 325, 475), 90),
            (BIG_CZI_FILE, (3, 1, 3, 5, 325, 475), 18),
            (BIG_LIF_FILE, (1, 4, 2, 1, 614, 614), 16),
        ],
    )
    def test_imread_from_delayed(
        resources_dir, filename, expected_shape, expected_tasks
    ):
        # Get filepath
        f = resources_dir / filename

        # Check that there are no open file pointers after init
        proc = Process()
        assert str(f) not in [f.path for f in proc.open_files()]

        # Load image as delayed dask array then as numpy array
        # Check computed task count
        with Profiler() as prof:
            img = imread(f)
            assert img.shape == expected_shape
            assert len(prof.results) == expected_tasks

        # Check that there are no open file pointers after basics
        assert str(f) not in [f.path for f in proc.open_files()]
