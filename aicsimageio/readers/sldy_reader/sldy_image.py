#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import yaml
from fsspec.spec import AbstractFileSystem

from ... import types
from .sldy_image import SldyImage

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class SldyImage:
    @staticmethod
    def yaml_mapping(loader, node, deep=False) -> dict:
        mapping = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            value = loader.construct_object(value_node, deep=deep)
            if key == "StartClass":
                key = value["ClassName"]

            if key in mapping:
                if not isinstance(mapping[key], list):
                    mapping[key] = [mapping[key]]

                mapping[key].append(value)
            else:
                mapping[key] = value

        return mapping

    @staticmethod
    def _get_yaml_contents(
        fs: AbstractFileSystem, yaml_path: Path, reraise_on_error=True
    ) -> Optional[dict]:
        try:
            with fs.open(yaml_path) as f:
                return yaml.load(f, Loader=yaml.Loader)
        except FileNotFoundError:
            if reraise_on_error:
                raise

            log.debug(f"Unable to load metadata file {yaml_path}, ignoring")
            return None

    @staticmethod
    def _get_dim_to_data_path_map(
        data_paths: Set[Path], dim_prefix: str
    ) -> Dict[int, List[Path]]:
        dim_to_data_paths: Dict[int, List[Path]] = {}
        for data_path in data_paths:
            file_name = data_path.stem
            dim_match = re.search(rf"{dim_prefix}(\d*)", file_name)
            if dim_match is not None:
                dim_match = dim_match.group(0)[len(dim_prefix) :]
                dim = int(dim_match)
                if dim not in dim_to_data_paths:
                    dim_to_data_paths[dim] = []

                dim_to_data_paths[dim].append(data_path)

        return dim_to_data_paths

    def __init__(
        self,
        fs: AbstractFileSystem,
        image_directory: types.PathLike,
        data_file_prefix: str,
    ):
        self._fs = fs
        self.image_directory = Path(image_directory)
        self.id = self.image_directory.stem
        self._annotation_record = SldyImage._get_yaml_contents(
            fs, image_directory / "AnnotationRecord.yaml"
        )
        self._channel_record = SldyImage._get_yaml_contents(
            fs, image_directory / "ChannelRecord.yaml"
        )
        self._image_record = SldyImage._get_yaml_contents(
            fs, image_directory / "ImageRecord.yaml"
        )

        z_step_size = self._annotation_record.get(
            "mInterplaneSpacing"
        )  # NOTE not found in test file, maybe no z dim? or is this wrong?
        m_micron_per_pixel = float(self._image_record["CLensDef70"]["mMicronPerPixel"])
        optovar_mag = float(self._image_record["COptovarDef70"]["mMagnification"])
        mx_factor = float(self._channel_record["CExposureRecord70"]["mXFactor"])
        my_factor = float(self._channel_record["CExposureRecord70"]["mYFactor"])
        # TODO: Is physical what Chris Frick was calling these?
        self.physical_pixel_size_x = m_micron_per_pixel / optovar_mag * mx_factor
        self.physical_pixel_size_y = m_micron_per_pixel / optovar_mag * my_factor
        self.physical_pixel_size_z = None if z_step_size is None else float(z_step_size)

        data_path_matcher = fs.glob(self.image_directory / f"{data_file_prefix}*.npy")
        self.data_paths = set([Path(data_path) for data_path in data_path_matcher])

        # Create mapping of channel / timepoint to their respective data paths
        self.channel_to_data_paths = SldyImage._get_dim_to_data_path_map(
            self.data_paths, "_Ch"
        )
        self.timepoint_to_data_paths = SldyImage._get_dim_to_data_path_map(
            self.data_paths, "_TP"
        )

        # Create simple sorted list of each timepoint and channel
        self.timepoints = sorted(self.timepoint_to_data_paths.keys())
        self.channels = sorted(self.channel_to_data_paths.keys())

    @property
    def metadata(self) -> Dict[str, dict]:
        if self._metadata is None:
            self._metadata = {
                "annotation_record": self._annotation_record,
                "aux_data": SldyImage._get_yaml_contents(
                    self._fs, self.image_directory / "AuxData.yaml", False
                ),
                "channel_record": self._channel_record,
                "elapsed_times": SldyImage._get_yaml_contents(
                    self._fs, self.image_directory / "ElapsedTimes.yaml", False
                ),
                "image_record": self._image_record,
                "mask_record": SldyImage._get_yaml_contents(
                    self._fs, self.image_directory / "MaskRecord.yaml", False
                ),
                "sa_position_data": SldyImage._get_yaml_contents(
                    self._fs, self.image_directory / "SAPositionData.yaml", False
                ),
                "stage_position_data": SldyImage._get_yaml_contents(
                    self._fs, self.image_directory / "StagePositionData.yaml", False
                ),
            }

        return self._metadata

    def get_data(
        self, timepoint: Optional[int], channel: Optional[int], delayed: bool
    ) -> np.ndarray:
        data_paths = self.data_paths
        if timepoint is not None:
            data_paths = data_paths.intersection(
                self.timepoint_to_data_paths[timepoint]
            )
        if channel is not None:
            data_paths = data_paths.intersection(self.channel_to_data_paths[channel])

        if not data_paths:
            raise ValueError(":(")

        if len(data_paths) > 1:
            raise ValueError("boo")

        # TODO: Test if accessing this like [1:] would pull in excess memory like [:]
        data = np.load(list(data_paths)[0], mmap_mode="r" if delayed else None)

        # Add empty Z dimension if not present already
        if len(data.shape) == 2:
            return np.array([data])

        return data
