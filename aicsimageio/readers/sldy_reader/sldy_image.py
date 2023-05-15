#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import yaml
from fsspec.spec import AbstractFileSystem

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class SldyImage:
    """
    Representation of a single acquisition in a 3i slidebook (SLDY) image.

    Parameters
    ----------
    fs: AbstractFileSystem
        The file system to used for reading.
    image_directory: types.PathLike
        Path to the image directory this is meant to represent.
    data_file_prefix: str, default = "ImageData"
        Prefix to the data files within this image directory to extract.
    """

    _metadata: Optional[Dict[str, Optional[dict]]] = None

    @staticmethod
    def _yaml_mapping(loader: yaml.Loader, node: yaml.Node, deep: bool = False) -> dict:
        """
        Static method intended to map key-value pairs found in image
        metadata yaml files to Python dictionaries.

        Necessary due to duplicate keys found in yaml files.

        Parameters
        ----------
        loader: yaml.Loader
            Loader to attach the mapping to and extract data using.
        node: Any
            Representation of the node at which this is at in the nested
            metadata tree.
        deep: bool default False
            Whether or not metadata will be deeply extractly.

        Returns
        -------
        mapping: dict
            Dictionary representation of the metadata in the node.
        """
        mapping: dict = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            value = loader.construct_object(value_node, deep=deep)
            # It seems slidebook classes are naively converted to yaml
            # files resulting in both duplicate keys mapped underneath
            # "StartClass" as well as duplicate classes
            if key == "StartClass":
                key = value["ClassName"]

            # Combine duplicate classes into a list
            if key in mapping:
                if not isinstance(mapping[key], list):
                    mapping[key] = [mapping[key]]

                mapping[key].append(value)
            else:
                mapping[key] = value

        return mapping

    @staticmethod
    def _get_yaml_contents(
        fs: AbstractFileSystem, yaml_path: Path, is_required: bool = True
    ) -> Optional[dict]:
        """
        Given a path to a yaml file will return a dictionary representation
        of the data found in the file.

        If the file does not exist will return `None` unless `is_required`
        is `True` in which case `FileNotFoundError`  will be allowed to
        bubble up out of this method.

        Parameters
        ----------
        fs: AbstractFileSystem
            The file system to used for reading.
        yaml_path: str
            The path to the file to read.
        is_required: bool default True
            If True, will not ignore `FileNotFoundError`s that occur while attempting
            to read in the yaml file.

        Returns
        -------
        yaml_contents: Optional[dict]
            Optional dictionary representation of the contents of the yaml file.
        """
        try:
            with fs.open(yaml_path) as f:
                return yaml.load(f, Loader=yaml.Loader)
        except FileNotFoundError:
            if is_required:
                raise

            log.debug(f"Unable to load metadata file {yaml_path}, ignoring")
            return None

    @staticmethod
    def _get_dim_to_data_path_map(
        data_paths: Set[Path], dim_prefix: str
    ) -> Dict[int, List[Path]]:
        """
        Returns a dictionary mapping from an arbitrary dimension index to the list of
        data paths matching that dimension.

        Parameters
        ----------
        data_paths: Set[Path]
            Set of data paths to compare against the dim_prefix.
        dim_prefix: str
            Prefix to the data paths, used to discern which dimension to read in.

        Returns
        -------
        dim_to_data_path_map: Dict[int, List[Path]]
            Dictionary mapping from an arbitrary dimension index to the list of
            data paths matching that dimension.
        """
        dim_to_data_paths: Dict[int, List[Path]] = {}
        for data_path in data_paths:
            file_name = data_path.stem
            search_result = re.search(rf"{dim_prefix}(\d*)", file_name)
            if search_result is not None:
                dim_match = search_result.group(0)[len(dim_prefix) :]
                dim = int(dim_match)
                if dim not in dim_to_data_paths:
                    dim_to_data_paths[dim] = []

                dim_to_data_paths[dim].append(data_path)

        return dim_to_data_paths

    @staticmethod
    def _cast_list(item: Any) -> List[Any]:
        if isinstance(item, list):
            return item

        return [item]

    def __init__(
        self,
        fs: AbstractFileSystem,
        image_directory: Path,
        data_file_prefix: str,
        channel_file_prefix: str = "_Ch",
        timepoint_file_prefix: str = "_TP",
    ):
        # Adjust mapping of yaml files to Python dictionaries to account
        # for duplicate keys found in slidebook yaml files
        yaml.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            SldyImage._yaml_mapping,
            yaml.Loader,
        )

        self._fs = fs
        self.image_directory = image_directory
        self.id = self.image_directory.stem
        self._channel_record = SldyImage._get_yaml_contents(
            fs, image_directory / "ChannelRecord.yaml"
        )
        self._image_record = SldyImage._get_yaml_contents(
            fs, image_directory / "ImageRecord.yaml"
        )

        # Ensure both are read in successfully
        if self._channel_record is None or self._image_record is None:
            raise ValueError(
                "Something unexpected went wrong reading in channel and image records"
            )

        lens_def = SldyImage._cast_list(self._image_record["CLensDef70"])[0]
        optovar_def = SldyImage._cast_list(self._image_record["COptovarDef70"])[0]
        exposure_record = SldyImage._cast_list(
            self._channel_record["CExposureRecord70"]
        )[0]
        micron_per_pixel = float(lens_def["mMicronPerPixel"])
        optovar_mag = float(optovar_def["mMagnification"])
        x_factor = float(exposure_record["mXFactor"])
        y_factor = float(exposure_record["mYFactor"])
        interplane_spacing = self._channel_record.get("mInterplaneSpacing")
        self.physical_pixel_size_x = micron_per_pixel / optovar_mag * x_factor
        self.physical_pixel_size_y = micron_per_pixel / optovar_mag * y_factor
        self.physical_pixel_size_z = (
            float(interplane_spacing) if interplane_spacing is not None else None
        )

        data_path_matcher = fs.glob(self.image_directory / f"{data_file_prefix}*.npy")
        self._data_paths = set([Path(data_path) for data_path in data_path_matcher])

        # Create mapping of timepoint / channel to their respective data paths
        self._timepoint_to_data_paths = SldyImage._get_dim_to_data_path_map(
            self._data_paths, timepoint_file_prefix
        )
        self._channel_to_data_paths = SldyImage._get_dim_to_data_path_map(
            self._data_paths, channel_file_prefix
        )

        # Create simple sorted list of each timepoint and channel
        self.timepoints = sorted(self._timepoint_to_data_paths.keys())
        self.channels = sorted(self._channel_to_data_paths.keys())

    @property
    def metadata(self) -> Dict[str, Optional[dict]]:
        """
        Returns a dictionary representing the metadata of this acquisition.

        Returns
        -------
        metadata: Dict[str, dict]
            Simple mapping of metadata file names to the metadata extracted
            from them. Possibly different than the actual yaml due to mapping
            the yaml to Python dictionaries, specifically with duplicate keys.
        """
        if self._metadata is None:
            self._metadata = {
                "annotation_record": SldyImage._get_yaml_contents(
                    self._fs, self.image_directory / "AnnotationRecord.yaml", False
                ),
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
        """
        Returns the image data for the given timepoint and channel if specified.
        If delayed, the data will be lazily read in.

        Parameters
        ----------
        timepoint: Optional[int]
            Optional timepoint to get data about.
        channel: Optional[int]
            Optional channel to get data about.
        delayed: bool
            If True, the data will be lazily read in.

        Returns
        -------
        data: np.ndarray
            Numpy representation of the image data found.
        """
        data_paths = self._data_paths
        if timepoint is not None:
            data_paths = data_paths.intersection(
                self._timepoint_to_data_paths[timepoint]
            )
        if channel is not None:
            data_paths = data_paths.intersection(self._channel_to_data_paths[channel])

        if len(data_paths) != 1:
            raise ValueError(
                f"Expected to find 1 data path for timepoint {timepoint} "
                f"and channel {channel}, but instead found {len(data_paths)}."
            )

        data = np.load(list(data_paths)[0], mmap_mode="r" if delayed else None)

        # Add empty Z dimension if not present already
        if len(data.shape) == 2:
            return np.array([data])

        return data
