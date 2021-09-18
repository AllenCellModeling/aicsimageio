from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Dict, List, Tuple, Union

    from ome_types import OME

    from .. import types


def get_dims_and_coords_from_ome(
    ome: OME, scene_index: int
) -> Tuple[List[str], Dict[str, Union[List[Any], Union[types.ArrayLike, Any]]]]:
    """
    Process the OME metadata to retrieve the dimension names and coordinate planes.

    Parameters
    ----------
    ome: OME
        A constructed OME object to retrieve data from.
    scene_index: int
        The current operating scene index to pull metadata from.

    Returns
    -------
    dims: List[str]
        The dimension names pulled from the OME metadata.
    coords: Dict[str, Union[List[Any], Union[types.ArrayLike, Any]]]
        The coordinate planes / data for each dimension.
    """
    import numpy as np

    from ..dimensions import DimensionNames
    from ..readers.reader import Reader

    # Select scene
    scene_meta = ome.images[scene_index]

    # Create dimension order by getting the current scene's dimension order
    # and reversing it because OME store order vs use order is :shrug:
    dims = [d for d in scene_meta.pixels.dimension_order.value[::-1]]

    # Get coordinate planes
    coords: Dict[str, Union[List[str], np.ndarray]] = {}

    # Channels
    # Channel name isn't required by OME spec, so try to use it but
    # roll back to ID if not found
    coords[DimensionNames.Channel] = [
        channel.name if channel.name is not None else channel.id
        for channel in scene_meta.pixels.channels
    ]

    # Time
    # If global linear timescale we can np.linspace with metadata
    if scene_meta.pixels.time_increment is not None:
        coords[DimensionNames.Time] = Reader._generate_coord_array(
            0, scene_meta.pixels.size_t, scene_meta.pixels.time_increment
        )
    # If non global linear timescale, we need to create an array of every plane
    # time value
    elif scene_meta.pixels.size_t > 1:
        if len(scene_meta.pixels.planes) > 0:
            t_index_to_delta_map = {
                p.the_t: p.delta_t for p in scene_meta.pixels.planes
            }
            coords[DimensionNames.Time] = list(t_index_to_delta_map.values())
        else:
            coords[DimensionNames.Time] = np.linspace(
                0,
                scene_meta.pixels.size_t - 1,
                scene_meta.pixels.size_t,
            )

    # Handle Spatial Dimensions
    if scene_meta.pixels.physical_size_z is not None:
        coords[DimensionNames.SpatialZ] = Reader._generate_coord_array(
            0, scene_meta.pixels.size_z, scene_meta.pixels.physical_size_z
        )
    if scene_meta.pixels.physical_size_y is not None:
        coords[DimensionNames.SpatialY] = Reader._generate_coord_array(
            0, scene_meta.pixels.size_y, scene_meta.pixels.physical_size_y
        )
    if scene_meta.pixels.physical_size_x is not None:
        coords[DimensionNames.SpatialX] = Reader._generate_coord_array(
            0, scene_meta.pixels.size_x, scene_meta.pixels.physical_size_x
        )

    return dims, coords


def physical_pixel_sizes(ome: OME, scene: int = 0) -> types.PhysicalPixelSizes:
    """
    Returns
    -------
    sizes: PhysicalPixelSizes
        Using available metadata, the floats representing physical pixel sizes for
        dimensions Z, Y, and X.

    Notes
    -----
    We currently do not handle unit attachment to these values. Please see the file
    metadata for unit information.
    """
    from ..types import PhysicalPixelSizes

    p = ome.images[scene].pixels
    return PhysicalPixelSizes(p.physical_size_z, p.physical_size_y, p.physical_size_x)
