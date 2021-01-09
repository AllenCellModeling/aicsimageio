#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union


def generate_ome_image_id(image_id: Union[str, int]) -> str:
    """
    Naively generates the standard OME image ID using a provided ID.

    Parameters
    ----------
    image_id: Union[str, int]
        A string or int representating the ID for an image.
        In the context of the usage of this function, this is usually used with the
        index of the scene / image.

    Returns
    -------
    ome_image_id: str
        The OME standard for image IDs.
    """
    return f"Image:{image_id}"
