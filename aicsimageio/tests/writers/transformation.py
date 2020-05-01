#!/usr/bin/env python

# authors: Dan Toloudis     danielt@alleninstitute.org
#          Zach Crabtree    zacharyc@alleninstitute.org

import math as m

import numpy as np


def transform(image):
    assert len(image.shape) == 4

    # the dimensions of input must be 'ZCYX'
    size_x = image.shape[2]
    size_y = image.shape[3]
    rgb_channels = 3

    transformed_image = np.ndarray([size_x, size_y, rgb_channels])

    # the png writer cannot handle slices with just a single channel.
    # this gives three channels that are identical, and stacks them on top of each other
    if image.shape[1] < rgb_channels:
        # if there is only one channel available, the channels are repeated three times
        # (to c = 3) R -> RRR
        # if there are two channels available, the channels are repeated twice
        # (to c = 4) RG -> RRGG
        # we eliminate the first R to have the RGG channels read in as RGB channels
        # (red and cyan colors)
        image = np.repeat(image, repeats=rgb_channels + 1 - image.shape[1], axis=1)
        if image.shape[1] == 4:
            image = np.delete(image, 0, 1)
        transformed_image = np.transpose(image[0, :, :, :], (2, 1, 0))

    for x in range(0, rgb_channels):
        # getting the middle slice from each channel
        transformed_image[:, :, x] = image[int(m.floor(image.shape[0] / 2)), x, :, :]
        # getting the maximum values from the intensity frequency histogram
        channel_max_value = transformed_image[:, :, x].max()
        # getting the minimum values from the intensity frequency histogram
        channel_min_value = transformed_image[:, :, x].min()
        # adjusting intensity frequency histogram to lie along the x axis
        transformed_image[:, :, x] -= channel_min_value
        # stretching the peaks along 0 - 255
        peak_stretch = int(255.0 / (channel_max_value - channel_min_value))
        transformed_image[:, :, x] *= peak_stretch

    return np.transpose(transformed_image, (2, 0, 1))
