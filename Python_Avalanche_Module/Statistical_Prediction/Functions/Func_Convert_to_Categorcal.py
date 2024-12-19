#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function to convert the output of neural network to a categorical type.
"""

# imports
import numpy as np


# function
def prob_to_cat(data):
    """
    Convert the output of neural network to a categorical type. This also works to convert one-hot formatted data to
    1d data.
    """
    if len(np.shape(data.squeeze())) == 1:
        out = np.zeros(np.shape(data.squeeze()))
        out[data.squeeze() > 0.5] = 1
    else:
        out = np.argmax(data, axis=1)
    # end if else

    return out

# end def