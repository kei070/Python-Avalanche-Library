#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper function for the confusion matrix to plot it in a heat map.
"""

# imports
import numpy as np

# function
def conf_helper(y, conf):

    # get the number of danger levels
    ndlev = len(np.unique(y))

    # get the support values
    supps = []
    for n in np.arange(ndlev):
        supps.append(np.sum(y == n))
    # end n
    supps = np.array(supps)

    gcounts = ["{0:0.0f}".format(value) for value in conf.flatten()]
    gperc = ["{0:.2%}".format(value) for value in conf.flatten()/np.repeat(supps, ndlev)]

    labels = [f"{v1}\n{v2}" for v1, v2 in zip(gcounts, gperc)]

    labels = np.asarray(labels).reshape(ndlev, ndlev)

    return labels

# end def

