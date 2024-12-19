#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find the closest index in 2d array with 2d lat and 2d lon.
"""

# imports
import numpy as np


# function
def find_closest_2d(lat, lon, lats, lons, dim=2):

    """
    Finds the closest index to a given lat-lon point.

    Parameters:
        lat   Float. The latitude of the to-be-found point.
        lon   Float. The longitude of the to-be-found point.
        lats  2d float array. The latitude array where the point is to be found.
        lons  2d float array. The longitude array where the point is to be found.
        dim   Integer. Either 1 or 2, determining the dimension of the returned index. If dim=1 the returned index
                       corresponds to the index in the flattened array.
    """

    # find the closest lat and lon
    lat_temp = np.array(lats) - lat
    lon_temp = np.array(lons) - lon

    # calculate the Euclidian distance
    avg_dis = np.sqrt(lat_temp**2 + lon_temp**2)

    # get the index
    if dim == 1:
        ind = np.argmin(avg_dis)
    elif dim == 2:
        ind = np.unravel_index(avg_dis.argmin(), avg_dis.shape)
    # end if elif

    return ind

# end def


def find_close_2d(lat, lon, lats, lons, dim=2, fac=1):

    """
    Finds the closest index as well as the surrounding indices of a given lat-lon point.

    The closest index being here C, the function returns C as well as the surrounding indices (X):

                                X  X  X
                                X  C  X
                                X  X  X

    To find C the function find_closest_2d is used.

    Parameters:
        lat   Float. The latitude of the to-be-found point.
        lon   Float. The longitude of the to-be-found point.
        lats  2d float array. The latitude array where the point is to be found.
        lons  2d float array. The longitude array where the point is to be found.
        dim   Integer. Either 1 or 2, determining the dimension of the returned index. If dim=1 the returned index
                       corresponds to the index in the flattened array.
        fac   Integer. Factor that determines the number of rows and columns around point C. In terms of the
                       illustration above, this determines the number of X's.
    """

    ind = find_closest_2d(lat, lon, lats, lons, dim=1)

    # get the "x-shift" required to jump to the next row
    jump_row = lats.shape[1]

    # get the surrounding indices
    inds = []
    for y in np.arange(ind - (jump_row*fac), ind + (jump_row*fac)+1, jump_row):
        for x in np.arange(y-fac, y+fac+1, 1):
            inds.append(x)
        # end for x
    # end for y

    lat_ind = []
    lon_ind = []
    if dim==2:
        for i in inds:
            temp = np.unravel_index(i, lats.shape)
            lat_ind.append(temp[0])
            lon_ind.append(temp[1])
        # end for i
        return np.array(lat_ind), np.array(lon_ind)
    elif dim==1:
        return inds
    # end if elif
# end def
