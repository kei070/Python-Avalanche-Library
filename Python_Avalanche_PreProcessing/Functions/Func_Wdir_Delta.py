#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function to calculate the difference in wind direction.
"""

# imports
import numpy as np


#%% version 1
def dwdir_simple(a, b):

    """
    Calculates the difference in wind direction in degrees.
    Expected input units: degrees from 0 to 360.

    Based on the following logic: The difference in wind direction on a wind rose cannot be larger than 180 degrees.
                                  Thus, if the difference is larger than 180 degrees (say, if the change in wind
                                  direction is from 10 to 350 degrees, which would give a delta of 340), the resulting
                                  delta is subtracted from 360 (in our example yielding 20 degrees, which is correct).

    Parameters:
        a   (Numpy array of) float or integer. Wind direction one.
        b   (Numpy array of) float or integer. Wind direction two.

    Returns:
        The non-directional (i.e., the number is always positive) difference of a and b on the wind rose.
    """

    # make sure that a and b are arrays
    if type(a) != np.ndarray:
        a = np.array([a])
    if type(b) != np.ndarray:
        b = np.array([b])
    # end if

    # calculate the delta array
    delta = np.abs(a - b)

    # generate a helper array
    temp = 360 - delta

    # for the elements > 180 set the values of the helper array
    delta[delta > 180] = temp[delta > 180]

    return delta

# end def dwdir


#%% version 2 --> using trigonometry
def dwdir(a, b):

    """
    Calculates the difference in wind direction in degrees using cos and arccos.
    Expected input units: degrees from 0 to 360.

    Based on the following logic: The difference in wind direction on a wind rose cannot be larger than 180 degrees.
                                  However, calculating the difference between, e.g., 10 and 200 degrees yields 190,
                                  which on the wind rose should yield 170. And the difference between 10 and 210 is
                                  200 while it should be 160. Applying the cosine function to the differences
                                  essentially provides this conversion. To retrieve the angle from the output of the
                                  cosine function, arccos function can be used.

    Parameters:
        a   (Numpy array of) float or integer. Wind direction one.
        b   (Numpy array of) float or integer. Wind direction two.

    Returns:
        The non-directional (i.e., the number is always positive) difference of a and b on the wind rose.
    """

    # make sure that a and b are arrays
    if type(a) != np.ndarray:
        a = np.array([a])
    if type(b) != np.ndarray:
        b = np.array([b])
    # end if

    # calculate the delta array
    delta = np.abs(a - b)

    return (np.arccos(np.cos(delta/180*np.pi))) / np.pi * 180

# end def


#%% helper function to calculate how many operations are needed to calculate the differences between all individual
#   hourly values in one day
def n_op(n):
    x = n-1
    return int(x**2 - (x**2 - x)/2)
# end def

# print(n_op(24))


#%% calculate the differences between all the individual values per day and then take the maximum
#   --> see the descriptions in the functions above for an explanation of the logic behind the procedure
def dwdir_df(series):
    series = np.array(series)
    delta = (series[:, np.newaxis]) - series
    return np.max((np.arccos(np.cos(delta/180*np.pi))) / np.pi * 180)
# end def


#%% test the function
"""
import pandas as pd
data = {"Datetime":pd.date_range(start="2020-01-01", periods=24, freq="h"),
        "Value":np.random.rand(24)*100}
df = pd.DataFrame(data)
df["Datetime"] = pd.to_datetime(df["Datetime"])
df.set_index("Datetime", inplace=True)

result = df.groupby(df.index.date).apply(lambda x: dwdir_df(x["Value"]))
"""

#%% test
# a = np.array([1, 2, 3])
# b = np.array([21, 62, 223])
# print(dwdir(a, b))