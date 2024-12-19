#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper function: methodology mostly suggested by ChatGPT. The CHatGPT function was originally written to work only for
December. My contribution lies in extending to to work for multiple months.
"""

# imports
import numpy as np
import xarray as xr

# function
def assign_winter_year(time, start_month):

    """
    The function works as follows:

        In step 1 the indices for the requested months are found. Not that if one submits e.g. start_month=11 both 11
                  and 12, i.e., November and December will be used. The result of this, i.e., the variable "inds" is
                  a numpy array that contains True where the month is 11 or 12 and False otherwise.
        In step 2 an array of years is set up that increases the year by 1 for the choses months (in the example from)
                  above this would be for November and December) and leaves it unchanged for the other months. This is
                  done via xr.where(cond, x, y) which returns x if cond==True and y if cond==False. Thus, in or case
                  it returns year+1 for November and December and year for the other months. With that November and
                  December have been reassigned.

    """

    try:
        year = np.array(time.dt.year)

        # shift the months of choice to the next year to group it with Jan/Feb of the next year
        month = np.array(time.dt.month)
    except:
        year = time.year

        # shift the months of choice to the next year to group it with Jan/Feb of the next year
        month = time.month
    # end try except

    # step 1 (see documentation above)
    inds = np.any(np.array([month == m for m in range(start_month, 13)]), axis=0)

    # step 2 (see documentation above)
    winter_year = xr.where(inds, year + 1, year)

    return winter_year
# end def

"""
# test
import pandas as pd

time = pd.date_range('1950-01-01', '1960-12-31', freq='ME')

test = assign_winter_year(time, start_month=11)
"""