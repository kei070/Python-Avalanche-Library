#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
An attempt to write a function that conveniently converts an entered date to the datetime format so it can be used as,
e.g., a threshold value in a datetime array.
"""

# imports
from datetime import datetime

# function 1
def date_dt(year, mon=None, day=None):

    """
    Returns the entered date in datetime format.

    Parameters:
        year   Integer.
        mon    Integer.
        day    Integer.
    """

    if mon is None:
        mon = "01"
    if day is None:
        day = "01"
    # end if

    return(datetime.strptime(f"{year}-{mon}-{day}", "%Y-%m-%d"))

# end def

# function 2
def date_dt2(date):

    """
    Returns the entered date in datetime format.

    Parameters:
        year   Integer.
        mon    Integer.
        day    Integer.
    """

    return(datetime.strptime(date, "%Y-%m-%d"))

# end def
