#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate the number of danger level days per season.
"""


#%% imports
import numpy as np
import pandas as pd

# import proprietary functions
from .Func_DatetimeSimple import date_dt


#%% function
def count_dl_days(df, sta_yr, end_yr, full_sta=12, full_end=5, out_df=False):

    """
    Counts the danger levels per season.

    Default (out_df=False):
    Returns dictionaries dl_count_full, dl_count_win, dl_count_spr, the keys of which are the danger levels (starting
    at 0) and the values are dataframes containing the number of the respective danger level per season. There are three
    dictionaries returned, one for the full avalanche season (Nov-Jun), one for the winter (Dec-Feb), and one for the
    spring (Mar-May).

    Parameters:
        df              Pandas Series containing the danger levels with a datetime index.
        sta_yr, end_yr  Integer. The sta_yr represents the year in which the first winter season starts and the
                                 end_yr represents the year in which the last season ends. That is, of sta_yr=1990
                                 and end_yr=1995 the first season is 1990/91 and the last season is 1994/95.
        full_sta        Integer. The month in which the full avalanche period starts. Default is 12, i.e. December.
                                 Consider using 11 (i.e., Nov) because in some years the first avalanche warnings are
                                 issued in November already. Note that what are considered winter and spring season is
                                 not affected by this parameter.
        full_end        Integer. The month in which the full period ends. Default is 5, i.e. May. Consider using 6
                                 (i.e., June) because in some years the last avalanche warnings are issued only in June.
                                 Note that what are considered winter and spring season is not affected by this
                                 parameter.
        out_df          Logical. If True the three outputs are three Pandas dataframes with the years as index and the
                                 danger levels as columns.
    """

    # get the unique danger levels
    ndlev = int(np.max(df) + 1)

    years = np.arange(sta_yr+1, end_yr+1, 1)

    dl_count_full = {dl:[] for dl in np.arange(ndlev)}
    dl_count_win = {dl:[] for dl in np.arange(ndlev)}
    dl_count_spr = {dl:[] for dl in np.arange(ndlev)}
    for yr in np.arange(sta_yr, end_yr):

        # get the indices for full, winter, and spring season
        full_ind = (df.index >= date_dt(yr, full_sta, 1)) & (df.index <= date_dt(yr+1, full_end, 1))
        win_ind = (df.index >= date_dt(yr, 12, 1)) & (df.index <= date_dt(yr+1, 2, 1))
        spr_ind = (df.index >= date_dt(yr+1, 3, 1)) & (df.index <= date_dt(yr+1, 5, 1))

        # ge the danger-level counts
        for dl in np.arange(ndlev):
            dl_count_full[dl].append(np.sum(df[full_ind] == dl))
            dl_count_win[dl].append(np.sum(df[win_ind] == dl))
            dl_count_spr[dl].append(np.sum(df[spr_ind] == dl))
        # end for dl

    # end for yr

    if out_df:
        dl_count_full = pd.DataFrame({dl:np.array(dl_count_full[dl]) for dl in np.arange(ndlev)}, index=years)
        dl_count_win = pd.DataFrame({dl:np.array(dl_count_win[dl]) for dl in np.arange(ndlev)}, index=years)
        dl_count_spr = pd.DataFrame({dl:np.array(dl_count_spr[dl]) for dl in np.arange(ndlev)}, index=years)
    else:
        for dl in np.arange(ndlev):
            dl_count_full[dl] = pd.DataFrame({"DL":np.array(dl_count_full[dl])}, index=years)
            dl_count_win[dl] = pd.DataFrame({"DL":np.array(dl_count_win[dl])}, index=years)
            dl_count_spr[dl] = pd.DataFrame({"DL":np.array(dl_count_spr[dl])}, index=years)
        # end dl
    # end if else

    # return
    return {"full":dl_count_full, "winter":dl_count_win, "spring":dl_count_spr}
# end def

