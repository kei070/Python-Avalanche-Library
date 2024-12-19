#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for calculating specific sets of predictors. One function for temperature-related predictors, one for
wind-related, and one for precipitation-related predictors.
"""


#%% imports
import sys
import glob
import numpy as np
import xarray as xr
import pandas as pd

from .Func_Progressbar import print_progress_bar
from .Func_Wdir_Delta import dwdir_df


#%% helper wrapper function for the aggregation of grid cells --> average? some percentile?
def agg_gridcells(data, agg_type="mean", perc=50, axis=0):

    # set up the functions
    funcs = {"mean":np.nanmean, "median":np.nanmedian, "percentile":np.nanpercentile}

    # perform the aggregation
    if agg_type == "mean":
        return funcs[agg_type](data, axis=axis)
    elif agg_type == "median":
        return funcs[agg_type](data, axis=axis)
    elif agg_type == "percentile":
        return funcs[agg_type](data, q=perc, axis=axis)
    # end if elif

# end def


#%% temperature-related predictors
def calc_temp_pred(data_path, reg_code, h_low=400, h_hi=900, agg_type="mean", perc=90, varn_dic={}, unclean=False):

    # get the variable names for wind speed and direction from the given dictionary
    t2m_n = varn_dic["t2m"]

    #% set region name according to region code
    regions = {3009:"NordTroms", 3010:"Lyngen", 3011:"Tromsoe", 3012:"SoerTroms", 3013:"IndreTroms"}

    # get the region name
    region = regions[reg_code]

    print(f"\nGenerate the temperature-related predictors for full time series for region {region}.\n")

    #% print data path
    print(data_path)
    print()

    #% load data --> should be a grid cell in the Lyngen Alps
    fn_list = sorted(glob.glob(data_path + f"*{region}*.nc"), key=str.casefold)

    if len(fn_list) == 0:
        sys.exit(f"No data for {reg_code} between {h_low} and {h_hi} m. Stopping execution.")
    # end if

    # load the nc file and the variable --> aggregate the grid cells depending on the given agg_type
    nc = xr.open_dataset(fn_list[0])
    at2m = nc[t2m_n]
    at2m = agg_gridcells(at2m, agg_type=agg_type, perc=perc, axis=0)

    # load the date
    date = nc.time

    #% set up the hourly dataframe with the multi-gridcell averaged values
    df_atm = pd.DataFrame({"date":date, "air_temperature_2m":at2m})

    df_atm.set_index("date", inplace=True)


    #% aggregate daily and calculate several quantities
    df_atm_day = pd.DataFrame({"date":pd.to_datetime(df_atm.groupby(df_atm.index.date).mean().index),
                               "t_mean":df_atm["air_temperature_2m"].groupby(df_atm.index.date).mean(),
                               "t_max":df_atm["air_temperature_2m"].groupby(df_atm.index.date).max(),
                               "t_min":df_atm["air_temperature_2m"].groupby(df_atm.index.date).min(),
                               "t_range":df_atm["air_temperature_2m"].groupby(df_atm.index.date).max() -
                               df_atm["air_temperature_2m"].groupby(df_atm.index.date).min()})

    # add the freeze-thaw cycle (1 if yes, 0 of no)
    ftc = np.zeros(len(df_atm_day))
    ftc[((df_atm_day["t_min"] < 273.15) & (df_atm_day["t_max"] > 273.15))] = 1
    df_atm_day["ftc"] = ftc

    df_atm_day.set_index(pd.to_datetime(df_atm_day.index), inplace=True)


    # loop over the available dates and set up the array
    # --> we are using the whole year so this is rather pointless
    # --> it is left in for possible later revision
    in_season_dates = []
    for idate in df_atm_day.index:
        if idate.month in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            in_season_dates.append(idate)
        # end if
    # end for idate


    #% set up a the predictors dataframe with the in_season_dates as index
    df_pred = pd.DataFrame({"date":in_season_dates})


    #% basically: extract the in-season dates from the df_atm_day dataset
    df_pred = df_atm_day.merge(df_pred, how="inner", on="date")
    df_pred.set_index(pd.to_datetime(df_pred.date), inplace=True)


    #% loop over the events and calculate also the accrued quantities
    dtemp1, dtemp2, dtemp3 = [], [], []  # temperature amplitude 1-3 days before an event
    dtempd1, dtempd2, dtempd3 = [], [], []  # temperature amplitude between 1-3 days before and day of the event
    t2, t3, t4, t5, t6, t7 = [], [], [], [], [], []  # average temperature 2-7 days before AND including the event
    tmax2, tmax3, tmax4, tmax5, tmax6, tmax7 =  [], [], [], [], [], []  # average of max-temperature 2-7 days
    pdd = []  # positive degree-days

    count = 0
    l = len(df_pred.date)
    print_progress_bar(0, l, prefix='Progress:', suffix='Complete', length=50)
    for da in df_pred.date:

        # find the index for the date of the event
        ind = np.where(df_atm_day.date == da)[0][0]

        # calculate the average temperature over multiple days
        t3.append(np.mean(df_atm_day["t_mean"].iloc[ind-2:ind+1]))
        t7.append(np.mean(df_atm_day["t_mean"].iloc[ind-6:ind+1]))

        # calculate the average maximum temperature over multiple days
        tmax3.append(np.mean(df_atm_day["t_max"].iloc[ind-2:ind+1]))
        tmax7.append(np.mean(df_atm_day["t_max"].iloc[ind-6:ind+1]))

        if unclean:
            t2.append(np.mean(df_atm_day["t_mean"].iloc[ind-1:ind+1]))
            t4.append(np.mean(df_atm_day["t_mean"].iloc[ind-3:ind+1]))
            t5.append(np.mean(df_atm_day["t_mean"].iloc[ind-4:ind+1]))
            t6.append(np.mean(df_atm_day["t_mean"].iloc[ind-5:ind+1]))
            tmax2.append(np.mean(df_atm_day["t_max"].iloc[ind-1:ind+1]))
            tmax4.append(np.mean(df_atm_day["t_max"].iloc[ind-3:ind+1]))
            tmax5.append(np.mean(df_atm_day["t_max"].iloc[ind-4:ind+1]))
            tmax6.append(np.mean(df_atm_day["t_max"].iloc[ind-5:ind+1]))
        # end if

        # calculate the temperature amplitude the day before and up to three days before the event
        dtemp1.append(df_atm_day["t_range"].iloc[ind-1])
        dtemp2.append(df_atm_day["t_range"].iloc[ind-2])
        dtemp3.append(df_atm_day["t_range"].iloc[ind-3])

        # calculate the temperature amplitude between 1-3 days before and the day of the event
        dtempd1.append(np.max([df_atm_day["t_max"].iloc[ind-1] - df_atm_day["t_min"].iloc[ind],
                               df_atm_day["t_max"].iloc[ind] - df_atm_day["t_min"].iloc[ind-1]]))
        dtempd2.append(np.max([df_atm_day["t_max"].iloc[ind-2] - df_atm_day["t_min"].iloc[ind],
                               df_atm_day["t_max"].iloc[ind] - df_atm_day["t_min"].iloc[ind-2]]))
        dtempd3.append(np.max([df_atm_day["t_max"].iloc[ind-3] - df_atm_day["t_min"].iloc[ind],
                               df_atm_day["t_max"].iloc[ind] - df_atm_day["t_min"].iloc[ind-3]]))

        # in lieu of having the "thaw periods" (whatever they exactly mean by that in the paper), calculate the sum of
        # the positive degree-days for a seven-day period before and including an event-day (i.e., six days prior and
        # the day itself)
        sev_temp = np.array(df_atm_day["t_mean"].iloc[ind-6:ind+1]) - 273.15
        pdd.append(np.sum(sev_temp[sev_temp > 0]))

        count += 1
        print_progress_bar(count, l, prefix='Progress:', suffix='Complete', length=50)
    # end for da

    print(f"\nWent through {count} iterations.\n")

    #% add the accrued quantities to the dataframe
    df_pred["t3"] = np.array(t3)
    df_pred["t7"] = np.array(t7)
    df_pred["tmax3"] = np.array(tmax3)
    df_pred["tmax7"] = np.array(tmax7)

    df_pred["dtemp1"] = np.array(dtemp1)
    df_pred["dtemp2"] = np.array(dtemp2)
    df_pred["dtemp3"] = np.array(dtemp3)

    df_pred["dtempd1"] = np.array(dtempd1)
    df_pred["dtempd2"] = np.array(dtempd2)
    df_pred["dtempd3"] = np.array(dtempd3)

    df_pred["pdd"] = np.array(pdd)

    if unclean:
        df_pred["t2"] = np.array(t2)
        df_pred["t4"] = np.array(t4)
        df_pred["t5"] = np.array(t5)
        df_pred["t6"] = np.array(t6)
        df_pred["tmax2"] = np.array(tmax2)
        df_pred["tmax4"] = np.array(tmax4)
        df_pred["tmax5"] = np.array(tmax5)
        df_pred["tmax6"] = np.array(tmax6)
    # end if

    return df_pred

# end calc_temp_pred


#%% wind-related predictors (except wind drift)
def calc_wind_pred(data_path, reg_code, h_low=400, h_hi=900, agg_type="mean", perc=90, varn_dic={}, unclean=False):

    # get the variable names for wind speed and direction from the given dictionary
    w10m_n = varn_dic["w10m"]
    wdir_n = varn_dic["wdir"]

    #% set region name according to region code
    regions = {3009:"NordTroms", 3010:"Lyngen", 3011:"Tromsoe", 3012:"SoerTroms", 3013:"IndreTroms"}

    # get the region name
    region = regions[reg_code]

    print(f"\nGenerate the wind-related predictors for full time series for region {region}.\n")

    #% print data path
    print(data_path)
    print()

    #% load data --> should be a grid cell in the Lyngen Alps
    fn_list = sorted(glob.glob(data_path + f"*{region}*.nc"), key=str.casefold)

    if len(fn_list) == 0:
        sys.exit(f"No data for {reg_code} between {h_low} and {h_hi} m. Stopping execution.")
    # end if

    # load the nc file and the variable --> aggregate the grid cells depending on the given agg_type
    nc = xr.open_dataset(fn_list[0])
    w10m = nc[w10m_n]
    w10m = agg_gridcells(w10m, agg_type=agg_type, perc=perc, axis=0)
    wdir = nc[wdir_n]
    wdir = agg_gridcells(wdir, agg_type=agg_type, perc=perc, axis=0)

    # load the date
    date = nc.time

    #% set up the hourly dataframe with the multi-gridcell averaged values
    df_atm = pd.DataFrame({"date":date, "wind_speed_10m":w10m, "wind_direction":wdir})

    df_atm.set_index("date", inplace=True)

    # calculate the wind direction variation per day
    dwdir = df_atm.groupby(df_atm.index.date).apply(lambda x: dwdir_df(x["wind_direction"]))
    df_atm["dwdir"] = dwdir

    #% aggregate daily and calculate several quantities
    df_atm_day = pd.DataFrame({"date":pd.to_datetime(df_atm.groupby(df_atm.index.date).mean().index),
                               "ws_mean":df_atm["wind_speed_10m"].groupby(df_atm.index.date).mean(),
                               "ws_min":df_atm["wind_speed_10m"].groupby(df_atm.index.date).min(),
                               "ws_max":df_atm["wind_speed_10m"].groupby(df_atm.index.date).max(),
                               "ws_range":df_atm["wind_speed_10m"].groupby(df_atm.index.date).max() -
                               df_atm["wind_speed_10m"].groupby(df_atm.index.date).min(),
                               "wind_direction":df_atm["wind_direction"].groupby(df_atm.index.date).mean(),
                               "dwdir":df_atm["dwdir"].groupby(df_atm.index.date).mean()})

    df_atm_day.set_index(pd.to_datetime(df_atm_day.index), inplace=True)


    # loop over the available dates and set up the array
    # --> we are using the whole year so this is rather pointless
    # --> it is left in for possible later revision
    in_season_dates = []
    for idate in df_atm_day.index:
        if idate.month in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            in_season_dates.append(idate)
        # end if
    # end for idate


    #% set up a the predictors dataframe with the in_season_dates as index
    df_pred = pd.DataFrame({"date":in_season_dates})


    #% basically: extract the in-season dates from the df_atm_day dataset
    df_pred = df_atm_day.merge(df_pred, how="inner", on="date")
    df_pred.set_index(pd.to_datetime(df_pred.date), inplace=True)

    #% loop over the events and calculate also the accrued quantities
    dws1, dws2, dws3 = [], [], []  # wind speed variation 1-3 days before an event
    dwdir1, dwdir2, dwdir3 = [], [], []  # wind direction variation 1-3 days before an event
    dwsd1, dwsd2, dwsd3 = [], [], []  # wind speed variability between 1-3 days before and day of the event
    w2, w3, w4, w5, w6, w7 = [], [], [], [], [], []  # average wind speed 2-5 days before AND including the event
    wmax2, wmax3, wmax4, wmax5, wmax6, wmax7 =  [], [], [], [], [], []  # average of max-wind speed 2-5 days before AND
    #                                                                     including the event

    count = 0
    l = len(df_pred.date)
    print_progress_bar(0, l, prefix='Progress:', suffix='Complete', length=50)
    for da in df_pred.date:

        # find the index for the date of the event
        ind = np.where(df_atm_day.date == da)[0][0]

        # calculate the average wind speed over multiple days
        w3.append(np.mean(df_atm_day["ws_mean"].iloc[ind-2:ind+1]))

        # calculate the average maximum wind speed over multiple days
        wmax3.append(np.mean(df_atm_day["ws_max"].iloc[ind-2:ind+1]))

        if unclean:
            w2.append(np.mean(df_atm_day["ws_mean"].iloc[ind-1:ind+1]))
            w4.append(np.mean(df_atm_day["ws_mean"].iloc[ind-3:ind+1]))
            w5.append(np.mean(df_atm_day["ws_mean"].iloc[ind-4:ind+1]))
            w6.append(np.mean(df_atm_day["ws_mean"].iloc[ind-5:ind+1]))
            w7.append(np.mean(df_atm_day["ws_mean"].iloc[ind-6:ind+1]))

            wmax2.append(np.mean(df_atm_day["ws_max"].iloc[ind-1:ind+1]))
            wmax4.append(np.mean(df_atm_day["ws_max"].iloc[ind-3:ind+1]))
            wmax5.append(np.mean(df_atm_day["ws_max"].iloc[ind-4:ind+1]))
            wmax6.append(np.mean(df_atm_day["ws_max"].iloc[ind-5:ind+1]))
            wmax7.append(np.mean(df_atm_day["ws_max"].iloc[ind-6:ind+1]))
        # end if

        # calculate the wind variation the day before and up to three days before the event
        dws1.append(df_atm_day["ws_range"].iloc[ind-1])
        dws2.append(df_atm_day["ws_range"].iloc[ind-2])
        dws3.append(df_atm_day["ws_range"].iloc[ind-3])
        dwdir1.append(df_atm_day["dwdir"].iloc[ind-1])
        dwdir2.append(df_atm_day["dwdir"].iloc[ind-2])
        dwdir3.append(df_atm_day["dwdir"].iloc[ind-3])

        # calculate the temperature amplitude between 1-3 days before and the day of the event
        dwsd1.append(np.max([df_atm_day["ws_max"].iloc[ind-1] - df_atm_day["ws_min"].iloc[ind],
                               df_atm_day["ws_max"].iloc[ind] - df_atm_day["ws_min"].iloc[ind-1]]))
        dwsd2.append(np.max([df_atm_day["ws_max"].iloc[ind-2] - df_atm_day["ws_min"].iloc[ind],
                               df_atm_day["ws_max"].iloc[ind] - df_atm_day["ws_min"].iloc[ind-2]]))
        dwsd3.append(np.max([df_atm_day["ws_max"].iloc[ind-3] - df_atm_day["ws_min"].iloc[ind],
                               df_atm_day["ws_max"].iloc[ind] - df_atm_day["ws_min"].iloc[ind-3]]))

        count += 1
        print_progress_bar(count, l, prefix='Progress:', suffix='Complete', length=50)
    # end for da

    print(f"\nWent through {count} iterations.\n")

    #% add the accrued quantities to the dataframe
    df_pred["w3"] = np.array(w3)
    df_pred["wmax3"] = np.array(wmax3)

    if unclean:
        df_pred["w2"] = np.array(w2)
        df_pred["w4"] = np.array(w4)
        df_pred["w5"] = np.array(w5)
        df_pred["w6"] = np.array(w6)
        df_pred["w7"] = np.array(w7)
        df_pred["wmax2"] = np.array(wmax2)
        df_pred["wmax4"] = np.array(wmax4)
        df_pred["wmax5"] = np.array(wmax5)
        df_pred["wmax6"] = np.array(wmax6)
        df_pred["wmax7"] = np.array(wmax7)
    # end if

    df_pred["dws1"] = np.array(dws1)
    df_pred["dws2"] = np.array(dws2)
    df_pred["dws3"] = np.array(dws3)

    df_pred["dwsd1"] = np.array(dwsd1)
    df_pred["dwsd2"] = np.array(dwsd2)
    df_pred["dwsd3"] = np.array(dwsd3)

    df_pred["dwdir1"] = np.array(dwdir1)
    df_pred["dwdir2"] = np.array(dwdir2)
    df_pred["dwdir3"] = np.array(dwdir3)

    return df_pred

# end calc_wind_pred


#%% precipitation-related predictors (except wind drift)
def calc_prec_pred(data_path, reg_code, h_low=400, h_hi=900, agg_type="mean", perc=90, varn_dic={}, unclean=False):

    # get the variable names for precipitation, rain, and snow from the given dictionary
    prec_n = varn_dic["prec"]
    snow_n = varn_dic["snow"]
    rain_n = varn_dic["rain"]

    #% set region name according to region code
    regions = {3009:"NordTroms", 3010:"Lyngen", 3011:"Tromsoe", 3012:"SoerTroms", 3013:"IndreTroms"}

    # get the region name
    region = regions[reg_code]

    print(f"\nGenerate the precip-related predictors for full time series for region {region}.\n")

    #% print data path
    print(data_path)
    print()

    #% load data --> should be a grid cell in the Lyngen Alps
    fn_list = sorted(glob.glob(data_path + f"*{region}*.nc"), key=str.casefold)

    if len(fn_list) == 0:
        sys.exit(f"No data for {reg_code} between {h_low} and {h_hi} m. Stopping execution.")
    # end if

    print(f"\nNumber of files: {len(fn_list)}.\n")


    # load the nc file and the variable --> average over the locations
    nc = xr.open_dataset(fn_list[0])
    prec = nc[prec_n]
    snow = nc[snow_n]
    rain = nc[rain_n]


    #% aggregate the grid cells depending on the given agg_type
    prec = agg_gridcells(prec, agg_type=agg_type, perc=perc, axis=0)
    snow = agg_gridcells(snow, agg_type=agg_type, perc=perc, axis=0)
    rain = agg_gridcells(rain, agg_type=agg_type, perc=perc, axis=0)

    # load the date
    date = nc.time


    #% set up the hourly dataframe with the multi-gridcell averaged values
    df_atm = pd.DataFrame({"date":date, "precipitation_amount_hourly":prec,
                           "solid_precip":snow, "liquid_precip":rain})

    df_atm.set_index("date", inplace=True)

    #% aggregate daily and calculate several quantities
    df_atm_day = pd.DataFrame({"date":pd.to_datetime(df_atm.groupby(df_atm.index.date).mean().index),
                               "total_prec_sum":df_atm["solid_precip"].groupby(df_atm.index.date).sum() +
                                                df_atm["liquid_precip"].groupby(df_atm.index.date).sum(),
                               "s1":df_atm["solid_precip"].groupby(df_atm.index.date).sum(),
                               "r1":df_atm["liquid_precip"].groupby(df_atm.index.date).sum()})

    df_atm_day.set_index(pd.to_datetime(df_atm_day.index), inplace=True)


    # loop over the available dates and set up the array
    # --> we are using the whole year so this is rather pointless
    # --> it is left in for possible later revision
    in_season_dates = []
    for idate in df_atm_day.index:
        if idate.month in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            in_season_dates.append(idate)
        # end if
    # end for idate


    #% set up a the predictors dataframe with the in_season_dates as index
    df_pred = pd.DataFrame({"date":in_season_dates})


    #% basically: extract the in-season dates from the df_atm_day dataset
    df_pred = df_atm_day.merge(df_pred, how="inner", on="date")
    df_pred.set_index(pd.to_datetime(df_pred.date), inplace=True)

    #% loop over the events and calculate also the accrued quantities
    r2, r3, r4, r5, r6, r7 = [], [], [], [], [], []  # liquid precip accrued 2-7 days
    s2, s3, s4, s5, s6, s7 = [], [], [], [], [], []  # solid precip accrued 2-7 days

    count = 0
    l = len(df_pred.date)
    print_progress_bar(0, l, prefix='Progress:', suffix='Complete', length=50)
    for da in df_pred.date:

        # find the index for the date of the event
        ind = np.where(df_atm_day.date == da)[0][0]

        # calculate the accrued liquid precipitation over multiple days
        r3.append(np.sum(df_atm_day["r1"].iloc[ind-2:ind+1]))
        r7.append(np.sum(df_atm_day["r1"].iloc[ind-6:ind+1]))

        # calculate the accrued solid precipitation over multiple days
        s3.append(np.sum(df_atm_day["s1"].iloc[ind-2:ind+1]))
        s7.append(np.sum(df_atm_day["s1"].iloc[ind-6:ind+1]))

        if unclean:
            r2.append(np.sum(df_atm_day["r1"].iloc[ind-1:ind+1]))
            r4.append(np.sum(df_atm_day["r1"].iloc[ind-3:ind+1]))
            r5.append(np.sum(df_atm_day["r1"].iloc[ind-4:ind+1]))
            r6.append(np.sum(df_atm_day["r1"].iloc[ind-5:ind+1]))
            s2.append(np.sum(df_atm_day["s1"].iloc[ind-1:ind+1]))
            s4.append(np.sum(df_atm_day["s1"].iloc[ind-3:ind+1]))
            s5.append(np.sum(df_atm_day["s1"].iloc[ind-4:ind+1]))
            s6.append(np.sum(df_atm_day["s1"].iloc[ind-5:ind+1]))
        # end if

        count += 1
        print_progress_bar(count, l, prefix='Progress:', suffix='Complete', length=50)
    # end for da

    print(f"\nWent through {count} iterations.\n")

    #% add the accrued quantities to the dataframe
    df_pred["r3"] = np.array(r3)
    df_pred["r7"] = np.array(r7)
    df_pred["s3"] = np.array(s3)
    df_pred["s7"] = np.array(s7)

    if unclean:
        df_pred["r2"] = np.array(r2)
        df_pred["r4"] = np.array(r4)
        df_pred["r5"] = np.array(r5)
        df_pred["r6"] = np.array(r6)
        df_pred["s2"] = np.array(s2)
        df_pred["s4"] = np.array(s4)
        df_pred["s5"] = np.array(s5)
        df_pred["s6"] = np.array(s6)
    # end if

    return df_pred

# end calc_prec_pred


#%% wind-drift predictors
def calc_wdrift_pred(reg_code, prec_pred, wind_pred, h_low=400, h_hi=900, unclean=False):

    """
    Requires s1, s2, s3 as well as wspeed_mean, w2, and w3 as calculated by the functions above.
    """

    # set up the output dataframe as a dictionary
    df = {"date":prec_pred["date"]}

    # calculate the wind-drift predictors
    df["wdrift"] = prec_pred["s1"] * wind_pred["ws_mean"]
    df["wdrift_3"] = prec_pred["s3"] * wind_pred["w3"]
    df["wdrift3"] = prec_pred["s1"] * wind_pred["ws_mean"]**3
    df["wdrift3_3"] = prec_pred["s3"] * wind_pred["w3"]**3

    if unclean:
        df["wdrift_2"] = prec_pred["s2"] * wind_pred["w2"]
        df["wdrift3_2"] = prec_pred["s2"] * wind_pred["w2"]**3
    # end if

    return pd.DataFrame(df)

# calc_wdrift_pred


#%% RH and radiation related predictors
def calc_rad_rh_pred(data_path, reg_code, h_low=400, h_hi=900, agg_type="mean", perc=90, varn_dic={}, unclean=False):

    # get the variable names for precipitation, rain, and snow from the given dictionary
    rh_n = varn_dic["rh"]
    nlw_n = varn_dic["nlw"]
    nsw_n = varn_dic["nsw"]

    #% set region name according to region code
    regions = {3009:"NordTroms", 3010:"Lyngen", 3011:"Tromsoe", 3012:"SoerTroms", 3013:"IndreTroms"}

    # get the region name
    region = regions[reg_code]

    print(f"\nGenerate the RH- and radiation-related predictors for full time series for region {region}.\n")

    #% print data path
    print(data_path)
    print()

    #% load data --> should be a grid cell in the Lyngen Alps
    fn_list = sorted(glob.glob(data_path + f"*{region}*.nc"), key=str.casefold)

    if len(fn_list) == 0:
        sys.exit(f"No data for {reg_code} between {h_low} and {h_hi} m. Stopping execution.")
    # end if

    # load the nc file and the variable --> average over the locations
    nc = xr.open_dataset(fn_list[0])

    rh = nc[rh_n]
    rh = agg_gridcells(rh, agg_type=agg_type, perc=perc, axis=0)

    nlw = nc[nlw_n]
    nlw = agg_gridcells(nlw, agg_type=agg_type, perc=perc, axis=0)

    nsw = nc[nsw_n]
    nsw = agg_gridcells(nsw, agg_type=agg_type, perc=perc, axis=0)

    # load the date --> accomodate for the 3h values from NorCP
    if "time3h" in list(nc[nlw_n].coords):
        date = nc["time3h"]
    elif "time" in list(nc[nlw_n].coords):
        date = nc["time"]
    # end if

    #% set up the hourly dataframe with the multi-gridcell averaged values
    df_atm = pd.DataFrame({"date":date, "NLW":nlw, "NSW":nsw, "RH":rh})

    df_atm.set_index("date", inplace=True)

    #% aggregate daily and calculate several quantities
    df_atm_day = pd.DataFrame({"date":pd.to_datetime(df_atm.groupby(df_atm.index.date).mean().index),
                               "RH":df_atm["RH"].groupby(df_atm.index.date).mean(),
                               "NLW":df_atm["NLW"].groupby(df_atm.index.date).mean(),
                               "NSW":df_atm["NSW"].groupby(df_atm.index.date).mean()})

    # check the NSW for invalid values
    nsw_ind = df_atm_day["NSW"] > 1e10
    nsw_sum = np.sum(nsw_ind)
    if nsw_sum > 0:
        print(f"\nNSW contains {nsw_sum} = {nsw_sum/len(df_atm_day)*100:.1f}% values > 1e10. Setting them to zero.\n")
        df_atm_day.loc[nsw_ind, "NSW"] = 0
    # end if

    df_atm_day.set_index(pd.to_datetime(df_atm_day.index), inplace=True)


    # loop over the available dates and set up the array
    # --> we are using the whole year so this is rather pointless
    # --> it is left in for possible later revision
    in_season_dates = []
    for idate in df_atm_day.index:
        if idate.month in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            in_season_dates.append(idate)
        # end if
    # end for idate


    #% set up a the predictors dataframe with the in_season_dates as index
    df_pred = pd.DataFrame({"date":in_season_dates})


    #% basically: extract the in-season dates from the df_atm_day dataset
    df_pred = df_atm_day.merge(df_pred, how="inner", on="date")
    df_pred.set_index(pd.to_datetime(df_pred.date), inplace=True)


    rh2, rh3, rh4, rh5, rh6, rh7 = [], [], [], [], [], []  # average RH 2-5 days before AND including the event

    nlw2, nlw3, nlw4, nlw5, nlw6, nlw7 = [], [], [], [], [], []  # average net lw radiation 2-5 days before AND including the event
    nsw2, nsw3, nsw4, nsw5, nsw6, nsw7 = [], [], [], [], [], []  # average net sw radiation 2-5 days before AND including the event

    count = 0
    l = len(df_pred.date)
    print_progress_bar(0, l, prefix='Progress:', suffix='Complete', length=50)
    for da in df_pred.date:

        # find the index for the date of the event
        ind = np.where(df_atm_day.date == da)[0][0]

        # calculate the average RH over multiple days
        rh3.append(np.mean(df_atm_day["RH"].iloc[ind-2:ind+1]))
        rh7.append(np.mean(df_atm_day["RH"].iloc[ind-6:ind+1]))

        # calculate the average NLW over multiple days
        nlw3.append(np.mean(df_atm_day["NLW"].iloc[ind-2:ind+1]))
        nlw7.append(np.mean(df_atm_day["NLW"].iloc[ind-6:ind+1]))

        # calculate the average NSW over multiple days
        nsw3.append(np.mean(df_atm_day["NSW"].iloc[ind-2:ind+1]))
        nsw7.append(np.mean(df_atm_day["NSW"].iloc[ind-6:ind+1]))

        if unclean:
            rh2.append(np.mean(df_atm_day["RH"].iloc[ind-1:ind+1]))
            rh4.append(np.mean(df_atm_day["RH"].iloc[ind-3:ind+1]))
            rh5.append(np.mean(df_atm_day["RH"].iloc[ind-4:ind+1]))
            rh6.append(np.mean(df_atm_day["RH"].iloc[ind-5:ind+1]))

            nlw2.append(np.mean(df_atm_day["NLW"].iloc[ind-1:ind+1]))
            nlw4.append(np.mean(df_atm_day["NLW"].iloc[ind-3:ind+1]))
            nlw5.append(np.mean(df_atm_day["NLW"].iloc[ind-4:ind+1]))
            nlw6.append(np.mean(df_atm_day["NLW"].iloc[ind-5:ind+1]))

            nsw2.append(np.mean(df_atm_day["NSW"].iloc[ind-1:ind+1]))
            nsw4.append(np.mean(df_atm_day["NSW"].iloc[ind-3:ind+1]))
            nsw5.append(np.mean(df_atm_day["NSW"].iloc[ind-4:ind+1]))
            nsw6.append(np.mean(df_atm_day["NSW"].iloc[ind-5:ind+1]))
        # end if

        count += 1
        print_progress_bar(count, l, prefix='Progress:', suffix='Complete', length=50)
    # end for da

    print(f"\nWent through {count} iterations.\n")

    #% add the accrued quantities to the dataframe
    df_pred["RH3"] = np.array(rh3)
    df_pred["RH7"] = np.array(rh7)
    df_pred["NLW3"] = np.array(nlw3)
    df_pred["NLW7"] = np.array(nlw7)
    df_pred["NSW3"] = np.array(nsw3)
    df_pred["NSW7"] = np.array(nsw7)

    if unclean:
        df_pred["RH2"] = np.array(rh2)
        df_pred["RH4"] = np.array(rh4)
        df_pred["RH5"] = np.array(rh5)
        df_pred["RH6"] = np.array(rh6)
        df_pred["NLW2"] = np.array(nlw2)
        df_pred["NLW4"] = np.array(nlw4)
        df_pred["NLW5"] = np.array(nlw5)
        df_pred["NLW6"] = np.array(nlw6)
        df_pred["NSW2"] = np.array(nsw2)
        df_pred["NSW4"] = np.array(nsw4)
        df_pred["NSW5"] = np.array(nsw5)
        df_pred["NSW6"] = np.array(nsw6)
    # end if

    return df_pred

# end calc_rad_rh_pred


#%% seNorge predictors
def calc_seNorge_pred(data_path, reg_code, h_low=400, h_hi=900, agg_type="mean", perc=90, unclean=False):

    #% set region name according to region code
    regions = {3009:"NordTroms", 3010:"Lyngen", 3011:"Tromsoe", 3012:"SoerTroms", 3013:"IndreTroms"}

    # get the region name
    region = regions[reg_code]

    print(f"\nGenerate the seNorge-based predictors for full time series for region {region}.\n")

    #% print seNorge data path
    print(data_path)
    print()

    #% load data --> should be a grid cell in the Lyngen Alps
    fn_list = sorted(glob.glob(data_path + f"*{region}*.nc"), key=str.casefold)

    if len(fn_list) == 0:
        sys.exit(f"No no seNorge data for {reg_code} between {h_low} and {h_hi} m. Stopping execution.")
    # end if

    # load the nc file and the variable --> average over the locations
    nc = xr.open_dataset(fn_list[0])

    swe = nc.SWE
    swe = agg_gridcells(swe, agg_type=agg_type, perc=perc, axis=0)

    sdepth = nc.SnowDepth
    sdepth = agg_gridcells(sdepth, agg_type=agg_type, perc=perc, axis=0)

    sdens = nc.SnowDens
    sdens = agg_gridcells(sdens, agg_type=agg_type, perc=perc, axis=0)

    me_re = nc.Melt_Refreeze
    me_re = agg_gridcells(me_re, agg_type=agg_type, perc=perc, axis=0)

    # load the date
    date = nc.time

    #% set up the hourly dataframe with the multi-gridcell averaged values
    df_atm = pd.DataFrame({"date":date, "SWE":swe, "SnowDepth":sdepth, "SnowDens":sdens, "MeltRefr":me_re})

    df_atm.set_index("date", inplace=True)

    #% aggregate daily and calculate several quantities
    df_atm_day = pd.DataFrame({"date":pd.to_datetime(df_atm.groupby(df_atm.index.date).mean().index),
                               "SWE":df_atm["SWE"].groupby(df_atm.index.date).mean(),
                               "SnowDepth":df_atm["SnowDepth"].groupby(df_atm.index.date).mean(),
                               "SnowDens":df_atm["SnowDens"].groupby(df_atm.index.date).mean(),
                               "MeltRefr":df_atm["MeltRefr"].groupby(df_atm.index.date).mean()})

    df_atm_day.set_index(pd.to_datetime(df_atm_day.index), inplace=True)


    # loop over the available dates and set up the array
    # --> we are using the whole year so this is rather pointless
    # --> it is left in for possible later revision
    in_season_dates = []
    for idate in df_atm_day.index:
        if idate.month in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            in_season_dates.append(idate)
        # end if
    # end for idate


    #% set up a the predictors dataframe with the in_season_dates as index
    df_pred = pd.DataFrame({"date":in_season_dates})


    #% basically: extract the in-season dates from the df_atm_day dataset
    df_pred = df_atm_day.merge(df_pred, how="inner", on="date")
    df_pred.set_index(pd.to_datetime(df_pred.date), inplace=True)


    swe2, swe3, swe4, swe5, swe6, swe7= [], [], [], [], [], []
    sd2, sd3, sd4, sd5, sd6, sd7= [], [], [], [], [], []
    sdens2, sdens3, sdens4, sdens5, sdens6, sdens7= [], [], [], [], [], []
    mr2, mr3, mr4, mr5, mr6, mr7= [], [], [], [], [], []


    count = 0
    l = len(df_pred.date)
    print_progress_bar(0, l, prefix='Progress:', suffix='Complete', length=50)
    for da in df_pred.date:

        # find the index for the date of the event
        ind = np.where(df_atm_day.date == da)[0][0]

        # calculate the mean of SWE over multiple days
        swe3.append(np.mean(df_atm_day["SWE"].iloc[ind-2:ind+1]))
        swe7.append(np.mean(df_atm_day["SWE"].iloc[ind-6:ind+1]))

        # calculate the mean of snow depth
        sd3.append(np.mean(df_atm_day["SnowDepth"].iloc[ind-2:ind+1]))
        sd7.append(np.mean(df_atm_day["SnowDepth"].iloc[ind-6:ind+1]))

        # calculate the mean snow density
        sdens3.append(np.mean(df_atm_day["SnowDens"].iloc[ind-2:ind+1]))
        sdens7.append(np.mean(df_atm_day["SnowDens"].iloc[ind-6:ind+1]))

        # calculate the mean melt or refreeze rate
        mr3.append(np.mean(df_atm_day["MeltRefr"].iloc[ind-2:ind+1]))
        mr7.append(np.mean(df_atm_day["MeltRefr"].iloc[ind-6:ind+1]))

        if unclean:
            swe2.append(np.mean(df_atm_day["SWE"].iloc[ind-1:ind+1]))
            swe4.append(np.mean(df_atm_day["SWE"].iloc[ind-3:ind+1]))
            swe5.append(np.mean(df_atm_day["SWE"].iloc[ind-4:ind+1]))
            swe6.append(np.mean(df_atm_day["SWE"].iloc[ind-5:ind+1]))
            sd2.append(np.mean(df_atm_day["SnowDepth"].iloc[ind-1:ind+1]))
            sd4.append(np.mean(df_atm_day["SnowDepth"].iloc[ind-3:ind+1]))
            sd5.append(np.mean(df_atm_day["SnowDepth"].iloc[ind-4:ind+1]))
            sd6.append(np.mean(df_atm_day["SnowDepth"].iloc[ind-5:ind+1]))
            sdens2.append(np.mean(df_atm_day["SnowDens"].iloc[ind-1:ind+1]))
            sdens4.append(np.mean(df_atm_day["SnowDens"].iloc[ind-3:ind+1]))
            sdens5.append(np.mean(df_atm_day["SnowDens"].iloc[ind-4:ind+1]))
            sdens6.append(np.mean(df_atm_day["SnowDens"].iloc[ind-5:ind+1]))
            mr2.append(np.mean(df_atm_day["MeltRefr"].iloc[ind-1:ind+1]))
            mr4.append(np.mean(df_atm_day["MeltRefr"].iloc[ind-3:ind+1]))
            mr5.append(np.mean(df_atm_day["MeltRefr"].iloc[ind-4:ind+1]))
            mr6.append(np.mean(df_atm_day["MeltRefr"].iloc[ind-5:ind+1]))
        # end if

        count += 1
        print_progress_bar(count, l, prefix='Progress:', suffix='Complete', length=50)
    # end for da

    print(f"\nWent through {count} iterations.\n")

    #% add the accrued quantities to the dataframe
    df_pred["SWE3"] = np.array(swe3)
    df_pred["SWE7"] = np.array(swe7)
    df_pred["SnowDepth3"] = np.array(sd3)
    df_pred["SnowDepth7"] = np.array(sd7)
    df_pred["SnowDens3"] = np.array(sdens3)
    df_pred["SnowDens7"] = np.array(sdens7)
    df_pred["MeltRefr3"] = np.array(mr3)
    df_pred["MeltRefr7"] = np.array(mr7)

    if unclean:
        df_pred["SWE2"] = np.array(swe2)
        df_pred["SWE4"] = np.array(swe4)
        df_pred["SWE5"] = np.array(swe5)
        df_pred["SWE6"] = np.array(swe6)
        df_pred["SnowDepth2"] = np.array(sd2)
        df_pred["SnowDepth4"] = np.array(sd4)
        df_pred["SnowDepth5"] = np.array(sd5)
        df_pred["SnowDepth6"] = np.array(sd6)
        df_pred["SnowDens2"] = np.array(sdens2)
        df_pred["SnowDens4"] = np.array(sdens4)
        df_pred["SnowDens5"] = np.array(sdens5)
        df_pred["SnowDens6"] = np.array(sdens6)
        df_pred["MeltRefr2"] = np.array(mr2)
        df_pred["MeltRefr4"] = np.array(mr4)
        df_pred["MeltRefr5"] = np.array(mr5)
        df_pred["MeltRefr6"] = np.array(mr6)
    # end if

    return df_pred

# end calc_seNorge_pred



