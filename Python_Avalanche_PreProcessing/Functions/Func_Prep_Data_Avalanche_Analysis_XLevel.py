#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function for loading and preparing/preprocessing the predictores/features for the analysis with the statistical models.
"""

# imports
import sys
import glob
import copy
import numpy as np
import pandas as pd

from Lists_and_Dictionaries.Features import features as all_features
from Lists_and_Dictionaries.Avalanche_Problems import ava_ps
from Lists_and_Dictionaries.Region_Codes import regions
from Lists_and_Dictionaries.Paths import path_par
from .Func_Discrete_Hist import disc_hist

# define the function
def load_feats_xlevel(reg_codes, ndlev=2, exposure=None, sel_feats=None,
                      a_p="danger_level",
                      start_date="09-2017", end_date="09-2023",
                      h_low=400, h_hi=900,
                      agg_type="mean",
                      perc=90,
                      out_type="array",
                      plot_y=False,
                      data_path_par=f"{path_par}/IMPETUS/NORA3/Avalanche_Predictors/",
                      pred_pl_path=""):

    """
    Parameters:
        reg_codes      List of integers. Region code(s) of the avalanche regions.
        ndlev          Number of danger levels. Possible levels are 2, 3, and 4.
        exposure       Either "west" or "east". Use only grid cells with western or eastern exposure, respectively.
                       Defaults to None, meaning all available grid cells are used. NOT USED
        sel_feats      List of features to be loaded.
        a_p            String or List. The avalanche problem for which the danger level is chosen. Defaults to
                               "danger_level", implying the general danger level. Use "all" to load all avalanche
                               problems. Or use a list of requested avalanche problems. Note that if more than one
                               avalanche problem is used no balancing is performed and the balanced data are returned
                               as NaN.
        agg_type       String. The type of the aggregation of the grid cells. Either mean, median, or percentile.
                               Defaults to mean.
        p              Integer. The percentile if the grid-cell aggregation type is percentile. Not used if the agg_type
                                is not percentile.
        out_type       String. Output type, either "array" or "dataframe" (selfexplanatory). Note that the output is a
                               dictionary in both cases, but either a dictionary of numpy arrays or pandas dataframes.
                               Note also that the keys are different for the two options. Defaults to "array".
        plot_y         Boolean. If True, a bar-plot of the y-values will be produced.
        data_path_par  Parent data path for the predictors/features.

    More information on the parameters:
        reg_code = 3009  # Nord-Troms
        reg_code = 3010  # Lyngen
        reg_code = 3011  # Tromsoe
        reg_code = 3012  # Soer-Troms
        reg_code = 3013  # Indre Troms
    """

    # if no features to select are provided use all features available
    if sel_feats == None:
        sel_feats = all_features
    # end if

    # if a_p == "all" load the list of avalanche problems, else make sure a_ps is a list
    if type(a_p) is list:
        a_ps = copy.deepcopy(a_p)

        # prepare the output
        a_p_out = ["y"] if a_p == ["danger_level"] else a_p
    elif a_p == "all":
        a_ps = copy.deepcopy(ava_ps)
        a_ps.insert(0, "danger_level")

        # prepare the output
        a_p_out = copy.deepcopy(ava_ps)
        a_p_out.insert(0, "y")
    else:
        a_ps = [a_p]

        # prepare the output
        a_p_out = ["y"] if a_p == "danger_level" else [a_p]
    # end if else

    # generate a name prefix/suffix depending on the gridcell aggregation
    # make sure that the percentile is 0 if the type is not percentile
    if agg_type != "percentile":
        perc = 0
    # end if

    agg_str = f"{agg_type.capitalize()}{perc if perc != 0 else ''}"

    # use western/eastern exposure condition?
    if exposure == "west":
        expos = "w_expos"
        expos_add = "_WestExposed"
    elif exposure == "east":
        expos = "e_expos"
        expos_add = "_EastExposed"
    else:
        expos_add = ""
    # end if

    # loop over the region codes and load the data
    data_x_l = []
    data_y_l = []

    dates_l = []
    dates_l = []

    reg_code_l = []

    for reg_code in reg_codes:

        # set region name according to region code
        region = regions[reg_code]

        # NORA3 data path
        data_path = data_path_par

        # load the data
        data_fns = sorted(glob.glob(data_path + f"/{agg_str}/Between{h_low}_and_{h_hi}m" +
                                  f"/{region}_Predictors_MultiCell{agg_str}_Between{h_low}_and_{h_hi}m{expos_add}.csv"))
        if len(data_fns) == 0:
            print(f"No predictors for {reg_code} between {h_low} and {h_hi} m. Continuing.")
            continue
        # end if

        # select and load a dataset
        data_i = 0
        data_fn = data_fns[data_i]

        data = pd.read_csv(data_fn)

        # convert date column to datetime
        data["date"] = pd.to_datetime(data["date"])

        # select the training data from training data set
        data_sub = data[all_features]

        # add the x and y data to list
        data_x_l.append(data_sub)
        data_y_l.append(np.array(data[a_ps]))

        # store the dates of the train and test data
        dates_l.append(data.date)

        # add the region coed
        reg_code_l.append(data["region"])

    # end for reg_code

    if len(data_x_l) == 0:
        sys.exit()
    # end if

    x_df = pd.concat(data_x_l)
    x_df["reg_code"] = pd.concat(reg_code_l, axis=0)
    data_y = np.concatenate(data_y_l)

    dates = pd.concat(dates_l)

    # convert the predictands to binary, 3-level, or 4-level
    # IMPORTANT: Like Python, we start at 0! --> the original danger-level list starts at 1
    y_bin = np.zeros(np.shape(data_y))

    if ndlev == 2:
        y_bin[data_y > 2] = 1
        # remainder is zero
    elif ndlev == 3:
        y_bin[data_y == 2] = 1
        y_bin[data_y > 2] = 2
        # remainder is zero
    elif ndlev == 4:
        y_bin[data_y == 2] = 1
        y_bin[data_y == 3] = 2
        y_bin[data_y > 3] = 3
        # remainder is zero
    # end if elif

    # make sure that instances of previous NaN remain NaN
    y_bin[np.isnan(data_y)] = np.nan

    # plot the number of the different values (0, 1, 2, 3) in the new classification
    if plot_y:
        disc_hist([y_bin])
    # end if

    # balancing can only be undertaken for one target varialbe, i.e., not for multiple avalanche problems at a time
    if len(a_ps) == 1:

        # squeeze to make the following operations possible
        # --> y_bin will be returned to its original shape at the end of this to ensure compatibility with the case
        #     without balancing
        # --> there are probably more elegant ways to do this
        y_bin = np.squeeze(y_bin)

        # first find out which level occurs the least often
        n_data = np.min([np.sum(y_bin == dlev) for dlev in np.arange(ndlev)])

        # to make sure the accuracy metrics are not just artifacts make sure there the data contain the same number of
        # danger levels
        y_bal = []
        dates_bal = []
        x_bal = []

        for dlev in np.arange(ndlev):

            # get the number of days with level dlev
            n_x = np.sum(y_bin == dlev)

            # select a random sample of length n_train as the training data -- NEW
            perm = np.random.choice(n_x, size=n_data, replace=False)

            # permute the predictands to get balanced predictands data
            y_bal.append(y_bin[y_bin == dlev][perm])

            # permutate the dates to get the dates of the balanced data
            dates_bal.append(dates.iloc[y_bin == dlev].iloc[perm])

            # extract the selected features
            x_bal.append(np.array(x_df[sel_feats])[y_bin == dlev, :][perm, :])

        # end for dlev
        y_bal = np.concatenate(y_bal)
        dates_bal = np.concatenate(dates_bal)
        x_bal = np.concatenate(x_bal)

        # return y_bin to its original shape to ensure compatibility with the code outside the if condition
        y_bin = y_bin[:, None]

    else:
        y_bal = [np.nan]
        dates_bal = [0]
        x_bal = {k:[np.nan] for k in sel_feats}
    # end if else

    x_all = np.array(x_df[sel_feats + ["reg_code"]])

    if out_type == "dataframe":
        output = {}

        # balanced data
        temp = pd.DataFrame(x_bal)
        temp.columns = sel_feats
        temp["date"] = dates_bal
        temp.set_index("date", inplace=True)
        temp["y_balanced"] = y_bal
        output["balanced"] = temp

        # all data
        temp = pd.DataFrame(x_all)
        temp.columns = sel_feats + ["reg_code"]
        temp.set_index(dates, inplace=True)
        for i, i_ap in enumerate(a_p_out):
            temp[i_ap] = y_bin[:, i]
        # end for i, i_ap
        output["all"] = temp

    elif out_type == "array":
        # add data to the function output

        # balanced data
        output = {"x_balanced":x_bal,
                  "x_all":x_all,
                  "dates_balanced":dates_bal,
                  "y_balanced":y_bal,
                  "y_all":y_bin,
                  "dates_all":dates}

        # all data
        output = {"x_all":x_all,
                  "y_all":y_bin,
                  "dates_all":dates}
    # end if elif

    # return
    return output

# end def
