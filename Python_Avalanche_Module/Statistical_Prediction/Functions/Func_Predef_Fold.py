#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function for setting up a predefined k-fold depending on season.
"""

# imports
import numpy as np
import pandas as pd
from sklearn.model_selection import PredefinedSplit
from .Func_Assign_Winter_Year import assign_winter_year

# function
def sea_fold(all_x, nfolds=3):

    # assign winter month
    all_winter = pd.Series(assign_winter_year(all_x.index, 8))
    obs_yrs = np.unique(all_winter)

    # assign fold labels --> depends on the year
    len_fold = int(len(obs_yrs) / nfolds)

    # the fold labels indicate which elements are part of the test (!) data for fold x
    # so if our data are
    # all_data = [1, 2, 3, 4, 5, 6, 7, 8]
    # and our folds labels are
    # fold_labels = [0, 0, 0, 0, 1, 1, 1, 1]
    # then the first four entries are the test data for the first fold the last four entries are the test data for the
    # second fold

    fold_label = np.zeros(len(all_winter))
    # OLD: for i, yr in enumerate(obs_yrs[::len_fold]):
    for i, j in enumerate(np.arange(len(obs_yrs))[::len_fold]):
        # OLD: yrs_temp = [yr_temp for yr_temp in np.arange(yr, yr+len_fold, 1)]
        yrs_temp = [obs_yrs[k] for k in np.arange(j, j+len_fold, 1)]
        fold_label[all_winter.isin(yrs_temp)] = i
    # end for i, yr

    # generate the predefined split
    ps = PredefinedSplit(fold_label)

    return ps

# end def