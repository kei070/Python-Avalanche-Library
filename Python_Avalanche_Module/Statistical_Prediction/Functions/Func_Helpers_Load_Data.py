#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A few helper functions for loading the features.
"""

# imports
import numpy as np
from .Func_Balance_Data import balance_data
from .Func_DatetimeSimple import date_dt


# extract a season
def extract_sea(all_df, sel_feats, split, balance_meth, target_n="y", k_neighbors=5, n_jobs=-1):

    """
    Parameters:

        sel_feats     List of features to select.
        split         Integer or float. Provide e.g. a list of years (e.g., [2021, 2023]) which will be extracted as
                                        test data.
        balance_meth  String. Set the method of balancing to be used. Choices are the following:
                              -None: no balancing
                              -undersample: [DOES NOT WORK ANYMORE] uses the custom undersample function
                              -SMOTE: uses the synthetic minority oversampling method from the imbalanced-learn library
                                      (default)
                              -SVMSMOTE: same as SMOTE but using an SVM algorithm to detect sample to use for generating
                                         new synthetic samples.
                              -KMeansSMOTE: Same as SMOTE but applies a KMeans clustering before to over-sample using
                                            SMOTE.
                              -ADASYN: uses the adaptive synthetic sampling method from the imbalanced-learn library
                              -ros: uses the random oversampling method from the imbalanced-learn library
                              -rus: uses the random undersampling method from the imbalanced-learn library
        target_n      String. The name of the target variable.
        k_neighbors   Integer. The number of neighbouring values SMOTE or ADASYN use to generate synthetic values.
                               Defaults to 5 and is only used if SMOTE or ADASYN is used as balancing method.
        n_jobs        Integer. Number of CPU cores used during the cross-validation loop. Defaults to -1, meaning all
                               all available cores will be used. Only used for SMOTE and ADASYN.
    """

    # make sure split is a list so that iteration is possible
    if type(split) == int:
        split = [split]
    # end if

    test_all_inds = []

    for sp in split:

        # make sure sp is an integer
        sp = int(sp)

        # extract the data for the requested avalanche season
        test_all_inds.append((all_df.index > date_dt(sp-1, 7, 1)) & (all_df.index < date_dt(sp, 7, 1)))

    # end for sp

    test_all_inds = np.logical_or.reduce(test_all_inds)

    test_all_df = all_df[test_all_inds]
    train_all_df = all_df[~test_all_inds]

    train_x_all = train_all_df[sel_feats]
    train_y_all = train_all_df[target_n]
    test_x_all = test_all_df[sel_feats]
    test_y_all = test_all_df[target_n]

    # return train_x_all, test_x_all, train_y_all, test_y_all

    # perform the balancing if requested
    if str(balance_meth) != "None":
        train_x, train_y = balance_data(train_x_all, train_y_all, method=balance_meth, k_neighbors=k_neighbors,
                                        n_jobs=n_jobs)
        test_x, test_y = balance_data(test_x_all, test_y_all, method=balance_meth, k_neighbors=k_neighbors,
                                      n_jobs=n_jobs)

        return train_x, test_x, train_y, test_y, train_x_all, test_x_all, train_y_all, test_y_all
    else:
        return train_x_all, test_x_all, train_y_all, test_y_all
    # end if else
# end def


# extract a region
def extract_reg(all_df, sel_feats, split, balance_meth, k_neighbors=5, target_n="y", n_jobs=-1):

    """
    See extract_sea for documentation.
    """

    # make sure split is a list so that iteration is possible
    if type(split) == int:
        split = [split]
    # end if

    test_all_inds = []

    for sp in split:

        # make sure sp is an integer
        sp = int(sp)

        # extract the data for the requested region
        test_all_inds.append(all_df["reg_code"] == sp)

    # end for sp

    test_all_inds = np.logical_or.reduce(test_all_inds)

    test_all_df = all_df[test_all_inds]
    train_all_df = all_df[~test_all_inds]

    # extract x and y data
    train_x_all = train_all_df[sel_feats]
    train_y_all = train_all_df[target_n]
    test_x_all = test_all_df[sel_feats]
    test_y_all = test_all_df[target_n]

    # perform the balancing if requested
    if str(balance_meth) != "None":
        train_x, train_y = balance_data(train_x_all, train_y_all, method=balance_meth, k_neighbors=k_neighbors,
                                        n_jobs=n_jobs)
        test_x, test_y = balance_data(test_x_all, test_y_all, method=balance_meth, k_neighbors=k_neighbors,
                                      n_jobs=n_jobs)

        return train_x, test_x, train_y, test_y, train_x_all, test_x_all, train_y_all, test_y_all
    else:
        return train_x_all, test_x_all, train_y_all, test_y_all
    # end if else

# end def