#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function for balancing a data set so it can be used as training/test data for a statistical model.
"""

# imports
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler, SVMSMOTE, KMeansSMOTE, BorderlineSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neighbors import NearestNeighbors

# helper balance function
def undersample(data_x, data_y, excl_fewer=100):

    """
    Balances the data by randomly selecting the number of elements corresponding to the number of elements in the
    smallest class (in so far it is larger than the number given in excl_fewer) from each individual class.

    Parameters:
        excl_fewer  Integer. Classes with a smaller number of elements than the number given here are excluded from the
                             balancing.
    """

    # print(f"\nClasses smaller than {excl_fewer} are excluded from the balancing.\n")

    y_unique = np.unique(data_y)

    data_ys = []
    data_xs = []
    data_lens = []
    y_unique_red = []
    for i_param in y_unique:
        ind = (data_y == i_param)
        y_temp = data_y[ind]
        x_temp = data_x[ind]

        if len(y_temp) < excl_fewer:
            # print(f"\n{col_name} value {i_param} has only {len(trainy_temp)} and will be excluded.\n")
            continue
        # end if
        data_ys.append(y_temp)
        data_xs.append(x_temp)
        data_lens.append(len(y_temp))
        y_unique_red.append(i_param)
    # end for i_param

    # return data_ys

    sampl_len = np.min(data_lens)

    # generate the random permutations for the samples
    ys_list = []
    xs_list = []

    for i, i_param in enumerate(y_unique_red):

        # get the permutation
        # data_perm = np.random.randint(low=0, high=data_lens[i], size=sampl_len)
        data_perm = np.random.choice(data_lens[i], size=sampl_len, replace=False)

        # extract the samples
        if np.ndim(data_x) > 1:
            xs_list.append(data_xs[i].iloc[data_perm, :])
        else:
            xs_list.append(data_xs[i].iloc[data_perm])
        # end if else
        ys_list.append(data_ys[i].iloc[data_perm])

    # end for i, i_param

    # concatenate the samples
    # x_balanced = np.concatenate(xs_list, axis=0)
    # y_balanced = np.concatenate(ys_list, axis=0)
    x_balanced = pd.concat(xs_list, axis=0)
    y_balanced = pd.concat(ys_list, axis=0)

    return x_balanced, y_balanced
# end def

# helper balance function
def balance_data(data_x, data_y, method="SMOTE", sample_strat="auto", excl_fewer=100, k_neighbors=10, n_jobs=-1):

    """
    Balances the data by randomly selecting the number of elements corresponding to the number of elements in the
    smallest class (in so far it is larger than the number given in excl_fewer) from each individual class.

    Parameters:
        method      String. Set the method of balancing to be used. Choices are the following:
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
        excl_fewer  Integer. Classes with a smaller number of elements than the number given here are excluded from the
                             balancing. Only used if method="undersample".
        k_neighbors   Integer. The number of neighbouring values SMOTE or ADASYN use to generate synthetic values.
                               Defaults to 5 and is only used if SMOTE or ADASYN is used as balancing method.
        n_jobs        Integer. Number of CPU cores used during the cross-validation loop. Defaults to -1, meaning all
                               all available cores will be used. Only used for SMOTE and ADASYN.
    """

    if method == "undersample":
        x_balanced, y_balanced = undersample(data_x, data_y, excl_fewer)
    elif method == "SMOTE":
        # set up a nearest-neighbour classifier separately
        nn = NearestNeighbors(n_neighbors=k_neighbors, algorithm='ball_tree', n_jobs=n_jobs)
        smote = SMOTE(random_state=42, k_neighbors=nn, sampling_strategy=sample_strat)
        x_balanced, y_balanced = smote.fit_resample(X=data_x, y=data_y)
    elif method == "BSMOTE":
        # set up a nearest-neighbour classifier separately
        # nn = NearestNeighbors(n_neighbors=k_neighbors, algorithm='ball_tree', n_jobs=n_jobs)
        smote = BorderlineSMOTE(random_state=42, k_neighbors=k_neighbors, sampling_strategy=sample_strat)
        x_balanced, y_balanced = smote.fit_resample(X=data_x, y=data_y)
    elif method == "SVMSMOTE":
        smote = SVMSMOTE(random_state=42, k_neighbors=k_neighbors, sampling_strategy=sample_strat)
        x_balanced, y_balanced = smote.fit_resample(X=data_x, y=data_y)
    elif method == "KMeansSMOTE":
        smote = KMeansSMOTE(random_state=42, k_neighbors=k_neighbors, n_jobs=n_jobs, sampling_strategy=sample_strat)
        x_balanced, y_balanced = smote.fit_resample(X=data_x, y=data_y)
    elif method == "ADASYN":
        adasyn = ADASYN(random_state=42, n_neighbors=k_neighbors, sampling_strategy=sample_strat)
        x_balanced, y_balanced = adasyn.fit_resample(data_x, data_y)
    elif method == "ros":
        ros = RandomOverSampler(random_state=42)
        x_balanced, y_balanced = ros.fit_resample(data_x, data_y)
    elif method == "rus":
        rus = RandomUnderSampler(random_state=42)
        x_balanced, y_balanced = rus.fit_resample(data_x, data_y)
    elif method == "SMOTEENN":
        smote = SMOTEENN(random_state=42, sampling_strategy=sample_strat)
        x_balanced, y_balanced = smote.fit_resample(X=data_x, y=data_y)
    elif method == "SMOTETomek":
        smote = SMOTETomek(random_state=42, sampling_strategy=sample_strat)
        x_balanced, y_balanced = smote.fit_resample(X=data_x, y=data_y)
    # end if elif

    return x_balanced, y_balanced
# end def