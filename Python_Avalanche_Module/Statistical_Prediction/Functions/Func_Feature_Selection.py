#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function for selection of avalanche danger prediction features based on random forest feature importance and Pearson
cross-correlation.
"""


# imports
import numpy as np
import pandas as pd

# function
def feat_sel(feats, cross_c, importances, thres=0.9, r_coeff="R**2", verbose=False):

    """
    Selection of avalanche danger prediction features based on random forest or decision tree feature importance and
    Pearson cross-correlation.

    Parameters:
        feats        Array of strings. All features the random forest was trained on.
        cross_c      Pandas dataframe. The cross-correlation matrix as containing the cross-correlation Rs of all
                                       features in feats.
        importances  Array of floats. The feature importances of all the features derived from the random forest.
        thres        Float. The threshold of the cross-correlation R above which features are dropped. Defaults to 0.9
        r_coef       String. Either R, R**2, or R^2 (the latter two are equivalent). Controls if the R or squared R is
                             used to drop features. Defaults to R**2.
        verbose      Logical. If True, a print statement reporting on the number of iterations is issued. Defaults to
                              False.
    """

    # depending on r_coeff set the power of the R coefficient
    if r_coeff == "R":
        r_power = 1
    elif (r_coeff == "R**2") | (r_coeff == "R^2"):
        r_power = 2
    # end if elif

    # get the sorting based on the importances
    imp_sort = np.argsort(importances)[::-1]

    # generate a Pandas dataframe to associate all features with their importances
    df = pd.DataFrame({"importance":importances[imp_sort]}, index=feats[imp_sort])
    df.index.name = "feature"

    # select the first feature as the start
    feat1 = feats[imp_sort][0]

    best_feats = [feat1]

    feats_low_corr = feats[imp_sort]

    for i in range(len(feats)):

        coll_feats = []

        for feat2 in feats_low_corr:
            if feat2 in best_feats:
                continue
            # end if

            if np.abs(cross_c[feat1][feat2]**r_power) >= thres:
                continue
            else:
                coll_feats.append(feat2)
            # end if else
        # end for feat2

        if len(coll_feats) == 0:
            if verbose:
                print(f"\nStopping after {i+1} iterations: No more features with {r_coeff} < {thres} \n")
            # end if
            break
        # end if

        feats_low_corr = coll_feats

        feat1 = coll_feats[0]

        best_feats.append(feat1)
        # end for

    # end for

    return df.loc[best_feats]

# end def
