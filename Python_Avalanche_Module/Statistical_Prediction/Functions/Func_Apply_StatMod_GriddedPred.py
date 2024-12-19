#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function for applying the stat_mod to the gridded predictors.
"""

# imports
import numpy as np
from joblib import load
from .Func_StatMod import stat_mod
from .Func_Helpers_Load_Data import extract_sea

# function
def apply_stat_mod_gridded(model_ty="RF", ndlev=4, reg_code=3009, perc=99, split=[2021, 2023], bal_meth="SVMSMOTE",
                           hyperp="", grid_search=False, grid_sample=0, class_weight=None, cv=3, return_model=False,
                           pred_path=""):

    """
    Function for applying the stat_mod to the gridded predictors.

    For now it is assumed that balancing of the data is requested.
    """

    # set the name of the y variable
    y_name = "danger_level"

    # load the predictors for the best grid cells, i.e., the cells with the highest accuracies in the intial training
    df_dl_and_pred = load(pred_path + f"NORA3_Gridded_Predictors_Best{perc}th_Perc.joblib")[reg_code]

    # get the features and remove the unneeded ones
    sel_feats = list(df_dl_and_pred.columns)

    # remove unnecessary features --> the accuracies and the danger_level
    # --> leave elevation in for now
    sel_feats.remove("test_all_acc")
    sel_feats.remove("test_bal_acc")
    sel_feats.remove("danger_level")

    # remove NaNs
    n_nan = np.sum(df_dl_and_pred.isna().any(axis=1))
    # print(f"\n{n_nan} rows with NaNs exist (={n_nan/len(df_dl_and_pred)*100:.1f}%) in the dataset. Dropping them...")
    suff = f"\n{n_nan:10} NaN-rows (={n_nan/len(df_dl_and_pred)*100:5.1f}%) removed\n"
    df_dl_and_pred.dropna(inplace=True)

    # print the number of removed NaNs
    print(suff)

    # ad-hoc adjustment to get to the 2- or 4-ADL case
    if ndlev == 4:
        df_dl_and_pred.loc[df_dl_and_pred["danger_level"] == 5, "danger_level"] = 4
        df_dl_and_pred["danger_level"] = df_dl_and_pred["danger_level"] - 1
    elif ndlev == 2:
        df_dl_and_pred.loc[df_dl_and_pred["danger_level"] < 3, "danger_level"] = 0
        df_dl_and_pred.loc[df_dl_and_pred["danger_level"] >= 3, "danger_level"] = 1
    # end if elif

    # perform the data split
    train_x, test_x, train_y, test_y, train_x_all, test_x_all, train_y_all, test_y_all = \
                                              extract_sea(df_dl_and_pred, sel_feats, split=split, balance_meth=bal_meth,
                                                                                 target_n="danger_level", k_neighbors=5)


    # define the model
    model = stat_mod(model_ty=model_ty, ndlev=ndlev, hyperp=hyperp, grid_search=grid_search,
                     grid_sample=grid_sample, class_weight=class_weight,
                     train_x=df_dl_and_pred[sel_feats], train_y=df_dl_and_pred[y_name], cv=cv)

    # fit the model
    model.fit(train_x, train_y)

    # perform the prediction
    pred_test = model.predict(test_x)
    pred_train = model.predict(train_x)
    pred_train_all = model.predict(train_x_all)
    pred_test_all = model.predict(test_x_all)

    if return_model:
        return pred_test, pred_train, pred_test_all, pred_train_all, train_x, train_y, test_x, test_y, train_x_all,\
                                                                              train_y_all, test_x_all, test_y_all, model
    else:
        return pred_test, pred_train, pred_test_all, pred_train_all, train_x, train_y, test_x, test_y, train_x_all,\
                                                                                     train_y_all, test_x_all, test_y_all
    # end if else

# end def