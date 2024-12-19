#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Set the hyperparameters dynamically in reponse to user input. The function is intended for usage in stat_mod
"""

import sys
from ..Lists_and_Dictionaries.Hyperp_Set import hypp_grid


# function
def set_hyperp(model_ty="RF", in_hypp=hypp_grid, grid_search=False, grid_sample=0, pipe=False, ndlev=0, verbose=True):

    """
    THE GRID SEARCH IS SO FAR ONLY IMPLEMENTED FOR THE RANDOM FOREST MODEL!
    Sets the hyperparameters dynamically in reponse to user input. The function is intended for usage in stat_mod.

    Parameters:
        model_ty     String. The type of statistical model that is used. Choices are RF, DT, LR, SVM, KNN.
                             Defaults to RF.
        in_hypp      Dictionary containing the hyperparameter(grids). Defaults to an empty dictionary.
        grid_search  Logical. If True, a grid search will be performed.
        grid_sample  Integer. If > 0 a large grid of hyperparameters is set up.
        pipe         Logical. If True, the hyperparameters are named such that they can be used in a pipeline. If False
                              (default) the hyperparameters are as used in the models.
        ndlev        Integer. The number of danger levels in the prediction. This can be used select some pre-set
                              hyperparameters or grids. Defaults to 0, meaning that some default set/grid is chosen.
                              Otherwise use either 2 or 4.
    """

    # since this function is only implemented for the RF model...
    if ((model_ty != "RF") & grid_search & (len(in_hypp) == 0)):
        cont = input("""\nDefault settings for a grid search are only implemented for the RF model.
                     Do you want to continue anyway? [Y/n]\n""")
        if cont.lower() in ("n", "no"):
            sys.exit("Stopping execution.")
        # end if
    # end if

    # set the default hyperparameters or hyperparameters grids depending on the chosen model
    if model_ty == "RF":
        # default hyperparameters
        # exp_hypps = {"n_estimators":200, "max_depth":15, "min_samples_leaf":10, "min_samples_split":10,
        #              "max_features":1.0, "bootstrap":True}
        if pipe:
            prefix = "randomforestclassifier__"
        else:
            prefix = ""
        # end if

        if ndlev == 0:
            # print("\nUsing default hyperparameter set/grid.\n")

            # more or less random grid of hyperparameters
            exp_hypps = {f"{prefix}n_estimators":1000, f"{prefix}max_depth":15, f"{prefix}min_samples_leaf":10,
                         f"{prefix}min_samples_split":10,
                         f"{prefix}max_features":"log2", f"{prefix}bootstrap":True}
            # used in the randomised grid search with 100 samples
            default_grids = {
                            f'{prefix}n_estimators': [200, 300, 400, 500, 600],  # [100, 200, 300],
                            f'{prefix}max_depth': [5, 10, 20, 30, 35],  # [2, 5, 8, 11, 14, 17, 20],
                            f'{prefix}min_samples_leaf': [2, 3, 4, 5],
                            f'{prefix}min_samples_split': [2, 5, 7],
                            f'{prefix}max_features': [1.0, "sqrt", "log2"],
                            f'{prefix}bootstrap': [True]
                            }

        elif ndlev == 2:
            # print("\nUsing pre-set hyperparameter set/grid for binary case.\n")

            # set for binary ADL case
            """
            # --> agg_type = mean
            exp_hypps = {f"{prefix}n_estimators":900, f"{prefix}max_depth":20, f"{prefix}min_samples_leaf":9,
                         f"{prefix}min_samples_split":2,
                         f"{prefix}max_features":"sqrt", f"{prefix}bootstrap":True}
            # grid for binary ADL case
            """
            # --> agg_type = percentile, perc = 90
            exp_hypps = {f"{prefix}n_estimators":350, f"{prefix}max_depth":70, f"{prefix}min_samples_leaf":2,
                         f"{prefix}min_samples_split":12,
                         f"{prefix}max_features":"log2", f"{prefix}bootstrap":True}
            """# --> targeted search for agg_type = mean
            default_grids = {
                            f'{prefix}n_estimators': [700, 750, 800, 850, 900],  # [100, 200, 300],
                            f'{prefix}max_depth': [10, 20, 30, 40, 50, 60],  # [2, 5, 8, 11, 14, 17, 20],
                            f'{prefix}min_samples_leaf': [8, 9, 10, 11, 12],
                            f'{prefix}min_samples_split': [2, 3, 4],
                            f'{prefix}max_features': [1.0, "sqrt", "log2"],
                            f'{prefix}bootstrap': [True]
                             }
            """
            # --> targeted search for agg_type = percential, perc = 90
            default_grids = {
                            f'{prefix}n_estimators': [200, 250, 300, 350, 400],  # [100, 200, 300],
                            f'{prefix}max_depth': [70, 75, 80, 85, 90],  # [2, 5, 8, 11, 14, 17, 20],
                            f'{prefix}min_samples_leaf': [2, 3, 4, 5],
                            f'{prefix}min_samples_split': [12, 13, 14, 15, 16],
                            f'{prefix}max_features': [1.0, "sqrt", "log2"],
                            f'{prefix}bootstrap': [True]
                             }
        elif ndlev == 4:
            # print("\nUsing pre-set hyperparameter set/grid for 4-ADL case.\n")

            # set for 4-ADL case
            """
            # --> agg_type = mean
            exp_hypps = {f"{prefix}n_estimators":500, f"{prefix}max_depth":35, f"{prefix}min_samples_leaf":2,
                         f"{prefix}min_samples_split":4,
                         f"{prefix}max_features":1.0, f"{prefix}bootstrap":True}
            """
            # --> agg_type = percentile, perc = 90
            exp_hypps = {f"{prefix}n_estimators":200, f"{prefix}max_depth":30, f"{prefix}min_samples_leaf":2,
                         f"{prefix}min_samples_split":2,
                         f"{prefix}max_features":"sqrt", f"{prefix}bootstrap":True}
            # grid for 4-ADL case
            """# --> targeted search for agg_type = mean
            default_grids = {
                            f'{prefix}n_estimators': [500, 550, 600, 650, 700],  # [100, 200, 300],
                            f'{prefix}max_depth': [35, 40, 45, 50],  # [2, 5, 8, 11, 14, 17, 20],
                            f'{prefix}min_samples_leaf': [2, 3, 4],
                            f'{prefix}min_samples_split': [4, 5, 6, 7],
                            f'{prefix}max_features': [1.0, "sqrt", "log2"],
                            f'{prefix}bootstrap': [True]
                             }
            """
            # --> targeted search for agg_type = percentile, perc = 90
            default_grids = {
                            f'{prefix}n_estimators': [100, 150, 200, 250, 300],  # [100, 200, 300],
                            f'{prefix}max_depth': [20, 25, 30, 35, 40],  # [2, 5, 8, 11, 14, 17, 20],
                            f'{prefix}min_samples_leaf': [2, 3, 4, 5],
                            f'{prefix}min_samples_split': [2, 3, 4, 5],
                            f'{prefix}max_features': [1.0, "sqrt", "log2"],
                            f'{prefix}bootstrap': [True]
                             }
        # end if elif

        if ((grid_sample > 0) & (len(in_hypp) == 0)):
            print("\nUsing large default hyperparameter set to sample.\n")
            default_grids = {
                            f'{prefix}n_estimators': [200, 300, 400, 500, 600, 700, 800, 900, 1000],
                            f'{prefix}max_depth': [10, 20, 30, 40, 50, 60, 70, 80],
                            f'{prefix}min_samples_leaf': [2, 4, 6, 8, 10, 12, 14, 16],
                            f'{prefix}min_samples_split': [2, 4, 6, 8, 10, 12, 14, 16],
                            f'{prefix}max_features': [1.0, "sqrt", "log2"],
                            f'{prefix}bootstrap': [True]
                             }
        # end if

    elif model_ty == "DT":

        if pipe:
            prefix = "decisiontreeclassifier__"
        else:
            prefix = ""
        # end if

        exp_hypps = {"max_depth":2, "min_samples_leaf":5}
        default_grids = {
                        'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10],
                        'min_samples_leaf': [2, 5, 10, 20]
                         }

    elif model_ty == "LR":

        if pipe:
            prefix = "logisticregression__"
        else:
            prefix = ""
        # end if

        exp_hypps = {"C":1}
        default_grids = {
                        'C': [0.1, 1, 10, 100],
                         }
    elif model_ty == "SVM":

        if pipe:
            prefix = "svc__"
        else:
            prefix = ""
        # end if

        exp_hypps = {"C":10}
        default_grids = {
                        'C': [0.1, 1, 10, 100],
                         }
    elif model_ty == "KNN":

        if pipe:
            prefix = "kneighborsclassifiert__"
        else:
            prefix = ""
        # end if

        exp_hypps = {"n_neighbors":10, "weights":"uniform"}
        default_grids = {
                        'n_neighbors': [2, 5, 10, 20, 30],
                        "weights":["uniform", "distance"]
                         }
    # end if "RF" elif "DT"


    # update the in_hypp with the prefix if necessary
    if ((len(in_hypp) > 0) & pipe):
        if verbose:
            print("\nUpdating the names in the in_hypp parameter with the pipe prefix...\n")
        # end if
        in_hypp = {prefix+k:e for k, e in zip(in_hypp.keys(), in_hypp.values())}
    # end if

    # set up the hyperparameter output as an empty dictionary
    param_grid = {}

    # set the hyperparameters
    if grid_search:
        if len(in_hypp) == 0:  # --> if no hyperparameter grids are provided by the user set the default
            # default grids for search
            param_grid = default_grids
            if verbose:
                print(f"\nUsing pre-set hyperparameter set/grid for {ndlev}-ADL case.\n")
                print(f"""The parameter grid is:
                      {param_grid}""")
            # end if
        else:  # --> if hyperparameter grids are given by the user use those and set the remainder to the expected ones
#                    --> note that this means that for the hyperparameters for which no grid is given no grid
#                        search will be performed, because the standard single value from exp_hypps is used
            for k in exp_hypps.keys():
                try:
                    # make sure the hyperparameters are given as lists
                    temp_hypp = in_hypp[k]
                    if type(temp_hypp) != list:
                        temp_hypp = [temp_hypp]
                    # end if

                    # if available take the parameter from the given set
                    param_grid[k] = temp_hypp
                except:
                    # if not take the default value
                    param_grid[k] = [exp_hypps[k]]
                # end try except
            # end for k
        # end if else
    else:  # --> if not grid_search

        if len(in_hypp) == 0:  # --> if no hyperparameters are provided by the user set the default
            param_grid = exp_hypps
        else:  # --> if hyperparameters are given by the user use those and set the remainder to the expected ones
            # loop over the expected hyperparameters
            for k in exp_hypps.keys():
                try:
                    # if available take the parameter from the given set
                    param_grid[k] = in_hypp[k]
                except:
                    # if not take the default value
                    param_grid[k] = exp_hypps[k]
                # end try except
            # end for k
        # end if else
    # end if else

    return param_grid
# end def


# test = set_hyperp(model_ty="SVM", in_hypp={"C":5}, grid_search=False)
# print(test)
