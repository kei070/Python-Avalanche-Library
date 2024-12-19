#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of a grid search procedure for hyperparameters.
"""

# imports
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from imblearn.over_sampling import SMOTE, SVMSMOTE, KMeansSMOTE, ADASYN, RandomOverSampler, BorderlineSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline

# import proprietary functions
from .Func_Predef_Fold import sea_fold


def perform_grid_search(model_ty, param_grid, ndlev, cv_type, cv, cv_score, grid_sample, balance_meth, class_weight,
                        train_x, train_y, kernel, sample_strat="auto", max_iter=1000, verbose=True):

    """
    For most of the parameters see the documentation of apply_stat_mod.
    """


    # set up an empty hyperparameter dictionary that will be filled later
    hyperparameters = {}

    # depending on the cv_type set the folds
    if cv_type == "seasonal":
        folds = sea_fold(train_x, nfolds=cv)
    elif cv_type == "stratified":
        # set up a stratifier for the cross-validation during the grid search
        folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    # end if elif

    # set up a dictionary associating the model_ty with the respective function
    models = {"DF":DecisionTreeClassifier(random_state=42, class_weight=class_weight),
              "LR":LogisticRegression(max_iter=max_iter, class_weight=class_weight),
              "SVM":SVC(kernel=kernel, class_weight=class_weight),
              "KNN":KNeighborsClassifier(),
              "RF":RandomForestClassifier(random_state=42, class_weight=class_weight)}

    # set up a dictionary associating the balance_meth with the respective over-/undersampler
    samplers = {"SMOTE":SMOTE(random_state=42, sampling_strategy=sample_strat),
                "ADASYN":ADASYN(random_state=42, sampling_strategy=sample_strat),
                "SVMSMOTE":SVMSMOTE(random_state=42, sampling_strategy=sample_strat),
                "BSMOTE":BorderlineSMOTE(random_state=42, sampling_strategy=sample_strat),
                "KMeansSMOTE":KMeansSMOTE(random_state=42, sampling_strategy=sample_strat),
                "ros":RandomOverSampler(random_state=42),
                "rus":RandomUnderSampler(random_state=42),
                "SMOTEENN":SMOTEENN(random_state=42, sampling_strategy=sample_strat),
                "SMOTETomek":SMOTETomek(random_state=42, sampling_strategy=sample_strat)}

    # set up the pipeline for the grid search
    pipeline = make_pipeline(samplers[balance_meth], models[model_ty])

    # setup the grid search
    if grid_sample > 0:
        grid_search = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid, cv=folds,
                                         verbose=1, scoring=cv_score, n_jobs=-1, n_iter=grid_sample)
    else:
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=folds,
                                   verbose=1, scoring=cv_score, n_jobs=-1)
    # end if else

    # Fit grid search
    best_model = grid_search.fit(train_x, train_y)

    # Best parameters and best score
    if verbose:
        print("Best parameters:", best_model.best_params_)
        print("Best score:", best_model.best_score_)
        print()
    # end if

    # set the hyperparameters
    hyperparameters = {k:best_model.best_params_[k] for k in best_model.best_params_.keys()}

    return hyperparameters

# end def
