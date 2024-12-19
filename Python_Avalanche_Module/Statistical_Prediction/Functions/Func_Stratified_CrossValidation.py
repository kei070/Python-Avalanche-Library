#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A wrapper for a stratified k-fold cross-validation.
"""


# imports
from sklearn.model_selection import cross_val_score, StratifiedKFold


# function
def stratified_cv(model, data_x, data_y, cv=5, return_split=False, scoring="accuracy"):

    # initialize a StratifiedKFold object
    stratified_kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    # Perform cross-validation
    scores = cross_val_score(model, data_x, data_y, cv=stratified_kfold, n_jobs=-1, scoring=scoring)

    if return_split:
        # split the data
        split = stratified_kfold.split(data_x, data_y)
        split_i = {f"fold {i}":{"train":train_i, "test":test_i} for i, (train_i, test_i) in enumerate(split)}

        return scores, split_i
    else:
        return scores
    # end if else

# end def


