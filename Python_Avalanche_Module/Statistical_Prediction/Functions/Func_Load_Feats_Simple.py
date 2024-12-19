#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Brief simple function for loading the predictive features for avalanche danger forecasting.
"""

# imports
import pandas as pd


# function
def load_features(path, features):
    """
    Brief simple function for loading the predictive features for avalanche danger forecasting.
    """

    feats_df = pd.read_csv(path)
    feats_df["date"] = pd.to_datetime(feats_df["date"])
    feats_df.set_index("date", inplace=True)
    feats_df = feats_df[features]
    feats_df[feats_df.isna()] = 0

    return feats_df

# end def
