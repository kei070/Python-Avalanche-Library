#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Here those dates of the predictor files generated with Calc_and_Store_NORA3_Predictors.py are extracted and combined
with the danger levels from Varsom to produce datasets containing both the predictors and the target variable which
can then be used to train and test the statistical models.
"""

#%% imports
import os
import sys
import pandas as pd
import argparse

from Lists_and_Dictionaries.Paths import path_par, obs_path


#%% set up the parser
parser = argparse.ArgumentParser(
                    prog="Predictors with Danger Levels",
                    description="""Combine danger levels and predictors.""")

# ...and add the arguments
parser.add_argument('--reg_code', default=3009, type=int, help='Code of the region.')
parser.add_argument("--h_low", type=int, default=400, help="The lower threshold of the grid cell altitude.")
parser.add_argument("--h_hi", type=int, default=900, help="The upper threshold of the grid cell altitude.")
parser.add_argument("--agg_type", default="percentile", type=str, choices=["mean", "median", "percentile"],
                    help="""The type of aggregation used for the grid cells.""")
parser.add_argument("--perc", type=int, default=90, help="""The percentile used in the grid-cell aggregation.""")
args = parser.parse_args()


#%% get the parameters from the parser
reg_code = args.reg_code
h_low = args.h_low
h_hi = args.h_hi
agg_type = args.agg_type
perc = args.perc


#%% get the region
regions = {3009:"NordTroms", 3010:"Lyngen", 3011:"Tromsoe", 3012:"SoerTroms", 3013:"IndreTroms"}
region = regions[reg_code]


#%% generate a name prefix/suffix depending on the gridcell aggregation

# make sure that the percentile is 0 if the type is not percentile
if agg_type != "percentile":
    perc = 0
# end if
agg_str = f"{agg_type.capitalize()}{perc if perc != 0 else ''}"


#%% set data path and read the avalanche predictor files
pred_path = f"/{path_par}/IMPETUS/NORA3/Avalanche_Predictors_Full_TimeSeries/{agg_str}/Between{h_low}_and_{h_hi}m/"
pred_name = f"{region}_Predictors_MultiCell{agg_str}_Between{h_low}_and_{h_hi}m.csv"

df = pd.read_csv(pred_path + pred_name)
df.date = pd.to_datetime(df.date)


#%% get all the danger-level "observations"
dl_path = f"/{obs_path}/IMPETUS/Avalanches_Danger_Files/"
dl_name = "Avalanche_Danger_List.csv"

dl_df = pd.read_csv(dl_path + dl_name)
dl_df.date = pd.to_datetime(dl_df.date)


#%% extract the region according to the region code
dl_df_reg = dl_df[dl_df.region == reg_code]


#%% perform an inner join with the predictors and the danger-level "observations" to merge the files
df_obs_data = df.merge(dl_df_reg, how="inner", on="date")
# df_obs_data.set_index("date", inplace=True)


#%% load the danger levels per avalanche problem
# ap_name = "Avalanche_Problems_ADL.csv"
ap_name = "Avalanche_Problems_ADL_Extended.csv"  # including the "wet" problem (loose & slab combined)

ap_df = pd.read_csv(dl_path + ap_name)
ap_df.date = pd.to_datetime(ap_df.date)


#%% extract the region according to the region code
ap_df_reg = ap_df[ap_df.region == reg_code].copy()
ap_df_reg.drop(columns="region", inplace=True)


#%% perform an left join with the predictors & general ADL on the left and the AP ADLs on the right
df_obs_data = df_obs_data.merge(ap_df_reg, how="left", on="date")
df_obs_data.set_index("date", inplace=True)


#%% store the predictors
out_path = f"/{path_par}/IMPETUS/NORA3/Avalanche_Predictors/{agg_str}/Between{h_low}_and_{h_hi}m/"

os.makedirs(out_path, exist_ok=True)

df_obs_data.to_csv(out_path + region + f"_Predictors_MultiCell{agg_str}_Between{h_low}_and_{h_hi}m.csv")


#%% make NaN test
# import pylab as pl
# pl.plot(df_obs_data.isna().sum())
