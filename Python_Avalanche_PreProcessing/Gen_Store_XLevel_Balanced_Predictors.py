#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate and store the x-level and balanced predicors data.
"""


#%% import
import os
import sys
import argparse

from Functions.Func_Prep_Data_Avalanche_Analysis_XLevel import load_feats_xlevel
from Lists_and_Dictionaries.Features import features as clean_feats
from Lists_and_Dictionaries.Paths import path_par


#%% set up the parser
parser = argparse.ArgumentParser(
                    prog="Predictor with Aggregated Danger Levels",
                    description="""Aggregate the danger levels to 2, 3, or 4 (from originally 5).
                    See the documentation to load_feats_xlevel for parameter descriptions.""")

# ...and add the arguments
parser.add_argument('--reg_code', default=0, type=int, help='Code of the region. Use 0 for all regions.')
parser.add_argument("--h_low", type=int, default=300, help="The lower threshold of the grid cell altitude.")
parser.add_argument("--h_hi", type=int, default=1300, help="The upper threshold of the grid cell altitude.")
parser.add_argument("--agg_type", default="mean", type=str, choices=["mean", "median", "percentile"],
                    help="""The type of aggregation used for the grid cells.""")
parser.add_argument("--perc", type=int, default=90, help="""The percentile used in the grid-cell aggregation.""")
parser.add_argument("--ndlev", type=int, default=2, help="""The number of danger levels to aggregate to.""")
parser.add_argument("--a_p", type=str, default="all", help="""The avalanche problem. Use 'all' for the general ADL.""")
parser.add_argument("--feats", nargs="*", default= ["all"], help="""Features to be used. Use 'all' for all, 'clean' for
                    the cleaned-up version or provide a list of your own choosing.""")
args = parser.parse_args()


#%% get the parameters from the parser
reg_code = args.reg_code
h_low = args.h_low
h_hi = args.h_hi
agg_type = args.agg_type
perc = args.perc
ndlev = args.ndlev
a_p = args.a_p
feats = args.feats

perm = None


#%% handle the features list
if feats[0] == "all":
    sel_feats = None
elif feats[0] == "clean":
    sel_feats = clean_feats
else:
    sel_feats = feats
# end if elif else


#%% set the store_balanced condition (i.e., should the balanced data be stored or not) based on the a_p variable
#   --> if more than one avalanche problem is used balancing does not work in the balanced data are not stored
if type(a_p) == list:
    store_balanced = True if len(a_p) == 1 else False
else:
    store_balanced = True if a_p != "all" else False
# end if else


#%% generate a name prefix/suffix depending on the gridcell aggregation

# make sure that the percentile is 0 if the type is not percentile
if agg_type != "percentile":
    perc = 0
# end if

agg_str = f"{agg_type.capitalize()}{perc if perc != 0 else ''}"


#%% set the region codes
if reg_code == 0:
    reg_codes = [3009, 3010, 3011, 3012, 3013]
else:
    reg_codes = [reg_code]
# end if else

print(f"\nGenerating {ndlev}-level predictor data: permutation {perm}, reg_code: {reg_codes}\n")


#%% select features and exposure
exposure = None


#%% add exposure part to the model name
add_expos = ""
if exposure == "west":
    add_expos = "_WestExpos"
elif exposure == "east":
    add_expos = "_EastExpos"
# end if elif


#%% load the data
data_path_par = f"/{path_par}/IMPETUS/NORA3/"
features = load_feats_xlevel(reg_codes=reg_codes, ndlev=ndlev, exposure=exposure, sel_feats=sel_feats,
                             a_p=a_p,
                             out_type="dataframe",
                             h_low=h_low, h_hi=h_hi,
                             agg_type=agg_type, perc=perc,
                             data_path_par=data_path_par + "Avalanche_Predictors/")


#%% create the folder
data_path = data_path_par + f"Avalanche_Predictors_{ndlev}Level/{agg_str}/Between{h_low}_and_{h_hi}m/"
os.makedirs(data_path, exist_ok=True)


#%% store the data as csv files
if len(reg_codes) == 1:

    print("Storing data for individual region in")
    print(data_path)

    # set region name according to region code
    if reg_codes[0] == 3009:
        region = "NordTroms"
    elif reg_codes[0] == 3010:
        region = "Lyngen"
    elif reg_codes[0] == 3011:
        region = "Tromsoe"
    elif reg_codes[0] == 3012:
        region = "SoerTroms"
    elif reg_codes[0] == 3013:
        region = "IndreTroms"
    # end if elif

    # training data balanced
    if store_balanced:
        features["balanced"].to_csv(data_path +
                        f"/Features_{ndlev}Level_Balanced_{agg_str}_Between{h_low}_{h_hi}m_{reg_codes[0]}_{region}.csv")
    # end if

    # training data all
    features["all"].to_csv(data_path +
                             f"/Features_{ndlev}Level_All_{agg_str}_Between{h_low}_{h_hi}m_{reg_codes[0]}_{region}.csv")

elif len(reg_codes) == 5:

    print("Storing data for all regions in")
    print(data_path)

    # training data balanced
    if store_balanced:
        features["balanced"].to_csv(data_path +
                                         f"/Features_{ndlev}Level_Balanced_{agg_str}_Between{h_low}_{h_hi}m_AllReg.csv")
    # end if

    # training data all
    features["all"].to_csv(data_path + f"/Features_{ndlev}Level_All_{agg_str}_Between{h_low}_{h_hi}m_AllReg.csv")

# end if elif


