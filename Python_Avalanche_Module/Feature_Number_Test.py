#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the change of model skill with the number of features included in model training.
"""


#%% imports
import os
import sys
import argparse
import numpy as np
import pandas as pd
import pylab as pl
from joblib import load
from sklearn.model_selection import PredefinedSplit, cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline

# import proprietary functions and variables
from Statistical_Prediction.Functions.Func_Round import r_2
from Statistical_Prediction.Functions.Func_Load_NORA3_XLevel_Predictors import load_xlevel_preds
from Statistical_Prediction.Functions.Func_Assign_Winter_Year import assign_winter_year
from Statistical_Prediction.Lists_and_Dictionaries.Paths import path_par


#%% set up the parser
parser = argparse.ArgumentParser(
                    prog="FeatureNumberTest",
                    description="""Performs a feature number test.""",
                    epilog="For more information consult the documentation of the function feat_sel.")

# ...and add the arguments
parser.add_argument("--a_p", default="y", type=str, choices=["y", "glide_slab", "new_loose", "new_slab", "pwl_slab",
                                                             "wet_loose", "wet_slab", "wind_slab"],
                    help="""The avalanche problem that is used. Standard is 'y', i.e., the general ADL.""")
parser.add_argument("--nan_handl", default="drop", type=str, choices=["drop", "zero"],
                    help="""How to handle NaNs in the danger level data.""")
parser.add_argument("--cv_type", type=str, default="seasonal", choices=["seasonal", "stratified"],
                    help="""The type of folds in the cross-validation during grid search.""")
parser.add_argument("--cv", type=int, default=6, help="""The number folds in the gridsearch cross-validation.""")
parser.add_argument("--cv_score", type=str, default="f1_macro", help="""The type of skill score used in the
                    cross-validation. For possible choices see
                    https://scikit-learn.org/stable/modules/model_evaluation.html""")
parser.add_argument("--cv_on_all", action="store_true", help="""If set, the k-fold cross-validation is performed on the
                    the whole dataset instead of only the training data.""")
parser.add_argument("--thres", type=float, default=0.9, help="""Value of cross-correlation R for dropping features""")
parser.add_argument("--r_coeff", default="R**2", type=str, choices=["R", "R**", "R^2"],
                    help="""The type of cross-correlation R to be considered as threshold""")
parser.add_argument("--model_ty", default="RF", choices=["RF", "DT"],
                    help="""The type of statistical model that is used.""")
parser.add_argument("--ndlev", type=int, default=4, choices=[2, 3, 4],
                    help="""The number of danger levels to be predicted.""")
parser.add_argument("--split", nargs="*", default=[2021, 2023],
                    help="""Fraction of data to be used as test data.""")
parser.add_argument("--balancing", type=str, default="external", choices=["none", "internal", "external"],
                    help="""Controls the kind of balancing to be performed.""")
parser.add_argument("--balance_meth", type=str, default="SVMSMOTE",
                    choices=["undersample", "rus", "ros", "SMOTE", "ADASYN"],
                    help="""The balancing method applied during the loading of the data. This is only relevant if
                    balancing is set to external.""")
parser.add_argument("--scale_x", action="store_true", help="Perform a predictor scaling.")
parser.add_argument("--sea", type=str, default="full", choices=["full", "winter", "spring"],
                    help="""The season for which the model is trained. The choices mean: full=Dec-May, winter=Dec-Feb,
                    spring=Mar-May.""")
parser.add_argument("--h_low", type=int, default=400, help="The lower threshold of the grid cell altitude.")
parser.add_argument("--h_hi", type=int, default=900, help="The upper threshold of the grid cell altitude.")
parser.add_argument("--reg_code", type=int, default=0, help="""The region code of the region for which the danger levels
                    will be predicted. Set 0 (default) to use all regions.""")
parser.add_argument("--agg_type", default="mean", type=str, choices=["mean", "median", "percentile"],
                    help="""The type of aggregation used for the grid cells.""")
parser.add_argument("--perc", type=int, default=90, help="""The percentile used in the grid-cell aggregation.""")

args = parser.parse_args()


#%% choose the model; choices are
#   DT  --> decision tree
#   RF  --> random forest
model_ty = args.model_ty

# the following should be superflous, but leave it in for the time being
if model_ty not in ("DT", "RF"):
    print("\nThe feature-importance is only available for DT and RF. Aborting.\n")
    sys.exit("model_ty not available.")
# end if


#%% set the train-test split
split = [float(i) for i in args.split]


#%% get the avalanche problem
a_p = args.a_p
nan_handl = args.nan_handl

a_p_str = a_p
if a_p == "y":
    a_p_str = "general"
# end if


#%% set parameters
ndlev = args.ndlev
cv = args.cv
cv_score = args.cv_score
cv_on_all = args.cv_on_all
h_low = args.h_low
h_hi = args.h_hi
sea = args.sea
balancing = args.balancing
balance_meth = args.balance_meth
scale_x = args.scale_x
agg_type = args.agg_type
perc = args.perc

if args.reg_code == 0:
    reg_code = "AllReg"  #  sys.argv[1]
    reg_codes = [3009, 3010, 3011, 3012, 3013]
else:
    reg_code = args.reg_code
    reg_codes = [reg_code]
# end if else


#%% generate a name prefix/suffix depending on the gridcell aggregation

# make sure that the percentile is 0 if the type is not percentile
if agg_type != "percentile":
    perc = 0
# end if

agg_str = f"{agg_type.capitalize()}{perc if perc != 0 else ''}"


#%% paths and names
data_path = f"{path_par}/IMPETUS/NORA3/"  # data path
mod_path = f"{data_path}/Stored_Models/{agg_str}/Between{h_low}_and_{h_hi}m/"  # directory to store the model in
pl_path = f"{data_path}/Plots/Feature_Testing/"


#%% prepare a suffix for the model name based on the data balancing
bal_suff = ""
if balancing == "internal":
    bal_suff = "_internal"
elif balancing == "external":
    bal_suff = f"_{balance_meth}"
# end if elif


#%% load the best-features list
best_path = f"{data_path}/Feature_Selection/"
best_name = f"BestFeats_{model_ty}_{ndlev}DL_{reg_code}_{agg_str}_Between{h_low}_and_{h_hi}m_{sea}{bal_suff}_" + \
                                                                                                        f"{a_p_str}"
# depending on availability load either the version without or with data
try:
    sel_feats = np.array(pd.read_csv(best_path + best_name + ".csv", index_col=0).index)
except:
    sel_feats = np.array(pd.read_csv(best_path + best_name + "_wData.csv", index_col=0).index)
# end try except

n_all_best = len(sel_feats)


#%% load data for cross-validation
"""
bal_x, bal_y, all_x, all_y = load_xlevel_preds(data_path + f"/Avalanche_Predictors_{ndlev}Level/{agg_str}/" + \
                                               f"Between{h_low}_and_{h_hi}m/",
                                               sel_feats, reg_code=reg_code, a_p=a_p, nan_handling=nan_handl,
                                               split=0, sea=sea,
                                               nlevels=ndlev, balance_meth=balance_meth, scale_x=scale_x,
                                               verbose=True)
"""
# load data
print("\nReloading data for cross-validation.")
if cv_on_all:
    print("\nUsing ALL data in the cross-validation.\n")
    bal_x, bal_y, all_x, all_y = load_xlevel_preds(data_path +
                                                   (f"/Avalanche_Predictors_{ndlev}Level" +
                                                                f"/{agg_str}/Between{h_low}_and_{h_hi}m/"),
                                                   sel_feats=sel_feats, reg_code=reg_code, a_p=a_p,
                                                   split=0, sea=sea,
                                                   nlevels=ndlev, balance_meth=balance_meth, scale_x=scale_x,
                                                   verbose=True)
else:
    print("\nUsing only the training data in the cross-validation.\n")
    train_x, test_x, train_y, test_y, train_x_all, test_x_all, train_y_all, test_y_all = load_xlevel_preds(data_path +
                                                   (f"/Avalanche_Predictors_{ndlev}Level" +
                                                                f"/{agg_str}/Between{h_low}_and_{h_hi}m/"),
                                                   sel_feats=sel_feats, reg_code=reg_code, a_p=a_p,
                                                   split=split, sea=sea,
                                                   nlevels=ndlev, balance_meth=balance_meth, scale_x=scale_x,
                                                   verbose=True)
    all_x = train_x_all
    all_y = train_y_all
    # perform the cross-validation only for the training data
# end if else


#%% perform a predefined-fold stratified cross-validation
print(f"\nPerforming a {cv}-fold cross-validation based on the {cv_score} score.\n")

# assign winter month
all_winter = pd.Series(assign_winter_year(all_x.index, 8))
obs_yrs = np.unique(all_winter)

# assign fold labels --> depends on the year
len_fold = int(len(obs_yrs) / cv)

fold_label = np.zeros(len(all_winter))


cv_scores = []
n_bests = np.arange(0, n_all_best, 10)
n_bests = np.arange(0, 41, 10)
n_bests[0] += 5
n_bests = list(n_bests)
n_bests.append(n_all_best)
for n_best in n_bests:

    nbest_suff = f"_{n_best:02}best"
    mod_name = f"{model_ty}_{ndlev}DL_{reg_code}_{agg_str}_Between{h_low}_and_{h_hi}m_{sea}{bal_suff}{nbest_suff}_" + \
                                                                                                      f"{a_p_str}_wData"

    bundle = load(f"{mod_path}/{mod_name}.joblib")

    for i, yr in enumerate(obs_yrs[::len_fold]):
        yrs_temp = [yr_temp for yr_temp in np.arange(yr, yr+len_fold, 1)]
        fold_label[all_winter.isin(yrs_temp)] = i
    # end for i, yr

    # generate the predefined split
    ps = PredefinedSplit(fold_label)

    pipeline = make_pipeline(SMOTE(), bundle["model"])

    # load the best feats
    best_path = f"{data_path}/Feature_Selection/"
    best_name = f"BestFeats_{model_ty}_{ndlev}DL_{reg_code}_{agg_str}_Between{h_low}_and_{h_hi}m_{sea}{bal_suff}_" + \
                                                                                                        f"{a_p_str}.csv"
    if n_best == 0:
        n_best = None
    # end if
    best_feats = np.array(pd.read_csv(best_path + best_name, index_col=0).index)[:n_best]

    # perform the cross-validation
    scores = cross_val_score(pipeline, all_x[best_feats], all_y, cv=ps, n_jobs=-1, scoring=cv_score)

    cv_scores.append(scores)

    print(f"\n{n_best} features:")
    print(f"\nCross-validation {cv_score} (train):")
    print("   " + " ".join([f"{r_2(k):.2f}" for k in scores]) +
          f"  avg: {r_2(np.mean(scores)):.2f}")

# end for i


#%% scores dict
cv_scores_dict = {k:i for k, i in zip(n_bests, cv_scores)}

cv_scores_df = pd.DataFrame(cv_scores_dict).T
cv_scores_df.index.name = "N_Best"

fnum_path = f"{data_path}/Feature_Numbers/"
os.makedirs(fnum_path, exist_ok=True)
fnum_name = f"FeatNum_{cv}Folds_{model_ty}_{ndlev}DL_{reg_code}_{agg_str}_" + \
                                                             f"Between{h_low}_and_{h_hi}m_{sea}{bal_suff}_{a_p_str}.csv"
cv_scores_df.to_csv(fnum_path + fnum_name)


#%% perform a stratified cross-validation
"""
print(f"\nPerforming a {cv}-fold cross-validation based on the {cv_score} score.\n")

scores_train = []
scores_test = []
n_bests = [5, 10, 20, 30, 40, n_all_best]
for n_best in n_bests:

    nbest_suff = f"_{n_best:02}best"
    mod_name = f"{model_ty}_{ndlev}DL_{reg_code}_Between{h_low}_and_{h_hi}m_{sea}{bal_suff}{nbest_suff}_wData"

    bundle = load(f"{mod_path}/{mod_name}.joblib")


    # perform cross validation
    cv_score_train_bal, split_tr = stratified_cv(bundle["model"], bundle["train_x"], bundle["train_y"], cv=cv,
                                                 scoring=cv_score, return_split=True)
    cv_score_test_bal, split_te = stratified_cv(bundle["model"], bundle["test_x"], bundle["test_y"], cv=cv,
                                                scoring=cv_score, return_split=True)

    scores_train.append(cv_score_train_bal)
    scores_test.append(cv_score_test_bal)

    print(f"\n{n_best} features:")
    print(f"\nCross-validation {cv_score} (train):")
    print("   " + " ".join([f"{r_2(k):.2f}" for k in cv_score_train_bal]) +
          f"  avg: {r_2(np.mean(cv_score_train_bal)):.2f}")

    print(f"\nCross-validation {cv_score} (test):")
    print("   " + " ".join([f"{r_2(k):.2f}" for k in cv_score_test_bal]) +
          f"  avg: {r_2(np.mean(cv_score_test_bal)):.2f}\n")

# end for i


#%% also load the full-features model
mod_name = f"{model_ty}_{ndlev}DL_{reg_code}_Between{h_low}_and_{h_hi}m_{sea}{bal_suff}_wData"

bundle = load(f"{mod_path}/{mod_name}.joblib")


# perform cross validation
cv_score_train_bal, split_tr = stratified_cv(bundle["model"], bundle["train_x"], bundle["train_y"], cv=cv,
                                             scoring=cv_score, return_split=True)
cv_score_test_bal, split_te = stratified_cv(bundle["model"], bundle["test_x"], bundle["test_y"], cv=cv,
                                            scoring=cv_score, return_split=True)

scores_train.append(cv_score_train_bal)
scores_test.append(cv_score_test_bal)

n_bests.append(len(bundle["model"].feature_names_in_))
print(f"\n{len(bundle['model'].feature_names_in_)} features:")

print(f"\nCross-validation {cv_score} (train):")
print("   " + " ".join([f"{r_2(k):.2f}" for k in cv_score_train_bal]) +
      f"  avg: {r_2(np.mean(cv_score_train_bal)):.2f}")

print(f"\nCross-validation {cv_score} (test):")
print("   " + " ".join([f"{r_2(k):.2f}" for k in cv_score_test_bal]) +
      f"  avg: {r_2(np.mean(cv_score_test_bal)):.2f}\n")


#%% plot the change
patch_artist = True

fig = pl.figure(figsize=(8, 5))
ax = fig.add_subplot(111)

bpl1 = ax.boxplot(scores_test, positions=np.arange(len(scores_test)), patch_artist=patch_artist, label="test",
                  medianprops={"color":"blue"})
bpl2 = ax.boxplot(scores_train, positions=np.arange(len(scores_test)), patch_artist=patch_artist, label="train",
                  medianprops={"color":"red"})

# fill with colors
for patch1, patch2 in zip(bpl1['boxes'], bpl2['boxes']):
    patch1.set_facecolor("red")
    patch2.set_facecolor("blue")
# end for

ax.legend()

ax.set_xticks(np.arange(len(scores_test)))
ax.set_xticklabels(n_bests)
ax.set_xlabel("N features")
ax.set_ylabel(f"{cv_score} score")

ax.set_title(f"{model_ty} {cv_score} {ndlev} ADLs")

os.makedirs(pl_path, exist_ok=True)
pl.savefig(pl_path + f"{model_ty}_{cv_score}_Saturation_{ndlev}ADL.png", bbox_inches="tight", dpi=150)

pl.show()
pl.close()

"""


#%% plot the change
patch_artist = True

fig = pl.figure(figsize=(8, 5))
ax = fig.add_subplot(111)

bpl1 = ax.boxplot(cv_scores, positions=np.arange(len(cv_scores)), patch_artist=patch_artist, label="test",
                  medianprops={"color":"blue"})

# fill with colors
for patch1 in bpl1['boxes']:
    patch1.set_facecolor("red")
# end for

ax.legend()

ax.set_xticks(np.arange(len(cv_scores)))
ax.set_xticklabels(n_bests)
ax.set_xlabel("N features")
ax.set_ylabel(f"{cv_score} score")

ax.set_title(f"{model_ty} {cv_score} {ndlev} ADLs")

os.makedirs(pl_path, exist_ok=True)
# pl.savefig(pl_path + f"{model_ty}_{cv_score}_Saturation_{ndlev}ADL_{a_p_str}.png", bbox_inches="tight", dpi=150)

pl.show()
pl.close()
