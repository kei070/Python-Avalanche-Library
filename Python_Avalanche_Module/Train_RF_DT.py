"""
Extra script for random forest and decision tree models because they return feature importances.

Example use of the script:
    python Train_RF_DT.py --ndlev 4 --no_plots --a_p wind_slab --nan_handl drop --h_low 600 --h_hi 1100
"""


#%% imports
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import pylab as pl
import argparse
import json
from joblib import dump
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.model_selection import PredefinedSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE, SVMSMOTE, ADASYN, RandomOverSampler, BorderlineSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline


#%% import proprietary functions variables
from Statistical_Prediction.Functions.Func_Apply_StatMod import apply_stat_mod
from Statistical_Prediction.Functions.Func_Discrete_Hist import disc_hist
from Statistical_Prediction.Functions.Func_Prep_Data_Avalanche_Analysis_XLevel import load_feats_xlevel
from Statistical_Prediction.Functions.Func_Load_NORA3_XLevel_Predictors import load_xlevel_preds
from Statistical_Prediction.Functions.Func_Stratified_CrossValidation import stratified_cv
from Statistical_Prediction.Functions.Func_Round import r_2
from Statistical_Prediction.Functions.Func_Assign_Winter_Year import assign_winter_year
from Statistical_Prediction.Lists_and_Dictionaries.Features import feats_paper1, feats_clean
from Statistical_Prediction.Lists_and_Dictionaries.Paths import path_par
from Statistical_Prediction.Functions.Func_ConfMat_Helper import conf_helper
from Statistical_Prediction.Lists_and_Dictionaries.Hyperp_Set import hypp_grid, hypp_set


#%% get the features from the paper 1 dictionary
features = list(feats_paper1.keys())
# features = list(feats_clean.keys())


#%% set up the parser
parser = argparse.ArgumentParser(
                    prog="Train_RF_DT",
                    description="""Performs the feature-importance test available for RF and DT.""",
                    epilog="For more information consult the documentation of the function apply_stat_mod.")

# ...and add the arguments
parser.add_argument("--model_ty", default="RF", type=str, choices=["RF", "DT"],
                    help="""The type of statistical model that is used.""")
parser.add_argument("--feats", default="all", type=str, choices=["all", "best"],
                    help="""Indicator for the feature to be considered. Either all or best.""")
parser.add_argument("--a_p", default="pwl_slab", type=str, choices=["y", "glide_slab", "new_loose", "new_slab",
                                                             "pwl_slab", "wet_loose", "wet_slab", "wind_slab",
                                                             "wet"],
                    help="""The avalanche problem that is used. Standard is 'y', i.e., the general ADL.""")
parser.add_argument("--nan_handl", default="drop", type=str, choices=["drop", "zero"],
                    help="""How to handle NaNs in the danger level data.""")
parser.add_argument("--n_best", type=int, default=0, help="""The number of best features to be used from the best
                    features list. Set to 0 (default) to use all 'best' parameters found in the feature search.""")
parser.add_argument("--grid_search", action="store_true", help="Perform a grid search for the hyperparameters.")
parser.add_argument("--grid_sample", type=int, default=0, help="""Number of random samples taken from the
                    hyperparameter grids. If set to 0 (default) the full grids will be searched.""")
parser.add_argument("--cv_on_all", action="store_true", help="""If set, the k-fold cross-validation is performed on the
                    the whole dataset instead of only the training data. This will also translate to the cross-
                    validation during the grid search.""")
parser.add_argument("--cv_type", type=str, default="seasonal", choices=["seasonal", "stratified"],
                    help="""The type of folds in the cross-validation during grid search.""")
parser.add_argument("--cv", type=int, default=3,
                    help="""The number folds in the cross-validation during grid search.""")
parser.add_argument("--cv_score", type=str, default="f1_macro", help="""The type of skill score used in the
                    cross-validation during grid search. For possible choices see
                    https://scikit-learn.org/stable/modules/model_evaluation.html""")
parser.add_argument("--hyperp", type=str, help="""The hyperparameters or hyperparameter grid for the statistical model.
                    Must be submitted as a JSON-string that can be interpreted as a dictionary, e.g.:
                        '{"max_depth":5}'. If empty, default values will be used.""")
parser.add_argument("--ndlev", type=int, default=4, choices=[2, 3, 4],
                    help="""The number of danger levels to be predicted.""")
parser.add_argument("--scale_x", action="store_true", help="Perform a predictor scaling.")
parser.add_argument("--balancing", type=str, default="external", choices=["none", "internal", "external"],
                    help="""Controls the kind of balancing to be performed.""")
parser.add_argument("--class_weight",  nargs="*", default=[None], help="""Weights of the individual classes.
                    Submit one value per class.
                    If empty, no balancing will be performed or automatic weights will be used,
                        depending on the parameter balancing (see above).""")
parser.add_argument("--split", nargs="*", default=[2021, 2023],
                    help="""Fraction of data to be used as test data.""")
parser.add_argument("--sea", type=str, default="full", choices=["full", "winter", "spring"],
                    help="""The season for which the model is trained. The choices mean: full=Dec-May, winter=Dec-Feb,
                    spring=Mar-May.""")
parser.add_argument("--h_low", type=int, default=400, help="The lower threshold of the grid cell altitude.")
parser.add_argument("--h_hi", type=int, default=900, help="The upper threshold of the grid cell altitude.")
parser.add_argument("--reg_code", type=int, default=0, help="""The region code of the region for which the danger levels
                    will be predicted. Set 0 (default) to use all regions.""")
parser.add_argument("--balance_meth", type=str, default="SMOTE",
                    choices=["undersample", "rus", "ros", "SMOTE", "BSMOTE", "SVMSMOTE", "ADASYN", "SMOTEENN",
                             "SMOTETomek"],
                    help="""The balancing method applied during the loading of the data. This is only relevant if
                    balancing is set to external.""")
parser.add_argument("--agg_type", default="mean", type=str, choices=["mean", "median", "percentile"],
                    help="""The type of aggregation used for the grid cells.""")
parser.add_argument("--perc", type=int, default=90, help="""The percentile used in the grid-cell aggregation.""")
parser.add_argument("--no_plots", action="store_true", help="Set this parameter to suppress plotting.")
parser.add_argument("--store_mod", action="store_true", help="""Set this parameter to store the statistical model. This
                    parameter is ignored if store_tr_te is set.""")
parser.add_argument("--store_tr_te", action="store_true", help="""Set this parameter to store the train-test split on
                    which model is trained. The data are stored as a .joblib file together with the model. Note that
                    the model is then stored irrespective of the store_mod parameter.""")

args = parser.parse_args()


#%% choose the model; choices are
#   DT  --> decision tree
#   RF  --> random forest
model_ty = args.model_ty

if model_ty not in ["DT", "RF"]:
    print("\nThe feature-importance is only available for DT and RF. Aborting.\n")
    sys.exit("model_ty not available.")
# end if


#%% get the features indicator
feats = args.feats
n_best = args.n_best

if feats == "all":
    n_best = 0
# end if


#%% get the avalanche problem
a_p = args.a_p
nan_handl = args.nan_handl

a_p_str = a_p
if a_p == "y":
    a_p_str = "general"
# end if


#%% enable grid search for best parameters?
grid_search = args.grid_search
grid_sample = args.grid_sample
cv_type = args.cv_type
cv = args.cv
cv_score = args.cv_score
cv_on_all = args.cv_on_all


#%% set hyperparameters
hyperp = {}
# convert the JSON string to a dictionary
if args.hyperp:
    try:
        hyperp = json.loads(args.hyperp)
    except json.JSONDecodeError:
        print("\nError while loading hyperp: Invalid JSON format\n")
    # end try except
else:
    print("\nNo hyperparameters provided, using pre-set values (see Hyperp_Set.py).\n")
    if grid_search:
        hyperp = hypp_grid
    else:
        hyperp = hypp_set
    # end if else
# end if else


#%% set the number of danger levels
ndlev = args.ndlev


#%% determine if the variables will be scaled or not
scale_x = args.scale_x


#%% set the balancing to be applied
balancing = args.balancing


#%% get the class weights if submitted
if args.class_weight[0] is None:
    class_weight = None
else:
    class_weight = {float(k):float(e) for k, e in zip(np.arange(len(args.class_weight)), args.class_weight)}
# end if else


#%% set the train-test split
split = [float(i) for i in args.split]


#%% set the season
sea = args.sea


#%% set the height thresholds
h_low = args.h_low
h_hi = args.h_hi


#%% set the balancing method
balance_meth = args.balance_meth


#%% read the region code
# reg_code = 3009  # Nord-Troms
# reg_code = 3010  # Lyngen
# reg_code = 3011  # Tromsoe
# reg_code = 3012  # Soer-Troms
# reg_code = 3013  # Indre Troms
if args.reg_code == 0:
    reg_code = "AllReg"
    reg_codes = [3009, 3010, 3011, 3012, 3013]
else:
    reg_code = args.reg_code
    reg_codes = [reg_code]
# end if else


#%% get the grid-cell aggregation type and the percentile
agg_type = args.agg_type
perc = args.perc


#%% plotting flag
no_plots = args.no_plots


#%% store statistical model flag
store_mod = args.store_mod


#%% store statistical model flag
store_tr_te = True  #  args.store_tr_te


#%% generate a name prefix/suffix depending on the gridcell aggregation

# make sure that the percentile is 0 if the type is not percentile
if agg_type != "percentile":
    perc = 0
# end if

agg_str = f"{agg_type.capitalize()}{perc if perc != 0 else ''}"


#%% print a parameter summary
print(f"""
      Parameter summary
      -----------------
      model_ty     {model_ty}
      feats        {feats}
      a_p          {a_p_str}
      nan_handl    {nan_handl}
      n_best       {n_best}
      grid_search  {grid_search}
      grid_sample  {grid_sample}
      cv_on_all    {cv_on_all}
      cv_type      {cv_type}
      cv           {cv}
      cv_score     {cv_score}
      hyperp       {hyperp}
      ndlev        {ndlev}
      scale_x      {scale_x}
      balancing    {balancing}
      class_weight {class_weight}
      split        {split}
      sea          {sea}
      h_low        {h_low}
      h_hi         {h_hi}
      reg_code     {reg_code}
      balance_meth {balance_meth}
      agg_type     {agg_type}
      perc         {perc}
      no_plots     {no_plots}
      store_mod    {store_mod}
      store_tr_te  {store_tr_te}
      """)


#%% set paths --> output or plot paths will be created if necessary
data_path = f"{path_par}/IMPETUS/NORA3/"  # data path
fi_path = f"{data_path}/Plots/Feature_Importance/"  # feature importance path
fc_path = f"{data_path}/Plots/Feature_Correlation/"  # feature correlation path
mod_path = f"{data_path}/Stored_Models/{agg_str}/Between{h_low}_and_{h_hi}m/"  # directory to store the model in


#%% prepare a suffix for the model name based on the data balancing
bal_suff = ""
if balancing == "internal":
    bal_suff = "_internal"
elif balancing == "external":
    bal_suff = f"_{balance_meth}"
# end if elif


#%% set the features
if feats == "all":
    sel_feats = np.array(features)
elif feats == "best":
    try:
        best_path = f"{data_path}/Feature_Selection/"
        best_name = f"BestFeats_{model_ty}_{ndlev}DL_{reg_code}_{agg_str}_" + \
                                                             f"Between{h_low}_and_{h_hi}m_{sea}{bal_suff}_{a_p_str}.csv"
        if n_best == 0:
            n_best = None
        # end if
        sel_feats = np.array(pd.read_csv(best_path + best_name, index_col=0).index)[:n_best]
        n_best = len(sel_feats)
    except:
        print("\nNo best-feature list found in\n")
        print(best_path + best_name)
        print("\nAborting\n")
        sys.exit()
    # end try except
# end if elif

# set the following for testing the script (takes much less time)
# sel_feats = np.array(["s3", "t7", "s7"])

# print number if features
print(f"\nNumber of features: {len(sel_feats)}\n")


#%% set the feature names in the plot
pl_feats = {f:f for f in sel_feats}


#%% perform the prediction
print("\nPerforming model training and prediction...\n")
pred_test, pred_train, pred_test_all, pred_train_all, train_x, train_y, test_x, test_y, train_x_all, train_y_all, \
    test_x_all, test_y_all, model = \
    apply_stat_mod(model_ty=model_ty, ndlev=ndlev, a_p=a_p, nan_handling=nan_handl,
                   hyperp=hyperp, grid_search=grid_search, grid_sample=grid_sample,
                   cv_on_all=cv_on_all, balancing=balancing, class_weight=class_weight,
                   split=split, h_low=h_low, h_hi=h_hi,
                   agg_type=agg_type, p=perc,
                   reg_code=reg_code, sel_feats=sel_feats,
                   scale_x=scale_x, sea=sea, data_path=data_path, assume_path=True, balance_meth=balance_meth,
                   cv=cv, cv_type=cv_type,
                   return_model=True)


#%% evaluate the model
acc_train = accuracy_score(train_y, pred_train)
print(f'Accuracy balanced training data: {(acc_train * 100)} %')

acc_test = accuracy_score(test_y, pred_test)
print(f'Accuracy balanced test data:     {(acc_test * 100)} %')

acc_train_all = accuracy_score(train_y_all, pred_train_all)
print(f'Accuracy all train data:         {(acc_train_all * 100)} %')

acc_test_all = accuracy_score(test_y_all, pred_test_all)
print(f'Accuracy all test data:          {(acc_test_all * 100)} %\n')

prf1_test_all = precision_recall_fscore_support(test_y_all, pred_test_all)


#%% perform cross validation --> kind of pointless...
"""
print(f"\nPerforming {cv}-fold cross-validation...\n")

cv_accs_train_bal, split_tr = stratified_cv(model, train_x, train_y, cv=cv, return_split=True)
cv_accs_test_bal, split_te = stratified_cv(model, test_x, test_y, cv=cv, return_split=True)

print("\nCross-validation accuracies (train):")
print("   " + " ".join([f"{r_2(k):.2f}" for k in cv_accs_train_bal]) + f"  avg: {r_2(np.mean(cv_accs_train_bal)):.2f}")

print("\nCross-validation accuracies (test):")
print("   " + " ".join([f"{r_2(k):.2f}" for k in cv_accs_test_bal]) +
      f"  avg: {r_2(np.mean(cv_accs_test_bal)):.2f}\n")


cv_f1_train_bal, split_tr = stratified_cv(model, train_x, train_y, cv=cv, scoring="f1_macro", return_split=True)
cv_f1_test_bal, split_te = stratified_cv(model, test_x, test_y, cv=cv, scoring="f1_macro", return_split=True)

print("\nCross-validation F1-macros (train):")
print("   " + " ".join([f"{r_2(k):.2f}" for k in cv_f1_train_bal]) + f"  avg: {r_2(np.mean(cv_f1_train_bal)):.2f}")

print("\nCross-validation F1-macros (test):")
print("   " + " ".join([f"{r_2(k):.2f}" for k in cv_f1_test_bal]) +
      f"  avg: {r_2(np.mean(cv_f1_test_bal)):.2f}\n")
"""

#%% load the data to perform predefined cross-validation

# load data
print("\nReloading data for cross-validation.")
if cv_on_all:
    print("\nUsing ALL data in the grid search.\n")
    bal_x, bal_y, all_x, all_y = load_xlevel_preds(data_path +
                                                   (f"/Avalanche_Predictors_{ndlev}Level" +
                                                                f"/{agg_str}/Between{h_low}_and_{h_hi}m/"),
                                                   sel_feats=sel_feats, reg_code=reg_code, a_p=a_p,
                                                   split=0, sea=sea,
                                                   nlevels=ndlev, balance_meth=balance_meth, scale_x=scale_x,
                                                   verbose=True)
else:
    print("\nUsing only the training data in the grid search.\n")
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


#%% assign winter month

# set up a dictionary to choose the right classifier
classifier = {"DT":DecisionTreeClassifier, "RF":RandomForestClassifier}

all_winter = pd.Series(assign_winter_year(all_x.index, 8))
obs_yrs = np.unique(all_winter)

# assign fold labels --> depends on the year
nfolds = cv
len_fold = int(len(obs_yrs) / nfolds)

fold_label = np.zeros(len(all_winter))
for i, j in enumerate(np.arange(len(obs_yrs))[::len_fold]):
    yrs_temp = [obs_yrs[k] for k in np.arange(j, j+len_fold, 1)]
    # print(yrs_temp)
    fold_label[all_winter.isin(yrs_temp)] = i
# end for i, yr

# generate the predefined split
ps = PredefinedSplit(fold_label)


# set the model for the CV pipeline
sampler = {"SMOTE":SMOTE(), "SVMSMOTE":SVMSMOTE(), "BSMOTE":BorderlineSMOTE(), "ADASYN":ADASYN(),
           "ros":RandomOverSampler(), "rus":RandomUnderSampler(),
           "SMOTEENN":SMOTEENN(sampling_strategy="auto"), "SMOTETomek":SMOTETomek(sampling_strategy="auto")}


pipeline = make_pipeline(sampler[balance_meth], classifier[model_ty](class_weight=None))

# perform the cross-validation
scores = cross_val_score(pipeline, all_x, all_y, cv=ps, n_jobs=-1, scoring=cv_score)

print(f"\nSeasonal {cv}-fold cross-validation:\n")
print(f"Scores: {scores}\n")


#%% print the classification_report
class_rep = classification_report(test_y_all, pred_test_all)
cr_vals = precision_recall_fscore_support(test_y_all, pred_test_all)
print(class_rep)


#%% set the keys for storing the classification report
cr_keys = ["precision", "recall", "f1", "support"]
cr_test_all = {k:cr_vals[i] for i, k in enumerate(cr_keys)}


#%% plot the different metrics per class
cr_test_all["f1"]


#%% discrete histogram
fig = pl.figure()
ax00 = fig.add_subplot(111)
ax00.bar(np.arange(ndlev), cr_test_all["f1"], facecolor="none", edgecolor="black", label="F1")
ax00.bar(np.arange(ndlev), cr_test_all["precision"], facecolor="none", edgecolor="blue", label="prec")
ax00.bar(np.arange(ndlev), cr_test_all["recall"], facecolor="none", edgecolor="red", label="rec")
ax00.legend()

ax00.set_xticks(np.arange(ndlev))
ax00.set_xticklabels(np.arange(ndlev)+1)
ax00.set_xlabel("Danger level")
ax00.set_ylabel("Metric value")

ax00.set_title(f"{a_p}")

pl.show()
pl.close()


#%% plot the discrete histograms of the test data
if not no_plots:
    disc_hist([test_y_all, pred_test_all], width=[0.3, 0.2],
              color=["black", "red"],
              labels=["test_y_all", "test_pred_all"],
              xlabel="Danger level", ylabel="Number of days",
              title=f"Occurence per danger level\nbalancing={balancing}")


    disc_hist([test_y, pred_test], width=[0.3, 0.2],
              color=["black", "red"],
              labels=["test_y", "test_pred"],
              xlabel="Danger level", ylabel="Number of days",
              title=f"Occurence per danger level\nbalancing={balancing}")

    disc_hist([train_y_all, pred_train_all], width=[0.3, 0.2],
              color=["black", "red"],
              labels=["train_y_all", "train_pred_all"],
              xlabel="Danger level", ylabel="Number of days",
              title=f"Occurence per danger level\nbalancing={balancing}")


    disc_hist([train_y, pred_train], width=[0.3, 0.2],
              color=["black", "red"],
              labels=["train_y", "train_pred"],
              xlabel="Danger level", ylabel="Number of days",
              title=f"Occurence per danger level\nbalancing={balancing}")
# end if


#%% investigate feature importance
importances = model.feature_importances_
print("\nFeature importances:", importances, "\n")# partly from https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
conf_train = confusion_matrix(train_y, pred_train)
conf_test = confusion_matrix(test_y, pred_test)
conf_test_all = confusion_matrix(test_y_all, pred_test_all)

labels = conf_helper(test_y_all, conf_test_all)


imp_sort = np.argsort(importances)[::-1]

if not no_plots:
    fig = pl.figure(figsize=(4, 14))
    ax00 = fig.add_subplot(111)
    # ax01 = fig.add_subplot(122)

    ax00.barh(sel_feats[imp_sort[::-1]], importances[imp_sort[::-1]])
    ax00.set_xlabel("Importance")
    ax00.set_title(f"Feature importance ndlev={ndlev}")

    ax00.tick_params(axis='x', labelrotation=0)

    ax00.set_ylim((-1, len(sel_feats)))

    # os.makedirs(fi_path, exist_ok=True)
    # pl.savefig(fi_path + f"{model_ty}_AllFeatImportance_{ndlev}Levels.png", bbox_inches="tight", dpi=200)

    pl.show()
    pl.close()
# end if


#%% prepare confusion matrix plot
if not no_plots:
    # partly from https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
    conf_train = confusion_matrix(train_y, pred_train)
    conf_test = confusion_matrix(test_y, pred_test)
    conf_test_all = confusion_matrix(test_y_all, pred_test_all)

    labels = conf_helper(test_y_all, conf_test_all)
# end if


#%% confusion matrix plot
# some explanation to the heat map:
#     The percentage given in each panel of the heat map correspond to the fraction the values of the panel with
#     respect to the number of true values in this class; that is, it is the number in the panel divided by the sum of
#     numbers in this row.
if not no_plots:
    fig = pl.figure(figsize=(7, 5))
    ax00 = fig.add_subplot()
    hm = sns.heatmap(conf_test_all, annot=labels, fmt="", cmap="Blues", ax=ax00)

    ax00.set_xlabel("Predicted danger")
    ax00.set_ylabel("True danger")
    ax00.set_title("Confusion matrix test_all")

    pl.show()
    pl.close()
# end if


#%% perform feature correlation ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


#%% generate a heat map
"""
if not no_plots:

    #% set exposure to None
    exposure = None


    #% load the data
    feat_df = load_feats_xlevel(reg_codes=reg_codes, ndlev=ndlev, exposure=exposure, sel_feats=list(sel_feats), a_p=a_p,
                                agg_type=agg_type, perc=perc,
                                out_type="dataframe",
                                h_low=h_low, h_hi=h_hi,
                                data_path_par=data_path + "Avalanche_Predictors/")["all"]

    feat_df.drop(a_p, axis=1, inplace=True)
    feat_df.drop("reg_code", axis=1, inplace=True)


    #% take the first n features and correlate them
    sel_sub = sel_feats[imp_sort]
    feat_sub = feat_df[sel_sub]


    #% calculate the cross-correlation
    cm_pearson = feat_sub.corr()
    cm_spearman = feat_sub.corr(method='spearman')
    cm_kendall = feat_sub.corr(method='kendall')


    #% combine the Pearson and the Spearman matrices: Pearson is lower left, Spearman is upper right
    cm_pear_spear = np.zeros(np.shape(cm_pearson))

    pearson_ind = np.tril_indices_from(cm_pearson, k=-1)
    spearman_ind = np.triu_indices_from(cm_spearman, k=1)

    cm_pear_spear[pearson_ind] = np.array(cm_pearson)[pearson_ind]
    cm_pear_spear[spearman_ind] = np.array(cm_spearman)[spearman_ind]

    cm_pear_spear_df = pd.DataFrame(cm_pear_spear, index=np.array(sel_sub), columns=np.array(sel_sub))


    #% combine the Pearson and the Kendall matrices: Pearson is lower left, Kendall is upper right
    cm_pear_ken = np.zeros(np.shape(cm_pearson))

    pearson_ind = np.tril_indices_from(cm_pearson, k=-1)
    kendall_ind = np.triu_indices_from(cm_kendall, k=1)

    cm_pear_ken[pearson_ind] = np.array(cm_pearson)[pearson_ind]
    cm_pear_ken[kendall_ind] = np.array(cm_kendall)[kendall_ind]

    cm_pear_ken_df = pd.DataFrame(cm_pear_ken, index=np.array(sel_sub), columns=np.array(sel_sub))


    # ... for the Pearson and Spearman tests
    fig = pl.figure(figsize=(10, 9))
    ax00 = fig.add_subplot(111)

    p00 = sns.heatmap(cm_pear_spear_df, ax=ax00, annot=False, cmap='coolwarm', vmin=-0.8, vmax=0.8)

    ax00.set_title("Upper: Spearman, Lower: Pearson")

    # os.makedirs(fc_path, exist_ok=True)
    # pl.savefig(fc_path + "Feature_CrossCorr_Pearson_Spearman.png", bbox_inches="tight", dpi=250)

    pl.show()
    pl.close()


    # ... for the Pearson and Kendall tests
    fig = pl.figure(figsize=(10, 9))
    ax00 = fig.add_subplot(111)

    p00 = sns.heatmap(cm_pear_ken_df, ax=ax00, annot=False, cmap='coolwarm', vmin=-0.8, vmax=0.8)

    ax00.set_title("Upper: Kendall, Lower: Pearson")

    # pl.savefig(fc_path + "Feature_CrossCorr_Pearson_Kendall.png", bbox_inches="tight", dpi=250)

    pl.show()
    pl.close()
# end if
"""

#%% store the model if requested
if store_mod & ~store_tr_te:

    # generate a suffix for the number of features
    nbest_suff = ""
    if feats == "best":
        nbest_suff = f"_{n_best:02}best"
    # end if

    # generate the directory
    os.makedirs(mod_path, exist_ok=True)
    mod_name = f"{model_ty}_{ndlev}DL_{reg_code}_{agg_str}_" + \
                                               "Between{h_low}_and_{h_hi}m_{sea}{bal_suff}{nbest_suff}_{a_p_str}.joblib"
    dump(model, f"{mod_path}/{mod_name}")

    print("\nModel stored in:")
    print("  " + mod_path)
    print("as:")
    print(f"  {mod_name}\n")
# end if


#%% store the train-test data if requested; this implicitly stores the model as well
if store_tr_te:

    # generate a suffix for the number of features
    nbest_suff = ""
    if feats == "best":
        nbest_suff = f"_{n_best:02}best"
    # end if

    # generate the bundle
    bundle = {"model":model,
              "train_x":train_x, "train_y":train_y,
              "test_x":test_x, "test_y":test_y,
              "train_x_all":train_x_all, "train_y_all":train_y_all,
              "test_x_all":test_x_all, "test_y_all":test_y_all}

    # generate the directory
    os.makedirs(mod_path, exist_ok=True)
    out_name = f"{model_ty}_{ndlev}DL_{reg_code}_{agg_str}_Between{h_low}_and_{h_hi}m" + \
                                                                  f"_{sea}{bal_suff}{nbest_suff}_{a_p_str}_wData.joblib"
    dump(bundle, f"{mod_path}/{out_name}")

    print("\nModel + data stored in:")
    print("  " + mod_path)
    print("as:")
    print(f"  {out_name}\n")
# end if

