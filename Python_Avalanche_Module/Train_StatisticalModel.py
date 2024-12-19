"""
Generate a statistical model of choice for the avalanche data. Implemented are:
    - decision tree
    - logistic regression
    - support vector machine
    - nearest neighbour
    - random forest
The number of danger levels can be chosen by the user (2-4).

Note that this script includes both the automatic train-test split and the option of selecting subseasons.
"""


#%% imports
import os
import numpy as np
import pandas as pd
import pylab as pl
import argparse
import seaborn as sns
import json
from joblib import dump
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import tree as sktree
from sklearn.model_selection import train_test_split, PredefinedSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline


#%% import proprietary functions and variables
from Statistical_Prediction.Functions.Func_Discrete_Hist import disc_hist
from Statistical_Prediction.Functions.Func_Apply_StatMod import apply_stat_mod
from Statistical_Prediction.Functions.Func_DatetimeSimple import date_dt
from Statistical_Prediction.Functions.Func_Balance_Data import balance_data
from Statistical_Prediction.Functions.Func_Assign_Winter_Year import assign_winter_year
from Statistical_Prediction.Functions.Func_Load_NORA3_XLevel_Predictors import load_xlevel_preds
from Statistical_Prediction.Functions.Func_Plot_PredictorSpace import plot_2predictor_space, plot_3predictor_space
from Statistical_Prediction.Functions.Func_Round import r_0, r_2
from Statistical_Prediction.Functions.Func_Stratified_CrossValidation import stratified_cv
from Statistical_Prediction.Functions.Func_ConfMat_Helper import conf_helper
from Statistical_Prediction.Lists_and_Dictionaries.Features import features
from Statistical_Prediction.Lists_and_Dictionaries.Paths import path_par
from Statistical_Prediction.Lists_and_Dictionaries.Hyperp_Set import hypp_grid, hypp_set


#%% set up the parser
parser = argparse.ArgumentParser(
                    prog="Train_StatisticalModel",
                    description="""Trains and applies a user-chosen statistical model.""",
                    epilog="For more information consult the documentation of the function apply_stat_mod.")

# ...and add the arguments
parser.add_argument("--sel_feats", nargs="*", default=["all"],
                    help="""The features to be selected.""")
parser.add_argument("--model_ty", default="RF", type=str, choices=["DT", "LR", "SVM", "KNN", "RF"],
                    help="""The type of statistical model that is used.""")
parser.add_argument("--a_p", default="y", type=str, choices=["y", "glide_slab", "new_loose", "new_slab", "pwl_slab",
                                                             "wet_loose", "wet_slab", "wind_slab"],
                    help="""The avalanche problem that is used. Standard is 'y', i.e., the general ADL.""")
parser.add_argument("--nan_handl", default="drop", type=str, choices=["drop", "zero"],
                    help="""How to handle NaNs in the danger level data.""")
parser.add_argument("--grid_search", action="store_true", help="Perform a grid search for the hyperparameters.")
parser.add_argument("--grid_sample", type=int, default=0, help="""Number of random samples taken from the
                    hyperparameter grids. If set to 0 (default) the full grids will be searched.""")
parser.add_argument("--train_gs", action="store_true", help="""If set, only the training data are used in the
                    cross-validation during the grid search.""")
parser.add_argument("--cv_type", type=str, default="seasonal", choices=["seasonal", "stratified"],
                    help="""The type of folds in the cross-validation during grid search.""")
parser.add_argument("--cv", type=int, default=5,
                    help="""The number folds in the cross-validation during grid search.""")
parser.add_argument("--cv_score", type=str, default="f1_macro", help="""The type of skill score used in the
                    cross-validation during grid search. For possible choices see
                    https://scikit-learn.org/stable/modules/model_evaluation.html""")
parser.add_argument("--hyperp", type=str, help="""The hyperparameters or hyperparameter grid for the statistical model.
                    Must be submitted as a JSON-string that can be interpreted as a dictionary, e.g.:
                        '{"max_depth":5}' in the case of a decision tree. If empty, default values will be used.""")
parser.add_argument("--ndlev", type=int, default=2, choices=[2, 3, 4],
                    help="""The number of danger levels to be predicted.""")
parser.add_argument("--scale_x", action="store_true", help="Perform a predictor scaling.")
parser.add_argument("--balancing", type=str, default="external", choices=["none", "internal", "external"],
                    help="""Controls the kind of balancing to be performed.""")
parser.add_argument("--split", nargs="*", default=[0.33], help="""Fraction of data to be used as test data.""")
parser.add_argument("--sea", type=str, default="full", choices=["full", "winter", "spring"],
                    help="""The season for which the model is trained. The choices mean: full=Dec-May, winter=Dec-Feb,
                    spring=Mar-May.""")
parser.add_argument("--h_low", type=int, default=400, help="The lower threshold of the grid cell altitude.")
parser.add_argument("--h_hi", type=int, default=900, help="The upper threshold of the grid cell altitude.")
parser.add_argument("--reg_code", type=int, default=0, help="""The region code of the region for which the danger levels
                    will be predicted. Set 0 (default) to use all regions.""")
parser.add_argument("--balance_meth", type=str, default="SMOTE",
                    choices=["undersample", "rus", "ros", "SMOTE", "ADASYN"],
                    help="""The balancing method applied during the loading of the data. This is only relevant if
                    balancing is set to external.""")
parser.add_argument("--agg_type", default="mean", type=str, choices=["mean", "median", "percentile"],
                    help="""The type of aggregation used for the grid cells.""")
parser.add_argument("--perc", type=int, default=90, help="""The percentile used in the grid-cell aggregation.""")
parser.add_argument("--no_plots", action="store_true", help="Set this parameter to suppress plotting.")
parser.add_argument("--store_mod", action="store_true", help="Set this parameter to store the statistical model.")

args = parser.parse_args()


#%% set the first features
sel_feats = args.sel_feats
sel_feats_info = sel_feats
if sel_feats == ["all"]:
    sel_feats = features  # ["wdrift3_3", "s4", "t5", "r7", "NSW5"]
    # sel_feats = ['wdrift_3', 't5', 'r7', 'RH2', 'dtempd1', 'wind_direction', 'dtempd3', 'dtemp2', 'SnowDepth4',
    #              'dtemp3', 'r1']
    sel_feats_info = "all"
# end if


#%% choose the model; choices are
#   DT  --> decision tree
#   LR  --> logistic regression
#   SVM --> support vector machine
#   KNN --> nearest neighbour
#   RF  --> random forest
model_ty = args.model_ty


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
train_gs = args.train_gs


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
    print("\nNo hyperparameters provided, using default.\n")
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
    reg_code = "AllReg"  #  sys.argv[1]
else:
    reg_code = args.reg_code
# end if else


#%% get the grid-cell aggregation type and the percentile
agg_type = args.agg_type
perc = args.perc


#%% plotting flag
no_plots = args.no_plots


#%% store statistical model flag
store_mod = args.store_mod


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
      sel_feats    {sel_feats_info}
      model_ty     {model_ty}
      a_p          {a_p_str}
      nan_handl    {nan_handl}
      grid_search  {grid_search}
      grid_sample  {grid_sample}
      train_gs     {train_gs}
      cv_type      {cv_type}
      cv           {cv}
      cv_score     {cv_score}
      hyperp       {hyperp}
      ndlev        {ndlev}
      scale_x      {scale_x}
      balancing    {balancing}
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
      """)


#%% set the feature names in the plot
pl_feats = {f:f for f in sel_feats}


#%% set paths --> output or plot paths will be created if necessary
data_path = f"{path_par}/IMPETUS/NORA3/"  # data path
pred_path = f"{data_path}/Avalanche_Predictors_{ndlev}Level/{agg_str}/Between{h_low}_and_{h_hi}m/"
fi_path = f"{data_path}/Plots/Feature_Importance/"  # feature importance path
tree_pl_path = f"{data_path}/Plots/Tree_Plots/"  # path for DT visualisation
obs_per_path = "{data_path}/Plots/Danger_Levels_2018_2023/"  # path for observation period prediction plots
mod_path = f"{data_path}/Stored_Models/{agg_str}/Between{h_low}_and_{h_hi}m/"  # directory to store the model in


#%% perform the prediction
print("\nPerforming model training and prediction...\n")
pred_test, pred_train, pred_test_all, pred_train_all, train_x, train_y, test_x, test_y, train_x_all, train_y_all, \
    test_x_all, test_y_all, model = \
    apply_stat_mod(model_ty=model_ty, ndlev=ndlev, a_p=a_p, nan_handling=nan_handl,
                   hyperp=hyperp, grid_search=grid_search, grid_sample=grid_sample,
                   train_gs=train_gs, balancing=balancing, split=split, h_low=h_low, h_hi=h_hi,
                   agg_type=agg_type, p=perc,
                   reg_code=reg_code, sel_feats=sel_feats,
                   scale_x=scale_x, sea=sea, data_path=data_path, assume_path=True, balance_meth=balance_meth,
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


#%% perform cross validation
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


#%% perform predefined cross-validation

# load data
print("\nReloading data for cross-validation.")
bal_x, bal_y, all_x, all_y = load_xlevel_preds(data_path + (f"/Avalanche_Predictors_{ndlev}Level" +
                                                            f"/{agg_str}/Between{h_low}_and_{h_hi}m/"),
                                               sel_feats=sel_feats, reg_code=reg_code, a_p=a_p,
                                               split=0, sea=sea,
                                               nlevels=ndlev, balance_meth=balance_meth, scale_x=scale_x,
                                               verbose=True)


#%% assign winter month

# set up a dictionary to choose the right classifier
classifier = {"DT":DecisionTreeClassifier(class_weight=None),
              "RF":RandomForestClassifier(class_weight=None),
              "SVM":SVC(class_weight=None),
              "LR":LogisticRegression(class_weight=None),
              "KNN":KNeighborsClassifier()}

all_winter = pd.Series(assign_winter_year(all_x.index, 8))
obs_yrs = np.unique(all_winter)

# assign fold labels --> depends on the year
nfolds = 3
len_fold = int(len(obs_yrs) / nfolds)

fold_label = np.zeros(len(all_winter))
for i, yr in enumerate(obs_yrs[::len_fold]):
    yrs_temp = [yr_temp for yr_temp in np.arange(yr, yr+len_fold, 1)]
    fold_label[all_winter.isin(yrs_temp)] = i
# end for i, yr

# generate the predefined split
ps = PredefinedSplit(fold_label)


pipeline = make_pipeline(SMOTE(), classifier[model_ty])

# perform the cross-validation
scores = cross_val_score(pipeline, all_x, all_y, cv=ps, n_jobs=-1, scoring=cv_score)

print(f"\nSeasonal {cv}-fold cross-validation:\n")
print(f"Scores: {scores}\n")


#%% load and scale the full data for later plotting purposes
# split
split_temp = 0.33

# load the data
df = pd.read_csv(pred_path + f"Features_{ndlev}Level_All_{agg_str}_Between{h_low}_{h_hi}m_AllReg.csv")

df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)
df_x = df[sel_feats]

if scale_x:
    scaler = StandardScaler()
    df_x_sc = scaler.fit_transform(df_x)
    df_x = pd.DataFrame({k:df_x_sc[:, i] for i, k in enumerate(df_x.columns)})
    df_x.set_index(df.index, inplace=True)
# end if

df = pd.concat([df_x, df["reg_code"], df[a_p]], axis=1)


#%% perform a simpler validation: generate random balanced and unbalanced splits of the data and check the model
#   performance

# loop over the three seasons
for seai in ["full", "winter", "spring"]:
    if seai == "full":
        df_t = df[((df.index.month == 12) | (df.index.month == 1) | (df.index.month == 2) |
                   (df.index.month == 3) | (df.index.month == 4) | (df.index.month == 5))]
    elif seai == "winter":
        df_t = df[((df.index.month == 12) | (df.index.month == 1) | (df.index.month == 2))]
    elif seai == "spring":
        df_t = df[((df.index.month == 3) | (df.index.month == 4) | (df.index.month == 5))]
    # end if elif

    df_y = df_t[a_p]
    df_x = df_t[sel_feats]

    acc_te_all, acc_te_bal, acc_tr_all, acc_tr_bal = [], [], [], []
    for _ in range(5):

        # balance the data
        bal_x, bal_y = balance_data(df_x, df_y, method=balance_meth)

        # perform the train-test split to extract a certain amount of data
        tr_x_all, te_x_all, tr_y_all, te_y_all = train_test_split(df_x, df_y, test_size=split_temp, shuffle=True,
                                                                  stratify=df_y)
        tr_x_bal, te_x_bal, tr_y_bal, te_y_bal = train_test_split(bal_x, bal_y, test_size=split_temp, shuffle=True,
                                                                  stratify=bal_y)

        acc_te_all.append(accuracy_score(te_y_all, model.predict(te_x_all)))
        acc_te_bal.append(accuracy_score(te_y_bal, model.predict(te_x_bal)))
        acc_tr_all.append(accuracy_score(tr_y_all, model.predict(tr_x_all)))
        acc_tr_bal.append(accuracy_score(tr_y_bal, model.predict(tr_x_bal)))
    # end for _

    print(f"\nRandom data selection accuracies {seai} season:\n")

    print(f"Using {(1-split_temp)*100}% of data, unbalanced:")
    print(" ".join([f"{r_2(k):.2f}" for k in acc_te_all]) + f"  avg: {r_2(np.mean(acc_te_all)):.2f}")

    print(f"\nUsing {(1-split_temp)*100}% of data, balanced:")
    print(" ".join([f"{r_2(k):.2f}" for k in acc_te_bal]) + f"  avg: {r_2(np.mean(acc_te_bal)):.2f}")

    print(f"\nUsing {(split_temp)*100}% of data, unbalanced:")
    print(" ".join([f"{r_2(k):.2f}" for k in acc_tr_all]) + f"  avg: {r_2(np.mean(acc_tr_all)):.2f}")

    print(f"\nUsing {(split_temp)*100}% of data, balanced:")
    print(" ".join([f"{r_2(k):.2f}" for k in acc_tr_bal]) + f"  avg: {r_2(np.mean(acc_tr_bal)):.2f}\n\n")

# end for seai


#%% print the classification_report
class_rep = classification_report(test_y_all, pred_test_all)
print(class_rep)


#%% investigate feature importance
if not no_plots:
    if model_ty in ["DT", "RF"]:
        importances = model.feature_importances_
        print("\nFeature importances:", importances, "\n")

        imp_sort = np.argsort(importances)

        fig = pl.figure(figsize=(12, 5))
        ax00 = fig.add_subplot(121)
        ax01 = fig.add_subplot(122)

        ax00.bar(np.array(sel_feats)[imp_sort], importances[imp_sort])
        ax00.set_ylabel("Importance")
        ax00.set_title("Feature importance")
        ax00.tick_params(axis='x', labelrotation=90)

        ax01.pie(importances, labels=sel_feats)

        os.makedirs(fi_path, exist_ok=True)
        pl.savefig(fi_path + f"{model_ty}_FeatImportance_{ndlev}Levels.png", bbox_inches="tight", dpi=200)

        pl.show()
        pl.close()
    # end if
# end if


#%% prepare confusion matrix plot
# partly from https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
conf_train = confusion_matrix(train_y, pred_train)
conf_test = confusion_matrix(test_y, pred_test)
conf_test_all = confusion_matrix(test_y_all, pred_test_all)
"""
# get the support values
supps_test = []
supps_test_all = []
for n in np.arange(ndlev):
    supps_test.append(np.sum(test_y == n))
    supps_test_all.append(np.sum(test_y_all == n))
# end n
supps_test = np.array(supps_test)
supps_test_all = np.array(supps_test_all)

gcounts_test_all = ["{0:0.0f}".format(value) for value in conf_test_all.flatten()]
gperc_test_all = ["{0:.2%}".format(value) for value in conf_test_all.flatten()/np.repeat(supps_test_all, ndlev)]

labels = [f"{v1}\n{v2}" for v1, v2 in zip(gcounts_test_all, gperc_test_all)]
"""
labels = conf_helper(test_y_all, conf_test_all)


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
# end if


#%% plot the decision tree if chosen
if not no_plots:
    if model_ty == "DT":

        if model.max_depth < 4:  # only generate the plot if the depth is smaller than 4
            fig = pl.figure(figsize=(12, 7.5))
            ax00 = fig.add_subplot()

            skp = sktree.plot_tree(model, feature_names=list(pl_feats.values()), class_names=["0", "1", "2", "3"],
                                   filled=True, ax=ax00)

            os.makedirs(tree_pl_path, exist_ok=True)
            pl.savefig(tree_pl_path + "DecisionTree.pdf", bbox_inches="tight", dpi=100)

            pl.show()
            pl.close()
        # end if
    # end if
# end if


#%% investigate the model performance per region
regions_dic = {3009:"Nord-Troms", 3010:"Lyngen", 3011:"Tromsoe", 3012:"Soer-Troms", 3013:"Indre Troms"}

if sea == "winter":
    df_t = df[((df.index.month == 12) | (df.index.month == 1) | (df.index.month == 2))]
elif sea == "spring":
    df_t = df[((df.index.month == 3) | (df.index.month == 4) | (df.index.month == 5))]
# end if elif

reg_accs_all = {}
reg_accs_bal = {}
for reg_code_pl in regions_dic.keys():

    # get the data for the current region
    df_x = df_t[df_t["reg_code"] == reg_code_pl]

    df_y = df_x[a_p]
    df_x = df_x[sel_feats]

    # balance the data
    bal_x, bal_y = balance_data(df_x, df_y, method="SMOTE")

    # make the prediction based on the x-data
    pred_y = model.predict(df_x)
    pred_y_bal = model.predict(bal_x)

    # calculate the accuracy
    reg_accs_all[reg_code_pl] = accuracy_score(df_y, pred_y)
    reg_accs_bal[reg_code_pl] = accuracy_score(bal_y, pred_y_bal)
# end for reg

print("\nAccuracy per region:")
print("                        all    bal")
for reg in regions_dic.keys():
    print(f"({reg}) {regions_dic[reg]:15}  {r_2(reg_accs_all[reg]):4}   {r_2(reg_accs_bal[reg])}")
# end for reg



#%% investigate the model performance per seasons
# set the indices for the months depending on the seasons
if sea == "full":
    dyr = 1
    sta_mon = 7
    end_mon = 7
elif sea == "winter":
    dyr = 1
    sta_mon = 12
    end_mon = 3
elif sea == "spring":
    dyr = 0
    sta_mon = 3
    end_mon = 6
# end if elif

inds = {}
yrs = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

reg_accs_all = {}
reg_accs_bal = {}
for yr in yrs:
    inds = (df.index >= date_dt(yr-dyr, sta_mon, 1)) & (df.index < date_dt(yr, end_mon, 1))

    # get the index for the current season
    df_x = df[inds]

    df_y = df_x[a_p]
    df_x = df_x[sel_feats]

    # balance the data
    bal_x, bal_y = balance_data(df_x, df_y, method="SMOTE")

    # make the prediction based on the x-data
    pred_y = model.predict(df_x)
    pred_y_bal = model.predict(bal_x)

    reg_accs_all[yr] = accuracy_score(df_y, pred_y)
    reg_accs_bal[yr] = accuracy_score(bal_y, pred_y_bal)
# end for yr

print("\nAccuracy per season:")
print("      all    bal (SMOTE)")
for yr in yrs:
    print(f"{yr}  {r_2(reg_accs_all[yr]):4}   {r_2(reg_accs_bal[yr])}")
# end for reg


#%% investigate the model performance per seasons for each individual region
inds = {}

reg_accs_all = {}
reg_accs_bal = {}
for reg_code_pl in regions_dic.keys():
    reg_accs_all[reg_code_pl] = {}
    reg_accs_bal[reg_code_pl] = {}

    # get the data for the current region
    df_x = df[df["reg_code"] == reg_code_pl]

    for yr in yrs:

        # get the index for the current season
        inds = (df_x.index >= date_dt(yr-dyr, sta_mon, 1)) & (df_x.index < date_dt(yr, end_mon, 1))
        df_x_yr = df_x[inds]

        df_y = df_x_yr[a_p]
        df_x_fin = df_x_yr[sel_feats]

        try:
            # balance the data
            bal_x, bal_y = balance_data(df_x_fin, df_y, method="SMOTE")
            pred_y_bal = model.predict(bal_x)
            reg_accs_bal[reg_code_pl][yr] = accuracy_score(bal_y, pred_y_bal)
        except:
            reg_accs_bal[reg_code_pl][yr] = np.nan
        # end try except

        # make the prediction based on the x-data
        pred_y = model.predict(df_x_fin)
        reg_accs_all[reg_code_pl][yr] = accuracy_score(df_y, pred_y)

    # end for yr
# end for reg_code_pl

print("\nAccuracy unbalanced data per season per region:")
reg_str = "             " + " ".join([f"{k:12}" for k in regions_dic.values()])
print(reg_str)

for yr in yrs:
    yr_str = str(yr) + " " + " ".join([f"{r_2(reg_accs_all[k][yr]):12}" for k in regions_dic.keys()])
    print(yr_str)
# end for reg

print("\nAccuracy balanced data per season per region (SMOTE):")
reg_str = "             " + " ".join([f"{k:12}" for k in regions_dic.values()])
print(reg_str)

for yr in yrs:
    yr_str = str(yr) + " " + " ".join([f"{r_2(reg_accs_bal[k][yr]):12}" for k in regions_dic.keys()])
    print(yr_str)
# end for reg

print("\nWhere there are nan values there was not enough data.\n")


#%% plot the accuracy per region per season
df_reg_accs_all = pd.DataFrame(reg_accs_all)
df_reg_accs_bal = pd.DataFrame(reg_accs_bal)
min_pl = np.min([df_reg_accs_all.min(axis=None), df_reg_accs_bal.min(axis=None)])
max_pl = np.max([df_reg_accs_all.max(axis=None), df_reg_accs_bal.max(axis=None)])
range_pl = max_pl - min_pl
dy = range_pl * 0.1
ylim = (min_pl-dy, max_pl+dy)

if not no_plots:

    fig = pl.figure(figsize=(9, 3.5))
    ax = fig.add_subplot(121)
    ax1 = fig.add_subplot(122)

    for reg in regions_dic.keys():
        ax.plot(yrs, reg_accs_all[reg].values(), label=regions_dic[reg], marker="o")
        ax1.plot(yrs, reg_accs_bal[reg].values(), label=regions_dic[reg], marker="o")
    # end for reg

    ax.legend(ncol=2)

    ax.set_ylabel("Accuracy")

    ax.set_ylim(ylim)
    ax1.set_ylim(ylim)
    ax1.set_yticklabels([])

    ax.set_title("Unbalanced data")
    ax1.set_title("Balanced data (SMOTE)")

    fig.suptitle("Accuracy per region and season")
    fig.subplots_adjust(wspace=0.05, top=0.85)

    pl.show()
    pl.close()

# end if


#%% plot the daily danger levels for a specific region
if not no_plots:

    # set the region
    reg_code_pl = 3009
    regions_dic = {3009:"Nord-Troms", 3010:"Lyngen", 3011:"Troms${\o}$", 3012:"S${\o}$r-Troms", 3013:"Indre Troms"}

    # get the data for the current region
    df_x = df[df["reg_code"] == reg_code_pl]

    df_y = df_x[a_p]
    df_x = df_x[sel_feats]

    # make the prediction based on the x-data
    pred_y = model.predict(df_x)

    # set the indices for the months depending on the seasons
    if sea == "full":
        dyr = 1
        sta_mon = 7
        end_mon = 7
    elif sea == "winter":
        dyr = 1
        sta_mon = 12
        end_mon = 3
    elif sea == "spring":
        dyr = 0
        sta_mon = 3
        end_mon = 6
    # end if elif

    inds = {}
    yrs = [2018, 2019, 2020, 2021, 2022, 2023]
    for yr in yrs:
        inds[yr] = (df_y.index >= date_dt(yr-dyr, sta_mon, 1)) & (df_y.index < date_dt(yr, end_mon, 1))
    # end for yr

    # calculate the range for the y-axis
    extend_y = ((np.max(df_y) - np.min(df_y)) + 1) / 4

    # 2018-2020
    fig = pl.figure(figsize=(10, 12))
    axes = [fig.add_subplot(311), fig.add_subplot(312), fig.add_subplot(313)]

    for i, yr in enumerate(yrs[:3]):

        axes[i].bar(df_y.index[inds[yr]], df_y[inds[yr]]+1, width=1, facecolor="lightgray",
                    edgecolor="gray", label="Truth (test)")
        axes[i].plot(df_y.index[inds[yr]], pred_y[inds[yr]]+1, c="black", linewidth=0.75, label="Prediction")

        axes[i].legend(ncol=2, loc="upper right")

        axes[i].set_xlim(np.min(df_y.index[inds[yr]]), np.max(df_y.index[inds[yr]]))
        axes[i].set_ylim((0, np.max(df_y) + 1 + extend_y))

        for y in np.unique(test_y + 1):
            axes[i].axhline(y=y, linestyle="--", c="black", linewidth=0.5)
        # end for y

        axes[i].set_yticks(np.unique(test_y + 1))

        axes[i].set_title(f"Season {yr-1}/{yr}")
        axes[i].set_ylabel("Danger level")
    # end for i, yr

    fig.suptitle(f"True and predicted danger level in {regions_dic[reg_code_pl]}")
    fig.subplots_adjust(top=0.93)

    # os.makedirs(obs_per_path, exist_ok=True)
    # pl.savefig(obs_per_path + f"DangerLevel{ndlev}_Sea{yr}_True_and_Pred_2018_20.png", bbox_inches="tight", dpi=200)

    pl.show()
    pl.close()

    # 2021-2023
    fig = pl.figure(figsize=(10, 12))
    axes = [fig.add_subplot(311), fig.add_subplot(312), fig.add_subplot(313)]

    for i, yr in enumerate(yrs[3:]):

        axes[i].bar(df_y.index[inds[yr]], df_y[inds[yr]]+1, width=1, facecolor="lightgray",
                    edgecolor="gray", label="Truth (test)")
        axes[i].plot(df_y.index[inds[yr]], pred_y[inds[yr]]+1, c="black", linewidth=0.75, label="Prediction")

        axes[i].legend(ncol=2, loc="upper right")

        axes[i].set_xlim(np.min(df_y.index[inds[yr]]), np.max(df_y.index[inds[yr]]))
        axes[i].set_ylim((0, np.max(df_y) + 1 + extend_y))

        for y in np.unique(test_y + 1):
            axes[i].axhline(y=y, linestyle="--", c="black", linewidth=0.5)
        # end for y

        axes[i].set_yticks(np.unique(test_y + 1))

        axes[i].set_title(f"Season {yr-1}/{yr}")
        axes[i].set_ylabel("Danger level")
    # end for i, yr

    fig.suptitle(f"True and predicted danger level in {regions_dic[reg_code_pl]}")
    fig.subplots_adjust(top=0.93)

    # pl.savefig(obs_per_path + f"DangerLevel{ndlev}_Sea{yr}_True_and_Pred_2021_23.png", bbox_inches="tight", dpi=200)

    pl.show()
    pl.close()
# end if


#%% if a specific year was extracted as test data perform a specific accuracy test for this year
""" --> no longer possible
if ((split > 1) & (split < 3000)):
    year_df = df[df.index.year == int(split)]

    year_x = year_df[sel_feats]
    year_y = year_df["y"]

    # unbalanced
    year_pred = model.predict(year_x)

    year_acc = accuracy_score(year_y, year_pred)

    # balanced
    year_x_bal, year_y_bal = balance_data(year_x, year_y, method=balance_meth, excl_fewer=100)

    year_pred_bal = model.predict(year_x_bal)

    year_acc_bal = accuracy_score(year_y_bal, year_pred_bal)

    print()
    print(f"Accuracy in year {int(split)} (unbalanced): {year_acc}")
    print(f"Accuracy in year {int(split)} (balanced):   {year_acc_bal}\n")
# end if


#%% if a specific region was extracted as test data perform a specific accuracy test for this region
if split > 3000:
    year_df = df[df["reg_code"] == int(split)]

    region_x = year_df[sel_feats]
    region_y = year_df["y"]

    if scale_x:
        scaler = StandardScaler()
        region_x = scaler.fit_transform(region_x)
    # end if

    # unbalanced
    region_pred = model.predict(region_x)

    region_acc = accuracy_score(region_y, region_pred)

    # balanced
    region_x_bal, region_y_bal = balance_data(region_x, region_y, method=balance_meth, excl_fewer=100)

    region_pred_bal = model.predict(region_x_bal)

    region_acc_bal = accuracy_score(region_y_bal, region_pred_bal)

    print()
    print(f"Accuracy for region {int(split)} (unbalanced): {region_acc}")
    print(f"Accuracy for region {int(split)} (balanced):   {region_acc_bal}\n")
# end if
"""

#%% when predicting more than 2 danger levels calculate the "distance" of the prediction to the truth: 1, 2, or 3 levels
if ndlev > 2:

    ddls = np.arange(-(ndlev-1), ndlev, 1)

    del_y_test = pred_test - test_y
    del_y_test_all = pred_test_all - test_y_all

    ndel_test = np.array([np.sum(del_y_test == i) for i in ddls])
    ndel_test_all = np.array([np.sum(del_y_test_all == i) for i in ddls])

    # fig = pl.figure(figsize=(8, 5))
    if not no_plots:
        xtick1 = [r_0(ndel_te / np.sum(ndel_test) * 100) for d, ndel_te in zip(ddls, ndel_test)]
        xtick2 = [r_0(ndel_te / np.sum(ndel_test_all) * 100) for d, ndel_te in zip(ddls, ndel_test_all)]
        xtick = [f"{int(xt1)}, {int(xt2)}" for xt1, xt2 in zip(xtick1, xtick2)]

        disc_hist([del_y_test, del_y_test_all], classes=ddls, width=[0.4, 0.3], color=["black", "gray"],
                  labels=["test bal", "test all"],
                  xlabel="Delta to truth", ylabel="Number of cases", title="Distance to true danger level",
                  add_xtick=xtick)
    # end if

    print("\nPrecentages of 'distances' to truth (balanced test):")
    print({d:f"{r_2(ndel_te / np.sum(ndel_test) * 100)} %" for d, ndel_te in zip(ddls, ndel_test)})

    print("\nPrecentages of 'distances' to truth (all test):")
    print({d:f"{r_2(ndel_te / np.sum(ndel_test_all) * 100)} %" for d, ndel_te in zip(ddls, ndel_test_all)})

# end if


#%% plot predictors in predictor space
if not no_plots:
    test_x_all_arr = np.array(test_x_all)
    # concatenate the training and test data to the full dataset
    data_x = np.concatenate([train_x_all, test_x_all_arr])

    if len(sel_feats) == 2:
        plot_2predictor_space(data_x=data_x, model=model, test_x=test_x_all_arr, test_y=test_y_all, sel_feats=sel_feats,
                              h=0.5)
    elif len(sel_feats) == 3:
        plot_3predictor_space(model=model, data_x=test_x_all_arr, data_y=test_y_all, sel_feats=sel_feats)
    # end if elif
# end if


#%% store the model if requested
if store_mod:
    os.makedirs(mod_path, exist_ok=True)
    dump(model, f"{mod_path}/{model_ty}_{ndlev}DL_{reg_code}_Between{h_low}_and_{h_hi}m_{sea}_{a_p_str}.joblib")

    print("\nModel stored in:")
    print("  " + mod_path)
    print("as:")
    print(f"  {mod_path}/{model_ty}_{ndlev}DL_{reg_code}_Between{h_low}_and_{h_hi}m_{sea}_{a_p_str}.joblib\n")
# end if


