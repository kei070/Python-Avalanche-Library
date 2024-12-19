"""
Generate an artificial neural network (ANN) tp predict avalanche danger levels.
The number of danger levels can be chosen by the user (2-4).

Note that this script includes both the automatic train-test split and the option of selecting subseasons.
"""


#%% imports
import os
import sys
import numpy as np
import pandas as pd
import pylab as pl
import argparse
import seaborn as sns
from joblib import dump
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import PredefinedSplit
from joblib import dump


#%% import proprietary functions
from Statistical_Prediction.Functions.Func_Apply_ANN_Sequential import apply_ANN
from Statistical_Prediction.Functions.Func_Count_DangerLevels import count_dl_days
from Statistical_Prediction.Functions.Func_Discrete_Hist import disc_hist
from Statistical_Prediction.Functions.Func_DatetimeSimple import date_dt
from Statistical_Prediction.Functions.Func_Plot_PredictorSpace import plot_2predictor_space, plot_3predictor_space
from Statistical_Prediction.Functions.Func_Round import r_2
from Statistical_Prediction.Functions.Func_Convert_to_Categorcal import prob_to_cat
from Statistical_Prediction.Functions.Func_Load_NORA3_XLevel_Predictors import load_xlevel_preds
from Statistical_Prediction.Functions.Func_Assign_Winter_Year import assign_winter_year
from Statistical_Prediction.Functions.Func_Progressbar import print_progress_bar
from Statistical_Prediction.Lists_and_Dictionaries.Features import feats_paper1
from Statistical_Prediction.Lists_and_Dictionaries.Paths import path_par


#%% get the features from the paper 1 dictionary
features = list(feats_paper1.keys())


#%% set up the parser
parser = argparse.ArgumentParser(
                    prog="Train_ANN",
                    description="""Trains and applies a neural network.""",
                    epilog="For more information consult the documentation of the function apply_ANN.")

# ...and add the arguments
parser.add_argument("--n_mod", type=int, default=500, help="""The number of models to be trained. Due to the initial
                    parameter set-up being random, the ANN has an intrinsic randomness. This implies more or less
                    different results for different trainings.""")
parser.add_argument("--feats", type=str, default="best", choices=["all", "best"],
                    help="""Indicator for the feature to be considered. Either all or best.""")
parser.add_argument("--n_best", type=int, default=20, help="""The number of best features to be used from the best
                    features list from the random forest model. Set to 0 (default) to use all 'best' parameters found in
                    the feature search.""")
parser.add_argument("--ndlev", type=int, default=2, choices=[2, 3, 4],
                    help="""The number of danger levels to be predicted.""")
parser.add_argument("--a_p", default="y", type=str, choices=["y", "glide_slab", "new_loose", "new_slab", "pwl_slab",
                                                             "wet_loose", "wet_slab", "wind_slab"],
                    help="""The avalanche problem that is used. Standard is 'y', i.e., the general ADL.""")
parser.add_argument("--nan_handl", default="drop", type=str, choices=["drop", "zero"],
                    help="""How to handle NaNs in the danger level data.""")
parser.add_argument("--cv_on_all", action="store_true", help="""If set, the k-fold cross-validation is performed on the
                    the whole dataset instead of only the training data. This will also translate to the cross-
                    validation during the grid search.""")
parser.add_argument("--cv_type", type=str, default="seasonal", choices=["seasonal", "stratified"],
                    help="""The type of folds in the cross-validation during grid search.""")
parser.add_argument("--cv", type=int, default=3, help="""The number folds in the cross-validation.""")
parser.add_argument("--cv_score", type=str, default="f1_macro", help="""The type of skill score used in the
                    cross-validation during grid search. For possible choices see
                    https://scikit-learn.org/stable/modules/model_evaluation.html""")
parser.add_argument("--n_in_nodes", type=int, default=8, help="The number of nodes in the input layer.")
parser.add_argument("--in_dropout", type=float, default=0.1, help="The dropout fraction for the input layer.")
parser.add_argument("--n_hid_nodes", type=int, default=[12], nargs="+", help="""A list of integers containing the number
                    of nodes for the hidden layers. The length of the list equates the number of hidden layers.""")
parser.add_argument("--dropouts", type=float, default=[0.1], nargs="+", help="""A list of dropout fractions for the
                    hidden layers. Must have the same length as the hidden-layers list.""")
parser.add_argument("--epochs", type=int, default=100, help="The number epochs during training.")
parser.add_argument("--batch_size", type=int, default=64, help="The size of the batches used in one epoch in training.")
parser.add_argument("--learning_rate", type=float, default=0.001, help="The learning rate of the ANN.")
parser.add_argument("--no_scale_x", action="store_true", help="Perform a predictor scaling.")
parser.add_argument("--balancing", type=str, default="external", choices=["none", "internal", "external"],
                    help="""Controls the kind of balancing to be performed.""")
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
                    choices=["undersample", "rus", "ros", "SMOTE", "SVMSMOTE", "BSMOTE", "ADASYN"],
                    help="""The balancing method applied during the loading of the data. This is only relevant if
                    balancing is set to external.""")
parser.add_argument("--agg_type", default="mean", type=str, choices=["mean", "median", "percentile"],
                    help="""The type of aggregation used for the grid cells.""")
parser.add_argument("--perc", type=int, default=90, help="""The percentile used in the grid-cell aggregation.""")
parser.add_argument("--no_plots", action="store_true", help="Set this parameter to suppress plotting.")
parser.add_argument("--store_mod", action="store_true", help="Set this parameter to store the statistical model.")
parser.add_argument("--store_tr_te", action="store_true", help="""Set this parameter to store the train-test split on
                    which model is trained. The data are stored as a .joblib file together with the model. Note that
                    the model is then stored irrespective of the store_mod parameter.""")
parser.add_argument("--Sharma", action="store_true", help="""If True, the ANN configuration proposed by Sharma et al.
                    (2023, see their Table. 4) is implemented. This corresponds to
                    n_in_nodes = 48
                    in_dropout = 0.2
                    n_hid_nodes = [24, 16]
                    dropouts = [0.1, 0.1]
                    epochs = 100
                    batch_size = 64
                    learning_rate = 0.001""")

args = parser.parse_args()


#%% get the number of models to be trained
n_mod = args.n_mod


#%% get the features indicator
feats = args.feats
n_best = args.n_best

if feats == "all":
    n_best = 0
# end if


#%% set the number of danger levels
ndlev = args.ndlev


#%% get the avalanche problem
a_p = args.a_p
nan_handl = args.nan_handl

a_p_str = a_p
if a_p == "y":
    a_p_str = "general"
# end if


#%% set the number of folds in the cross-validation and its type
cv_type = args.cv_type
cv = args.cv
cv_score = args.cv_score
cv_on_all = args.cv_on_all


#%% set the number of input-layer nodes
n_in_nodes = args.n_in_nodes


#%% set the dropout fraction of the input layer
in_dropout = args.in_dropout


#%% set the number of nodes in the hidden layers and the number of hidden layers
n_hid_nodes = args.n_hid_nodes


#%% set the dropouts for the hidden layers
dropouts = args.dropouts


#%% set the epochs
epochs = args.epochs


#%% set the epochs
batch_size = args.batch_size


#%% set the learning rate of the neural network
learning_rate = args.learning_rate


#%% determine if the variables will be scaled or not
scale_x = not args.no_scale_x


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


#%% get the grid-cell aggregation type and the percentile
agg_type = args.agg_type
perc = args.perc


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


#%% plotting flag
no_plots = args.no_plots


#%% store statistical model flag
store_mod = args.store_mod


#%% store statistical model flag
store_tr_te = args.store_tr_te


#%% get the Sharma parameter
print("\n\nSHARMA IS TRUE ANYWAY!!!\n\n")
if True:# args.Sharma:
    print("\nUsing the Sharma et al. (2023) setup.\n")
    n_in_nodes = 48
    in_dropout = 0.2
    n_hid_nodes = [24, 16]
    dropouts = [0.1, 0.1]
    epochs = 100
    batch_size = 64
    learning_rate = 0.001
# end if


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
      n_mod         {n_mod}
      feats         {feats}
      n_best        {n_best}
      ndlev         {ndlev}
      n_in_nodes    {n_in_nodes}
      in_dropout    {in_dropout}
      n_hid_nodes   {n_hid_nodes}
      dropouts      {dropouts}
      epochs        {epochs}
      batch_size    {batch_size}
      learning_rate {learning_rate}
      scale_x       {scale_x}
      balancing     {balancing}
      split         {split}
      sea           {sea}
      cv_on_all     {cv_on_all}
      cv_type       {cv_type}
      cv            {cv}
      cv_score      {cv_score}
      h_low         {h_low}
      h_hi          {h_hi}
      reg_code      {reg_code}
      agg_type      {agg_type}
      perc          {perc}
      balance_meth  {balance_meth}
      store_mod     {store_mod}
      store_tr_te   {store_tr_te}
      """)


#%% set the feature names in the plot
pl_feats = {f:f for f in feats}


#%% set paths --> output or plot paths will be created if necessary
data_path = f"{path_par}/IMPETUS/NORA3/"  # data path
pred_path = f"{data_path}/Avalanche_Predictors_{ndlev}Level/{agg_str}/Between{h_low}_and_{h_hi}m/"
obs_per_path = "{data_path}/Plots/Danger_Levels_2017_2024/"  # path for observation period prediction plots
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
        best_name = f"BestFeats_RF_{ndlev}DL_{reg_code}_{agg_str}_" + \
                                                             f"Between{h_low}_and_{h_hi}m_{sea}{bal_suff}_{a_p_str}.csv"
        if n_best == 0:
            n_best = None
        # end if
        sel_feats = np.array(pd.read_csv(best_path + best_name, index_col=0).index)[:n_best]
        n_best = len(sel_feats)

        print("\nUsing the best features from the RF model...\n")
    except:
        print("\nNo best-feature list found in\n")
        print(best_path + best_name)
        print("\nAborting\n")
        sys.exit()
    # end try except
# end if elif


#%% set some dependent parameters for the neural network
if ndlev == 2:
    loss = "binary_crossentropy"
    activ_out = "sigmoid"
else:
    loss = "categorical_crossentropy"
    activ_out = "softmax"
# end if else


#%% perform the prediction
acc_train_l = []
acc_test_l = []
acc_train_all_l = []
acc_test_all_l = []
model_l = []
test_acc = 0
print(f"\nTraining {n_mod} ANNs...\n")
print_progress_bar(0, n_mod)
for i in np.arange(n_mod):
    # print(f"\n{i} / {n_mod}\n")

    pred_test_t, pred_train_t, pred_test_all_t, pred_train_all_t, train_x, train_y, test_x, test_y, train_x_all, \
                                                                    train_y_all, test_x_all, test_y_all, model = \
        apply_ANN(ndlev, n_in_nodes, in_dropout, n_hid_nodes, dropouts,
                  activ_out=activ_out, learning_rate=learning_rate, loss=loss, epochs=epochs, batch_size=batch_size,
                  balancing=balancing, split=split,
                  h_low=h_low, h_hi=h_hi, reg_code=reg_code, sel_feats=sel_feats, scale_x=scale_x, sea=sea,
                  agg_type=agg_type, p=perc,
                  data_path=data_path, assume_path=True, balance_meth=balance_meth,
                  return_model=True)

    model_l.append(model)

    # store the model and the predictions only if it is better than the last one
    if accuracy_score(test_y_all, pred_test_all_t) > test_acc:
        pred_test, pred_train, pred_test_all, pred_train_all, model = pred_test_t, pred_train_t, pred_test_all_t, \
                                                                      pred_train_all_t, model
    # end if
    test_acc = accuracy_score(test_y_all, pred_test_all)

    acc_train_l.append(accuracy_score(train_y, pred_train_t))
    acc_test_l.append(accuracy_score(test_y, pred_test_t))
    acc_train_all_l.append(accuracy_score(train_y_all, pred_train_all_t))
    acc_test_all_l.append(accuracy_score(test_y_all, pred_test_all_t))

    print_progress_bar(i, n_mod)
# end for


#%% store the 500 models together with the training and the test data
dump({"train_x":train_x, "train_y":train_y, "test_x":test_x, "test_y":test_y, "train_x_all":train_x_all,
      "train_y_all":train_y_all, "test_x_all":test_x_all, "test_y_all":test_y_all, "acc_test_all_l":acc_test_all_l,
      "model_l":model_l},
     mod_path + f"ANNx{n_mod}_{ndlev}ADL_Train_and_Test_Data.joblib")


#%% evaluate the model
acc_train = np.mean(acc_train_l)  # accuracy_score(train_y, pred_train)
print(f'\nAccuracy balanced training data: {(acc_train * 100)} %  (averaged over {n_mod} values)')

acc_test = np.mean(acc_test_l)  # accuracy_score(test_y, pred_test)
print(f'Accuracy balanced test data:     {(acc_test * 100)} %  (averaged over {n_mod} values)')

acc_train_all = np.mean(acc_train_all_l)  # accuracy_score(train_y_all, pred_train_all)
print(f'Accuracy all train data:         {(acc_train_all * 100)} %  (averaged over {n_mod} values)')

acc_test_all = np.mean(acc_test_all_l)  # accuracy_score(test_y_all, pred_test_all)
print(f'Accuracy all test data:          {(acc_test_all * 100)} %  (averaged over {n_mod} values)\n')


#%% plot a histogram of the accuracies
std = np.std(acc_test_all_l)

pl_path = "/home/kei070/Documents/IMPETUS/Publishing/The Cryosphere/Supplementary_Paper_1_Revision/Figures/"
fig = pl.figure(figsize=(6, 4))
ax00 = fig.add_subplot(111)

ax00.hist(acc_test_all_l, density=False, histtype="step")
ax00.axvline(x=np.mean(acc_test_all_l), c="black", label=f"mean: {np.mean(acc_test_all_l):.2f}")
ax00.axvline(x=np.median(acc_test_all_l), c="red", label=f"median: {np.median(acc_test_all_l):.2f}")

ax00.axvline(x=np.mean(acc_test_all_l) + std, c="gray", linestyle="--")
ax00.axvline(x=np.mean(acc_test_all_l) - std, c="gray", linestyle="--")

ax00.legend()

ax00.set_xlabel("Accuracy")
ax00.set_ylabel("Frequency")
ax00.set_title("Histogram ANN accuracies")

pl.savefig(pl_path + f"Histogram_ANN_{n_mod}_{ndlev}ADL_ACCs.pdf", bbox_inches="tight", dpi=200)

pl.show()
pl.close()


#%% perform predefined cross-validation
"""
# load data
bal_x, bal_y, all_x, all_y = load_xlevel_preds(data_path +
                                               f"/Avalanche_Predictors_{ndlev}Level/{agg_str}/" +
                                               f"Between{h_low}_and_{h_hi}m/",
                                               sel_feats, reg_code=reg_code,
                                               split=0, sea=sea,
                                               nlevels=ndlev, balance_meth=balance_meth, scale_x=scale_x,
                                               verbose=True)
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
all_winter = pd.Series(assign_winter_year(all_x.index, 8))
obs_yrs = np.unique(all_winter)

# assign fold labels --> depends on the year
nfolds = cv
len_fold = int(len(obs_yrs) / nfolds)

fold_label = np.zeros(len(all_winter))
for i, yr in enumerate(obs_yrs[::len_fold]):
    yrs_temp = [yr_temp for yr_temp in np.arange(yr, yr+len_fold, 1)]
    fold_label[all_winter.isin(yrs_temp)] = i
# end for i, yr

# generate the predefined split
ps = PredefinedSplit(fold_label)


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

df = pd.concat([df_x, df["reg_code"], df["y"]], axis=1)


#%% stratified cross-validation --> cross-validation is (so far) not implemented in Keras or Tensorflow, so this had to
#   be coded here
"""
print(f"\n{cv}-fold cross-validation:\n")

skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
scores = []

df_x = df[sel_feats]
df_y = df["y"]

input_shape = len(sel_feats)
if ndlev == 2:
    n_out_nodes = 1
else:
    n_out_nodes = ndlev
# end if else

for train_index, test_index in skf.split(df_x, df_y):
    model = ANN(input_shape, n_in_nodes, in_dropout, n_hid_nodes, dropouts, n_out_nodes, activ_out, learning_rate, loss)

    X_train, X_test = df_x.iloc[train_index], df_x.iloc[test_index]
    y_train, y_test = to_categorical(df_y.iloc[train_index]), to_categorical(df_y.iloc[test_index])

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)  # Train model
    score = model.evaluate(X_test, y_test, verbose=0)  # Evaluate model on test data
    scores.append(score[1])  # Assume accuracy is the second item in metrics
# end for train_index, test_index

print(f"cross-validated scores: {' '.join([f'{k:.2f}' for k in scores])}")
print(f"mean accuracy: {np.mean(scores):.2f}")
"""

#%% perform a simpler validation: generate random balanced and unbalanced splits of the data and check the model
#   performance
"""
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

    # df_t.set_index("date", inplace=True)
    df_y = df_t["y"]
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

        acc_te_all.append(accuracy_score(te_y_all, prob_to_cat(model.predict(te_x_all, verbose=0))))
        acc_te_bal.append(accuracy_score(te_y_bal, prob_to_cat(model.predict(te_x_bal, verbose=0))))
        acc_tr_all.append(accuracy_score(tr_y_all, prob_to_cat(model.predict(tr_x_all, verbose=0))))
        acc_tr_bal.append(accuracy_score(tr_y_bal, prob_to_cat(model.predict(tr_x_bal, verbose=0))))
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
"""

#%% print the classification_report
class_rep = classification_report(test_y_all, pred_test_all)
print(class_rep)


#%% prepare confusion matrix plot
# partly from https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
conf_train = confusion_matrix(train_y, pred_train)
conf_test = confusion_matrix(test_y, pred_test)
conf_test_all = confusion_matrix(test_y_all, pred_test_all)

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

labels = np.asarray(labels).reshape(ndlev, ndlev)


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


#%% plot the daily danger levels for a specific region
if not no_plots:

    # set the region
    reg_code_pl = 3009
    regions_dic = {3009:"Nord-Troms", 3010:"Lyngen", 3011:"Troms${\o}$", 3012:"S${\o}$r-Troms", 3013:"Indre Troms"}

    # get the data for the current region
    df_x = df[df["reg_code"] == reg_code_pl]

    df_y = df_x["y"]
    df_x = df_x[sel_feats]

    # make the prediction based on the x-data
    pred_y = prob_to_cat(model.predict(df_x))

    hind_dl = pd.DataFrame({"y":pred_y}, index=df_x.index)["y"]

    sta_yr = 2016
    end_yr = 2024
    dl_counts = count_dl_days(hind_dl, sta_yr, end_yr)
    # ava = {"full":dl_count_full[1].squeeze(), "winter":dl_count_win[1].squeeze(), "spring":dl_count_spr[1].squeeze()}
    ava = {k:dl_counts[k][1].squeeze() for k in dl_counts.keys()}

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
    for yr in yrs:
        inds[yr] = (df_y.index >= date_dt(yr-dyr, sta_mon, 1)) & (df_y.index < date_dt(yr, end_mon, 1))
    # end for yr

    # calculate the range for the y-axis
    extend_y = ((np.max(df_y) - np.min(df_y)) + 1) / 4

    # 2017-2020
    fig = pl.figure(figsize=(10, 14))
    axes = [fig.add_subplot(411), fig.add_subplot(412), fig.add_subplot(413), fig.add_subplot(414)]

    for i, yr in enumerate(yrs[:4]):

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
    fig.subplots_adjust(top=0.95)

    # os.makedirs(obs_per_path, exist_ok=True)
    # pl.savefig(obs_per_path + f"DangerLevel{ndlev}_Sea{yr}_True_and_Pred_2018_20.png", bbox_inches="tight", dpi=200)

    pl.show()
    pl.close()

    # 2021-2024
    fig = pl.figure(figsize=(10, 14))
    axes = [fig.add_subplot(411), fig.add_subplot(412), fig.add_subplot(413), fig.add_subplot(414)]

    for i, yr in enumerate(yrs[4:]):

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
    fig.subplots_adjust(top=0.95)

    # pl.savefig(obs_per_path + f"DangerLevel{ndlev}_Sea{yr}_True_and_Pred_2021_23.png", bbox_inches="tight", dpi=200)

    pl.show()
    pl.close()
# end if


#%% investigate the model performance per seasons
"""
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

reg_accs_all = {}
reg_accs_bal = {}
for yr in yrs:
    inds = (df.index >= date_dt(yr-dyr, sta_mon, 1)) & (df.index < date_dt(yr, end_mon, 1))

    # get the index for the current season
    df_x = df[inds]

    df_y = df_x["y"]
    df_x = df_x[sel_feats]

    # balance the data
    bal_x, bal_y = balance_data(df_x, df_y, method="SMOTE")

    # make the prediction based on the x-data
    pred_y = prob_to_cat(model.predict(df_x, verbose=0))
    pred_y_bal = prob_to_cat(model.predict(bal_x, verbose=0))

    reg_accs_all[yr] = accuracy_score(df_y, pred_y)
    reg_accs_bal[yr] = accuracy_score(bal_y, pred_y_bal)
# end for yr

print("\nAccuracy per season:")
print("      all    bal (SMOTE)")
for yr in yrs:
    print(f"{yr}  {r_2(reg_accs_all[yr]):4}   {r_2(reg_accs_bal[yr])}")
# end for reg
"""

#%% investigate the model performance per seasons for each individual region
"""
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

        df_y = df_x_yr["y"]
        df_x_fin = df_x_yr[sel_feats]

        try:
            # balance the data
            bal_x, bal_y = balance_data(df_x_fin, df_y, method="SMOTE")
            pred_y_bal = prob_to_cat(model.predict(bal_x, verbose=0))
            reg_accs_bal[reg_code_pl][yr] = accuracy_score(bal_y, pred_y_bal)
        except:
            reg_accs_bal[reg_code_pl][yr] = np.nan
        # end try except

        # make the prediction based on the x-data
        pred_y = prob_to_cat(model.predict(df_x_fin, verbose=0))
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
"""

#%% if a specific year was extracted as test data perform a specific accuracy test for this year
"""
if split > 1:
    year_df = df[df.index.year == int(split)]

    year_x = year_df[sel_feats]
    year_y = year_df["y"]

    # unbalanced
    year_pred = prob_to_cat(model.predict(year_x, verbose=0))

    year_acc = accuracy_score(year_y, year_pred)

    # balanced
    year_x_bal, year_y_bal = balance_data(year_x, year_y, method=balance_meth, excl_fewer=100)

    year_pred_bal = model.predict(year_x_bal, verbose=0)

    year_acc_bal = accuracy_score(year_y_bal, year_pred_bal)

    print()
    print(f"Accuracy in {int(split)} (unbalanced): {year_acc}")
    print(f"Accuracy in {int(split)} (balanced):   {year_acc_bal}\n")
# end if
"""

#%% when predicting more than 2 danger levels calculate the "distance" of the prediction to the truth: 1, 2, or 3 levels
if not no_plots:
    if ndlev > 2:
        del_y_test = pred_test - test_y
        del_y_test_all = pred_test_all - test_y_all

        ndel_test = np.array([np.sum(del_y_test == i) for i in np.unique(del_y_test)])
        ndel_test_all = np.array([np.sum(del_y_test_all == i) for i in np.unique(del_y_test_all)])

        # fig = pl.figure(figsize=(8, 5))

        disc_hist([del_y_test, del_y_test_all], width=[0.4, 0.3], color=["black", "gray"], labels=["test", "test_all"],
                  xlabel="Delta to truth", ylabel="Number of cases", title="Distance to true danger level")

        print("\nPrecentages of 'distances' to truth (balanced test):")
        print({d:f"{r_2(ndel_te / np.sum(ndel_test) * 100)} %" for d, ndel_te in zip(np.unique(del_y_test), ndel_test)})

        print("\nPrecentages of 'distances' to truth (all test):")
        print({d:f"{r_2(ndel_te / np.sum(ndel_test_all) * 100)} %" for d, ndel_te in zip(np.unique(del_y_test_all),
                                                                                         ndel_test_all)})

    # end if
# end if


#%% plot predictors in predictor space
if not no_plots:
    test_x_all = np.array(test_x_all)
    # concatenate the training and test data to the full dataset
    data_x = np.concatenate([train_x_all, test_x_all])

    if len(sel_feats) == 2:
        plot_2predictor_space(data_x=data_x, model=model, test_x=test_x_all, test_y=test_y_all, sel_feats=sel_feats,
                              h=0.5)
    elif len(sel_feats) == 3:
        plot_3predictor_space(model=model, data_x=test_x_all, data_y=test_y_all, sel_feats=sel_feats)
    # end if elif
# end if


#%% store the model if requested
"""
if store_mod:
    os.makedirs(mod_path, exist_ok=True)
    model.save(mod_path + f"/ANN_{ndlev}DL_{reg_code}_{agg_str}_Between{h_low}_and_{h_hi}m_{sea}.keras")

    print("\nModel stored in:")
    print("  " + mod_path)
    print("as:")
    print(f"  {mod_path}/ANN_{ndlev}DL_{reg_code}_{agg_str}_Between{h_low}_and_{h_hi}m_{sea}.keras\n")
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
    mod_name = f"ANN_{ndlev}DL_{reg_code}_{agg_str}_" + \
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
    out_name = f"ANN_{ndlev}DL_{reg_code}_{agg_str}_Between{h_low}_and_{h_hi}m" + \
                                                                  f"_{sea}{bal_suff}{nbest_suff}_{a_p_str}_wData.joblib"
    dump(bundle, f"{mod_path}/{out_name}")

    print("\nModel + data stored in:")
    print("  " + mod_path)
    print("as:")
    print(f"  {out_name}\n")
# end if