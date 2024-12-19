#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function for loading the x-level predictors with the option of extracting winter and spring as well as the possibility
of choosing the one-winter-out train-test split.
"""

# imports
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# import proprietary functions
from .Func_Balance_Data import balance_data
from .Func_Helpers_Load_Data import extract_reg, extract_sea


# function
def load_xlevel_preds(data_path, sel_feats, reg_code, a_p="y", split=0.33, sea="Full", nlevels=2, nan_handling="drop",
                      balance_meth="SMOTE", sample_strat="auto", scale_x=False, verbose=True):

    """
    Function for loading the predictors (implictly performing a train-test split) for a given danger level.

    Parameters:
        data_path
        sel_feats List of the features that will be loaded.
        reg_code  Region code (3009, 3010, ...)
        a_p            String. The avalanche problem for which the danger level is chosen. Defaults to "y",
                               implying the general danger level. The names of the available avalanche problems are
                               stored in Avalanche_Problems.py under Lists_and_Dictionaries.
        split     Possibility (1): 0. No train-test split is performed. Note that this changes the output. Note that
                                   this option is itntroduced for special purposes in can (so far) not be used as an
                                   option in scripts like RF_DT_FeatureImportance or Train_StatisticalModel.
                  Possibility (2): Float between 0 and 1 (excluding these boundaries) to perform the train-test split
                                   using the scikit-learn function train_test_split. The value given to split then
                                   represents the fraction of the data that is used as test data.
                  Possibility (3): Integer representing the year of the winter to be excluded. Note that the year is
                                   interpreted as the year in which the season ends. That is, for the winter of 2019/20
                                   one must submit 2020.
                  Possibility (4): Integer representing the region code of the region to be extracted as test data and
                                   removed from the training data. Note that this works slightly differently than the
                                   other two methods as the data must be balanced only AFTER the requested region has
                                   been extracted as test data.
                  Default: 0.33
        sea:          String. Full (default) for the full avalanche season, winter for Dec-Feb, and spring for Mar-May.
        nlevels       Integer. Number of danger levels to convert the raw data to. For 2, levels 1 & 2 are set to 0 and
                               3-5 are set to 1. For 3, level 1=0, 2=1, 3-5=2. For 4, level 1=0, 2=1, 3=2, 4&5=3.
        nan_handling  String. Controls how NaNs are handled. Two possibilities: 1) "drop" (default), i.e., dropping all
                              rows where the danger level is NaN and 2) "zero", i.e., converting the NaNs to 0s.
        balance_meth  String. The method used for balancing the data. Choices are the following:
                               -None data are not balanced
                               -undersample: uses the custom undersample function NOT FUNCTIONAL
                               -SMOTE: (default) uses the synthetic minority oversampling method from the
                                       imbalanced-learn library
                               -BSMOTE: BorderlineSMOTE
                               -SVMSMOTE: SMOTE using an SVM
                               -ADASYN: uses the adaptive synthetic sampling method from the imbalanced-learn library
                               -ros: uses the random oversampling method from the imbalanced-learn library
                               -rus: uses the random undersampling method from the imbalanced-learn library
                               -SMOTEENN: combination of over- and undersampling
                               -SMOTETomek: combination of over- and undersampling
                              There is considerable difference in procedure between the undersample case and the other
                              cases: In the undersample case the predictors have already been balanced (script
                              Gen_Store_XLevel_Balanced_Predictors.py). In the other cases the predictors are only
                              balanced here.
       sample_strat   Float, String, or dict: The strategy used in the class balancing algorithm.
                                              Float only for binary classification. String possibilities: auto, all,
                                              minority, not_majority, not_minority. dict: Keys indicate the targeted
                                              classes and the values to desired number of samples per class.
       scale_x        Logical. If true, the predictors will be scaled using the scikit-learn scaler. If false (default),
                              the predictors are used without being scaled.
       verbose        Logical. If True (default) print statements will made. If False they are suppressed.
    """

    """ --> probably remove this part
    if balance_meth == "undersample":

        #% load the data
        df = pd.read_csv(glob.glob(data_path + f"Features_{nlevels}Level_Balanced_*_{reg_code}*.csv")[0])
        all_df = pd.read_csv(glob.glob(data_path + f"Features_{nlevels}Level_All_*_{reg_code}*.csv")[0])


        #% convert the date to datetime
        df.date = pd.to_datetime(df.date)
        all_df.date = pd.to_datetime(all_df.date)
        df.set_index("date", inplace=True)
        all_df.set_index("date", inplace=True)

        # perform the scaling if requested
        df_x = df[sel_feats]
        all_df_x = all_df[sel_feats]
        if scale_x:
            scaler = StandardScaler()
            df_x_sc = scaler.fit_transform(df_x)
            df_x = pd.DataFrame({k:df_x_sc[:, i] for i, k in enumerate(df_x.columns)})
            df_x.set_index(df.index, inplace=True)

            all_df_x_sc = scaler.fit_transform(all_df_x)
            all_df_x = pd.DataFrame({k:all_df_x_sc[:, i] for i, k in enumerate(all_df_x.columns)})
            all_df_x.set_index(all_df.index, inplace=True)
        # end if

        df = pd.concat([df_x, df["y_balanced"]], axis=1)
        all_df = pd.concat([all_df_x, all_df["reg_code"], all_df["y"]], axis=1)

        #% extract the subseason (or not)
        if sea == "winter":
            df = df[((df.index.month == 12) | (df.index.month == 1) | (df.index.month == 2))]
            all_df = all_df[((all_df.index.month == 12) | (all_df.index.month == 1) | (all_df.index.month == 2))]
        elif sea == "spring":
            df = df[((df.index.month == 3) | (df.index.month == 4) | (df.index.month == 5))]
            all_df = all_df[((all_df.index.month == 3) | (all_df.index.month == 4) | (all_df.index.month == 5))]
        # end if elif

        # if the split is not supposed to by OWO or ORO perform it with the standard function from scikit-learnd
        if split == 0:  # no train-test split

            if verbose:
                print("\nNo train-test split performed..\n")
            # end if

            #% extract the required features and prepare the data for the decision tree
            odata_x = df[sel_feats]
            odata_y = df["y_balanced"]

            #% make sure the data are balanced; note that they might not be because of the sub-season extraction
            # --> in case of the full season this should not have any effect
            bal_x, bal_y = balance_data(odata_x, odata_y, method="undersample")

            all_x = all_df[sel_feats]
            all_y = all_df["y"]

        elif (split > 0) & (split < 1):  # split = fraction of data to be extracted as test data

            if verbose:
                print(f"\n{split*100}% of data is randomly extracted as test data.\n")
            # end if

            #% extract the required features and prepare the data for the decision tree
            odata_x = df[sel_feats]
            odata_y = df["y_balanced"]

            #% make sure the data are balanced; note that they might not be because of the sub-season extraction
            # --> in case of the full season this should not have any effect
            odata_x, odata_y = balance_data(odata_x, odata_y, method="undersample")

            odata_x_all = all_df[sel_feats]
            odata_y_all = all_df["y"]

            train_x, test_x, train_y, test_y = train_test_split(odata_x, odata_y, test_size=split, shuffle=True,
                                                                stratify=odata_y)
            train_x_all, test_x_all, train_y_all, test_y_all = train_test_split(odata_x_all, odata_y_all,
                                                                                test_size=split, shuffle=True,
                                                                                stratify=odata_y_all)
        elif split < 3000:  # split = year to be extracted as test data

            if verbose:
                print(f"\nYear {split} is extracted as test data.\n")
            # end if

            # extract the data for the requested avalanche season
            test_inds = (df.index > date_dt(split-1, 7, 1)) & (df.index < date_dt(split, 7, 1))
            test_all_inds = (all_df.index > date_dt(split-1, 7, 1)) & (all_df.index < date_dt(split, 7, 1))
            test_df = df[test_inds]
            test_all_df = all_df[test_all_inds]
            train_df = df[~test_inds]
            train_all_df = all_df[~test_all_inds]

            #% extract the required features and prepare the data for the decision tree
            train_x = train_df[sel_feats]
            train_y = train_df["y_balanced"]
            test_x = test_df[sel_feats]
            test_y = test_df["y_balanced"]

            #% make sure the data are balanced; note that they might not be because of the sub-season extraction
            # --> in case of the full season this should not have any effect
            train_x, train_y = balance_data(train_x, train_y, method="undersample")
            test_x, test_y = balance_data(test_x, test_y, method="undersample")

            train_x_all = train_all_df[sel_feats]
            train_y_all = train_all_df["y"]
            test_x_all = test_all_df[sel_feats]
            test_y_all = test_all_df["y"]

        else:  # split = region to be extracted as test data
            if verbose:
                print(f"\nRegion {split} is extracted as test data.\n")
            # end if

            # extract the data for the requested region
            # --> this only makes sense for the unbalanced data since the balanced data were randomly selected from the
            #     regions
            # --> this means we first extract the data from the unbalanced files and perform the train-test splint and
            #     balance the data only after
            test_all_inds = all_df["reg_code"] == split
            test_all_df = all_df[test_all_inds]
            train_all_df = all_df[~test_all_inds]

            # extract x and y data
            train_x_all = train_all_df[sel_feats]
            train_y_all = train_all_df["y"]
            test_x_all = test_all_df[sel_feats]
            test_y_all = test_all_df["y"]

            # generate the balanced data
            train_x, train_y = balance_data(train_x_all, train_y_all, method="undersample")
            test_x, test_y = balance_data(test_x_all, test_y_all, method="undersample")

        # end if elif else
    """

    # else:  # balance the data using scikit-learn's methods (ros, SMOTE, ADASYN, ...)

    # make sure that split is a list
    if type(split) != list:
        split = [split]
    # end if

    all_df = pd.read_csv(glob.glob(data_path + f"Features_{nlevels}Level_All_*_{reg_code}*.csv")[0])

    #% convert the date to datetime
    all_df.date = pd.to_datetime(all_df.date)
    all_df.set_index("date", inplace=True)

    # perform the scaling if requested
    all_df_x = all_df[sel_feats]
    if scale_x:
        scaler = StandardScaler()
        all_df_x_sc = scaler.fit_transform(all_df_x)
        all_df_x = pd.DataFrame({k:all_df_x_sc[:, i] for i, k in enumerate(all_df_x.columns)})
        all_df_x.set_index(all_df.index, inplace=True)
    # end if

    all_df = pd.concat([all_df_x, all_df["reg_code"], all_df[a_p]], axis=1)

    # rename the a_p column to "y" for later simpler use
    if a_p != "y":
        all_df.rename(columns={a_p:"y"}, inplace=True)
    # end if

    # HOW SHOULD NANs BE TREATED?
    # --> NaN essentially means the avalanche problem in question was not identified on that day
    if nan_handling == "drop":
    # 1) drop them
        all_df.dropna(axis=0, inplace=True)
    elif nan_handling == "zero":
    # 2) convert to zero
        all_df[all_df.isna()] = 0
    # end if

    #% extract the subseason (or not)
    if sea == "winter":
        all_df = all_df[((all_df.index.month == 12) | (all_df.index.month == 1) | (all_df.index.month == 2))]
    elif sea == "spring":
        all_df = all_df[((all_df.index.month == 3) | (all_df.index.month == 4) | (all_df.index.month == 5))]
    # end if elif

    # extract the data
    all_x = all_df[sel_feats]
    all_y = all_df["y"]

    # balance the data if requested
    if str(balance_meth) != "None":
        bal_x, bal_y = balance_data(all_x, all_y, method=balance_meth, sample_strat=sample_strat)
    # end if


    if split[0] == 0:  # no train-test split

        if verbose:
            print("\nNo train-test split performed..\n")
        # end if

    elif (split[0] > 0) & (split[0] < 1):  # split = fraction of data to be extracted as test data
        if verbose:
            print(f"\n{split[0]*100}% of data is randomly extracted as test data.\n")
        # end if

        # perform the split into training and test data
        if str(balance_meth) != "None":
            train_x, test_x, train_y, test_y = train_test_split(bal_x, bal_y, test_size=split[0], shuffle=True,
                                                                stratify=bal_y)
        # end if
        train_x_all, test_x_all, train_y_all, test_y_all = train_test_split(all_x, all_y, test_size=split[0],
                                                                            shuffle=True, stratify=all_y)

    elif split[0] < 3000:  # split = year to be extracted as test data
        if verbose:
            print(f"\nYear(s) {split} is/are extracted as test data.\n")
        # end if

        if str(balance_meth) == "None":
            train_x_all, test_x_all, train_y_all, test_y_all = extract_sea(all_df, sel_feats, split, balance_meth)
        else:
            train_x, test_x, train_y, test_y, train_x_all, test_x_all, train_y_all, test_y_all =\
                                                                extract_sea(all_df, sel_feats, split, balance_meth)
        # end if else
    else:  # split = region to be extracted as test data
        if verbose:
            print(f"\nRegion(s) {split} is/are extracted as test data.\n")
        # end if

        if str(balance_meth) == "None":
            train_x_all, test_x_all, train_y_all, test_y_all = extract_reg(all_df, sel_feats, split, balance_meth)
        else:
            train_x, test_x, train_y, test_y, train_x_all, test_x_all, train_y_all, test_y_all =\
                                                                extract_reg(all_df, sel_feats, split, balance_meth)
        # end if else
    # end if elif else

    # return
    if split[0] != 0:  # train-test split performed
        if str(balance_meth) == "None":
            return train_x_all, test_x_all, train_y_all, test_y_all
        else:
            return train_x, test_x, train_y, test_y, train_x_all, test_x_all, train_y_all, test_y_all
        # end if else
    else:  # NO train-test split performed
        if str(balance_meth) == "None":
            return all_x, all_y
        else:
            return bal_x, bal_y, all_x, all_y
        # end if else
    # end if else

# end def
