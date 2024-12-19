#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parent directory path for avalanche analysis --> change when porting to another system.
For the scripts to work properly (without further changes), the directory structure must be as follows:

    The avalanche features/predictors must be stored under
        h_low = 400  # change depending on requested height threshold
        h_hi = 900  # change depending on requested height threshold
        agg_type = "Mean"
        perc = "0"
        agg_str = ...
        feat_path = f"{path_par}/IMPETUS/NORA3/Avalanche_Predictors/{agg_str}/Between_{h_low}_and_{h_hi}m/"

    Note that these are the features together with the original 5-level target variable. The features with reduced
    danger level number are in
        ndlev = 2
        feat_path = f"{path_par}/IMPETUS/NORA3/Avalanche_Predictors_{ndlev}Level/{agg_str}/Between_{h_low}_and_{h_hi}m/"

    The naming convention of the feature/predictor files is
        region = "IndreTroms"  # "NordTroms", ...
        f_name = f"{region}_Predictors_MultiCell{agg_str}_Between{h_low}_and_{h_hi}.csv"

    in case of the 5-level target and
        reg_code = 3013  # 3009, ...
        f_name_all = f"Features_{ndlev}Level_All_{agg_str}_Between400_900m_{reg_code}_{region}.csv"
        f_name_bal = f"Features_{ndlev}Level_{agg_str}_Balanced_Between400_900m_{reg_code}_{region}.csv"

    where "All" indicates all data and "Balanced" indicates that a balancing by undersampling majority classes was
    undertaken. That is, the number of elements per class (i.e., danger level) was equalised to the number of elements
    in the smallest class. Note that this reduces that data that is used to train the model and is (so far) only
    feasible for the case of ndlev = 2, because for 3 or 4 danger levels too much data is removed. In the latter cases,
    oversampling or internal class balancing is suggested.
    Note that there also exist concatenated files containing the data for all regions combined:
        f_name_all = f"Features_{ndlev}Level_All_Between400_900m_AllReg.csv"
        f_name_bal = f"Features_{ndlev}Level_Balanced_Between400_900m_AllReg.csv"

    This is a redundancy and might be changed in the future.


    There is a further path to the "observations", that is the avalanche danger levels provided by Varsom. The expected
    data structure is:
        f"/{obs_path}/IMPETUS/Avalanche_Danger_Files/Avalanche_Danger_List.csv"

    Finally, there is the path to the Python scripts in this library that is used in the batch scripts.
"""


# set path
path_par = "/"

# path to danger level "observations"
obs_path = "/"

# python scripts path  |  ADJUST!
py_path_par = "ADJUST/Avalanche_Libraries_Python/Python_Avalanche_PreProcessing/"
