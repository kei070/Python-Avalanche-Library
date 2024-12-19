#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to call all scripts for the predictor preparation given that the NORA3 data and the seNorge data.
"""


#%% imports
import argparse
import subprocess
import numpy as np

from Lists_and_Dictionaries.Paths import py_path_par


#%% set up the parser
parser = argparse.ArgumentParser(
                    prog="Predictor_Prep",
                    description="""Based on the NORA3 and seNorge data perform the full predictive feature
                    preparation.""")

# ...and add the arguments
parser.add_argument("--h_low", type=int, default=300, help="The lower threshold of the grid cell altitude.")
parser.add_argument("--h_hi", type=int, default=1300, help="The upper threshold of the grid cell altitude.")
parser.add_argument("--agg_type", default="percentile", type=str, choices=["mean", "median", "percentile"],
                    help="""The type of aggregation used for the grid cells.""")
parser.add_argument("--perc", type=int, default=90, help="""The percentile used in the grid-cell aggregation.""")
parser.add_argument("--sta_yr1", type=int, default=1970, help="""First start year for the extracted grid cells.
                    The reason for two start and end years is: Due to memory limitations the NORA3 dataset is here
                    separated into smaller chunks. In the standard case the two start years are 1970 and 1972, i.e.,
                    the chunk length is 3 years.""")
parser.add_argument("--sta_yr2", type=int, default=1970, help="Second start year for the extracted grid cells.")
parser.add_argument("--end_yr1", type=int, default=2024, help="First end year for the extracted grid cells.")
parser.add_argument("--end_yr2", type=int, default=2024, help="Second end year for the extracted grid cells.")
parser.add_argument("--no_search", action="store_true", help="Set this parameter to suppress grid-cell search.")
parser.add_argument("--no_ex_n3", action="store_true", help="""Set this parameter to suppress NORA3 grid-cell
                    extraction.""")
parser.add_argument("--no_ex_sn", action="store_true", help="""Set this parameter to suppress seNorge grid-cell
                    extraction.""")
parser.add_argument("--no_conc", action="store_true", help="""Set this parameter to suppress the period
                    concatination.""")
parser.add_argument("--no_r1s1", action="store_true", help="""Set this parameter to suppress the r1 and s1
                    calculation.""")
parser.add_argument("--no_calc", action="store_true", help="""Set this parameter to suppress the predictor
                    calculation.""")
parser.add_argument("--no_comb", action="store_true", help="""Set this parameter to suppress the combination of the
                    predictive features with the danger levels.""")
parser.add_argument("--no_agg", action="store_true", help="""Set this parameter to suppress the aggregation of the
                    danger levels.""")
args = parser.parse_args()


#%% set the paths to the scripts
py_path = f"{py_path_par}/"


#%% get the parameters from the parser
h_low = args.h_low
h_hi = args.h_hi
agg_type = args.agg_type
perc = args.perc
no_search = args.no_search
no_ex_n3 = args.no_ex_n3
no_ex_sn = args.no_ex_sn
no_conc = args.no_conc
no_r1s1 = args.no_r1s1
no_calc = args.no_calc
no_comb = args.no_comb
no_agg = args.no_agg


#%% set up the NORA3 period years
sta_yr1 = args.sta_yr1
sta_yr2 = args.sta_yr2
end_yr1 = args.end_yr1
end_yr2 = args.end_yr2


#%% find the gridcells between the height thresholds
# --> Perform_Full_Find_GridCells_Procedure.py
if not no_search:
    print(f"\nSearching for NORA3 grid cells between {h_low} and {h_hi}...\n")
    for reg_code in np.arange(3009, 3013+1):
        subprocess.call(["python", py_path + "Find_GridCells_Above_or_Between_CertainHeight.py",
                         "--reg_code", str(reg_code), "--low", str(h_low), "--high", str(h_hi)])
        subprocess.call(["python", py_path + "Convert_Height_shps_to_DataFrame.py",
                         "--reg_code", str(reg_code), "--low", str(h_low), "--high", str(h_hi)])
    # end for reg_code
# end if


#%% extract the NORA3 gridcells
# --> Batch_Extract_NORA3_GridCells.py
if not no_ex_n3:
    print("""\nExtracting the NORA3 grid cells found above and storing the data as netcdf files.
          Due to memory limitations this has to be done for 3 years at a time.\n""")
    for syr, eyr in zip(np.arange(sta_yr1, end_yr1+1, 3), np.arange(sta_yr2, end_yr2+1, 3)):
        print(f"\nYears {syr}-{eyr}\n")

        subprocess.call(["python", py_path + "Extract_NORA3_GridCells_Between_NC.py",
                         "--start_year", str(sta_yr1), "--end_year", str(end_yr2),
                         "--low", str(h_low), "--high", str(h_hi)])
    # end for syr, eyr
# end if


#%% extract the seNorge gridcells
if not no_ex_sn:
    print("\nExtracting grid cells for the seNorge data...\n")
    subprocess.call(["python", py_path + "Extract_seNorge_GridCells_Between.py",
                     "--start_year", str(sta_yr1), "--end_year", str(end_yr2),
                     "--low", str(h_low), "--high", str(h_hi), "--dataset", "NORA3"])
# end if


#%% concatenate the extracted NORA3 data
if not no_conc:
    print("\nConcatenating the 3-year nc-files generated above to one file per region...\n")
    subprocess.call(["python", py_path + "Concat_Extracted_NORA3_GridCell_Data_NC.py",
                     "--low", str(h_low), "--high", str(h_hi),
                     "--sta_yr1", str(sta_yr1), "--sta_yr2", str(sta_yr2),
                     "--end_yr1", str(end_yr1), "--end_yr2", str(end_yr2)])
# end if


#%% calculate the predictors only for those days with available danger assessment
# --> Batch_Calc_and_Store_Predictors.py
# --> this will take a lot of time; although a lot of that time is due to the r1-s1 calculations
print_calc = False
r1s1_str = ""
if not no_r1s1:
    r1s1_str = "Performing r1-s1 calculations"
    print_calc = True
if (not no_calc) & (not no_r1s1):
    pred_calc_str = " and generating remaining features"
if (not no_calc) & (no_r1s1):
    pred_calc_str = "Generating predictive features"
    print_calc = True
# end if

if print_calc:
    full_pred_calc_str = (r1s1_str + pred_calc_str +
                          " for the full NORA3 time series. This will take quite some time...\n")
    print("\n" + full_pred_calc_str)
# end if

for reg_code in [3009, 3010, 3011, 3012, 3013]:
    if not no_r1s1:
        subprocess.call(["python", py_path + "Calc_s1_r1.py", "--reg_code", str(reg_code),
                         "--low", str(h_low), "--high", str(h_hi)])
    if not no_calc:
        subprocess.call(["python", py_path + "Calc_and_Store_NORA3_Predictors.py",
                         "--reg_code", str(reg_code), "--low", str(h_low), "--high", str(h_hi),
                         "--agg_type", agg_type, "--perc", str(perc)])
    # end if
# end for reg_code


#%% combine predictors and danger levels
# --> Batch_Gen_Predictors_With_DangerLevels.py
if not no_comb:
    print("\nCombining the features with the danger levels...\n")
    for reg_code in [3009, 3010, 3011, 3012, 3013]:
        print(reg_code)
        subprocess.call(["python", py_path + "Gen_Preds_With_DangerLevels.py",
                         "--reg_code", str(reg_code), "--h_low", str(h_low), "--h_hi", str(h_hi),
                         "--agg_type", agg_type, "--perc", str(perc)])
    # end for reg_code
# end if


#%% generate the aggregated danger levels (BCL, 3-level ADL, 4-level ADL)
# --> Batch_XLevel_Predictor_Calc.py
if not no_agg:
    print("\nGenerating the binary, 3-, and 4-level files...\n")
    for ndlev in [2, 3, 4]:
        for reg_code in [0, 3009, 3010, 3011, 3012, 3013]:
            subprocess.call(["python", py_path + "Gen_Store_XLevel_Balanced_Predictors.py",
                             "--reg_code", str(reg_code),
                             "--h_low", str(h_low), "--h_hi", str(h_hi),
                             "--ndlev", str(ndlev), "--agg_type", agg_type, "--perc", str(perc)])
        # end for reg_code
    # end for ndlev
# end if
