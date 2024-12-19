#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Set up a hyperparameter grid that can be used by any function (mainly targeted at set_hyperp)
"""

# fine grid SVMSMOTE ndlev = 4
hypp_grid = {'n_estimators': [350, 400, 450],  # [100, 200, 300],
             'max_depth': [65, 70, 75],  # [2, 5, 8, 11, 14, 17, 20],
             'min_samples_leaf': [2, 3, 4],
             'min_samples_split': [3, 4, 5],
             'max_features': ["sqrt", "log2"],
             'bootstrap': [True]
             }

# fine grid SMOTE ndlev = 4
hypp_grid = {'n_estimators': [350, 400, 450],  # [100, 200, 300],
             'max_depth': [40, 45, 50],  # [2, 5, 8, 11, 14, 17, 20],
             'min_samples_leaf': [2, 3, 4],
             'min_samples_split': [3, 4, 5],
             'max_features': ["sqrt", "log2"],
             'bootstrap': [True]
             }

# fine grid ADASYN ndlev = 4
hypp_grid = {'n_estimators': [550, 600, 650],  # [100, 200, 300],
             'max_depth': [75, 80, 85],  # [2, 5, 8, 11, 14, 17, 20],
             'min_samples_leaf': [2, 3, 4],
             'min_samples_split': [2, 3, 4],
             'max_features': ["sqrt", "log2"],
             'bootstrap': [True]
             }


# fine grid ADASYN ndlev = 2
hypp_grid = {'n_estimators': [850, 900, 950],  # [100, 200, 300],
             'max_depth': [35, 40, 45],  # [2, 5, 8, 11, 14, 17, 20],
             'min_samples_leaf': [2, 3, 4],
             'min_samples_split': [13, 14, 15],
             'max_features': ["sqrt", "log2"],
             'bootstrap': [True]
             }

# fine grid SMOTE ndlev = 2
hypp_grid = {'n_estimators': [850, 900, 950],  # [100, 200, 300],
             'max_depth': [55, 60, 65],  # [2, 5, 8, 11, 14, 17, 20],
             'min_samples_leaf': [2, 3, 4],
             'min_samples_split': [13, 14, 15],
             'max_features': ["sqrt", "log2"],
             'bootstrap': [True]
             }


"""
# full grid --> do a grid sample on this to reduce computational requirements
hypp_grid = {'n_estimators': [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100],  # [100, 200, 300],
             'max_depth': [40, 45, 50, 55, 60, 65, 70, 75, 80],  # [2, 5, 8, 11, 14, 17, 20],
             'min_samples_leaf': [2, 3, 4, 6, 8, 10, 12, 14],
             'min_samples_split': [3, 4, 6, 8, 10, 12, 14],
             'max_features': ["sqrt", "log2"],
             'bootstrap': [True]
             }

"""

hypp_set = {'n_estimators':800,  # [100, 200, 300],
            'max_depth':70,  # [2, 5, 8, 11, 14, 17, 20],
            'min_samples_leaf':3,
            'min_samples_split':4,
            'max_features':"log2",
            'bootstrap':True
            }

hypp_set = {'n_estimators':750,  # [100, 200, 300],
            'max_depth':45,  # [2, 5, 8, 11, 14, 17, 20],
            'min_samples_leaf':2,
            'min_samples_split':5,
            'max_features':"sqrt",
            'bootstrap':True
            }

hypp_set = {'n_estimators':600,  # [100, 200, 300],
            'max_depth':55,  # [2, 5, 8, 11, 14, 17, 20],
            'min_samples_leaf':2,
            'min_samples_split':4,
            'max_features':"sqrt",
            'bootstrap':True
            }


hypp_set = {'n_estimators':400,  # [100, 200, 300],
            'max_depth':70,  # [2, 5, 8, 11, 14, 17, 20],
            'min_samples_leaf':2,
            'min_samples_split':4,
            'max_features':"sqrt",
            'bootstrap':True
            }

# SVMSMOTE ndlev = 4
hypp_set = {'n_estimators':450,  # [100, 200, 300],
            'max_depth':65,  # [2, 5, 8, 11, 14, 17, 20],
            'min_samples_leaf':2,
            'min_samples_split':3,
            'max_features':"sqrt",
            'bootstrap':True
            }

# SMOTE ndlev = 4
hypp_set = {'n_estimators':350,  # [100, 200, 300],
            'max_depth':40,  # [2, 5, 8, 11, 14, 17, 20],
            'min_samples_leaf':2,
            'min_samples_split':5,
            'max_features':"sqrt",
            'bootstrap':True
            }

# ADASYN ndlev = 4
hypp_set = {'n_estimators':650,  # [100, 200, 300],
            'max_depth':75,  # [2, 5, 8, 11, 14, 17, 20],
            'min_samples_leaf':2,
            'min_samples_split':2,
            'max_features':"sqrt",
            'bootstrap':True
            }

# ADASYN ndlev = 2
hypp_set = {'n_estimators':850,  # [100, 200, 300],
            'max_depth':35,  # [2, 5, 8, 11, 14, 17, 20],
            'min_samples_leaf':2,
            'min_samples_split':14,
            'max_features':"log2",
            'bootstrap':True
            }

# SMOTE ndlev = 2
hypp_set = {'n_estimators':850,  # [100, 200, 300],
            'max_depth':55,  # [2, 5, 8, 11, 14, 17, 20],
            'min_samples_leaf':2,
            'min_samples_split':13,
            'max_features':"sqrt",
            'bootstrap':True
            }

