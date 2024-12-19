#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# imports
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# import proprietary functions
from .Func_Set_Hyperparameters import set_hyperp
from .Func_Grid_Search import perform_grid_search

# function
def stat_mod(model_ty="RF", ndlev=2, hyperp={}, grid_search=False, grid_sample=0, class_weight=None,
             train_x=None, train_y=None, cv_type="seasonal", cv=3, cv_score="accuracy", balance_meth="SMOTE",
             verbose=True):

    """
    This function implements the use of the following statistical models specifically on the avalanche predictors as
    prepared by the functions load_xlevel_preds:
        - DT  = decision tree (set as default)
        - LR  = logistic regression
        - SVM = support vector machine
        - KNN = nearest neighbour
        - RF  = random forest

    The implementation without a function can be found in Gen_and_Store_StatisticalModel.py.

    The function here is intended to be used, i.a., in the feature testing, i.e., to find out which features best
    predict avalanche danger.

    A grid search for the optimal hyperparameters can be performed if required by the user. However, it must be noted
    that likely not all possible hyperparmeters will be searched and that the possible values given may not include the
    optimal one. The parameter search will be performed using a 5-fold crossvalidation (default). Note that if a grid
    search is performed, the training data (predictors and target) must be provided. Otherwise, no data are required by
    this function.

    Parameters:
        model_ty     String. The type of model to set up. Choices and their abbreviations are given above.
        ndlev        Integer. The number of classes in the target variable, i.e., the number of danger levels. This can
                              be either 1, 2, 3, or 4. This is here only used for the SVM to determine of the kernel is
                              linear (ndlev=2) or rbf (ndlev > 2).
        hyperp       Dictionary. Dictionary containing the hyperparameters or the hyperparameter grid (if grid search
                                 is True) for the chosen statistical model. Parameters available so far are:
                                     DT: max_depth, min_leaf_samp
                                     LR: C
                                     SVM: C
                                     KNN: n_neighbors, weights
                                     RF: n_estimators, max_depth, min_leaf_samp
        grid_search  Logical. If true, a grid search for optimal hyperparameters will be performed as described above.
                              In case the hyperp dictionary is not empty, it will be assumed to provide the parameter
                              grid for the grid search.
        grid_sample  Integer. If > 0, the grid search will be randomised (RandomizedSearchCV from scikit-learn) with
                              grid_sample being the number of parameter settings sampled. If set to 0 (default) the full
                              grids will be searched.
        class_weight None, "balanced", or dictionary. If None the data are assumed to be balanced (i.e., external
                                                      balancing). If "balanced" the data will be assumed to be
                                                      unbalanced (although this should also work if the data are
                                                      balanced; this should simply imply that the weights will be the
                                                      same for each class) and an internal balancing via automatically
                                                      calculated class weights is performed. If a dictionary is
                                                      submitted, this corresponds to manually setting the class weights.
                                                      Thus, make sure that the names of the elements correspond to the
                                                      class names.
                                                      Note that for the KNN no internal balancing is implemented as of
                                                      yet, meaning that in the case of "balanced" in fact the unbalanced
                                                      data are used WITHOUT internal balancing.
        train_x      Pandas DataFrame containing the predictors. Only required if grid_search=True, that is, the grid
                                      search is performed for these data. Default is None.
                                      If class_weight is None this should be the balanced data, if it is "balanced" it
                                      should be the unbalanced (i.e., full) data.
        train_y      Pandas DataFrame containing the target variable (i.e., the avalanche danger level). Only required
                                      if grid_search=True. That is, the grid search is performed for these data. Default
                                      is None.
                                      If class_weight is None this should be the balanced data, if it is "balanced" it
                                      should be the unbalanced (i.e., full) data.
        cv_type      String. The type of folds to generated. Choices are "stratified", meaning that the class
                             frequencies of the folds will be equal to those in the original data, or "seasonal",
                             meaning that the folds are predefined based on the years of the avalanche seasons.
                             Defaults to seasonal.
        cv           Integer. The number of folds in the gridsearch crossvalidation. Defaults to 5.
        cv_score     String or score object. The score to be used in the cross-validation. For possible choices see
                                             https://scikit-learn.org/stable/modules/model_evaluation.html
        balance_meth String. The method used for balancing the data when loading the predictors. Note that this does
                             not control what is happening during the model training, which is controlled by the
                             balancing parameter. If balancing is set to none or internal, the balance_meth parameter
                             becomes irrelevant.
                             Choices are the following:
                               -undersample: uses the custom undersample function
                               -SMOTE: uses the synthetic minority oversampling method from the imbalanced-learn library
                               -SVMSMOTE: SMOTE using an SVM
                               -ADASYN: uses the adaptive synthetic sampling method from the imbalanced-learn library
                               -ros: uses the random oversampling method from the imbalanced-learn library
                               -rus: uses the random undersampling method from the imbalanced-learn library
                             Default is SMOTE.
                             There is considerable difference in procedure between the undersample case and the other
                             cases: In the undersample case the predictors have already been balanced (script
                             Gen_Store_XLevel_Balanced_Predictors.py). In the other cases the predictors are only
                             balanced here.

    Output:
        The statistical model as returned by the scikit-learn functions.
    """

    # set the hyperparameters
    if cv_type == "seasonal":
        pipe = True
    else:
        pipe = True
    # end if

    if grid_sample > 0:
        hy_ndlev = 0
    else:
        hy_ndlev = ndlev
    # end if
    param_grid = set_hyperp(model_ty=model_ty, in_hypp=hyperp, grid_search=grid_search, grid_sample=grid_sample,
                            pipe=pipe, ndlev=hy_ndlev, verbose=verbose)

    # set some model parameters:

    # kernel for the SVM
    if ndlev == 2:
        kernel = "linear"
    elif ndlev > 2:
        kernel = "rbf"
    # end if elif

    # max iterations for the LR
    max_iter = 1000


    if grid_search:
        # if grid_search=True, check if train_x and train_y are provided.
        if ((train_x is None) | (train_y is None)):
            print("\nNo predictors or traget provided. This is required for the hyperparameter search. Aborting.\n")
            sys.exit("No data for gridsearch.")
        # end if

        hyperparameters = perform_grid_search(model_ty, param_grid, ndlev, cv_type, cv, cv_score, grid_sample,
                                              balance_meth, class_weight, train_x, train_y, kernel, max_iter=max_iter,
                                              verbose=verbose)
    else:
        hyperparameters = param_grid
    # end if else

    # make the keys in hyperparameters compatible with non-pipeline usage
    try:  # Python >= 3.9
        hyperparameters = hyperparameters | {k.split("__")[-1]:hyperparameters[k] for k in hyperparameters.keys()}
    except: # Python >= 3.5
        hyperparameters = {**hyperparameters, **{k.split("__")[-1]:hyperparameters[k] for k in hyperparameters.keys()}}
    # end try except

    # select the model
    if model_ty == "DT":

        # define the decision tree model
        model = DecisionTreeClassifier(criterion="gini", max_depth=hyperparameters["max_depth"],
                                       min_samples_leaf=hyperparameters["min_samples_leaf"],
                                       class_weight=class_weight)

    elif model_ty == "LR":

        # define the logistic regression model
        model = LogisticRegression(max_iter=max_iter, class_weight=class_weight, C=hyperparameters["C"])

    elif model_ty == "SVM":

        # create the SVM Classifier
        model = SVC(kernel=kernel, class_weight=class_weight, C=hyperparameters["C"])

    elif model_ty == "KNN":

        # create a NN Classifier
        model = KNeighborsClassifier(n_neighbors=hyperparameters["n_neighbors"], weights=hyperparameters["weights"])

    elif model_ty == "RF":

        # define the random forest model according to the best hyperparameters
        model = RandomForestClassifier(n_estimators=hyperparameters["n_estimators"],
                                       max_depth=hyperparameters["max_depth"],
                                       min_samples_leaf=hyperparameters["min_samples_leaf"],
                                       min_samples_split=hyperparameters["min_samples_split"],
                                       max_features=hyperparameters["max_features"],
                                       random_state=42, class_weight=class_weight)
    # end if elif

    # print the hyperparameters
    if verbose:
        print("\nThe following set/gird of hyperparameters is used:")
        print(hyperparameters)
        print()
    # end if

    # return the model
    return model
# end def
