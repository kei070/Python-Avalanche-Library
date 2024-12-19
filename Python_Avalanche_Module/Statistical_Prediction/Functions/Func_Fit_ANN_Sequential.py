#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function to set up an artificial neural network (ANN) based on the Keras Sequential methodology.
"""

# imports
import os
import sys
import numpy as np
from keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from keras.utils import to_categorical


#%% proprietary functions
from .Func_ANN_Sequential import ANN

# function
def fit_ANN(train_x, train_y, test_x, test_y, test_x_all, test_y_all,
            input_shape, n_in_nodes, in_dropout, n_hid_nodes, dropouts, n_out_nodes,
            activ_out="softmax", learning_rate=0.001, loss="categorical_crossentropy",
            epochs=100, batch_size=64, stop_early=True, es_params={"monitor":"val_loss", "patience":20},
            class_weight=None, shuffle=True, verbose=-1):


    """
    Set up and fit an artificial neural network (ANN) based on the Keras Sequential methodology. The function ANN is
    used to set up the model. Both the model and the fitting history are returned.

    Parameters:
        train_x       DataFrame. The predictors on which the ANN is trained. Note that the classes should be balanced if
                                 class_weight is set to None (default).
        train_y       DataFrame. The target variable (in one-hot format! --> if not, use the to_categorical function
                                 from keras.utils to convert to one-hot) data set on which the ANN is trained. Note that
                                 the classes should be balanced if class_weight is set to None (default).
        test_x        DataFrame. The predictors which are used to validate the ANN.
        test_y        DataFrame. The target variable data set (in one-hot format! --> if not, use the to_categorical
                                 function from keras.utils to convert to one-hot) used to validate the ANN.
        input_shape   Integer. Equals the number of features that are used to predict the target variable.
        n_in_nodes    Integer. The number of nodes in the input layer.
        in_dropout    Float. The dropout fraction of the input layer.
        n_hid_nodes   List of integers. The number of nodes in the hidden layers. The length of the list determines the
                                        number of hidden layers.
        dropouts      List of floats. The dropout fractions of the hidden layers. The legnth of this list must equate
                                      the length of n_hid_nodes.
        activ_out      String. The activation function in the output layer. For a binary target variable this should be
                               set to "sigmoid", while for a multi-class problem (i.e., n_out_nodes > 2) it should be
                               "softmax" (default).
        learning_rate  Float. Learning rate to be applied in the optimiser (which in this function is fixed to Adam).
                              Defaults to 0.001.
        loss           String. The loss function to be applied. For a binary target variable this should be
                               "binary_crossentropy" while for a multi-class problem (i.e., n_out_nodes > 2) it should be
                               "categorical_crossentropy" (default).
        epochs         Integer. The number of epochs the ANN iterates over during training. Defaults to 100.
        batch_size     Integer. The batch size that the ANN uses during one iteration in the training process. Defaults
                                to 64.
        stop_early     Logical. If True, an early-stopping procedure is used which interupts the training process if the
                                loss (default) no longer reduces sufficiently after a given number of epochs (default:
                                20). The weights of the best iteration are restored.
        es_params      Dictionary of parameters to be used in the early-stopping procedure. Defaults to
                                  monitor="val_loss" (i.e., the loss with respect to the validation data is used as a
                                  heuristic for the early stopping) and patience=20, i.e., if there is no decrease of the
                                  loss after 20 iterations/epochs the training is stopped. Note that there are more
                                 parameters which are not implemented yet.
        class_weight   None or "balanced". If None the data are assumed to be balanced (i.e., external balancing), if
                                           "balanced" the data will be assumed to be unbalanced and an internal
                                           balancing via automatically calculated class weights is performed. Note that
                                           for the KNN no internal balancing is implemented as of yet, meaning that in
                                           the case of "balanced" in fact the unbalanced data are used WITHOUT internal
                                           balancing.
        shuffle        Logical. If True (default), the training data are shuffled to make sure that they are not sorted,
                                which may influence model training.
        verbose        Integer. The degree to which print statements are made during training. Set to -1 (default) for no
                                prints at all, except for the those during the fitting procedure.
    """

    # compute class weights
    if class_weight is None:
        class_weights = np.repeat(1, len(np.unique(train_y)))
    elif class_weight == "balanced":
        # since the data are in one-hot format convert them back to the standard
        temp_y = np.argmax(train_y, axis=1)
        class_weights = compute_class_weight('balanced', classes=np.unique(temp_y), y=temp_y)
    # end if elif
    class_weight_dict = dict(enumerate(class_weights))

    if verbose > -1:
        print("\nClass weights:")
        print(class_weight_dict)
        print()
    else:
        verbose = 0
    # end if

    # set up the early stopping
    if stop_early:
        early_stopping = [EarlyStopping(monitor=es_params["monitor"], patience=es_params["patience"], verbose=verbose,
                                        mode='min', restore_best_weights=True)]
    else:
        early_stopping = []
    # end if

    # convert the train and test data to the one-hot format
    ndlev = len(np.unique(train_y))
    if ndlev > 2:
        train_y_1h = to_categorical(train_y, num_classes=ndlev)
        test_y_1h = to_categorical(test_y, num_classes=ndlev)
        test_y_all_1h = to_categorical(test_y_all, num_classes=ndlev)
    else:
        train_y_1h = train_y
        test_y_1h = test_y
        test_y_all_1h = test_y_all
    # end if else

    # use the ANN function to set up the model
    model = ANN(input_shape, n_in_nodes, in_dropout, n_hid_nodes, dropouts, n_out_nodes, activ_out, learning_rate, loss)

    # fit the model to the data
    history = model.fit(train_x, train_y_1h, epochs=epochs, batch_size=batch_size, verbose=0,
                        callbacks=early_stopping, shuffle=shuffle, validation_data=(test_x_all, test_y_all_1h),
                        class_weight=class_weight_dict)

    return model, history

# end def
