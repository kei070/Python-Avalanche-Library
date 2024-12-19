#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function to set up an artificial neural network (ANN) based on the Keras Sequential methodology.
"""

# imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.initializers import HeNormal
from keras.optimizers import Adam

# function
def ANN(input_shape, n_in_nodes, in_dropout, n_hid_nodes, dropouts, n_out_nodes,
        activ_out="softmax", learning_rate=0.001, loss="categorical_crossentropy"):


    """
    Set up an artificial neural network (ANN) based on the Keras Sequential methodology. The ANN model is returned and
    can then be used to fit the data.

    Parameters:
        input_shape   Integer. Equals the number of features that are used to predict the target variable.
        n_in_nodes    Integer. The number of nodes in the input layer.
        in_dropout    Float. The dropout fraction of the input layer.
        n_hid_nodes   List of integers. The number of nodes in the hidden layers. The length of the list determines the
                                        number of hidden layers.
        n_out_nodes   Integer. The number of the nodes of the output layer. This must correspond to the number of
                               classes in the target variable. This corresponds to the number danger levels that are
                               predicted. Note that for a binary problem (i.e., two classes) this must be 1, while for
                               three or more classes this must equate the number of classes.
        dropouts      List of floats. The dropout fractions of the hidden layers. The legnth of this list must equate
                                      the length of n_hid_nodes.
        n_out_nodes   Integer. The number of the nodes of the output layer. This must correspond to the number of
                               classes in the target variable. This corresponds to the number danger levels that are
                               predicted. Note that for a binary problem (i.e., two classes) this must be 1, while for
                               three or more classes this must equate the number of classes.
        activ_out      String. The activation function in the output layer. For a binary target variable this should be
                               set to "sigmoid", while for a multi-class problem (i.e., n_out_nodes > 2) it should be
                               "softmax" (default).
        learning_rate  Float. Learning rate to be applied in the optimiser (which in this function is fixed to Adam).
                              Defaults to 0.001.
        loss           String. The loss function to be applied. For a binary target variable this should be
                               "binary_crossentropy" while for a multi-class problem (i.e., n_out_nodes > 2) it should
                               be "categorical_crossentropy" (default).
    """


    model = Sequential()  # build the model sequentially, kind of 'by hand'

    # first layer
    model.add(Dense(n_in_nodes, input_shape=(input_shape,), activation='relu', kernel_initializer=HeNormal()))
    Dropout(in_dropout)

    # add hidden layers using a for-loop
    for n_n, drop in zip(n_hid_nodes, dropouts):
        model.add(Dense(n_n, activation='relu', kernel_initializer=HeNormal()))
        Dropout(drop)
    # end for n_n, drop

    # the output layer has one node and uses the sigmoid activation function
    model.add(Dense(n_out_nodes, activation=activ_out))

    # Create the optimizer with a specific learning rate
    optimizer = Adam(learning_rate=learning_rate)

    # compile the model
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    return model

# end def
