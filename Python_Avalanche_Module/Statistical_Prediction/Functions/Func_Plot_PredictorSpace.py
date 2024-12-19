#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function for plotting the predicted danger-level classes in the predictor space.
"""

# imports
import numpy as np
import pylab as pl
from matplotlib.colors import ListedColormap

def plot_2predictor_space(data_x, model, test_x, test_y, sel_feats, h=0.5):

    """
    Plots the predictor space in a 2d-meshgrid as well as the locations target variable in this space.
    The function is based on a suggestion by ChatUiT.

    Parameters:
        data_x     The full predictor data set. If load_xlevel_preds or apply_stat_mod is used to load the data, just
                   concatenate train_x_all and test_x_all.
        model      The statistical model used for the prediction. Use e.g. the output of apply_stat_mod.
        test_x     The test predictors. These will be used to plot the locations of the target variable in the predictor
                   space.
        test_y     The target variable corresponding to the predictor values given in test_x.
        sel_feats  The selected features (i.e., predictors).
        h          Float. The step size in the predictor space mesh.
    """

    # get the number of classes
    ndlev = np.unique(test_y)

    # create a mesh grid
    x_min, x_max = data_x[:, 0].min() - 1, data_x[:, 0].max() + 1
    y_min, y_max = data_x[:, 1].min() - 1, data_x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # predict class labels for each point in the mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # define the colors
    # cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    # cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    # Light red, light green, light blue, golden rod
    background_colors = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#DAA520'])

    # Red, green, blue, gold
    data_point_colors = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#FFD700'])


    # plot the decision boundary
    fig = pl.figure(figsize=(8, 5))
    ax00 = fig.add_subplot(111)

    # span up the predictor space
    ax00.pcolormesh(xx, yy, Z, cmap=background_colors)

    # plot also the target locations
    p00 = ax00.scatter(test_x[:, 0], test_x[:, 1], c=test_y, cmap=data_point_colors, edgecolor='k', s=20)
    ax00.legend(*p00.legend_elements(), title="Classes")
    ax00.set_xlim(xx.min(), xx.max())
    ax00.set_ylim(yy.min(), yy.max())
    ax00.set_title(f"{len(ndlev)}-Class classification")
    try:
        ax00.set_xlabel(sel_feats[0])
        ax00.set_ylabel(sel_feats[1])
    except:
        print("\nNo feature names given.\n")
    # end try
    pl.show()
    pl.close()
# end def


def plot_3predictor_space(model, data_x, data_y, sel_feats):

    """
    Plots the target variable in a 3d-predictor space.
    The function is based on a suggestion by ChatUiT.

    Parameters:
        model      The statistical model used for the prediction. Use e.g. the output of apply_stat_mod.
        data_x     Predictors that will be used to plot the locations of the target variable in the predictor
                   space.
        data_y     The target variable corresponding to the predictor values given in test_x.
        sel_feats  The selected features (i.e., predictors).
    """

    # Red, green, blue, gold
    data_point_colors = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#FFD700'])

    fig = pl.figure()

    ax00 = fig.add_subplot(111, projection='3d')

    p00 = ax00.scatter(data_x[:, 0], data_x[:, 1], data_x[:, 2], c=data_y, cmap=data_point_colors)

    legend = ax00.legend(*p00.legend_elements(), title="Classes")
    ax00.add_artist(legend)

    try:
        ax00.set_xlabel(sel_feats[0])
        ax00.set_ylabel(sel_feats[1])
        ax00.set_zlabel(sel_feats[2])
    except:
        print("\nNo feature names given.\n")
    # end try

    pl.show()
    pl.show()
# end def