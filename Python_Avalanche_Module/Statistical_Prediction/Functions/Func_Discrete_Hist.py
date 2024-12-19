#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function to plot a histogram-style bar-plot for discrete classes.
"""

# imports
import numpy as np
import pylab as pl


# function 1 -- data preperation
def disc_hist_data(data_list, classes=None):

    data_uni_list = []
    data_hist_list = []
    for data in data_list:

        if classes is None:
            data_uni = np.unique(data)
            data_uni_list.append(data_uni)
        else:
            data_uni = classes
            data_uni_list.append(data_uni)
        # end if else

        data_hist = []

        for val in data_uni:
            data_hist.append(np.sum(np.array(data) == val))
        # end for val
        data_hist_list.append(data_hist)
    # end for data

    return data_uni_list, data_hist_list

# end def

# function 2 -- the plot
def disc_hist(data_list, classes=None, width=[0.5], color=["black"], labels=[None], legend=True,
              figsize=(8, 5), xlabel="", ylabel="", title="", add_xtick=None, linewidth=1):

    # use disc_hist_data to prepare the data
    data_uni_list, data_hist_list = disc_hist_data(data_list=data_list, classes=classes)

    # plot
    fig = pl.figure(figsize=figsize)

    ax00 = fig.add_subplot(111)

    i = 0
    for data_uni, data_hist in zip(data_uni_list, data_hist_list):
        ax00.bar(np.array(data_uni), np.array(data_hist), width=width[i], color="none", edgecolor=color[i],
                 label=labels[i], linewidth=linewidth)
        i += 1
    # end for

    if legend:
        ax00.legend()
    # end if

    ax00.set_xticks(data_uni)
    if add_xtick != None:
        ax00.set_xticklabels([f"{i}\n{tick}%" for i, tick in zip(data_uni, add_xtick)])
    else:
        ax00.set_xticklabels(data_uni)
    # end if else

    ax00.set_xlabel(xlabel)
    ax00.set_ylabel(ylabel)
    ax00.set_title(title)

    pl.show()
    pl.close()
# end def
