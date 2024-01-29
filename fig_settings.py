# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 10:11:05 2021

@author: John Meluso
"""

import os

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt


import matplotlib as mpl
import matplotlib.colors as mcolors
from cycler import cycler
import cmasher as cmr

 


#color styling


def set_colors(n_colors = 2):
    global cmap 
    global pallette
    cmap = 'cmr.redshift'
    qualitative_cmap = cmr.get_sub_cmap(cmap, 0.2, 0.8, N=n_colors)

    pallette = qualitative_cmap.colors
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color= pallette)


# cmap = mcolors.LinearSegmentedColormap.from_list('my_cmap',[pallete[0],pallete[1]])
# cmap = cmr.ember

def set_fontsize():
    plt.rcParams['axes.labelsize'] = 30
    # Set the global default size of the tick labels
    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15
    plt.rcParams['axes.titlesize'] = 25
    plt.rcParams['legend.fontsize'] = 25



def set_fonts(extra_params={}):
    params = {
        "font.family": "Sans-Serif",
        "font.sans-serif": ["Tahoma", "DejaVu Sans", "Lucida Grande", "Verdana"],
        "mathtext.fontset": "cm",
        "legend.fontsize": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.titlesize": 16,
    }
    for key, value in extra_params.items():
        params[key] = value
    pylab.rcParams.update(params)


def fig_size(frac_width, frac_height, n_cols=1, n_rows=1):
    # Set default sizes
    page_width = 8.5
    page_height = 11
    side_margins = 1
    tb_margins = 1
    middle_margin = 0.25
    mid_marg_width = middle_margin * (n_cols - 1)
    mid_marg_height = middle_margin * (n_rows - 1)

    # Width logic
    if frac_width == 1:
        width = page_width - side_margins
    else:
        width = (page_width - side_margins - mid_marg_width) * frac_width

    # Height logic
    if frac_height == 1:
        height = page_height - tb_margins
    else:
        height = (page_height - tb_margins - mid_marg_height) * frac_height

    return (width, height)






def get_formats():
    return ["eps", "jpg", "pdf", "png", "tif"]


def set_border(ax, top=False, bottom=False, left=False, right=False):
    ax.spines["top"].set_visible(top)
    ax.spines["right"].set_visible(right)
    ax.spines["bottom"].set_visible(bottom)
    ax.spines["left"].set_visible(left)


def save_publication_fig(name, dpi=1200, **kwargs):
    save_fig(name, dpi, fig_type="publication", **kwargs)


def save_presentation_fig(name, dpi=1200, **kwargs):
    save_fig(name, dpi, fig_type="presentation", **kwargs)


def save_fig(name, dpi=1200, fig_type=None, **kwargs):
    for ff in get_formats():
        if fig_type:
            path = f"../figures/{fig_type}/{ff}"
        else:
            path = f"../figures/{ff}"
        if not os.path.exists(path):
            os.makedirs(path)
        fname = f"{path}/{name}.{ff}"
        plt.savefig(fname, format=ff, dpi=dpi, **kwargs)
