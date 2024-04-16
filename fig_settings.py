# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 10:11:05 2021

@author: John Meluso
"""

import cmasher as cmr
import matplotlib as mpl
import matplotlib.pylab as pylab


# color styling
def set_colors(n_colors=2):
    global cmap
    global pallette
    cmap = "cmr.redshift"
    qualitative_cmap = cmr.get_sub_cmap(cmap, 0.2, 0.8, N=n_colors)

    pallette = qualitative_cmap.colors
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=pallette)


def set_fonts(extra_params={}):
    params = {
        "font.family": "Serif",
        # "font.sans-serif": ["Tahoma", "DejaVu Sans", "Lucida Grande", "Verdana"],
        "mathtext.fontset": "cm",
        "legend.fontsize": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.titlesize": 12,
    }
    for key, value in extra_params.items():
        params[key] = value
    pylab.rcParams.update(params)
