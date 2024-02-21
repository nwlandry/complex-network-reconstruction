import json

import matplotlib.pyplot as plt
import numpy as np
import xgi
from matplotlib.gridspec import GridSpec

import fig_settings as fs
from lcs import *

fs.set_colors()
fs.set_fonts({"font.family": "sans-serif"})
cmap = fs.cmap


models = ["SBM", "Watts-Strogatz"]
cfs = [
    "SIS",
    r"Threshold, $\tau=2$",
    r"Threshold, $\tau=3$",
]
keys = ["epsilon", "p"]
titles = ["SBM", "Small-World"]
labels = [r"$\epsilon$", r"$p$"]
xticks = [
    [0, 0.5, 1],
    [-6, -4, -2, 0],
]
xticklabels = [
    ["0", "0.5", "1"],
    [r"$10^{-6}$", r"$10^{-4}$", r"$10^{-2}$", r"$10^{0}$"],
]
convert_to_log = [False, True]


def visualize_networks(i, ax):
    n = 50
    match i:
        case 0:
            A = sbm(n, 10, 0.9, seed=0)
            e = [(i, j) for i, j in nx.Graph(A).edges]
        case 1:
            A = watts_strogatz(n, 6, 0.03, seed=0)
            e = [(i, j) for i, j in nx.Graph(A).edges]

    H = xgi.Hypergraph(e)

    node_size = 4
    dyad_lw = 0.5
    node_lw = 0.5

    match i:
        case 0:
            pos = xgi.pca_transform(xgi.pairwise_spring_layout(H, seed=2))
        case 1:
            pos = xgi.circular_layout(H)
    xgi.draw(H, ax=ax, pos=pos, node_size=node_size, node_lw=node_lw, dyad_lw=dyad_lw)


fig = plt.figure(figsize=(6, 9))
gs = GridSpec(len(cfs) + 1, len(models), wspace=0.2, hspace=0.2)

for i, m in enumerate(models):
    with open(f"Data/{m.lower()}.json") as file:
        data = json.load(file)
    var = np.array(data[keys[i]], dtype=float)
    b = np.array(data["beta"], dtype=float)
    sps = np.array(data["sps"], dtype=float)

    if convert_to_log[i]:
        var = np.log10(var)

    for j, cf in enumerate(cfs):
        sps_summary = sps[j].mean(axis=2).T
        ax = fig.add_subplot(gs[j + 1, i])
        im = ax.imshow(
            to_imshow_orientation(sps_summary),
            extent=(min(var), max(var), min(b), max(b)),
            vmin=0,
            vmax=1,
            aspect="auto",
            cmap=cmap,
        )
        ax.set_xlim([min(var), max(var)])
        ax.set_ylim([min(b), max(b)])
        ax.set_xticks(xticks[i], xticklabels[i])
        ax.set_yticks([0, 0.5, 1], [0, 0.5, 1])

        if i == 0:
            ax.set_ylabel(f"{cfs[j]}\n" + r"$\beta$")
        else:
            ax.set_yticks([], [])

        if j + 1 == len(cfs):
            ax.set_xlabel(labels[i])
        else:
            ax.set_xticks([], [])

cbar_ax = fig.add_axes([0.91, 0.11, 0.015, 0.57])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label(r"F-Score", fontsize=15, rotation=270, labelpad=25)
cbar_ax.set_yticks([0, 0.5, 1], [0, 0.5, 1], fontsize=15)

for i, m in enumerate(models):
    ax = fig.add_subplot(gs[0, i])
    visualize_networks(i, ax)
    ax.set_title(titles[i])

plt.savefig("Figures/Fig3/fig3.png", dpi=1000)
plt.savefig("Figures/Fig3/fig3.pdf", dpi=1000)
plt.show()
