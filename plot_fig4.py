import json

import matplotlib.pyplot as plt
import numpy as np
import xgi
from matplotlib.gridspec import GridSpec

import fig_settings as fs
from lcs import *

metric_name = "auprc"
axis_limits = [0, 1]

axislabel_fontsize = 20
tick_fontsize = 18
fs.set_fonts(
    {
        "font.family": "serif",
        "axes.labelsize": axislabel_fontsize,
        "xtick.labelsize": tick_fontsize,
        "ytick.labelsize": tick_fontsize,
    }
)
fs.set_colors()
cmap = fs.cmap


models = ["Erdos-Renyi", "CM", "clustered_network", "SBM", "Watts-Strogatz"]
cfs = [
    "SIS",
    r"Threshold, $\tau=2$",
    r"Threshold, $\tau=3$",
]
keys = ["p", "alpha", "size", "epsilon", "p"]
titles = ["Erdös-Rényi", "Power-law CM", "Clustered", "SBM", "Small-World"]
labels = [r"$p$", r"$\alpha$", r"$s$", r"$\epsilon$", r"$p$"]
xticks = [
    [0, 0.5, 1],
    [-4, -3.5, -3, -2.5, -2, -1.5],
    [1, 7, 13, 19],
    [0, 0.5, 1],
    [-6, -4, -2, 0],
]
xticklabels = [
    ["0", "0.5", "1"],
    ["-4", "-3.5", "-3", "-2.5", "-2", "-1.5"],
    ["1", "7", "13", "19"],
    ["0", "0.5", "1"],
    [
        r"$\mathregular{10^{-6}}$",
        r"$\mathregular{10^{-2}}$",
        r"$\mathregular{10^{-2}}$",
        r"$\mathregular{10^{0}}$",
    ],
]
convert_to_log = [False, False, False, False, True]


def visualize_networks(i, ax):
    n = 50
    match i:
        case 0:
            A = erdos_renyi(n, 0.1, seed=0)
            e = [(i, j) for i, j in nx.Graph(A).edges]
        case 1:
            A = truncated_power_law_configuration(n, 2, 20, -3, seed=0)
            e = [(i, j) for i, j in nx.Graph(A).edges]
        case 2:
            k = 2  # each node belongs to two cliques
            clique_size = 4
            k1 = k * np.ones(n)
            num_cliques = round(sum(k1) / clique_size)
            k2 = clique_size * np.ones(num_cliques)
            A = clustered_network(k1, k2, seed=0)
            e = [(i, j) for i, j in nx.Graph(A).edges]
        case 3:
            A = sbm(n, 10, 0.9, seed=0)
            e = [(i, j) for i, j in nx.Graph(A).edges]
        case 4:
            A = watts_strogatz(n, 6, 0.03, seed=0)
            e = [(i, j) for i, j in nx.Graph(A).edges]

    H = xgi.Hypergraph(e)

    node_size = 5
    dyad_lw = 0.5
    node_lw = 0.5

    match i:
        case 0:
            pos = xgi.pairwise_spring_layout(H, seed=2)
        case 1:
            pos = xgi.pairwise_spring_layout(H, seed=2)
        case 2:
            pos = xgi.pairwise_spring_layout(H, seed=2)
        case 3:
            pos = xgi.pca_transform(xgi.pairwise_spring_layout(H, seed=2))
        case 4:
            pos = xgi.circular_layout(H)
    xgi.draw(H, ax=ax, pos=pos, node_size=node_size, node_lw=node_lw, dyad_lw=dyad_lw)


fig = plt.figure(figsize=(16, 10))
plt.subplots_adjust(left=0.09, right=0.9, bottom=0.1, top=0.95, wspace=0.4, hspace=0.4)

gs = GridSpec(len(cfs) + 1, len(models), wspace=0.2, hspace=0.2)

for i, m in enumerate(models):
    with open(f"Data/{m.lower()}.json") as file:
        data = json.load(file)
    var = np.array(data[keys[i]], dtype=float)
    b = np.array(data["beta"], dtype=float)
    recovery_metric = np.array(data[metric_name], dtype=float)

    if convert_to_log[i]:
        var = np.log10(var)

    for j, cf in enumerate(cfs):
        recovery_average = recovery_metric[j].mean(axis=2).T
        ax = fig.add_subplot(gs[j + 1, i])
        im = ax.imshow(
            to_imshow_orientation(recovery_average),
            extent=(min(var), max(var), min(b), max(b)),
            vmin=axis_limits[0],
            vmax=axis_limits[1],
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

cbar_ax = fig.add_axes([0.91, 0.1, 0.015, 0.63])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label(r"AUPRC", fontsize=axislabel_fontsize, rotation=270, labelpad=25)
cbar_ax.set_yticks([0, 0.5, 1], [0, 0.5, 1], fontsize=tick_fontsize)

for i, m in enumerate(models):
    ax = fig.add_subplot(gs[0, i])
    visualize_networks(i, ax)
    ax.set_title(titles[i])

plt.savefig(f"Figures/Fig4/fig4.png", dpi=1000)
plt.savefig(f"Figures/Fig4/fig4.pdf", dpi=1000)
# plt.show()
