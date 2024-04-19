import json

import matplotlib.pyplot as plt
import numpy as np
import xgi
from matplotlib.gridspec import GridSpec

import fig_settings as fs
from lcs import *

axis_limits = [0, 1]

fs.set_fonts()
fs.set_colors()
cmap = fs.cmap


models = [
    "Erdos-Renyi",
    "CM",
    "clustered_network",
]
cfs = [
    "SIS",
    r"Threshold, $\tau=2$",
]
keys = ["p", "alpha", "size"]
titles = ["Erdös-Rényi", "Power-law CM", "Clustered"]
labels = [r"$p$", r"$\alpha$", r"$s$"]
xticks = [
    [0, 0.5, 1],
    [-4, -3.5, -3, -2.5, -2, -1.5],
    [1, 7, 13, 19],
]
xticklabels = [
    ["0", "0.5", "1"],
    ["-4", "-3.5", "-3", "-2.5", "-2", "-1.5"],
    ["1", "7", "13", "19"],
]
convert_to_log = [False, False, False]


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
    xgi.draw(H, ax=ax, pos=pos, node_size=node_size, node_lw=node_lw, dyad_lw=dyad_lw)


fig = plt.figure(figsize=(8, 8))
plt.subplots_adjust(left=0.12, right=0.85, bottom=0.1, top=0.95, wspace=0.4, hspace=0.4)

gs = GridSpec(3, len(models), wspace=0.2, hspace=0.2)

for i, m in enumerate(models):
    with open(f"Data/{m.lower()}.json") as file:
        data = json.load(file)
    var = np.array(data[keys[i]], dtype=float)
    b = np.array(data["beta"], dtype=float)

    performance = np.array(data["auprc"], dtype=float)

    # plot the difference in auprc
    mean_difference = performance[1].mean(axis=2).T - performance[0].mean(axis=2).T
    ax = fig.add_subplot(gs[1, i])
    im1 = ax.imshow(
        to_imshow_orientation(mean_difference),
        extent=(min(var), max(var), min(b), max(b)),
        vmin=-0.5,
        vmax=0.5,
        aspect="auto",
        cmap=cmap,
    )
    ax.set_xlim([min(var), max(var)])
    ax.set_ylim([min(b), max(b)])
    ax.set_xticks(xticks[i], xticklabels[i])
    ax.set_yticks([0, 0.5, 1], [0, 0.5, 1])

    if i == 0:
        ax.set_ylabel(r"$\beta$")
    else:
        ax.set_yticks([], [])

    ax.set_xticks([], [])

    # plot the density
    rho_samples = np.array(data["rho-samples"], dtype=float)
    rho = np.array(data["rho"], dtype=float)
    density_error = np.abs(rho_samples - rho)

    mean_difference = density_error[0].mean(axis=2) - density_error[1].mean(axis=2)
    ax = fig.add_subplot(gs[2, i])
    im2 = ax.imshow(
        to_imshow_orientation(mean_difference),
        extent=(min(var), max(var), min(b), max(b)),
        vmin=-0.5,
        vmax=0.5,
        aspect="auto",
        cmap=cmap,
    )
    ax.set_xlim([min(var), max(var)])
    ax.set_ylim([min(b), max(b)])
    ax.set_xticks(xticks[i], xticklabels[i])
    ax.set_yticks([0, 0.5, 1], [0, 0.5, 1])

    if i == 0:
        ax.set_ylabel(r"$\beta$")
    else:
        ax.set_yticks([], [])

    ax.set_xlabel(labels[i])

cbar_ax1 = fig.add_axes([0.86, 0.45, 0.015, 0.325])
cbar = fig.colorbar(im1, cax=cbar_ax1)
cbar.set_label(
    r"$\mathregular{AUPRC_{Complex} - AUPRC_{Simple}}$", rotation=270, labelpad=25
)
# cbar_ax1.set_yticks([0, 0.5, 1], [0, 0.5, 1])

cbar_ax2 = fig.add_axes([0.86, 0.1, 0.015, 0.325])
cbar = fig.colorbar(im2, cax=cbar_ax2)
cbar.set_label(
    r"$|\rho-\rho_{\mathregular{Simple}}| - |\rho-\rho_{\mathregular{Complex}}|$",
    rotation=270,
    labelpad=25,
)
# cbar_ax2.set_yticks([-1, 0, 1], [-1, 0, 1])

for i, m in enumerate(models):
    ax = fig.add_subplot(gs[0, i])
    visualize_networks(i, ax)
    ax.set_title(titles[i])

plt.savefig(f"Figures/Fig2/fig2.png", dpi=1000)
plt.savefig(f"Figures/Fig2/fig2.pdf", dpi=1000)
plt.show()
