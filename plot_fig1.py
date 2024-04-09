import json

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xgi
from matplotlib.gridspec import GridSpec

import fig_settings as fs
from lcs import *

axislabel_fontsize = 12
tick_fontsize = 12
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

fig = plt.figure(figsize=(7, 5))
plt.subplots_adjust(left=0.1, right=0.86, bottom=0.15, top=0.95)

gs = GridSpec(2, 2, hspace=0.6, wspace=0.5)

"""
Panel 1: Network Viz
"""
ax1 = fig.add_subplot(gs[0])

ax1.text(-0.29, 1.1, "(a)", transform=ax1.transAxes, fontsize=13, fontweight='bold', va="top")

el = zkc(format="edgelist")
H = xgi.Hypergraph(el)
A = zkc()
n = A.shape[0]

i = 13
t = 56

gamma = 0.2
b = 0.07
contagion_function = lambda nu, b: 1 - (1 - b) ** nu
c = contagion_function(np.arange(n), b)
x0 = np.zeros(n)
x0[0] = 1

x = contagion_process(A, gamma, c, x0, tmin=0, tmax=100, random_seed=2)

infected_color = "C1"
susceptible_color = "white"
subgraph_color = "black"
graph_color = (0.7, 0.7, 0.7, 0.5)
subgraph_node_lc = "black"
graph_node_lc = (0.3, 0.3, 0.3)

sg = H.nodes.memberships(i)
nbrs = H.nodes.neighbors(i)
nbrs.add(i)

pos = xgi.pca_transform(xgi.pairwise_spring_layout(H, seed=5, k=0.25))
# rotate pos by 30 degree3
angle = -1.15 * np.pi / 6
rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
pos = {k: np.dot(v, rotation_matrix) for k,v in pos.items()}

node_fc = [infected_color if x[t, i] else susceptible_color for i in H.nodes]
node_ec = [subgraph_node_lc if n in nbrs else graph_node_lc for n in H.nodes]
node_fc[12] = "C0"

dyad_color = [subgraph_color if e in sg else graph_color for e in H.edges]


xgi.draw(
    H,
    pos=pos,
    node_size=6.5,
    node_fc=node_fc,
    dyad_color=dyad_color,
    dyad_lw=0.8,
    node_ec=node_ec,
    node_lw=0.8,
    ax=ax1
)
plt.scatter(pos[13][0], pos[13][1], s=50, c='C0', edgecolors='black', linewidths=0.8, zorder=10, marker='s')

ax1.set_xlim([1.01 * min([pos[i][0] for i in pos]), 1.01 * max([pos[i][0] for i in pos])])


"""
Panel 2: 
"""
ax2 = fig.add_subplot(gs[1])
ax2.text(-0.3, 1.1, "(b)", transform=ax2.transAxes, fontsize=12, fontweight='bold',  va="top")


with open("Data/zkc_infer_contagion_functions.json") as file:
    data = json.load(file)
A = np.array(data["A"], dtype=float)
c1 = np.array(data["c1"], dtype=float)
c2 = np.array(data["c2"], dtype=float)
x1 = np.array(data["x1"], dtype=int)
x2 = np.array(data["x2"], dtype=int)
A1_samples = np.array(data["A1-samples"], dtype=int)
A2_samples = np.array(data["A2-samples"], dtype=int)
gamma1_samples = np.array(data["gamma1-samples"], dtype=float)
gamma2_samples = np.array(data["gamma2-samples"], dtype=float)
c1_samples = np.array(data["c1-samples"], dtype=float)
c2_samples = np.array(data["c2-samples"], dtype=float)
l1 = np.array(data["l1"], dtype=float)
l2 = np.array(data["l2"], dtype=float)

kmax = np.max(degrees(A))
n = A.shape[0]

nus = np.arange(0, n, 1)

# simple contagion
c1_mean = c1_samples.mean(axis=0)
ax2.plot(nus, c1, "-", color="C0", lw=5, alpha=0.5)

err_c1 = np.zeros((2, n))
c1_mode = np.zeros(n)
for i in range(n):
    interval = az.hdi(c1_samples[:, i], hdi_prob=0.95)
    x, y = interval
    err_c1[0, i] = max(c1_mean[i] - x, 0)
    err_c1[1, i] = max(y - c1_mean[i], 0)

    
offset_distance = 0.15
ax2.errorbar(nus - offset_distance, c1_mean, err_c1, color="C0", fmt="o",
             capsize=3, markersize=5, markeredgecolor="#315b7d", label="Simple")

# threshold contagion, tau=2
c2_mean = c2_samples.mean(axis=0)
ax2.plot(nus, c2, "-", color="C1", lw=5, alpha=0.5)

err_c2 = np.zeros((2, n))
c2_mode = np.zeros(n)
for i in range(n):
    interval = az.hdi(c2_samples[:, i], alpha=0.05, roundto=4)
    x, y = interval
    err_c2[0, i] = max(c2_mean[i] - x, 0)
    err_c2[1, i] = max(y - c2_mean[i], 0)
ax2.errorbar(nus + offset_distance, c2_mean, err_c2, color="C1", fmt="o",
             capsize=3, markersize=5, markeredgecolor="#391c23", label="Complex")


ax2.set_xticks(np.arange(0, n, 5))
ax2.set_xlabel(r"# of infected neighbors $\nu$")
ax2.set_ylabel(r"Contagion $c(\nu)$")

ax2.set_xlim([0, 15.5])
ax2.set_ylim([0, 1])
ax2.set_yticks([0, 0.5, 1], [0, 0.5, 1])

ax2.legend(loc="upper left", handletextpad=0.1, frameon=False)

sns.despine()

""""
Panel 3: recovery vs. tmax
"""
ax3 = fig.add_subplot(gs[2])
ax3.text(-0.29, 1.1, "(c)", transform=ax3.transAxes, fontsize=12, fontweight='bold',  va="top")


with open("Data/zkc_infer_vs_tmax.json") as file:
    data = json.load(file)


tmax = data["tmax"]
sps = np.array(data["sps"], dtype=float)
ps = np.array(data["ps"], dtype=float)
fce = np.array(data["fce"], dtype=float)


ax3.semilogx(tmax, sps[0].mean(axis=1), color="C0", label="Simple")
ax3.semilogx(tmax, sps[1].mean(axis=1), color="C1", label="Complex")
ax3.fill_between(
    tmax,
    sps[0].mean(axis=1) - sps[0].std(axis=1),
    sps[0].mean(axis=1) + sps[0].std(axis=1),
    alpha=0.3,
    color="C0",
    edgecolor="none"
)
ax3.fill_between(
    tmax,
    sps[1].mean(axis=1) - sps[1].std(axis=1),
    sps[1].mean(axis=1) + sps[1].std(axis=1),
    alpha=0.3,
    color="C1",
    edgecolor="none"
)
ax3.set_ylabel("Performance")
ax3.set_xlabel(r"$t_{max}$")
ax3.set_xlim([10, 10**5])
ax3.set_xticks(
    [10, 1000, 100000],
    [r"$\mathregular{10^1}$", r"$\mathregular{10^3}$", r"$\mathregular{10^5}$"],
)
ax3.set_ylim([0, 1])
ax3.set_yticks([0, 0.5, 1], [0, 0.5, 1])

ax3.legend(loc="lower right", markerfirst=False, frameon=False, handlelength=0.8)
sns.despine()

"""
Panel 4: heatmap of recover vs. beta and f
"""
ax4 = fig.add_subplot(gs[3])
ax4.text(-0.31, 1.1, "(d)", transform=ax4.transAxes, fontsize=12, fontweight='bold',  va="top")

with open("Data/zkc_frac_vs_beta.json") as file:
    data = json.load(file)
beta = np.array(data["beta"], dtype=float)
frac = np.array(data["fraction"], dtype=float)
ps = np.array(data["ps"], dtype=float)
sps = np.array(data["sps"], dtype=float)
fce = np.array(data["fce"], dtype=float)

cmap = cmap

sps_summary = sps.mean(axis=2)


c = ax4.imshow(
    to_imshow_orientation(sps_summary),
    extent=(min(frac), max(frac), min(beta), max(beta)),
    aspect="auto",
    cmap=cmap,
    vmin=0,
    vmax=1,
)
ax4.set_xlabel(r"Complexity $\lambda$")
ax4.set_ylabel(r"Infectivity $\beta$")

ax4.set_xticks([0, 0.5, 1], [0, 0.5, 1])
ax4.set_yticks([0, 0.5, 1], [0, 0.5, 1])


cbar_ax = fig.add_axes([0.875, 0.15, 0.015, 0.31])  # x, y, width, height
cbar = plt.colorbar(c, cax=cbar_ax)
cbar.set_label(r"Performance", fontsize=axislabel_fontsize, rotation=270, labelpad=15)
cbar_ax.set_yticks([0, 0.5, 1], [0, 0.5, 1], fontsize=tick_fontsize)

plt.savefig("Figures/Fig1/fig1.png", dpi=1000)
plt.savefig("Figures/Fig1/fig1.pdf", dpi=1000)
# plt.show()
