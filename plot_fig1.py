import json

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xgi
from matplotlib.gridspec import GridSpec

import fig_settings as fs
from lcs import *

axislabel_fontsize = 20
tick_fontsize = 18
fs.set_fonts(
    {
        "font.family": "sans-serif",
        "axes.labelsize": axislabel_fontsize,
        "xtick.labelsize": tick_fontsize,
        "ytick.labelsize": tick_fontsize,
    }
)
fs.set_colors()
cmap = fs.cmap

fig = plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=0.1, right=0.86, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)

gs = GridSpec(2, 2, hspace=0.4, wspace=0.4)

"""
Panel 1: Network Viz
"""
ax1 = fig.add_subplot(gs[0])

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
graph_color = (0.1, 0.1, 0.1, 0.1)
subgraph_node_lc = "black"
graph_node_lc = (0.3, 0.3, 0.3)

sg = H.nodes.memberships(i)
nbrs = H.nodes.neighbors(i)
nbrs.add(i)

pos = xgi.pca_transform(xgi.pairwise_spring_layout(H, seed=5, k=0.3))
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
    dyad_lw=0.5,
    node_ec=node_ec,
    node_lw=0.5,
    ax=ax1,
)


"""
Panel 2: 
"""
ax2 = fig.add_subplot(gs[1])

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
ax2.plot(nus, c1, "-", color="C0", label="Simple contagion")

err_c1 = np.zeros((2, n))
c1_mode = np.zeros(n)
for i in range(n):
    interval = az.hdi(c1_samples[:, i], hdi_prob=0.95)
    x, y = interval
    err_c1[0, i] = max(c1_mean[i] - x, 0)
    err_c1[1, i] = max(y - c1_mean[i], 0)
ax2.errorbar(nus, c1_mean, err_c1, color="C0", fmt="o")

# threshold contagion, tau=2
c2_mean = c2_samples.mean(axis=0)
ax2.plot(nus, c2, "-", color="C1", label="Complex contagion")

err_c2 = np.zeros((2, n))
c2_mode = np.zeros(n)
for i in range(n):
    interval = az.hdi(c2_samples[:, i], alpha=0.05, roundto=4)
    x, y = interval
    err_c2[0, i] = max(c2_mean[i] - x, 0)
    err_c2[1, i] = max(y - c2_mean[i], 0)
ax2.errorbar(nus, c2_mean, err_c2, color="C1", fmt="o")

ax2.set_xticks(np.arange(0, n, 5))
ax2.set_xlabel(r"$\nu$")
ax2.set_ylabel(r"$c(\nu)$")

ax2.set_xlim([0, kmax + 2.5])
ax2.set_ylim([0, 1])
ax2.set_yticks([0, 0.5, 1], [0, 0.5, 1])

ax2.legend(loc="upper left")

sns.despine()

""""
Panel 3: recovery vs. tmax
"""
ax3 = fig.add_subplot(gs[2])


with open("Data/zkc_infer_vs_tmax.json") as file:
    data = json.load(file)


tmax = data["tmax"]
sps = np.array(data["sps"], dtype=float)
ps = np.array(data["ps"], dtype=float)
fce = np.array(data["fce"], dtype=float)


ax3.semilogx(tmax, sps[0].mean(axis=1), color="C0", label="Simple contagion")
ax3.semilogx(tmax, sps[1].mean(axis=1), color="C1", label="Complex contagion")
ax3.fill_between(
    tmax,
    sps[0].mean(axis=1) - sps[0].std(axis=1),
    sps[0].mean(axis=1) + sps[0].std(axis=1),
    alpha=0.3,
    color="C0",
)
ax3.fill_between(
    tmax,
    sps[1].mean(axis=1) - sps[1].std(axis=1),
    sps[1].mean(axis=1) + sps[1].std(axis=1),
    alpha=0.3,
    color="C1",
)
ax3.set_ylabel("F-Score")
ax3.set_xlabel(r"$t_{max}$")

ax3.legend(loc="upper left")
sns.despine()

"""
Panel 4: heatmap of recover vs. beta and f
"""
ax4 = fig.add_subplot(gs[3])

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
ax4.set_xlabel(r"$f$")
ax4.set_ylabel(r"$\beta$")

ax4.set_xticks([0, 0.5, 1], [0, 0.5, 1])
ax4.set_yticks([0, 0.5, 1], [0, 0.5, 1])


cbar_ax = fig.add_axes([0.875, 0.11, 0.015, 0.32])  # x, y, width, height
cbar = plt.colorbar(c, cax=cbar_ax)
cbar.set_label(r"F-Score", fontsize=axislabel_fontsize, rotation=270, labelpad=25)
cbar_ax.set_yticks([0, 0.5, 1], [0, 0.5, 1], fontsize=tick_fontsize)

# plt.savefig("Figures/Fig1/illustration.png", dpi=1000)
# plt.savefig("Figures/Fig1/illustration.pdf", dpi=1000)
plt.show()
