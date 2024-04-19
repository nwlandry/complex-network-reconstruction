import cmasher as cmr
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import xgi

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

with open(f"Data/zkc_tmax_comparison.json") as file:
    data = json.load(file)
    tmax = np.array(data["tmax"], dtype=float)
    A = np.array(data["A"], dtype=float)
    node_error_simple = np.array(data["node-error-simple"], dtype=float)
    node_error_complex = np.array(data["node-error-complex"], dtype=float)

# with open("Data/zkc_infer_vs_tmax.json") as file:
#     data = json.load(file)
# tmax = np.array(data["tmax"], dtype=float)
# performance = np.array(data["auprc"], dtype=float)
# slice_idx = np.argmax(performance[1].mean(axis=1) - performance[0].mean(axis=1))
slice_idx = 11

xmin = tmax.min() - 1.2
xmax = tmax.max()
ymin = -0.2
ymax = 0.2

n, n_t, n_r = node_error_simple.shape

G = nx.Graph(A.astype(int))
kc = nx.core_number(G)
coreness = np.zeros(n)
coreness[list(kc)] = list(kc.values())

H = xgi.Hypergraph([[i, j] for i, j in G.edges])

# plotting settings
colormap = cmr.redshift
clist = [colormap(0.15), colormap(0.3), colormap(0.7), colormap(0.85)]

## Coreness difference

ms = 4
alpha = 1

x = tmax

plt.figure(figsize=(5.5, 8))
plt.subplots_adjust(left=0, bottom=0, right=1, top=1)

plt.subplot(211)
ax = plt.gca()
ax.set_position([0.18, 0.5, 0.75, 0.45])

ax.text(
    -0.2,
    1.0,
    "(a)",
    transform=ax.transAxes,
    fontsize=13,
    fontweight="bold",
    va="top",
)

core_values = np.unique(coreness)

for idx, k in enumerate(core_values):
    n_k = sum(coreness == k)
    y = (node_error_simple[coreness == k] - node_error_complex[coreness == k]).T
    ymean = y.mean(axis=2).mean(axis=0)
    ystd = y.mean(axis=2).std(axis=0)
    plt.semilogx(
        x,
        ymean,
        "o-",
        markersize=ms,
        color=clist[k - 1],
        alpha=alpha,
        label=f"{int(k)}-core nodes",
    )
    plt.fill_between(x, ymean - ystd, ymean + ystd, color=clist[k - 1], alpha=0.1)

plt.semilogx([xmin, xmax], [0, 0], "k--")

plt.plot([tmax[slice_idx], tmax[slice_idx]], [ymin, ymax], color="grey", alpha=0.3)

plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
plt.xticks(
    [10**1, 10**2, 10**3, 10**4],
    [
        r"$\mathregular{10^1}$",
        r"$\mathregular{10^2}$",
        r"$\mathregular{10^3}$",
        r"$\mathregular{10^4}$",
    ],
)
plt.yticks([-0.2, -0.1, 0, 0.1])

plt.legend(frameon=False)
plt.ylabel(r"$\varepsilon_{SC} - \varepsilon_{CC}$")
plt.xlabel(r"$t_{max}$")
sns.despine()


plt.subplot(212)
ax = plt.gca()
ax.text(
    0.025,
    0.9,
    "(b)",
    transform=ax.transAxes,
    fontsize=13,
    fontweight="bold",
    va="top",
)

pos = xgi.pca_transform(xgi.pairwise_spring_layout(H, seed=5, k=0.3))

node_prop = (node_error_simple[slice_idx] - node_error_complex[slice_idx]).mean(axis=0)
better_color = clist[0]
worse_color = clist[2]
ycolor = [colormap(d) for d in node_prop]

ax = plt.gca()
ax, collections = xgi.draw(
    H,
    pos=pos,
    node_size=10,
    dyad_lw=1,
    node_fc=node_prop,
    node_fc_cmap=colormap,
)
node_collection, _, _ = collections
plt.colorbar(node_collection)

sns.despine()
# plt.savefig("Figures/Fig3/fig3.png", dpi=1000)
# plt.savefig("Figures/Fig3/fig3.pdf", dpi=1000)
plt.show()
