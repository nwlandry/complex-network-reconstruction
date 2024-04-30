import arviz as az
import cmasher as cmr
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import xgi

import fig_settings as fs
from lcs import *


fs.set_fonts()

with open(f"Data/zkc_tmax_comparison.json") as file:
    data = json.load(file)
    tmax = np.array(data["tmax"], dtype=float)
    A = np.array(data["A"], dtype=float)
    node_performance_simple = np.array(data["node-performance-simple"], dtype=float)
    node_performance_complex = np.array(data["node-performance-complex"], dtype=float)

xmin = tmax.min()
xmax = tmax.max()
ymin = -0.2
ymax = 0.2

n, n_t, n_r = node_performance_simple.shape

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

plt.figure(figsize=(5.5, 4))

core_values = np.unique(coreness)

for idx, k in enumerate(core_values):
    n_k = sum(coreness == k)
    y = node_performance_complex[coreness == k] - node_performance_simple[coreness == k]
    # combine first and last axes.
    y = np.array([y[i, :, j] for i in range(n_k) for j in range(n_r)]).T
    ymean = np.median(y, axis=1)

    hdpi_a = np.zeros_like(tmax)
    hdpi_b = np.zeros_like(tmax)
    for i in range(len(tmax)):
        interval = az.hdi(y[i], hdi_prob=0.5)
        a, b = interval
        hdpi_a[i] = a
        hdpi_b[i] = b
    plt.semilogx(
        x,
        ymean,
        "o-",
        markersize=ms,
        color=clist[idx],
        alpha=alpha,
        label=f"{int(k)}-core nodes",
    )
    plt.fill_between(x, hdpi_a, hdpi_b, color=clist[idx], alpha=0.1)

plt.semilogx([xmin, xmax], [0, 0], "k--")

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
plt.yticks([-0.2, -0.1, 0, 0.1, 0.2])

plt.legend(frameon=False, loc="upper left")
plt.ylabel(r"$\Delta\,\phi_i$")
plt.xlabel(r"$T$")
sns.despine()
plt.tight_layout()

plt.savefig("Figures/Fig3/fig3.png", dpi=1000)
plt.savefig("Figures/Fig3/fig3.pdf", dpi=1000)
# plt.show()
