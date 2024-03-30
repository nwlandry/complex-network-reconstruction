import cmasher as cmr
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

from lcs import *

with open(f"Data/zkc_tmax_comparison.json") as file:
    data = json.load(file)
    tmax = np.array(data["tmax"], dtype=float)
    A = np.array(data["A"], dtype=float)[0, 0, 0]
    Q = np.array(data["Q"], dtype=float)

n_c, n_t, n_r, n, _ = Q.shape

G = nx.Graph(A.astype(int))

cc = clustering_coefficient(A)
deg = degrees(A)

kc = nx.core_number(G)
coreness = np.zeros(n)
coreness[list(kc)] = list(kc.values())

# plotting settings

colormap = cmr.redshift
clist = [colormap(0.15), colormap(0.3), colormap(0.7), colormap(0.85)]

## Coreness difference

ms = 4
alpha = 1

x = tmax

y1 = np.zeros([n_r, n_t, n])
y2 = np.zeros([n_r, n_t, n])
for i in range(n_r):
    y1[i] = [nodal_performance(Q[0, j, i], A) for j in range(n_t)]
    y2[i] = [nodal_performance(Q[1, j, i], A) for j in range(n_t)]

# plt.figure(figsize=(4, 3))
plt.figure(figsize=(4, 6))
plt.subplot(211)

core_values = np.unique(coreness)

for idx, k in enumerate(core_values):
    n_k = sum(coreness == k)
    y = np.zeros([n_r, n_t, n_k])
    for i in range(n_r):
        y[i] = (y1[i, :, coreness == k] - y2[i, :, coreness == k]).T
    ymean = y.mean(axis=2).mean(axis=0)
    ystd = y.mean(axis=2).std(axis=0)
    plt.semilogx(
        x,
        ymean,
        "o-",
        markersize=ms,
        color=clist[idx],
        alpha=alpha,
        label=f"{int(k)}-core nodes",
    )
plt.semilogx(tmax, np.zeros_like(tmax), "k--")

plt.xlim([tmax.min() - 0.5, tmax.max() + 1000])
plt.yticks([-0.2, -0.1, 0, 0.1])
plt.legend()
plt.ylabel(r"$\varepsilon_{SC} - \varepsilon_{CC}$")
plt.xlabel(r"$t_{max}$")
sns.despine()
# plt.tight_layout()
# plt.savefig("Figures/Fig3/figure3a.png", dpi=1000)
# plt.savefig("Figures/Fig3/figure3a.pdf", dpi=1000)

plt.subplot(212)
# Degree difference
ms = 4
alpha = 1

x = tmax
y1 = np.zeros([n_r, n_t, n])
y2 = np.zeros([n_r, n_t, n])
for i in range(n_r):
    y1[i] = [nodal_performance(Q[0, j, i], A) for j in range(n_t)]
    y2[i] = [nodal_performance(Q[1, j, i], A) for j in range(n_t)]

# plt.figure(figsize=(4, 3))

deg_bounds = [[1, 5], [6, 10], [11, 15], [16, 20]]
for idx, d in enumerate(deg_bounds):
    n_d = sum((d[0] <= deg) & (deg <= d[1]))
    y = np.zeros([n_r, n_t, n_d])
    for i in range(n_r):
        y[i] = (
            y1[i, :, (d[0] <= deg) & (deg <= d[1])]
            - y2[i, :, (d[0] <= deg) & (deg <= d[1])]
        ).T
    ymean = y.mean(axis=2).mean(axis=0)
    ystd = y.mean(axis=2).std(axis=0)
    plt.semilogx(
        x,
        ymean,
        "o-",
        markersize=ms,
        color=clist[idx],
        alpha=alpha,
        label=rf"{int(d[0])}$\leq k\leq${int(d[1])}",
    )
plt.semilogx(tmax, np.zeros_like(tmax), "k--")

plt.xlim([tmax.min() - 0.5, tmax.max() + 1000])
plt.yticks([-0.1, 0, 0.1])
plt.ylabel(r"$\varepsilon_{SC} - \varepsilon_{CC}$")
plt.xlabel(r"$t_{max}$")
plt.legend()
plt.tight_layout()
sns.despine()
plt.savefig("Figures/Fig3/figure3b.png", dpi=1000)
plt.savefig("Figures/Fig3/figure3b.pdf", dpi=1000)
