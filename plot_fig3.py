import cmasher as cmr
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import xgi

from lcs import *

with open(f"Data/zkc_tmax_comparison.json") as file:
    data = json.load(file)
    tmax = np.array(data["tmax"], dtype=float)
    A = np.array(data["A"], dtype=float)[0, 0, 0]
    Q = np.array(data["Q"], dtype=float)

xmin = tmax.min() - 1.2
xmax = tmax.max() + 15000
ymin = -0.2
ymax = 0.2

slice_idx = 11

n_c, n_t, n_r, n, _ = Q.shape

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

y1 = np.zeros([n_r, n_t, n])
y2 = np.zeros([n_r, n_t, n])
for i in range(n_r):
    y1[i] = [nodal_performance(Q[0, j, i], A) for j in range(n_t)]
    y2[i] = [nodal_performance(Q[1, j, i], A) for j in range(n_t)]


plt.figure(figsize=(4, 6))
plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
plt.subplot(211)
ax = plt.gca()
ax.set_position([0.18, 0.5, 0.75, 0.45])

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
plt.semilogx([xmin, xmax], [0, 0], "k--")

plt.plot([tmax[slice_idx], tmax[slice_idx]], [ymin, ymax], color="grey", alpha=0.3)

plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
plt.xticks(
    [10**1, 10**2, 10**3, 10**4, 10**5],
    [
        r"$\mathregular{10^1}$",
        r"$\mathregular{10^2}$",
        r"$\mathregular{10^3}$",
        r"$\mathregular{10^4}$",
        r"$\mathregular{10^5}$",
    ],
)
plt.yticks([-0.2, -0.1, 0, 0.1])

plt.legend(frameon=False)
plt.ylabel(r"$\varepsilon_{SC} - \varepsilon_{CC}$")
plt.xlabel(r"$t_{max}$")
sns.despine()


plt.subplot(212)

pos = xgi.pca_transform(xgi.pairwise_spring_layout(H, seed=5, k=0.3))

node_prop = (y1[slice_idx] - y2[slice_idx]).mean(axis=0)
better_color = clist[0]
worse_color = clist[2]
ycolor = [better_color if d > 0 else worse_color for d in node_prop]

ax = plt.gca()
ax, collections = xgi.draw(
    H,
    pos=pos,
    node_size=6.5,
    dyad_lw=1,
    node_fc=ycolor,
)

sns.despine()
plt.savefig("Figures/Fig3/fig3.png", dpi=1000)
plt.savefig("Figures/Fig3/fig3.pdf", dpi=1000)
plt.show()
