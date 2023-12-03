import json
import os

import numpy as np

from lcs import *

plist = set()
clist = set()
rlist = set()
beta = []
frac = []


data_dir = "Data/clustering"

for f in os.listdir(data_dir):
    d = f.split(".json")[0].split("_")
    try:
        p = float(d[0])
        c = int(d[1])
        r = int(d[2])
    except:
        p = float(d[0] + "-" + d[1])
        c = int(d[2])
        r = int(d[3])

    plist.add(p)
    clist.add(c)
    rlist.add(r)

clist = sorted(clist)
plist = sorted(plist)
rlist = sorted(rlist)

c_dict = {c: i for i, c in enumerate(clist)}
p_dict = {p: i for i, p in enumerate(plist)}
r_dict = {r: i for i, r in enumerate(rlist)}


ps = np.zeros((len(clist), len(plist), len(rlist)))
sps = np.zeros((len(clist), len(plist), len(rlist)))

for f in os.listdir(data_dir):
    d = f.split(".json")[0].split("_")
    try:
        p = float(d[0])
        c = int(d[1])
        r = int(d[2])
    except:
        p = float(d[0] + "-" + d[1])
        c = int(d[2])
        r = int(d[3])

    i = c_dict[c]
    j = p_dict[p]
    k = r_dict[r]

    fname = os.path.join(data_dir, f)

    with open(fname, "r") as file:
        data = json.loads(file.read())

    A = np.array(data["A"])
    samples = np.array(data["samples"])

    ps[i, j, k] = posterior_similarity(samples, A)
    sps[i, j, k] = samplewise_posterior_similarity(samples, A)

data = {}
data["clique_number"] = plist
data["sps"] = sps.tolist()
data["ps"] = ps.tolist()
datastring = json.dumps(data)

with open("Data/clustering.json", "w") as output_file:
    output_file.write(datastring)
