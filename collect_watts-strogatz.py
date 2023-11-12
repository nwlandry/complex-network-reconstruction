import json
import os

import numpy as np

from lcs import *

plist = set()
clist = set()
rlist = set()
beta = []
frac = []


data_dir = "Data/watts-strogatz/"

for f in os.listdir(data_dir):
    d = f.split(".json")[0].split("-")
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

c_dict = dict(zip(clist, range(len(clist))))
p_dict = dict(zip(plist, range(len(plist))))
r_dict = dict(zip(rlist, range(len(rlist))))


ps = np.zeros((len(clist), len(plist), len(rlist)))
sps = np.zeros((len(clist), len(plist), len(rlist)))

for f in os.listdir(data_dir):
    d = f.split(".json")[0].split("-")
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

    ps[i, j, k] = posterior_similarity(A, samples)
    sps[i, j, k] = samplewise_posterior_similarity(A, samples)

data = {}
data["p"] = plist
data["sps"] = sps.tolist()
data["ps"] = ps.tolist()
datastring = json.dumps(data)

with open("Data/watts-strogatz.json", "w") as output_file:
    output_file.write(datastring)
