# Complex contagions can outperform simple contagions for network reconstruction with dense networks or saturated dynamics

This repository accompanies the preprint "Complex contagions can outperform simple contagions for network reconstruction with dense networks or saturated dynamics" by Nicholas Landry, Will Thompson, Laurent Hébert-Dufresne, and Jean-Gabriel Young.

### The structure of this repository is as follows:
* The `Data` folder contains all of the data corresponding to the figures in the manuscript.
* The `Figures` folder contains PDF and PNG files for each of the figures in the paper.
* The `tests` folder contains unit tests to validate the code written for generating our results.
* The `lcs` (Learning Complex Structure) folder contains all of the code necessary for the generation of time series, the generation of random networks, the inference of networks, and the measurement of reconstruction performance.
* The `Extra` folder contains scripts and notebooks which are not used in the manuscript.
* The `convergence` folder contains notebooks used for heuristically determining what the values for burn-in and sampling gap should be for our MCMC sampler.

### General things:
* To run the unit tests, you need to pip install the package locally. Navigate to the local folder on your computer and run "pip install -e ."
* To run the unit tests, run "pytest" in the command line.
* The package is referenced as `lcs` (Learning complex structure) when accessing the functionality.

### Scripts
* `plot_fig#.py` generates all of the figures displayed in the manuscript.
* The following scripts generate ensembles of random networks, contagion time-series data, and then attempts to infer these networks from the time-series data:
  * `clustered_network.py`
  * `cm.py`
  * `erdos_renyi.py`
  * `sbm.py`
  * `watts-strogatz.py`
* The following scripts collect the above data and measure the performance of the reconstructions:
  * `collect_clustered_network.py`
  * `collect_cm.py`
  * `collect_erdos_renyi.py`
  * `collect_sbm.py`
  * `collect_watts-strogatz.py`
* `zkc_*.py` generates the data used in Figs. 1 and 3.
* `collect_tmax_comparison.py` collect the data generated vs. tmax and measures the nodal performance displayed in Fig. 3.
* `collect_zkc_infer_vs_tmax.py` and `collect_zkc_frac_vs_beta` collect this data and measure the performance of the reconstructions for Figs. 1(c) and 1(d) respectively.

### Notebooks
* `run_dynamical_inference.ipynb` runs a single inference for a single network and time series.