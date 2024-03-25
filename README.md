# _Code repository:_ Signatures of hierarchical temporal processing in the mouse visual system


This repository contains the analysis pipeline and plotting routines for our preprint

[Arxiv]()

```
@misc{rudelt_signatures_2023,
  doi = {},
  url = {},
}
```

## Refactoring Note

We are currently cleaning this repository.

This requires converting all scripts to a new data backend, using our prepared spiking data in a minimal format (to avoid AllenSDK dependencies to just reproduce the paper).

The current state is as follows:

- [`branching_network/`](branching_network) BN simulation results are independent, self-contained and up to date.
- [`experiment_analysis/`](experiment_analysis) contains updated scripts.
- [`experiment_legacy/`](experiment_legacy) contains the old scripts, which are not compatible with the new data format, and instead the AllenSDK.
- we provide the new data format and the intermediate analysis results in the old format (for legacy plot scripts), in the data repository on [gin.g-node.org](https://gin.g-node.org/pspitzner/mouse_visual_timescales).


## Data

We analyse data of mouse visual cortex from the [Allen Brain Atlas](https://atlas.brain-map.org/).
The data is accessed using the [Allen SDK](http://alleninstitute.github.io/AllenSDK/install.html).
For convenience, we provide a copy of the preprocessed data that is compatible with
our analysis pipeline on [gin.g-node.org](https://gin.g-node.org/pspitzner/mouse_visual_timescales).
Loading these files requires only minimal dependencies and should be easy to setup using
our `environment.yaml`.
All use of these data must comply with the orignal sources' [Terms of Use](https://alleninstitute.org/terms-of-use/).

Instructions on how to download the data can be found in the [docs](https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html).
In the folder `experiment_analysis/data` we provide scripts that download the spike data and related metrics
- `download_csv_files.py` downloads csv files containing information regarding the experimental sessions, probes, channels and sorted unit, as well as other analysis metrics such as stimulus selectivity. This will create a file `brain_observatory_unit_metrics_filtered.csv`, which is required for further analysis regarding stimulus selectivity. 
- `download_session_data.py` downloads the full session data containing spike data for each experimental session of both the `functional_connectivity` and `brain_observatory_1_1` experiments
- if this does not work well, you can also execute `download_session_data_via_http.py` to download the data directly using http (see [docs](https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html) for more info)

TODO: Add info about write_spike_times.py to create the h5 files from the raw spike data. 

## Plotting

- [ ] Figure 1, legacy code
  - `allen_single_units.py`
  - `allen_single_units_stimulus.py`

- [ ] Figure 2, legacy code
  - require processed data using old pipeline, or the results from `stats.zip`
  - `allen_grouped.py`
  - `allen_hierarchy.py`
  - `allen_hierarchical_bayes_model_comparison.py` creates hdf5 files, to plot with
  - `allen_bayes.ipynb`


- [x] Figure 3, up to date
  - `branching_network/notebooks/bn_plot_measures_vs_a.ipynb`
  - `branching_network/notebooks/bn_raster_examples.ipynb`

- [ ] Figure 4, legacy code
  - `measures_vs_allen_metrics_scatter.py`

- [x] Figure S7, up to date
  - Illustrates the autocorrelation fits and the new dataformat.
  - `experiment_analysis/notebooks/single_unit_autocorrelation.ipynb`

----

Under construction, not yet up to date:

## Analysis

All analysis run on our preprocessed data and all requirements can be installed
by creating a new conda environment

```
conda env create -f environment.yaml --name mouse_visual_timescales
```
