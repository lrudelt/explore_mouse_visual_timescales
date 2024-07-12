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

We analyse data of mouse visual cortex from the [Allen Brain Atlas](https://atlas.brain-map.org/). The data is accessed using the [Allen SDK](http://alleninstitute.github.io/AllenSDK/install.html).
**For convenience, we provide a copy of the preprocessed data that is compatible with
our analysis pipeline on [gin.g-node.org](https://gin.g-node.org/pspitzner/mouse_visual_timescales).**
Loading these files requires only minimal dependencies and should be easy to setup using our `environment.yaml`. All use of these data must comply with the orignal sources' [Terms of Use](https://alleninstitute.org/terms-of-use/).

Instructions on how to download the original data can be found in the [docs](https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html).
In the folder [`experiment_analysis/download/`](/experiment_analysis/download/) we provide scripts that download and preprocess the spike data and related metrics. For these scripts to work you will have to change the path in `data_directory = os.path.abspath("/path/to/repo/experiment_analysis/dat/")` to your local path. 

- [`download_csv_files.py`](/experiment_analysis/download/download_csv_files.py) downloads csv files containing information regarding the experimental sessions, probes, channels and sorted unit, as well as other analysis metrics such as stimulus selectivity. This will create a file `brain_observatory_unit_metrics_filtered.csv`, which is required for further analysis regarding stimulus selectivity.
- [`download_session_data.py`](/experiment_analysis/download/download_session_data.py) downloads the full session data containing spike data for each experimental session of both the `functional_connectivity` and `brain_observatory_1_1` experiments
- if this does not work well, you can also execute [`download_session_data_via_http.py`](/experiment_analysis/download/download_session_data_via_http.py) to download the data directly using http (see [docs](https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html) for more info)
- [`write_spike_times_hdf5.py`](/experiment_analysis/download/write_spike_times_hdf5.py) to create h5 files from the raw data that contain the spiketimes for each session and are required for further analysis. Alternatively, these files can be downloaded from [gin.g-node.org](https://gin.g-node.org/pspitzner/mouse_visual_timescales) under the folder `experiment_analysis/dat/spikes/`.

## Analysis

### Experiments

Because the analysis takes some compute, we provide the [results in the data repository](https://gin.g-node.org/pspitzner/mouse_visual_timescales/src/6f278b915440a63988e0cbf658ed22150fec2538/experiment_analysis/dat/all_units_merged_blocks_with_spont.h5).
This data does not include the metric csvs, which are also needed. Use [`download_csv_files.py`](/experiment_analysis/download/download_csv_files.py).

All analysis run on our preprocessed spiking data. Requirements can be installed
by creating a new conda environment

```bash
conda env create -f environment.yaml --name mouse_visual_timescales
conda activate mouse_visual_timescales
```

There are two notebooks (with only minor differences) to analyse stimulated and spontaneous activity.

- [`analyse_all_units_blocks_merged.ipynb`](/experiment_analysis/notebooks/analyse_all_units_blocks_merged.ipynb)
- [`analyse_all_units_spontaneous_for_merged.ipynb`](/experiment_analysis/notebooks/analyse_all_units_spontaneous_for_merged.ipynb)

In short, they:

- set data directories (adjust this as needed!)
- load spikes and metadata
- setup the hde and mr estimators
- run the analysis using dask.
- if you want to explore this on your laptop, the `full_analysis()` function is of interest, and you can use local compute by altering the last big cell:

```python
with ExitStack() as stack:
    dask_cluster = stack.enter_context(
        LocalCluster(local_directory=f"{tempfile.gettempdir()}/dask/")
    )
    dask_cluster.scale(cores=num_cores)
    dask_client = stack.enter_context(Client(dask_cluster))

    final_df = main(dask_client, meta_df)
```

### Branching Network

We again provide the simulation results [in the data repository](https://gin.g-node.org/pspitzner/mouse_visual_timescales/src/6f278b915440a63988e0cbf658ed22150fec2538/branching_network/dat/res_dset_bn_code_cleaned_merged.zarr.zip)

The full analysis is contained in [`bn_analysis.ipynb`](/branching_network/notebooks/bn_analysis.ipynb).
It is also designed to loop on a dask cluster.

- The full analysis for one neuron takes one hour on one core.
- The notebook scans the directory specified, extracts the coordinates to pass what filenames each dask worker has to analyse.
- the [`hdestimator_wrapper`](/branching_network/ana/hdestimator_wrapper.py) is _planned_ to become part of hdesitmator itself. For now, it temporarily stores all the output to disk and then reads it back to provide a dict, and cleans up.


## Plotting

Here we summarize which plot comes from which notebook so you can trace back the ingredients that went in.

### Main paper 

- Figure 1 
  - [`example_units_natural_movie.ipynb`](/experiment_analysis/notebooks/example_units_natural_movie.ipynb)
- Figure 2
  - [`experiment_analysis/notebooks/plot_fig2.ipynb`](/experiment_analysis/notebooks/plot_fig2.ipynb)
- Figure 3
  + [`branching_network/notebooks/bn_plot_measures_vs_a.ipynb`](/branching_network/notebooks/bn_plot_measures_vs_a.ipynb)
  + [`branching_network/notebooks/bn_raster_examples.ipynb`](/branching_network/notebooks/bn_raster_examples.ipynb)
- Figure 4
  + [`experiment_analysis/notebooks/plot_fig4.ipynb`](/experiment_analysis/notebooks/plot_fig4.ipynb)

### Supplementary material

- Figure S1 
  + [`experiment_legacy/plotting/allen_fc_number_of_units.py]

- Figure S7 (Illustrates the autocorrelation fits)
  + [`experiment_analysis/notebooks/single_unit_autocorrelation.ipynb`](/experiment_analysis/notebooks/single_unit_autocorrelation.ipynb)

- Figure S11 (Similar to Fig 2 but for the natural movie three condition of the Brain Observatory 1.1 data set)
  + [`experiment_analysis/notebooks/plot_fig2.ipynb`](/experiment_analysis/notebooks/plot_fig2.ipynb)
  + [`experiment_analysis/notebooks/bayes_model_comparison.ipynb`](/experiment_analysis/notebooks/bayes_model_comparison.ipynb)

- Figure S12 (Similar to Fig 2 but for spontaneous activity in the Functional Connectivity data set)
  + [`experiment_analysis/notebooks/plot_fig2.ipynb`](/experiment_analysis/notebooks/plot_fig2.ipynb)
  + [`experiment_analysis/notebooks/bayes_model_comparison.ipynb`](/experiment_analysis/notebooks/bayes_model_comparison.ipynb)

- Figures S13-S16 (Posterior distributions for hierarchical parameters for both models)
  + [`experiment_analysis/notebooks/bayes_plotting.ipynb`](experiment_analysis/notebooks/bayes_plotting.ipynb)

- Figures S17 and S18 (Posterior distributions for non-hierarchical parameters for both models)
  + [`experiment_analysis/notebooks/bayes_plotting.ipynb`](experiment_analysis/notebooks/bayes_plotting.ipynb)

- Figures S19-S21 (Posterior predictive checks for both hierarchical models for all three stimulus conditions)
  + [`experiment_analysis/notebooks/bayes_plotting.ipynb`](experiment_analysis/notebooks/bayes_plotting.ipynb)

- Figure S22 (Area-wise comparison of timescales and predictability between stimulus conditions)
  + [`experiment_analysis/notebooks/plot_stimulus_comparison_areawise.ipynb`](experiment_analysis/notebooks/plot_stimulus_comparison_areawise.ipynb)

- Figure S23 and S25 (Areawise plot of timescales and predictability vs stimulus selectivity measures)
  + [`experiment_analysis/notebooks/plot_stimulus_selectivity_areawise.ipynb`](experiment_analysis/notebooks/plot_stimulus_selectivity_areawise.ipynb)

- Figure S24 (Plot of timescales and predictability vs image selectivity)
  + [`experiment_analysis/notebooks/plot_fig4.ipynb`](experiment_analysis/notebooks/plot_fig4.ipynb)
