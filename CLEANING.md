# information timescales

## Overall todos 23-04-26
- [ ] update mr estimator
- [ ] update hd estimator
- [ ] repo files

## cleaning plan 23-03-09

- one environment for preprocessing allen data to hdf5
- second environment for our stuff:
    - reading the prepared hdf5 from above
    - all analysis (including pymc) based on those
    - plotting
    - branching network
- All BN related work in its own folder, no dependencies outside this 'sandbox'
    - requires manual checking of MRE / HDE analysis pipeline. this should be better.
    - [ ] toolbox updates. @ps talk to daniel. make a branch that we install via pip. check idtxl.


- overall workflow for experiment_analysis:
    1. convert allen data from sdk to hdf5 (to avoid dependencies for future us). using allen_sdk environment.
    2. run analysis, writing results to `dat/stats`, using our environment.
        - this can have multiple stages if needed (e.g. bayesian model?)
    3. plotting only loads results from `dat/stats` with only minimal further prepping.

- folder structure: root
    - `branching_network`
    - `experiment_analysis`
        - `src` ? will see.
        - `plt` ? folder for plot helpers that are used over and over again? depending on the scale, this could just live directly in the notebooks.
        - `ana` "implement the analysis"
            - utilities, get data helper.
            - `mre_analysis.py`
            - one script for each block of analysis to run
            - analysis helpers that are called multiple times.
        - `run` "call the scripts from ana`
            - parameters
        - `dat`
            - `allen` |
            - `raw` | our hdf5 prepared from allen sdk, currently "share/allen_visual_coding/...."
                - ps 23-04-26: would suggest `spikes`, since we already prepared them to fit our format (and, thus, they are not "raw")
            - `prepped` | `stats` ~ analysis results before plotting them
        - `notebooks`
            - create_figure_xyz
            - entry point for people trying to find out where things are defined / done
        - `fig` plot outputs

### Notes / Todos as we go along:
- [ ] update the bn/env.yaml (move to root at the end)
- [x] added a trash folder at project root, where we can store unneded things for future reference before delting
- [ ] `/data.nst/pspitzner/information_timescales/cluster-jobs/allen-fc-merged/write_spike_times_hdf5.py` is the file we need to get running with the allen_sdk environment


## step-by-step guide to the analysis, example 1: hippocampus data

### prepare data

1. edit `dirs.yaml` and set `allen_data_dir` to the location of the raw data
2. edit `get_neuron_list.py` to include the areas you want to look at, here CA1 and DG
3. run `python3 get_neuron_list.py unmerged` to obtain `allen_neuron_list_unmerged.tsv`
4. run `python3 split_neuron_list.py allen_neuron_list_unmerged.tsv -v analysis_v1 -l 810` to obtain `allen_neuron_list_unmerged_analysis_v1_nm_sp.tsv` (excludes data shorter than 810 seconds, ie 15min with up to 10% deviation)

### run hdestimator

1. check vars and dirs in `analyse.sh` and the settings yaml file, then run `analyse.sh` to create analysis files
2. run `python3 merge_csv_stats_files.py  /path/to/analysis_files` (from hdestimator folder) to create `statistics_merged.csv`
3. copy `statistics_merged.csv` to analysis folder and rename eg to `statistics_allen_hippocampus.csv`

### analyse data

now the analysis can be performed on the data in `statistics_allen_hippocampus.csv`


## step-by-step guide to the analysis, example 2: brain observatory data

### prepare data

1. edit `dirs.yaml` and set `allen_data_dir` to the location of the raw data
2. edit `get_neuron_list.py` to include the areas you want to look at, here the visual areas ['VISp','VISl','VISrl','VISal','VISpm','VISam','LGd','LP']
3. run `python3 get_neuron_list.py merged` to obtain `allen_neuron_list_merged.tsv`
4. run `python3 split_neuron_list.py allen_neuron_list_merged.tsv -v analysis_v2 -l 1080` to obtain `allen_neuron_list_merged_analysis_v2.tsv` (excludes data shorter than 1080 seconds, ie 20min with up to 10% deviation)

### run hdestimator

1. check vars and dirs in `analyse.sh` and the settings yaml file, then run `analyse.sh` to create analysis files
2. run `python3 merge_csv_stats_files.py  /path/to/analysis_files` (from hdestimator folder) to create `statistics_merged.csv`
3. copy `statistics_merged.csv` to analysis folder and rename eg to `statistics_allen_brain_observatory.csv`

### analyse data

now the analysis can be performed on the data in `statistics_allen_brain_observatory.csv`
