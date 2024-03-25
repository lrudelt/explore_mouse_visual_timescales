# Setup

so I had to tweak the hdestimator and mrestimator a bit to avoid a lot of printing when using dask.
they should both work if you activate the environment i created.

```
conda activate /data.nst/pspitzner/miniforge/envs/its_bn/
```

# Running the Branching network

all ingredients to run simulations are in the [`run/`](run) folder.
The model is implemented in [`src/`](src).
- `cd mouse_visual_timescales/branching_network/`, some scripts assume being run from this base directory.
- [`run_bn.py`](run/run_bn.py) is a thin wrapper to enable running from the command line (serial queue).
- `python3 ./run/create_parameters.py` creates the tsv that has all the parameter combinations you want to sweep. one line per run.
    - check [`create_parameters.py`](run/create_parameters.py) to e.g. change to the 5ms timesteps, or storage paths and the likes.
- the script says how many jobs will be needed. to send e.g. 100 jobs into a slurm queue `qsub -t 1-100 ./run/submit_to_cluster.sh`

# Analysis

The full analysis is contained in [`notebooks/bn_analysis.ipynb`](notebooks/bn_analysis.ipynb).
It is designed to loop on a dask cluster.

- The full analysis for one neuron takes one hour on on core.
- The notebook scans the directory specified, extracts the coordinates to pass what filenames each dask worker has to analyse.
- the [`ana/hdestimator_wrapper.py`](ana/hdestimator_wrapper.py) is planned to become part of hdesitmator itself. For now, it temporarily stores all the output to disk and then reads it back to provide a dict, and cleans up.

