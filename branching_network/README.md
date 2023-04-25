# Setup

so I had to tweak the hdestimator and mrestimator a bit to avoid a lot of printing when using dask.
they should both work if you activate the environment i created.

```
conda activate /data.nst/pspitzner/miniforge/envs/its_bn/
```

# Running the Branching network

all ingredients to run simulations are in the `run` folder.
- `cd mouse_visual_timescales/branching_network/` some scripts assume being run from this base directory.
- `run_bn.py` is a thin wrapper to enable running from the command line (serial queue).
- `python3 ./run/create_parameters.py` creates the tsv that has all the parameter combinations you want to sweep. one line per run.
    - check `create_parameters.py` to e.g. change to the 5ms timesteps, or storage paths and the likes.
- the script says how many jobs will be needed. to send e.g. 100 jobs into the queue `qsub -t 1-100 ./run/submit_to_cluster.sh`
- Note on the 10k Neuron network: requires more ~15GB ram, so we cant have 32 jobs on one node. works if we spread it. i have a script for that in `/home/pspitzner/bin/resubmit.sh`

# Analysis

Most is contained in `notebooks/bn_analysis.ipynb`.
- It scans the directory specified, extracts the coordinates and know what filenames each dask worker has to analyse.
- the `ana/hdestimator_wrapper.py` just dumps all the output to disk and then reads it back to provide a dict, and cleans up.
- analysing 100 simulations with 100 neurons took ~5 hours on the 256 cores. we cant get more cores (there is a file limit our admins wont/cant increase)

# Next steps
- rerun the analysis for the large systems, the ipynb _should_ work but i could not test it anymore yesterday. (minor bugs possible)
- this dataset has only the correlated inputs. for separating / filtering the file crawler, we will have to add another condition in the beginning.
- reminders for johannes of what we wanted to do:
    - change implementation of the BN to use actual transfer function
    - then add memory (in the current way we write it, out adaptation would affect external rates, too.)
        - reopening question for you two: maybe it should? if im refactory, also external inputs cant trigger me.


