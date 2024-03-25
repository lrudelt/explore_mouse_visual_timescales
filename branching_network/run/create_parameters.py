# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2023-02-09 17:12:09
# @Last Modified: 2023-03-09 10:55:14
# ------------------------------------------------------------------------------ #
# This creates a `parameter.tsv` where each line contains one parameter set,
# which can be directly called from the command line (i.e. on a cluster).
# ------------------------------------------------------------------------------ #

import os
import numpy as np
import hashlib
from uuid import uuid1
from datetime import datetime
from itertools import product
import sys


# set directory to the location of this script file to use relative paths
os.chdir(os.path.dirname(__file__))
script_path = os.path.abspath("../src/branching_network.py")
out_path = os.path.abspath(
    "/data.nst/lucas/history_dependence/signatures_of_temporal_processing_paper_code/data/bn_code_cleaned/"
)

# each repetition gets its own seed
num_reps = 20

# parameters to scan
l_a = np.linspace(5, 40.0, 20)  # translated to m
l_target_m = 1 - 1 / l_a
l_reps = np.arange(num_reps)
l_input_types = ["OU", "constant"]

# more parameters, same for all runs
meta_note = "2023-03-09, lowered k to 10 for both, N 1_000 and 10_000"
l_N = [1_000, 10_000]
k = 10
N_to_record = 100
dt = 5 / 1000
# this will be ignored and automatically overwritten for constant input
tau_ext = 30.0 / 1000


print("l_target_m  ", l_target_m)
print("num_reps    ", num_reps)

# we want to make sure we don't have any duplicates
hashes = []
# start filenames with date so they appear in a sorted order. (to check for obsoletes)
now = f"{datetime.now().strftime('%y%m%d_%H%M%S')}"

arg_list = list(product(l_input_types, l_N, l_target_m, l_reps))

with open("./parameters.tsv", "w") as par_file:
    par_file.write("# commands to run, one line per realization\n")

    for args in arg_list:
        input_type = args[0]
        N = args[1]
        m = args[2]
        rep = args[-1]

        line = (
            f"python3 {script_path} "
            + f"N={N} "
            + f"k={k} "
            + f"m={m:.5f} "
            + f"rep={rep} "
            + f"input_type={input_type} "
            + f"input_tau={tau_ext} "
            + f"dt={dt} "
            + f"N_to_record={N_to_record} "
            # bash requires ancient string escaping ...
            + f'meta_note="{meta_note}" '
            # keep the filepath last, we add to it, below!
            + f"path_for_neuron_spiketimes={out_path}/"
        )

        # filename
        # we explcitly do not _want_ meaningful filenames, everything is in the metadata!
        # add a checksum of our command line call, this should be pretty unique,
        # but reproducable on subsequent runs
        hash = hashlib.md5(line.encode("utf-8")).hexdigest()[:8]
        hashes.append(hash)
        line += f"{now}_{hash}.zarr"
        line += f"\n"

        par_file.write(line)

# longer hash might be needed
assert len(hashes) == len(
    np.unique(hashes)
), "hashes are not unique! are there parameter duplicates?"

print(f"number of jobs: {len(arg_list)}")
