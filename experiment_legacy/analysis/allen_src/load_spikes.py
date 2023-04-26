from sys import stderr, argv, exit
from os.path import realpath, dirname, exists
import os
import numpy as np
import pandas as pd
import h5py

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

import yaml

### settings

# stimuli = {'brain_observatory_1.1' : ["natural_movie_three"],
#            'functional_connectivity' : ["natural_movie_one_more_repeats",
#                                         "spontaneous"]}

# stimulus order brain_observatory_1.1                  block   approx. length[min]
# * drifting_gratings                                   1       10
# * natural_movie_three (10 min = 5* 120s)              1       10
# * natural_movie_one (5 min = 10* 30s)                 1       5
# * drifting_gratings                                   2       10
# * other
# * natural_movie_three (10 min = 5* 120s)              2       10
# * drifting_gratings                                   3       10
# * other
# * natural_movie_one (5 min = 10* 30s)                 2       5

# stimulus order functional_connectivity
# * drifting_gratings_contrast                          1       15
# * other
# * natural_movie_one_more_repeats (15 min = 30* 30s)   1       15
# * natural_movie_one_shuffled (5 min = 10* 30s)        1       5
# * drifting_gratings_75_repeats                        1       15
# * other
# * drifting_gratings_75_repeats                        2       15
# * natural_movie_one_shuffled (5 min = 10* 30s)        2       5
# * natural_movie_one_more_repeats (15 min = 30* 30s)   2       15

return_values = {
    "SUCCESS": 0,
    "ERR_EMPTY": 1,
    "ERR_REC_LEN": 2,
    "ERR_INVALID_TIMES": 3,
    "ERR_NON_STATIONARY": 4,
    "ERR_LO_FIR_RATE": 5,
}

# cache: which neuron has which index in the spike file

# lowest level, this requires full knowledge of the unit, (single block)
def load_spike_data_hdf5(filepath, session_id, stimulus, stimulus_block, unit_index):
    filename = f"{filepath}/session_{session_id}/session_{session_id}_spike_data.h5"
    f = h5py.File(os.path.expanduser(filename), "r", libver="latest")
    spike_data = f[
        f"session_{session_id}_stimulus_{stimulus}_stimulus_block_{stimulus_block}_spiketimes"
    ]
    spike_times_unit = spike_data[unit_index]
    spike_times_unit = spike_times_unit[np.isfinite(spike_times_unit)].astype(float)
    f.close()
    metadata = pd.read_hdf(
        filename,
        f"session_{session_id}_stimulus_{stimulus}_stimulus_block_{stimulus_block}_metadata",
    )
    metadata_unit = metadata.loc[unit_index]
    return spike_times_unit, metadata_unit

# higher, for a certain unitid, get all spikes for the given blocks (blocks are a list)
# this already merges blocks (check carefully)
def get_spike_times(
    session_id, unit, stimulus, stimulus_blocks, target_length, transient, data_directory
):
    num_blocks = len(stimulus_blocks)
    spike_times_merged = []
    filename = f"{data_directory}/session_{session_id}/session_{session_id}_spike_data.h5"
    for stimulus_block in stimulus_blocks:
        metadata_block = pd.read_hdf(
            filename,
            f"session_{session_id}_stimulus_{stimulus}_stimulus_block_{stimulus_block}_metadata",
        )
        unit_index = np.where(metadata_block["unit_id"].values == unit)[0]
        spike_times, metadata = load_spike_data_hdf5(
            data_directory, session_id, stimulus, stimulus_block, unit_index
        )
        if len(spike_times) > 0:
            transient_threshold_time = spike_times[0] + transient
            length_threshold_time = (
                spike_times[0] + transient + target_length / num_blocks
            )
            spike_times_merged += [
                [
                    spt
                    for spt in spike_times
                    if spt >= transient_threshold_time and spt <= length_threshold_time
                ]
            ]
        else:
            spike_times_merged += []
    return spike_times_merged


# Assert appropriateness for each stimulus block independently
# check whether a give unitid is allowed (according to our custom rules)
# for a given stimulus block.

# - decent rate
# - stopping spiking 10% before ending of recording
# - invalid spike times (they might violate refactory times, indicating artifacts)
# - stationarity check: how different are the firing rates between blocks?

def assert_appropriateness(
    unit, session_id, stimulus, stimulus_blocks, target_length, transient, data_directory
):
    num_blocks = len(stimulus_blocks)
    target_length = target_length / num_blocks + transient
    filename = f"{data_directory}/session_{session_id}/session_{session_id}_spike_data.h5"
    firing_rates = []
    for stimulus_block in stimulus_blocks:
        metadata_block = pd.read_hdf(
            filename,
            f"session_{session_id}_stimulus_{stimulus}_stimulus_block_{stimulus_block}_metadata",
        )
        unit_index = np.where(metadata_block["unit_id"].values == unit)[0]
        spike_times, metadata = load_spike_data_hdf5(
            data_directory, session_id, stimulus, stimulus_block, unit_index
        )
        ecephys_structure_acronym = metadata["ecephys_structure_acronym"].values[0]
        unit = metadata["unit_id"].values[0]
        firing_rate = metadata["firing_rate"].values[0]
        rec_len = metadata["recording_length"].values[0]
        firing_rates += [firing_rate]
        # first check recording length
        if rec_len == 0:
            print(
                "empty recording for {}, {}, {}, {}".format(
                    session_id, unit, stimulus, stimulus_block
                )
            )
            return "ERR_EMPTY"
        elif rec_len < 0.9 * target_length:
            print(
                "short recording ({} s) for {}, {}, {}, {}".format(
                    rec_len, session_id, unit, stimulus, stimulus_block
                )
            )
            return "ERR_REC_LEN"
        if firing_rate < 0.01:
            print(
                "low firing rate ({} Hz) for {}, {}, {}, {}".format(
                    firing_rate, session_id, unit, stimulus, stimulus_block
                )
            )
            return "ERR_LO_FIR_RATE"

        # check for invalid times in analysed spike times
        if metadata["invalid_spiketimes_check"].values[0] != "SUCCESS":
            print(
                f"invalid spike times in {session_id}, {unit},"
                f" {ecephys_structure_acronym}, {stimulus}, {stimulus_block}"
            )
            return "ERR_INVALID_TIMES"

    # check for evidence of non-stationarity
    if num_blocks > 1:
        rel_firing_rate_diff = np.abs(firing_rates[0] - firing_rates[1]) / np.mean(
            firing_rates
        )
        if rel_firing_rate_diff > 0.5:
            print(
                f"non-stationarity ({firing_rates[0]} Hz, {firing_rates[1]} Hz) in"
                f" {session_id}, {unit}, {ecephys_structure_acronym}, {stimulus},"
                f" {stimulus_block}"
            )
            return "ERR_NON_STATIONARY"

    return "SUCCESS"

# potentially useful for debugging or using hdestimator via cli
def print_spike_times(
    session_id,
    unit_index,
    stimulus,
    stimulus_blocks,
    target_length,
    transient,
    data_directory,
):
    spike_times_merged = get_spike_times(
        session_id,
        unit_index,
        stimulus,
        stimulus_blocks,
        target_length,
        transient,
        data_directory,
    )
    first_print = True
    for spike_times in spike_times_merged:
        if len(spike_times) > 0:
            if first_print:
                first_print = False
            else:
                print("----------")

            for spt in spike_times:
                print(spt)
    exit()

# @ps check if they do the same as my
# lucas said there was a bug. important: both blocks need the same number of bins)
def get_binned_spike_times(spike_times, bin_size):

    # spike_times is list of lists (num_blocks, num_spikes ragged)
    num_blocks = len(spike_times)
    max_block_duration = np.amax(
        [spike_times[block][-1] - spike_times[block][0] for block in range(num_blocks)]
    )
    # max_spt = np.max(np.hstack(spike_times))
    binned_spike_times = np.zeros([num_blocks, int(max_block_duration / bin_size) + 1])
    for block in range(num_blocks):
        # binned_spt = np.zeros(int(spt[-1] / bin_size) + 1, dtype=int)
        spike_times[block] = (
            spike_times[block] - spike_times[block][0]
        )  # align spikes to beginning of recording (first recorded spike)
        for spike_time in spike_times[block]:
            binned_spike_times[block, int(spike_time / bin_size)] += 1
            # binned_spt = np.array([binned_spt])
    return binned_spike_times


# filename = f"{data_directory}/session_{session_id}/session_{session_id}_spike_data.h5"
# spike_times_merged = []
# for stimulus_block in stimulus_blocks:
#     metadata = pd.read_hdf(filename, f"session_{session_id}_stimulus_{stimulus}_stimulus_block_{stimulus_block}_metadata")
#     unit_index = np.where(metadata["unit_id"].values == unit)[0]
#     spike_times, metadata = utl.load_spike_data_hdf5(data_directory, session_id, stimulus, stimulus_block, unit_index)
#     spike_times_merged += [spike_times]
