from sys import stderr, argv, exit
from os.path import realpath, dirname, exists
import os
import shutil

import numpy as np
import pandas as pd
import h5py
import logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s",
    datefmt="%y-%m-%d %H:%M",
)
log = logging.getLogger(__name__)
log.setLevel("DEBUG")

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

import yaml

SCRIPT_DIR = dirname(realpath(__file__))
# SCRIPT_DIR = "./"
with open('{}/../../dirs.yaml'.format(SCRIPT_DIR), 'r') as dir_settings_file:
    dir_settings = yaml.load(dir_settings_file, Loader=yaml.BaseLoader)

## settings
data_directory = dir_settings['allen_data_dir']
manifest_path = os.path.join(data_directory, "manifest.json")
session_type = 'functional_connectivity'
areas = ['VISp','VISl','VISrl','VISal','VISpm','VISam','LGd','LP']
# 'brain_observatory_1.1'
# Natural Movie One was a 30 second clips repeated 20 times (2 blocks of 5),
# while Natural Movie Three was a 120 second clip repeated 10 times (2 blocks of 10)

# 'functional_connectivity'
# The Natural Movie One stimulus was presented a total of 60 times, with
# an additional 20 repeats of a temporally shuffled version.

stimuli = {'brain_observatory_1.1' : ["natural_movie_three"],
           'functional_connectivity' : ["natural_movie_one_more_repeats",
                                        "spontaneous"]}

# np.unique(presentations['stimulus_name'])
# 'brain_observatory_1.1'
# array(['drifting_gratings', 'flashes', 'gabors', 'natural_movie_one',
#        'natural_movie_three', 'natural_scenes', 'spontaneous',
#        'static_gratings'], dtype=object)
# 'functional_connectivity'
# array(['dot_motion', 'drifting_gratings_75_repeats',
#        'drifting_gratings_contrast', 'flashes', 'gabors',
#        'natural_movie_one_more_repeats', 'natural_movie_one_shuffled',
#        'spontaneous'], dtype=object)

quality_metrics = {'unmerged': {'presence_ratio_minimum': 0.9, # was -np.inf,  # default: 0.9
                                'amplitude_cutoff_maximum': 0.01,   # default: 0.1,
                                'isi_violations_maximum': 0.5},     # default: 0.5
                   'merged': {'presence_ratio_minimum': 0.9,
                              'amplitude_cutoff_maximum': 0.01,
                              'isi_violations_maximum': 0.5}}
list_version = 'unmerged'
# stimulus block format
def sbfmt(sb):
    try:
        formatted_sb = float(sb)
        return formatted_sb
    except:
        return sb


def get_stimulus_blocks_indices(stimulus_epochs, stimulus_blocks):
    stimulus_blocks_idx = []
    for stimulus_block in stimulus_blocks:
        # in some cases a single stimulus block is split into two
        # parts (due to invalid presentations in between)
        # therefore use the index to have a unique identifier
        # let the index be that of the longer part
        stimulus_blocks_idx += [
            stimulus_epochs[stimulus_epochs["stimulus_block"] == stimulus_block][
                "duration"
            ].idxmax()
        ]
    return stimulus_blocks_idx


def get_stimulus_presentation_ids(
    presentations, stimulus_epochs, session_type, stimulus, stimulus_blocks
):
    stimulus_presentation_ids = {}
    for stimulus_block in stimulus_blocks.split(","):
        stimulus_presentation_ids[stimulus_block] = []

    if presentations.empty:
        return stimulus_presentation_ids

    for stimulus_block, stimulus_block_idx in zip(
        stimulus_blocks.split(","),
        get_stimulus_blocks_indices(
            stimulus_epochs,
            [sbfmt(stimulus_block) for stimulus_block in stimulus_blocks.split(",")],
        ),
    ):
        block_start_time = stimulus_epochs.iloc[stimulus_block_idx]["start_time"]
        block_stop_time = stimulus_epochs.iloc[stimulus_block_idx]["stop_time"]

        stimulus_presentation_ids[stimulus_block] = presentations[
            (presentations["start_time"] >= block_start_time)
            & (presentations["stop_time"] <= block_stop_time)
        ].index.values

    return stimulus_presentation_ids

def check_invalid_spiketimes(spike_times,
                           session,
                           session_id,
                           unit,
                           session_type,
                           stimulus,
                           stimulus_blocks,
                           target_length,
                           transient):

    ret_msg = 'SUCCESS'
    # first check recording length
    rec_len = sum([spt[-1] - spt[0] for spt in spike_times.values() if len(spt) > 1])

    # check for invalid times in analysed spike times
    units = session.units
    probe_id = str(units.loc[unit]['probe_id'])

    invalid_times = session.invalid_times.copy()

    if not invalid_times.empty:
        probe_ids = np.array([str(a[1]) for a in invalid_times['tags'].values])
        invalid_times.insert(3, 'probe_id', probe_ids)


        for spt in spike_times.values():
            if len(spt) > 0:
                if len(invalid_times[(invalid_times['probe_id'] == probe_id) &
                                     (invalid_times['start_time'] < spt[-1]) &
                                     (invalid_times['stop_time'] > spt[0])]) > 0:
                    print("invalid spike times in {}, {}, {}, {}".format(session_id,
                                                                         unit,
                                                                         stimulus,
                                                                         stimulus_blocks))
                    ret_msg = 'ERR_INVALID_TIMES'


    # check for evidence of non-stationarity
    # if len(stimulus_blocks.split(',')) > 1:
    #     firing_rates = [len(spt) / (spt[-1] - spt[0]) for spt in spike_times.values()]

    #     firing_rate_diff = np.abs(firing_rates[0] - firing_rates[1]) / min(firing_rates)

    #     if firing_rate_diff > 0.2:
    #         return return_values['ERR_NON_STATIONARY']

    return ret_msg


def get_spike_times(session,
                    unit,
                    stimulus_presentation_ids,
                    target_length,
                    transient):

    num_nonzero_blocks = sum([1 for stimulus_block in stimulus_presentation_ids
                              if len(stimulus_presentation_ids[stimulus_block]) > 0])

    spike_times = {}
    for stimulus_block in stimulus_presentation_ids:
        spike_times[stimulus_block] = []

        if not len(stimulus_presentation_ids[stimulus_block]) > 0:
            continue

        spikes = session.presentationwise_spike_times(
            stimulus_presentation_ids=stimulus_presentation_ids[stimulus_block],
            unit_ids = unit
        )

        if not len(spikes) > 0:
            print('??? for {}, {}'.format(session, unit))
            continue

        transient_threshold_time = spikes.index.values[0] + transient
        length_threshold_time = spikes.index.values[0] + transient + target_length/num_nonzero_blocks

        spike_times[stimulus_block] = [spt for spt in spikes.index.values
                                       if spt >= transient_threshold_time
                                       and spt <= length_threshold_time]

    return spike_times

def get_spikes_and_attribute_lists(unit_ids, session, session_id, session_type, stimulus_presentation_ids,target_length, stimulus, stimulus_block):
    spikes_list = []
    ecephys_structure_acronym_list = []
    invalid_spiketimes_check_list = []
    rec_len_list = []
    firing_rate_list = []
    for unit in unit_ids:
        ecephys_structure_acronym_list += [session.units.loc[unit]['ecephys_structure_acronym']]

        spike_times_unit = get_spike_times(session,
                                           unit,
                                           stimulus_presentation_ids,
                                           target_length,
                                           transient=0)
        spikes_list += [spike_times_unit[stimulus_block]]
        invalid_spiketimes_check_list += [check_invalid_spiketimes(spike_times_unit,
                                              session,
                                              session_id,
                                              unit,
                                              session_type,
                                              stimulus,
                                              stimulus_block,
                                              target_length,
                                              transient=0)]


        rec_len_list += [spt[-1] - spt[0]
                             if len(spt) > 1
                             else 0.0
                             for spt in spike_times_unit.values()]

        firing_rate_list += [len(spt) / (spt[-1] - spt[0])
                                 if len(spt) > 1
                                 else 0.0
                                 for spt in spike_times_unit.values()]

    return spikes_list, ecephys_structure_acronym_list, invalid_spiketimes_check_list, rec_len_list, firing_rate_list

# to read the file, use something like this
def load_spike_data_hdf5(filepath, session_id, stimulus, stimulus_block, unit_index):
    filename = f"{filepath}/session_{session_id}/session_{session_id}_spike_data.h5"
    f = h5py.File(os.path.expanduser(filename), "r", libver="latest")
    spike_data = f[f"/{session_id}/{stimulus}/{stimulus_block}/spiketimes"]
    metadata = pd.read_hdf(filename, f"/{session_id}/{stimulus}/{stimulus_block}/metadata")
    spike_times_unit = spike_data[unit_index]
    spike_times_unit = spike_times_unit[np.isfinite(spike_times_unit)]
    metadata_unit = metadata.loc[unit_index]
    f.close()
    return spike_times_unit, metadata_unit

#     # Note: these guys are not numpy arrays, but h5py datasets
#     # they behave mostly the same
#     # But: without the [:] the data is not loaded into memory
#     # until it is accesed. so you could have a huge dataset
#
#     # read the attributes
#     description = file["/sessions/123/spiketimes"].attrs["description"]

def write_session_spikedata_to_h5(filepath, session, session_type, session_id, stimulus, stimulus_block, unit_ids,  stimulus_presentation_ids, target_length):
    filename = f"{filepath}/session_{session_id}/session_{session_id}_spike_data.h5"
    log.debug(f"writing session {session_id} to {filename}")
    # this takes care of closing the file, even when something does not complete
    if exists(filename):
        file = h5py.File(os.path.expanduser(filename), "a", libver="latest")
    else:
        file = h5py.File(os.path.expanduser(filename), "w", libver="latest")
    if h5py.version.hdf5_version_tuple >= (1, 9, 178):
        # A file which is being written to in Single-Write-Multliple-Read (swmr) mode
        # is guaranteed to always be in a valid (non-corrupt) state for reading.
        # This has the added benefit of leaving a file in a valid state even if
        # the writing application crashes before closing the file properly.
        # SWMR requires HDF5 version >= 1.9.178
        file.swmr_mode = True
        log.debug("using swmr mode")
    else:
        log.debug("SWMR requires HDF5 version >= 1.9.178")

    spikes_list, ecephys_structure_acronym_list, invalid_spiketimes_check_list, rec_len_list, firing_rate_list = get_spikes_and_attribute_lists(unit_ids, session, session_id, session_type, stimulus_presentation_ids,target_length, stimulus, stimulus_block)
    # creating dummy data for 2d nan-padded example
    num_units = len(unit_ids)
    max_num_spikes = 0
    for i in range(num_units):
        max_num_spikes = np.amax([max_num_spikes, len(spikes_list[i])])
    # init the empty thing
    spike_times = np.nan * np.ones(
        shape=(num_units, max_num_spikes), dtype=np.float32
    )

    # copy the spikes. this is not the efficient way but you get the idea
    for udx, unit in enumerate(unit_ids):
        for sdx, spike in enumerate(spikes_list[udx]):
            spike_times[udx, sdx] = spike

    # 2d array with first dim neuron id, second dim spike times
    spikes_dataset = file.create_dataset(
        f"/session_{session_id}_stimulus_{stimulus}_stimulus_block_{stimulus_block}_spiketimes",
        data=spike_times,
        compression="gzip",
        compression_opts=9,
    )
    file.close()
    # Store metadata for each unit via pandas dataframe
    d = {"unit_id": unit_ids, "ecephys_structure_acronym": ecephys_structure_acronym_list, "invalid_spiketimes_check": invalid_spiketimes_check_list, "recording_length": rec_len_list, "firing_rate": firing_rate_list}
    metadata = pd.DataFrame(data = d)
    metadata.to_hdf(filename, f"session_{session_id}_stimulus_{stimulus}_stimulus_block_{stimulus_block}_metadata")

    #
    # # and here the flat example
    # spike_list = []
    # unit_list = []
    #
    # for udx, unit in enumerate(units):
    #     spike_list.extend(unit_spikes)
    #     unit_list.extend([unit] * len(unit_spikes))
    #
    # # create the array with two columns
    # spiketimes_as_list = np.array([spike_list, unit_list]).T
    #
    # # flat list with first column neuron id, second column spike time
    # target_b = file.create_dataset(
    #     f"/sessions/{session}/spiketimes_as_list",
    #     data=spiketimes_as_list,
    #     compression="gzip",
    #     compression_opts=9,
    # )

    # if you do not know whats coming later, we can still create the dataset
    # and extend it later.
    # datasets behave like np arrays

    # create an empty dataset
    # target_c = file.create_dataset(
    #     f"/sessions/{session}/spiketimes_as_list_2",
    #     shape=(0, 2),
    #     maxshape=(None, 2),
    #     compression="gzip",
    #     compression_opts=9,
    # )
    #
    # # extend it
    # target_c.resize((spiketimes_as_list.shape[0], 2))
    # target_c[:] = spiketimes_as_list

if __name__ == "__main__":
    ## load data
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
    sessions = cache.get_session_table()

    ## write neuron lists
    isi_violations_maximum = quality_metrics[list_version]['isi_violations_maximum']
    amplitude_cutoff_maximum = quality_metrics[list_version]['amplitude_cutoff_maximum']
    presence_ratio_minimum = quality_metrics[list_version]['presence_ratio_minimum']

    session_ids = sessions[sessions['session_type'] == session_type].index.values

    for session_id in session_ids[11:]:
        session = cache.get_session_data(session_id,
                                         isi_violations_maximum = isi_violations_maximum,
                                         amplitude_cutoff_maximum = amplitude_cutoff_maximum,
                                         presence_ratio_minimum = presence_ratio_minimum)

        session_type = session.session_type

        stimulus_epochs = session.get_stimulus_epochs()

        units = session.units[session.units["ecephys_structure_acronym"].isin(areas)]
        unit_ids = units.index.values

        # Iterate over all stimuli of interest
        for stimulus in stimuli[session_type]:
            presentations = session.get_stimulus_table(stimulus)
            stimulus_blocks = [str(s) for s in
                               stimulus_epochs[stimulus_epochs['stimulus_name']
                                               == stimulus]['stimulus_block'].unique()]
            # Iterate over all stimulus blocks
            for stimulus_block in stimulus_blocks:
                stimulus_blocks_idxs \
                    = get_stimulus_blocks_indices(stimulus_epochs,
                                                       [sbfmt(stimulus_block)])
                block_durations \
                    = [block_duration
                       for block_duration
                       in stimulus_epochs[stimulus_epochs.index.isin(
                        stimulus_blocks_idxs)]['duration'].values]
                target_length = sum(block_durations)
                block_durations = ",".join(["{:.2f}".format(s) for s in block_durations])

                stimulus_presentation_ids \
                    = get_stimulus_presentation_ids(presentations,
                                                         stimulus_epochs,
                                                         session_type,
                                                         stimulus,
                                                         stimulus_block)

                write_session_spikedata_to_h5(data_directory, session, session_type, session_id, stimulus, stimulus_block, unit_ids,  stimulus_presentation_ids, target_length)
                log.debug(f"Successfully added spike data for stimulus {stimulus} and stimulus block {stimulus_block} to session {session_id}")
                # spikes_list, ecephys_structure_acronym_list, invalid_spiketimes_check_list, rec_len_list, firing_rate_list = get_spikes_and_attribute_lists(unit_ids, session, session_id, session_type, stimulus_presentation_ids,target_length, stimulus, stimulus_block)

    exit()

#
# def print_usage_and_exit(script_name):
#     print('usage is python3 {} list_version [test]'.format(script_name))
#     print('with list_version one of [unmerged, merged]')
#     print("if 'test' is passed, it is run for two sessions only")
#     exit()
#
# if __name__ == "__main__":
#     if len(argv) < 2 or len(argv) > 3 \
#        or not argv[1] in ['unmerged', 'merged']:
#         print_usage_and_exit(argv[0])
#
#     test = False
#     if len(argv) == 3:
#         if not argv[2] == 'test':
#             print_usage_and_exit(argv[0])
#         else:
#             test = True
#
#     exit(get_neuron_list(list_version=argv[1],
#                          test=test))
