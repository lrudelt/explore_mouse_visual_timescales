from sys import stderr, argv, exit
from os.path import realpath, dirname, exists
import os
import numpy as np
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
with open("{}/../../dirs.yaml".format(SCRIPT_DIR), "r") as dir_settings_file:
    dir_settings = yaml.load(dir_settings_file, Loader=yaml.BaseLoader)
### settings
data_directory = dir_settings["allen_data_dir"]
manifest_path = os.path.join(data_directory, "manifest.json")

stimuli = {
    "brain_observatory_1.1": [
        "natural_movie_one",
        "natural_movie_three",
        "drifting_gratings",
        "gabors",
    ],
    "functional_connectivity": [
        "natural_movie_one_more_repeats",
        "natural_movie_one_shuffled",
        "drifting_gratings_contrast",
        "drifting_gratings_75_repeats",
        "gabors",
        "spontaneous",
    ],
}

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


def load_spike_times(session_id, unit, spike_data_dir):
    spiketimes = np.load(
        "{}/spike_data_{}_{}_{}_{}.npy".format(
            spike_data_dir, session_id, unit, stimulus, stimulus_blocks
        )
    )


def get_spike_times(
    session,
    stimulus_presentation_ids,
    stimulus_block=None,
):

    num_nonzero_blocks = sum(
        [
            1
            for stimulus_block in stimulus_presentation_ids
            if len(stimulus_presentation_ids[stimulus_block]) > 0
        ]
    )

    spike_times = {}
    # iterate over stim_blocks, if stimulus_block is None
    # else only fill the one we requested
    for sb in stimulus_presentation_ids:
        if stimulus_block is not None and sb != stimulus_block:
            continue

        spike_times[sb] = []

        if not len(stimulus_presentation_ids[sb]) > 0:
            continue

        spikes = session.presentationwise_spike_times(
            stimulus_presentation_ids=stimulus_presentation_ids[sb],
            # None gives all
            unit_ids=None,
        )

        if not len(spikes) > 0:
            print("??? for {}, {}".format(session))
            continue

        spike_times[sb] = spikes.index.values

    return spike_times


def write_block_to_h5(filename, session, stimulus_block):

    log.debug(f"writing session {session} to {filename}")

    # for now we only do resting
    stimulus = "spontaneous"

    # todo for lucas

    # get_spike_times always returns a dict (but not with all stim blocks)
    spikes_2d = get_spike_times(
        session=session,
        stimulus_presentation_ids=stimulus_ids,
        stimulus_block=stimulus_block,
    )

    # this takes care of closing the file, even when something does not complete
    with h5py.File(os.path.expanduser(filename), "w", libver="latest") as file:
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

        # creating dummy data for 2d nan-padded example
        num_units = 53
        max_num_spikes = 1234
        # init the empty thing
        spike_times = np.nan * np.ones(
            shape=(num_units, max_num_spikes), dtype=np.float32
        )

        # copy the spikes. this is not the efficient way but you get the idea
        for udx, unit in enumerate(units):
            for sdx, spike in enumerate(spikes):
                spike_times[udx, sdx] = spike

        # 2d array with first dim neuron id, second dim spike times
        target_a = file.create_dataset(
            f"/sessions/{session}/spiketimes",
            data=spikes_2d,
            compression="gzip",
            compression_opts=9,
        )

        # lets add some attributes. think meta data
        target_a.attrs["description"] = "lorem ipsum dolor"
        target_a.attrs["block_time_blah"] = 42




        # and here the flat example
        spike_list = []
        unit_list = []

        for udx, unit in enumerate(units):
            spike_list.extend(unit_spikes)
            unit_list.extend([unit] * len(unit_spikes))

        # create the array with two columns
        spiketimes_as_list = np.array([spike_list, unit_list]).T

        # flat list with first column neuron id, second column spike time
        target_b = file.create_dataset(
            f"/sessions/{session}/spiketimes_as_list",
            data=spiketimes_as_list,
            compression="gzip",
            compression_opts=9,
        )



        # if you do not know whats coming later, we can still create the dataset
        # and extend it later.
        # datasets behave like np arrays

        # create an empty dataset
        target_c = file.create_dataset(
            f"/sessions/{session}/spiketimes_as_list_2",
            shape=(0, 2),
            maxshape=(None, 2),
            compression="gzip",
            compression_opts=9,
        )

        # extend it
        target_c.resize((spiketimes_as_list.shape[0], 2))
        target_c[:] = spiketimes_as_list


    # to read the file, use something like this
    with h5py.File(os.path.expanduser(filename), "r", libver="latest") as file:
        # read the data
        data = file["/sessions/123/spiketimes"][:]

        # Note: these guys are not numpy arrays, but h5py datasets
        # they behave mostly the same
        # But: without the [:] the data is not loaded into memory
        # until it is accesed. so you could have a huge dataset

        # read the attributes
        description = file["/sessions/123/spiketimes"].attrs["description"]

def assert_appropriateness(
    spike_times,
    session,
    session_id,
    unit,
    session_type,
    stimulus,
    stimulus_blocks,
    target_length,
    transient,
):

    # first check recording length
    rec_len = sum([spt[-1] - spt[0] for spt in spike_times.values() if len(spt) > 1])

    if rec_len == 0:
        print(
            "empty recording for {}, {}, {}, {}".format(
                session_id, unit, stimulus, stimulus_blocks
            )
        )
        return return_values["ERR_EMPTY"]
    elif rec_len < 0.9 * target_length:
        print(
            "short recording ({} s) for {}, {}, {}, {}".format(
                rec_len, session_id, unit, stimulus, stimulus_blocks
            )
        )
        return return_values["ERR_REC_LEN"]
    elif rec_len > 1.1 * target_length:
        print(
            "long recording ({} s) for {}, {}, {}, {}".format(
                rec_len, session_id, unit, stimulus, stimulus_blocks
            )
        )
        return return_values["ERR_REC_LEN"]

    for spt in spike_times.values():
        if not len(spt) > 1:
            continue
        firing_rate = len(spt) / (spt[-1] - spt[0])
        if firing_rate < 0.01:
            print(
                "low firing rate ({} Hz) for {}, {}, {}, {}".format(
                    firing_rate, session_id, unit, stimulus, stimulus_blocks
                )
            )
            return return_values["ERR_LO_FIR_RATE"]

    # check for invalid times in analysed spike times
    units = session.units
    probe_id = str(units.loc[unit]["probe_id"])

    invalid_times = session.invalid_times.copy()

    if not invalid_times.empty:
        probe_ids = np.array([str(a[1]) for a in invalid_times["tags"].values])
        invalid_times.insert(3, "probe_id", probe_ids)

        for spt in spike_times.values():
            if len(spt) > 0:
                if (
                    len(
                        invalid_times[
                            (invalid_times["probe_id"] == probe_id)
                            & (invalid_times["start_time"] < spt[-1])
                            & (invalid_times["stop_time"] > spt[0])
                        ]
                    )
                    > 0
                ):
                    print(
                        "invalid spike times in {}, {}, {}, {}".format(
                            session_id, unit, stimulus, stimulus_blocks
                        )
                    )
                    return return_values["ERR_INVALID_TIMES"]

    # check for evidence of non-stationarity
    # if len(stimulus_blocks.split(',')) > 1:
    #     firing_rates = [len(spt) / (spt[-1] - spt[0]) for spt in spike_times.values()]

    #     firing_rate_diff = np.abs(firing_rates[0] - firing_rates[1]) / min(firing_rates)

    #     if firing_rate_diff > 0.2:
    #         return return_values['ERR_NON_STATIONARY']

    return return_values["SUCCESS"]


def print_spike_times(
    session_id, unit, stimulus, stimulus_blocks, target_length, transient, spike_data_dir
):
    # check if soikes have been stored
    if exists(
        "{}/spike_data_{}_{}_{}_{}.npy".format(
            spike_data_dir, session_id, unit, stimulus, stimulus_blocks
        )
    ):
        spike_times_npy = np.load(
            "{}/spike_data_{}_{}_{}_{}.npy".format(
                spike_data_dir, session_id, unit, stimulus, stimulus_blocks
            ),
            allow_pickle=True,
        )
    # if not, load spikes from session data
    else:
        cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

        # filters were already applied to get the neuron list, so no need to apply again
        session = cache.get_session_data(
            session_id,
            isi_violations_maximum=np.inf,
            amplitude_cutoff_maximum=np.inf,
            presence_ratio_minimum=-np.inf,
        )
        session_type = session.session_type
        stimulus_epochs = session.get_stimulus_epochs()

        assert stimulus in stimuli[session_type]
        for stimulus_block in stimulus_blocks.split(","):
            assert (
                sbfmt(stimulus_block)
                in stimulus_epochs[stimulus_epochs["stimulus_name"] == stimulus][
                    "stimulus_block"
                ].values
            )
        assert unit in session.units.index.values

        presentations = session.get_stimulus_table(stimulus)

        stimulus_presentation_ids = get_stimulus_presentation_ids(
            presentations, stimulus_epochs, session_type, stimulus, stimulus_blocks
        )

        spike_times = get_spike_times(
            session, unit, stimulus_presentation_ids, target_length, transient
        )

        if (
            not assert_appropriateness(
                spike_times,
                session,
                session_id,
                unit,
                session_type,
                stimulus,
                stimulus_blocks,
                target_length,
                transient,
            )
            == return_values["SUCCESS"]
        ):
            exit()
        # save spikes in npy format
        spike_times_npy = [spike_times[stimulus_block] for stimulus_block in spike_times]
        print(spike_times_npy)
        np.save(
            "{}/spike_data_{}_{}_{}_{}.npy".format(
                spike_data_dir, session_id, unit, stimulus, stimulus_blocks
            ),
            spike_times_npy,
        )

    # print the spike times
    # first_print = True
    # for stimulus_block in spike_times:
    #     if len(spike_times[stimulus_block]) > 0:
    #         if first_print:
    #             first_print = False
    #         else:
    #             print('----------')
    #
    #         for spt in spike_times[stimulus_block]:
    #             print(spt)
    first_print = True
    for spike_times_block in spike_times_npy:
        if len(spike_times_block) > 0:
            if first_print:
                first_print = False
            else:
                print("----------")

            for spt in spike_times_block:
                print(spt)
    exit()


def print_usage_and_exit(script_name):
    print(
        "usage is python3 {} session_id unit stimulus stimulus_blocks target_length"
        " transient".format(script_name)
    )
    print(
        "with stimulus one of {} for the brain_observatory_1.1 stimulus set,".format(
            stimuli["brain_observatory_1.1"]
        )
    )
    print(
        "or one of {} for the functional_connectivity stimulus set.".format(
            stimuli["functional_connectivity"]
        )
    )
    print("target_length and transient in seconds.")
    exit()


if __name__ == "__main__":
    # format of the neuron list is
    # session_id session_type unit ecephys_structure_acronym stimulus stimulus_blocks ...

    if (
        not len(argv) == 8
        or not argv[1].isdecimal()
        or not argv[2].isdecimal()
        or (
            (not argv[3] in stimuli["brain_observatory_1.1"])
            & (not argv[3] in stimuli["functional_connectivity"])
        )
    ):
        print_usage_and_exit(argv[0])

    for stimulus_block in argv[4].split(","):
        try:
            float(stimulus_block)
        except:
            if not stimulus_block == "null":
                print(
                    'stimulus_blocks must be a number or "null" or several instanced'
                    " thereof, separated by commas"
                )
                exit()

    for val, name in zip([argv[5], argv[6]], ["target_length", "transient"]):
        try:
            float(val)
        except:
            print("{} must be a number (please provide it in seconds)".format(name))
            exit()
    spike_data_dir = argv[7]
    try:
        exists(spike_data_dir)
    except:
        print("Spike data directory {} does not exist!")

    exit(
        print_spike_times(
            session_id=int(argv[1]),
            unit=int(argv[2]),
            stimulus=argv[3],
            stimulus_blocks=argv[4],
            target_length=float(argv[5]),
            transient=float(argv[6]),
            spike_data_dir=spike_data_dir,
        )
    )
