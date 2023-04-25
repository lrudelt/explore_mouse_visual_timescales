# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2023-03-09 18:33:59
# @Last Modified: 2023-04-25 16:46:27
# ------------------------------------------------------------------------------ #


import logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s",
    level=logging.WARNING,
)
log = logging.getLogger(__name__)
log.setLevel("DEBUG")

import re
import glob
import h5py
import sys
import numpy as np
import xarray as xr
import pandas as pd
import warnings
import numpy as np
from numba import jit
from tqdm import tqdm
from humanize import naturalsize

# silence numba deprications, numpy overflows
warnings.filterwarnings("ignore")

sys.path.append("../")

# ------------------------------------------------------------------------------ #
# Loading of hdf5 data, maybe we put this into its own file.
# ------------------------------------------------------------------------------ #


_meta_df = None


def all_unit_metadata(
    dir="/data.nst/share/data/allen_visual_coding_neuropixels/",
    reload=False,
):
    """
    Returns a pandas dataframe holding the overall index,
    an overview of sessions and units.

    Index is a unit, columns are sessions, stimuli, blocks, needed file path,
    and various tests and metadata.
    """

    # lets be smart about this and not load this from disk every times
    global _meta_df
    if not reload and _meta_df is not None:
        log.debug("Using cached metadata")
        return _meta_df.copy()

    files = glob.glob(dir + "/**/*.h5", recursive=True)
    log.debug(f"Found {len(files)} hdf5 files in {dir}")

    meta_df = []

    for fp in tqdm(files, desc="Fetching metadata from sessions"):
        meta_df.append(load_session(fp, meta_only=True))

    meta_df = pd.concat(meta_df, axis=0, ignore_index=True)
    meta_df.set_index("unit_id", inplace=True, drop=False)
    _meta_df = meta_df.copy()

    return meta_df


def load_session(
    filepath, meta_only=False, pad_spikes_to=None, filter=None, as_dict=False
):
    """
    Load a single session as dictionary.
    optionally, only get metadata.


    # Parameters
    filepath (str): path to hdf5 file holding the session,
        best get this from all_unit_metadata
    meta_only (bool): if True, only load metadata, not spiketimes

    # Returns
    session_dict (dict): nested dictionary with the following structure:
        session_dict[session][stimulus][block][kind]
        kind is either "meta" (holding a pandas dataframe)
        or "data" (holding a xarray DataArray with spiketimes)
    """

    session_dict = dict()
    num_spikes = dict()
    if filter is None:
        filter = dict()

    # in my current implementation, we touch the file a bunch of times :/
    with h5py.File(filepath, "r") as f:
        # there should only be one session per file, but this makes it easier to handle, later
        keys = list(f.keys())

        # we want the number of spikes in the metadata, but they are in the spiketimes
        for key in keys:
            if key.endswith("_spiketimes"):
                num_spikes[key.replace("_spiketimes", "")] = f[key].shape[1]

    # start by loading the metadata and setting the directory structure
    for key in keys:
        # keys are specific for stimulus-block pairs, and come in pairs:
        # one ending with _metadata (a panadas dataframe)
        # one ending with _spiketimes (a nan-padded 2d array, nid x spikes)
        if not key.endswith("_metadata"):
            continue

        parts = _key_to_parts(key)
        session = parts["session"]
        stimulus = parts["stimulus"]
        block = parts["block"]

        # only continue if we match the filter criteria.
        # filter looks like this: dict(session=["session_1"], block=["block_1", "block_2"])
        if np.any([parts[k] not in v for k, v in filter.items() if k in parts]):
            continue

        if session not in session_dict:
            session_dict[session] = dict()
            # this becomes a problem when trying to get the metadata across all units
            # session_dict[session]["_filepath"] = filepath

        if stimulus not in session_dict[session]:
            session_dict[session][stimulus] = dict()

        if block not in session_dict[session][stimulus]:
            session_dict[session][stimulus][block] = dict()

        # load the metadata.
        # pandas operates on the file -> have to open twice
        # which will only work if we saved in swmr mode.
        meta_df = pd.read_hdf(filepath, key=key.replace("_spiketimes", "_metadata"))
        # set unit_id as index, but keep it as a column for convenience
        meta_df.set_index("unit_id", inplace=True, drop=False)

        meta_df["filepath"] = filepath

        # we will later need the max number of spikes, so we can pad the arrays
        meta_df["num_spikes"] = num_spikes[key.replace("_metadata", "")]
        session_dict[session][stimulus][block]["meta"] = meta_df

        # on the block level, units should be unique
        assert len(meta_df) == len(meta_df["unit_id"].unique())

    if meta_only:
        if as_dict:
            return session_dict
        else:
            return _session_dict_to_df(session_dict)

    # now load the spiketimes
    with h5py.File(filepath, "r") as f:
        for key in keys:
            if not key.endswith("_spiketimes"):
                continue

            parts = _key_to_parts(key)
            session = parts["session"]
            stimulus = parts["stimulus"]
            block = parts["block"]

            # check the filter
            if np.any([parts[k] not in v for k, v in filter.items() if k in parts]):
                continue

            # load data to ram, and convert to xarray so we can to conveniently
            # index via unit_id
            da = xr.DataArray(
                data=f[key][:],
                dims=["unit_id", "spiketimes"],
                coords={"unit_id": meta_df.index.values},
            )
            # add the stimulus and block as dimensions, so we can later merge
            da = da.expand_dims(
                {"session": [session], "stimulus": [stimulus], "block": [block]}
            )

            if pad_spikes_to is not None and len(da["spiketimes"]) < pad_spikes_to:
                # extend the dimesions to that length, so that it can be merged
                # outside
                # the xarray pad api is likely to change. written for version 2023.1.0
                da = da.pad(
                    spiketimes=(0, pad_spikes_to - len(da["spiketimes"])),
                    mode="constant",
                    constant_values=np.nan,
                )

            session_dict[session][stimulus][block]["data"] = da

    if as_dict:
        return session_dict
    else:
        return _session_dict_to_df(session_dict)


def _key_to_parts(key):
    """
    We have a convention to store data in hdf5 files.
    Each group key looks like this:
    `session_774875821_stimulus_natural_movie_one_more_repeats_ \
    stimulus_block_8.0_spiketimes`

    with this helper we use regex to get the individual parts as a dict:
    session (int):
    stimulus (str):
    block (str):
    kind (str): either metadata or spiketimes
    """

    parts = dict()

    # regex magic to get parts of the key
    # session is the integer sequence after `session_`
    parts["session"] = re.search(r"session_(\d+)_", key).group(1)
    parts["stimulus"] = re.search(r"stimulus_(.+)_stimulus_block", key).group(1)
    parts["block"] = re.search(r"stimulus_block_(.+)_", key).group(1)
    # word characters after last `_` and before the end of the key
    parts["kind"] = re.search(r"_([a-zA-Z0-9.-]+)$", key).group(1)

    if parts["kind"] not in ["metadata", "spiketimes"]:
        raise ValueError(f"Unknown kind {parts['kind']} for key '{key}'")

    # blocks shall always remain strings, make sure sessions are integers?
    # units from metadata dataframe are int64, anyway. mightaswell be consistent.
    parts["session"] = int(parts["session"])

    return parts


def load_spikes(meta_df):
    """
    After filtering the global index for desired untis / criteria,
    provide the filtered meta dataframe here, to load spiketimes as xarray,
    and add them as a new column.

    # Example
    ```
    meta_df = all_unit_metadata()
    filtered_df = meta_df.query("stimulus == 'spontaneous'")
    loaded_df = load_spikes(filtered_df)
    ```

    # Parameters
    meta_df (pd.DataFrame): filtered meta dataframe

    # Returns
    df : pd.DataFrame
        same rows as meta_df, but with an additional column
        `spiketimes` containing the spiketimes as xr.DataArray

    """

    # lessons learned: tried to return a single xarray for this, with dimensions:
    # session, stimulus, block, unit_id, spiketimes. Not the best idea,
    # there will be a lot of not-used indices (nans) requiring tons of ram.
    # instead: add an xr array as a column to the pandas dataframe

    max_num_spikes = meta_df["num_spikes"].max()
    sessions = meta_df["session"].unique()
    stimuli = meta_df["stimulus"].unique()
    blocks = meta_df["block"].unique()
    units = meta_df["unit_id"].unique()
    files = meta_df["filepath"].unique()

    res_df = meta_df.copy()
    # add column to hold objects (arrays)
    res_df["spiketimes"] = None
    res_df["spiketimes"] = res_df["spiketimes"].astype(object)

    res_df.set_index(["session", "stimulus", "block", "unit_id"], inplace=True)

    filter = dict(
        # lets try to be smart with the filtering. units: no!
        # sessions neither, as we iterate them.
        # blocks and stimuli are sensible.
        stimulus=list(stimuli),
        block=list(blocks),
    )

    log.debug(
        f"Loading spikes for {len(units)} units, {len(res_df)} rows for pandas dataframe."
    )

    # create an empty datarray with our known coordinates

    num_rows = 0

    for fdx, file in enumerate(tqdm(files, desc="Loading spiking data")):
        session_dict = load_session(file, filter=filter, as_dict=True)

        # iterate the dict, find out where to put each units dataframe.
        session = list(session_dict.keys())[0]
        sd = session_dict[session]
        for stim in sd.keys():
            for block in sd[stim].keys():
                da = sd[stim][block]["data"]
                for unit in da["unit_id"].values:
                    index = (session, stim, block, unit)
                    if not index in res_df.index:
                        continue

                    res_df.loc[index, "spiketimes"] = da.sel(unit_id=unit)
                    num_rows += 1

    if not num_rows == len(res_df):
        raise ValueError(f"Loaded {num_rows} rows, but expected {len(res_df)}")

    res_df.reset_index(inplace=True, drop=False)
    return res_df


def _session_dict_to_df(session_dict):
    """
    create the metadata dataframe from a single session dict.
    """

    meta_df = []
    for session in session_dict:
        for stimulus in session_dict[session]:
            for block in session_dict[session][stimulus]:
                df = session_dict[session][stimulus][block]["meta"]
                df["session"] = session
                df["stimulus"] = stimulus
                df["block"] = block
                meta_df.append(df)

    meta_df = pd.concat(meta_df, axis=0, ignore_index=True)
    meta_df.set_index("unit_id", inplace=True, drop=False)
    return meta_df


def _session_dict_to_xr(session_dict):
    """
    takes a single session dict, and merges the dataframes into a combined xarray df.
    takes all stimuli and blocks, expanding the dimensions.

    the root key of the provided session_dict is expected to be the session id.
    (as you get from load_session)

    # Note
    The merged xr.DataArray will need notably more RAM then the separate ones
    in the session_dict, because we pad nans in all dimensions (session, stim, spikes ...)
    """
    if len(session_dict) != 1:
        raise ValueError("Expected a single session as first key, got more")

    session = list(session_dict.keys())[0]
    sd = session_dict[session]

    # resulting dataframe, we add all block-level frames, after expanding spike dim.
    session_df = None

    max_num_spikes = 0
    for stim in sd.keys():
        for block in sd[stim].keys():
            if "data" not in sd[stim][block]:
                raise ValueError(
                    f"Missing spiketimes for {stim} block {block} in {session}"
                )
            max_num_spikes = max(
                max_num_spikes, len(sd[stim][block]["data"]["spiketimes"])
            )

    for stim in sd.keys():
        for block in sd[stim].keys():
            df = sd[stim][block]["data"]
            # the padding is usually avoided if we provided `pad_spikes_to` in
            # `load_session()`
            if max_num_spikes > len(df["spiketimes"]):
                df = df.pad(
                    spiketimes=(0, max_num_spikes - len(df["spiketimes"])),
                    mode="constant",
                    constant_values=np.nan,
                )
            if session_df is None:
                session_df = df
            else:
                session_df = xr.concat([session_df, df], dim="block")

    return session_df


def default_filter(meta_df, target_length=1080, transient_length=360, trim=True):
    """
    Apply our default set of quality controls to the metadata frame.

    # Parameters
    meta_df: the metadata dataframe
    target_length: the target length for recording duration, in seconds
        required duration for stimuli with two blocks is 0.5 target_length for each block
    transient_length: in seconds, added to target_length (to account for changes at the
        beginning of each block)
    trim : if True (default)
        remove rows that failed the quality checks.
        to get the full-length dataframe, set to False, and
        this will update the `invalid_spiketimes_check` column. The you can query like:
        `meta_df.query("invalid_spiketimes_check == 'SUCCESS'")`

    # Returns
    meta_df: the modified dataframe

    """

    meta_df = meta_df.copy()

    def _num_good(df):
        return len(df.query("invalid_spiketimes_check == 'SUCCESS'"))

    log.debug(f"Default quality checks, valid rows before: {_num_good(meta_df)}")

    # We find a minimal firing rate of approximately 0.02 Hz and a maximal firing rate of approximately 90 Hz, with 95\% of firing rates in the range of 0.19 Hz to 21.11 Hz. The highest firing rates are certainly biologically implausible, but units with these values are few so that they should not distort results in any significant way\
    # This check effectively filters units without _any_ spiking
    meta_df.loc[
        meta_df["firing_rate"] < 0.01, "invalid_spiketimes_check"
    ] = "ERR_LO_FIR_RATE"
    log.debug(f"After rate check: {_num_good(meta_df)}")

    meta_df.loc[
        meta_df["recording_length"] == 0, "invalid_spiketimes_check"
    ] = "ERR_EMPTY"
    log.debug(f"After zero-length check: {_num_good(meta_df)}")

    # stationarity.
    # group by units and stimuli, and if the stim has two blocks,
    # check that the rate is not too far off between blocks
    # requires "neutral" index

    # meta_df.reset_index(inplace=True, drop=True)
    meta_df.set_index(["unit_id", "stimulus"], inplace=True, drop=True)

    # including stationarity checks did not change results.
    # for (unit_id, stimulus), df in tqdm(
    #     meta_df.groupby(["unit_id", "stimulus"]), desc="Stationarity check"
    # ):
    #     if len(df) < 2:
    #         continue

    #     # get the firing rate for the first and second block
    #     fr1 = df.iloc[0]["firing_rate"]
    #     fr2 = df.iloc[1]["firing_rate"]

    #     # if the difference is too big, mark the unit as invalid
    #     if abs(fr1 - fr2) / np.mean([fr1, fr2]) > 0.5:
    #         meta_df.loc[
    #             (unit_id, stimulus), "invalid_spiketimes_check"
    #         ] = "ERR_STATIONARITY"

    meta_df.reset_index(inplace=True, drop=False)

    log.debug(f"After stationarity check: {_num_good(meta_df)}")

    # check the duration for single blocks
    idx = meta_df.query(
        f"stimulus == 'spontaneous'"
        + f" & recording_length < {0.9 * (target_length + transient_length)}"
    ).index
    meta_df.loc[idx, "invalid_spiketimes_check"] = "ERR_REC_LEN"

    # check total duration of two=block stimuli
    idx = meta_df.query(
        f"stimulus != 'spontaneous'"
        + f" & recording_length < {0.9 * (target_length / 2 + transient_length)}"
    ).index
    meta_df.loc[idx, "invalid_spiketimes_check"] = "ERR_REC_LEN"

    log.debug(f"After minmum-duration check: {_num_good(meta_df)}")

    if trim:
        meta_df = meta_df.query("invalid_spiketimes_check == 'SUCCESS'")

    # reset the index to the default (unit_id)
    meta_df.set_index("unit_id", inplace=True, drop=False)

    return meta_df


def merge_blocks(meta_df, target_length=1080, transient_length=360):
    """
    Merge the spiking data from two blocks (for stimuli with two blocks).

    do this last.

    # Parameters
    meta_df: the metadata dataframe, data already loaded, checks performed.
        Also, we expect only two-block stimuli to be present.
    """

    new_rows = []
    dropped_units = 0

    try:
        groupby = meta_df.groupby(["session", "stimulus", "unit_id"])
    except ValueError:
        # this happens when the index is set to unit_id (and unit_id is still a column)
        # resetting is not great, it will modify the outer df which seems unexpected.
        meta_df = meta_df.reset_index(drop=True)
        groupby = meta_df.groupby(["session", "stimulus", "unit_id"])

    for df_idx, df in tqdm(groupby, desc="Merging blocks", total=len(groupby)):
        if len(df) == 1:
            # the unit was only recorded in one block
            dropped_units += 1
            continue
        elif len(df) != 2:
            raise ValueError("Expected two blocks for each stimulus")

        # squeeze effectively gets rid of len-1 dimensions
        spikes1 = df.iloc[0]["spiketimes"].copy().squeeze()
        spikes2 = df.iloc[1]["spiketimes"].copy().squeeze()

        # remove the nan-padding
        spikes1 = spikes1[np.isfinite(spikes1)]
        spikes2 = spikes2[np.isfinite(spikes2)]

        # align to first spike
        spikes1 = spikes1 - spikes1[0]
        spikes2 = spikes2 - spikes2[0]

        # remove the transient
        spikes1 = spikes1[spikes1 > transient_length] - transient_length
        spikes2 = spikes2[spikes2 > transient_length] - transient_length

        # limit the duration
        spikes1 = spikes1[spikes1 < target_length]
        spikes2 = spikes2[spikes2 < target_length]

        # align second spike train to the end of the first @LR
        spikes2 = spikes2 + spikes1[-1]

        # assign the new coordinates (block) to xarray
        new_block = f"merged_{df.iloc[0]['block']}_and_{df.iloc[1]['block']}"
        spikes1.coords["block"] = new_block
        spikes2.coords["block"] = new_block
        spikes = xr.concat([spikes1, spikes2], dim="spiketimes")

        # create the new row, updating columns
        new_row = df.iloc[0].copy()
        new_row["spiketimes"] = spikes
        new_row["block"] = new_block
        new_row["num_spikes"] = len(spikes)
        new_row["recording_length"] = (spikes[-1] - spikes[0]).values[()]
        new_row["firing_rate"] = len(spikes) / new_row["recording_length"]
        new_rows.append(new_row)

    res_df = pd.DataFrame(new_rows)
    log.debug(
        f"Dropped {dropped_units} units (only found in one block)."
        f" {len(res_df['unit_id'].unique())} units remaining."
    )

    return res_df


# ------------------------------------------------------------------------------ #
# spike processing
# ------------------------------------------------------------------------------ #


def binned_spike_count(spiketimes, bin_size):
    """
    Similar to a population_rate, but we get a number of spike counts, per neuron,
    as needed for e.g. cross-correlations.

    Parameters
    ----------
    spiketimes : list of lists or 2d nan-padded array
        where first dim is neuron/block and second index are spiketimes.
        for a single neuron, pass `[your_spiketimes_as_array]`
    bin_size :
        float, in units of spiketimes

    Returns
    -------
    counts : 2d array
        time series of the counted number of spikes per bin,
        one row for each neuron, in steps of bin_size
    """
    # type checking is easier here than in numba
    if len(spiketimes) == 0:
        return np.array([])
    elif not isinstance(spiketimes[0], (np.ndarray, xr.DataArray)):
        raise ValueError("spiketimes must be a list of numpy arrays or a 2d array")

    if isinstance(spiketimes[0], xr.DataArray):
        # this is only needed if we want to use numba.
        spiketimes = [spiketimes[n_id].to_numpy() for n_id in range(len(spiketimes))]

    return _binned_spike_count(spiketimes, bin_size)


@jit(nopython=True, parallel=True, fastmath=False, cache=True)
def _binned_spike_count(spiketimes, bin_size):
    num_n = len(spiketimes)

    t_min = np.inf
    t_max = 0

    for n_id in range(0, num_n):
        t_min = min(t_min, np.nanmin(spiketimes[n_id]))
        t_max = max(t_max, np.nanmax(spiketimes[n_id]))

    num_bins = int((t_max - t_min) / bin_size) + 1
    counts = np.zeros(shape=(num_n, num_bins))

    for n_id in range(0, num_n):
        train = spiketimes[n_id]
        for t in train:
            if not np.isfinite(t):
                break
            # @LR decision: do we align to a global min (pauls)
            # or to a local one, per block (lucas)
            # t_idx = int((t - t_min) / bin_size)
            t_idx = int((t - train[0]) / bin_size)
            counts[n_id, t_idx] += 1

    return counts


# ------------------------------------------------------------------------------ #
# misc helpers
# ------------------------------------------------------------------------------ #


# on the cluster we dont want to see the progress bar
# https://stackoverflow.com/questions/37091673/silence-tqdms-output-while-running-tests-or-running-the-code-via-cron
def disable_tqdm():
    """Disable tqdm progress bar."""
    global tqdm
    from functools import partialmethod

    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


# disable by default
# disable_tqdm()


def enable_tqdm():
    """Enable tqdm progress bar."""
    global tqdm
    from functools import partialmethod

    tqdm.__init__ = partialmethod(tqdm.__init__, disable=False)
