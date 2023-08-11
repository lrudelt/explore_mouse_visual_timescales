# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2023-03-09 18:33:59
# @Last Modified: 2023-08-11 18:20:03
# ------------------------------------------------------------------------------ #


import logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s",
    level=logging.WARNING,
)
log = logging.getLogger("its_utility")
log.setLevel("WARNING")

import os
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

# sys.path.append("../")

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

    Uses `load_session` on every found hdf5 file.

    # Parameters:
    dir (str): directory holding the hdf5 files
    reload (bool): if True, reload the metadata from disk, otherwise use the module cache

    # Columns:
    unit_id : int
        unique unit id, e.g. 951013153. unique per block, might reoccur in different blocks
    session : int
        session number, e.g. 787025148
    num_spikes : int
        note that this is the max number of spikes any unit had in that block,
        because spiking data is nan-padded (and we cannot load the actual number
        for a single unit from meta-data alone).
    stimulus : str/Object
        stimulus name, e.g. "natural_movie_one_more_repeats"
    block : str/O
        block description, e.g. "3.0" or "null"
    ecephys_structure_acronym : str/O
        area, e.g. "VISam"
    invalid_spiketimes_check : str/O
        overall status. we update this later, when certain checks fail. default: "SUCCESS"
    recording_length : float
        in seconds
    firing_rate : float
        in spikes per second
    filepath : str
        path to the hdf5 file holding the session

    # Notes:
    - to get the `spiketimes` column, call `load_spikes(meta_df)`, best after filtering,
        to avoid loading data you don't need.
    """

    # lets be smart about this and not load this from disk every times
    global _meta_df
    if not reload and _meta_df is not None:
        log.debug("Using cached metadata")
        return _meta_df.copy()

    dir = os.path.abspath(os.path.expanduser(dir))
    files = glob.glob(dir + "/**/*.h5", recursive=True)
    log.debug(f"Found {len(files)} hdf5 files in {dir}")
    assert len(files) > 0, f"Found no hdf5 files in {dir}"

    meta_df = []

    for fp in tqdm(files, desc="Fetching metadata from sessions"):
        try:
            session_df = load_session(fp, meta_only=True)
        except ValueError as e:
            log.info(f"Skipping {fp}. This might be a hdf5 file with no session data.")
            continue
        meta_df.append(session_df)

    meta_df = pd.concat(meta_df, axis=0, ignore_index=True)
    # unit_ids are not unique, so lets avoid using them as the index.
    meta_df.reset_index(inplace=True, drop=True)

    _meta_df = meta_df.copy()

    return meta_df


def load_session(
    filepath, meta_only=False, pad_spikes_to=None, filter=None, as_dict=False
):
    """
    Load a single session as pandas dataframe (or a dictionary)
    optionally, only get metadata.


    # Parameters
    filepath (str): path to hdf5 file holding the session,
        best get this from all_unit_metadata
    filter : dict or None (default)
        only load keys that are in the given filters.
        e.g. `filter=dict(stimulus=["stim_1"], block=["block_1", "block_2"])`
    meta_only (bool): if True, only load metadata, not spiketimes
    as_dict (bool): if True, return a dictionary instead of a dataframe.
    pad_spikes_to (int): if not None, pad the spiketimes with nan to this length.
        might help when merging xarrays, later.

    # Returns
    meta_df (pandas.DataFrame): dataframe holding the metadata, OR:
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
        # note that num_spikes is the max number of spikes any unit had in that block
        # due to the nan-padding
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
            # update num_spikes column in the metadata
            for uid in session_dict[session][stimulus][block]["meta"]["unit_id"].unique():
                session_dict[session][stimulus][block]["meta"].loc[uid, "num_spikes"] = (
                    np.isfinite(da.sel(unit_id=uid)).sum().item()
                )

    if as_dict:
        return session_dict
    else:
        return _session_dict_to_df(session_dict)


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
        - same rows as meta_df, but with an additional column (object)
            `spiketimes` containing the spiketimes as xr.DataArray
        - the contained xarray.DataArray have dimensions:
            (session, stimulus, block, spiketimes)


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

    res_df.set_index(["session", "stimulus", "block", "unit_id"], inplace=True, drop=True)

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

    for fdx, file in enumerate(tqdm(files, desc="Loading spikes for sessions")):
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

                    # assign data to df, and update the num_spikes column
                    arr = da.sel(unit_id=unit)
                    if len(arr.squeeze().shape) != 1:
                        log.warning("spiketimes should have one variable dimension")
                    res_df.loc[index, "spiketimes"] = arr
                    res_df.loc[index, "num_spikes"] = np.isfinite(arr).sum().item()

                    num_rows += 1

    if not num_rows == len(res_df):
        raise ValueError(f"Loaded {num_rows} rows, but expected {len(res_df)}")

    # we dropped the columns above, so here we need to re-insert by drop=False
    res_df.reset_index(inplace=True, drop=False)
    return res_df


def default_filter(
    meta_df, target_length=1080, transient_length=360, trim=True, inplace=False
):
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
        this will update the `invalid_spiketimes_check` column. Then you can query like:
        `meta_df.query("invalid_spiketimes_check == 'SUCCESS'")`
    inplace : if True, modify the dataframe in place. Otherwise, return a copy.

    # Returns
    meta_df: the modified dataframe

    """

    if not inplace:
        meta_df = meta_df.copy()

    if not np.all(meta_df["invalid_spiketimes_check"] == "SUCCESS"):
        log.warning("Some units already have invalid_spiketimes_check, we overwrite.")
        meta_df["invalid_spiketimes_check"] = "SUCCESS"

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
    # log.debug(f"After stationarity check: {_num_good(meta_df)}")

    # check the duration for single blocks
    idx = meta_df.query(
        f"stimulus == 'spontaneous'"
        + f" & recording_length < {0.9 * (target_length + transient_length)}"
    ).index
    meta_df.loc[idx, "invalid_spiketimes_check"] = "ERR_REC_LEN"

    # find stimuli that have two different blocks:
    stims_with_two_blocks = []
    for stim in meta_df["stimulus"].unique():
        if len(meta_df.query(f"stimulus == '{stim}'")["block"].unique()) > 1:
            stims_with_two_blocks.append(stim)

    # check total duration of two-block stimuli
    idx = meta_df.query(
        f"stimulus in {stims_with_two_blocks}"
        + f" & recording_length < {0.9 * (target_length / 2 + transient_length)}"
    ).index
    meta_df.loc[idx, "invalid_spiketimes_check"] = "ERR_REC_LEN"
    log.debug(f"After minmum-duration check: {_num_good(meta_df)}")

    # how many units did we exclude per stimulus?
    for stim in meta_df["stimulus"].unique():
        num_total = len(meta_df.query(f"stimulus == '{stim}'"))
        num_bad = len(
            meta_df.query(
                f"stimulus == '{stim}' & invalid_spiketimes_check != 'SUCCESS'"
            )
        )
        log.debug(f"Excluded {num_bad} / {num_total} units for stimulus {stim}")

    if trim:
        meta_df = meta_df.query("invalid_spiketimes_check == 'SUCCESS'")

    return meta_df


def merge_blocks(temp_df, target_length=1080, transient_length=360, inplace=True):
    """
    Merge the spiking data from two blocks (for those stimuli that have two blocks).

    best do this after filtering etc.

    # Returns
    meta_df: a new dataframe. to combine with the original one use
        `pd.concat([meta_df, meta_df2])`

    # Parameters
    meta_df: the metadata dataframe, data already loaded, checks performed.
        Also, we expect only two-block stimuli to be present.
    """

    new_rows = []
    dropped_units = 0

    if not inplace:
        meta_df = meta_df.copy()

    assert "spiketimes" in temp_df.columns, "call `load_spikes` first"

    try:
        groupby = temp_df.groupby(["session", "stimulus", "unit_id"])
    except ValueError:
        # this happens when the index is set to unit_id (and unit_id is still a column)
        # resetting is not great, it will modify the outer df which seems unexpected.
        temp_df = temp_df.reset_index(drop=True)
        groupby = temp_df.groupby(["session", "stimulus", "unit_id"])

    for df_idx, df in tqdm(groupby, desc="Merging blocks for units", total=len(groupby)):
        if len(df) == 1:
            # the unit was only recorded in one block
            dropped_units += 1
            continue
        elif len(df) != 2:
            raise ValueError("Expected two blocks for each stimulus")

        try:
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

        except IndexError:
            # if the unit has bad spike times or is shorter than our target lengths...
            # log.debug(f"Unit {df.iloc[0]['unit_id']} could not be merged.")
            dropped_units += 1
            continue

    res_df = pd.DataFrame(new_rows)
    log.debug(
        f"Did not merge {dropped_units} units (only found in one block)."
        f" {len(res_df['unit_id'].unique())} units remained."
    )

    temp_df = pd.concat([temp_df, res_df])
    temp_df.reset_index(inplace=True, drop=True)

    return temp_df


# ------------------------------------------------------------------------------ #
# pandas dataframe realted helpers
# ------------------------------------------------------------------------------ #


def load_metrics(meta_df, data_dir, inplace=False, csvs=None, cols=None):
    """
    Load metrics from the Allen Institute csv files,
    (brain_observatory_1.1_analysis_metrics.csv and
    functional_connectivity_analysis_metrics.csv)

    # Parameters
    - meta_df : dataframe holding the metadata, with at least a column "unit_id"
    - data_dir : directory holding the csv files
    - inplace : if True, modify the dataframe in place. Otherwise, return a copy.
    - csvs : list of csv filenames to load. default : None, standard file names
    - cols : list of columns to load.
        default : None -> ["g_dsi_dg", "image_selectivity_ns", "mod_idx_dg"]

    # Notes
    - focus is on numerical columns of the csv.
        non-numeric types not tested.
    """

    assert "unit_id" in meta_df.columns, "unit_id column missing, try `reset_index()`?"

    if not inplace:
        meta_df = meta_df.copy()

    if csvs is None:
        csvs = [
            "brain_observatory_unit_metrics_filtered.csv",
            "functional_connectivity_analysis_metrics.csv",
        ]

    # load dataframes. we only need columns for stimulus selectivity
    if cols is None:
        cols = ["g_dsi_dg", "image_selectivity_ns", "mod_idx_dg"]

    loaded_dfs = []
    loaded_csvs = []
    for csv in csvs:
        csv_path = os.path.abspath(os.path.join(data_dir, csv))
        df = _load_metrics_from_csv(csv_path, cols=cols)
        log.debug(f"Loaded columns {df.columns.to_list()} from {csv_path}")
        loaded_dfs.append(df)
        loaded_csvs.append(csv_path)

    # some sanity checks
    cols_in_multiple_dfs = []
    for col in cols:
        dfs_with_col = [df for df in loaded_dfs if col in df.columns]
        if len(dfs_with_col) == 0:
            raise ValueError(f"Column {col} not found in any of the loaded dataframes.")
        elif len(dfs_with_col) > 1:
            cols_in_multiple_dfs.append(col)
            log.info(f"Column {col} found in multiple dataframes.")

        if col in meta_df.columns and meta_df[col].isfinite().sum() > 0:
            raise ValueError(f"Column {col} already exists in meta_df, and has values.")

        meta_df[col] = np.nan

    # merge the dataframes, by unit_id. units likely occur multiple times in meta_df.
    # merging can be finicky, doing this manually may a bit slower but should be safe
    cols_copied = []
    for idx, df in enumerate(loaded_dfs):
        if len(df.columns) == 1 or len(df) == 0:
            # only unit_id or no rows
            continue
        same_units = meta_df["unit_id"].isin(df["unit_id"])  # not unique units but rows
        log.debug(f"Matched {same_units.sum()} rows from meta_df in {loaded_csvs[idx]}")

        # columns of interest, for this dataframe
        cois = [c for c in cols if c in df.columns]
        skip = [c for c in cois if c in cols_copied]
        if len(skip) > 0:
            log.info(
                f"2nd occurence of cols {skip}, ignoring values from {loaded_csvs[idx]}"
            )
        cois = [c for c in cois if c not in skip]

        if len(cois) > 0:
            for unit in meta_df["unit_id"].unique():
                if unit in df["unit_id"].values:
                    meta_df.loc[meta_df["unit_id"] == unit, cois] = df.loc[
                        df["unit_id"] == unit, cois
                    ].values

        cols_copied.extend(cois)

    for col in cols:
        if len(meta_df[col].dropna()) == 0:
            log.warning(f"Column {col} only has NaNs after merging.")

    return meta_df


def strict_merge_dfs_by_index(left, right):
    """
    Merge two columns of two similar dataframes, by index.
    - length must match
    - index must match (use `set_index()`)
    - columns that exist in both frames are checked to hold the same values
    """

    if not left.index.names == right.index.names:
        raise ValueError(
            f"Indices left {left.index.names} and right {right.index.names} do not match."
            " Try `set_index()` to match them."
        )

    assert left.index.is_unique, "Indices should be unique"
    assert right.index.is_unique, "Indices should be unique"

    log.debug(
        f"left length before data frame merge: {len(left)}, right length: {len(right)}."
        " They should match."
    )

    # make sure the index order is the same:
    right = right.reindex(left.index)

    # merge the dataframes
    cols_to_copy = right.columns.difference(left.columns)
    cols_redundant = left.columns.intersection(right.columns)
    log.debug(f"Copying columns: {cols_to_copy}")

    # check the values in the other columns match:
    for col in cols_redundant:
        num_diffs = np.sum(left[col] != right[col])
        if num_diffs > 0:
            log.warning(f"Column {col} differs in {num_diffs} rows.")

    merged_df = left.merge(right[cols_to_copy], left_index=True, right_index=True)

    return merged_df


# ------------------------------------------------------------------------------ #
# helpers for session handling etc.
# ------------------------------------------------------------------------------ #


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


def _load_metrics_from_csv(csv_path, cols=None):
    """
    Loads Allen Institute metrics from the specified csv file, keeping only the
    desired columns (None for all).

    cols may contain columns not found in the csv, these are ignored.
    """
    metric_df = pd.read_csv(csv_path, index_col=None)

    # rename columns to match our format
    metric_df.rename(columns={"ecephys_unit_id": "unit_id"}, inplace=True)
    assert "unit_id" in metric_df.columns

    # drop columns that we don't need
    if cols is not None:
        cols = cols.copy()
        assert isinstance(cols, list)
        cols = ["unit_id"] + cols
        cols_to_drop = [c for c in metric_df.columns if c not in cols]
        metric_df.drop(columns=cols_to_drop, inplace=True)

    return metric_df


# ------------------------------------------------------------------------------ #
# Saving the dataframe
# ------------------------------------------------------------------------------ #


def save_dataframe(meta_df, path, cols_to_skip=None):
    """
    Thin wrapper around pandas df.to_hdf
    """

    if cols_to_skip is None:
        cols_to_skip = []
    cols_to_save = [c for c in meta_df.columns if c not in cols_to_skip]
    df_to_save = meta_df[cols_to_save]
    df_to_save.to_hdf(path, key="meta_df", mode="w", complib="zlib", complevel=9)


# ------------------------------------------------------------------------------ #
# spike processing
# ------------------------------------------------------------------------------ #


def binned_spike_count(spiketimes, bin_size):
    """
    Get a number of spike counts in a time bin, per neuron.

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

    Notes
    -----
    - to aggregate across neurons, simply call `np.sum(counts, axis=0)` afterwards
    - pulling out individual neurons after they were binned together with other ones
        is not guaranteed to give the same result as binning separately, because
        the spike times are aligned to the first one in the block.
    """
    # type checking is easier here than in numba
    if len(spiketimes) == 0:
        return np.array([])

    # this is only needed because we want to use numba, which doesnt like xarray
    if isinstance(spiketimes, xr.DataArray):
        spiketimes = spiketimes.to_numpy()
    elif isinstance(spiketimes, list):
        if isinstance(spiketimes[0], xr.DataArray):
            spiketimes = [spiketimes[n_id].to_numpy() for n_id in range(len(spiketimes))]

    spiketimes = np.array(spiketimes)
    # now we are sure to have a numpy array. check it's 2d
    try:
        len(spiketimes[0])
    except TypeError:
        spiketimes = np.array([spiketimes])

    if len(spiketimes.shape) > 2:
        raise ValueError(
            "spiketimes must have at most 2 dimensions: (neuron, time), "
            f"found {len(spiketimes.shape)}. Consider `np.squeeze()`"
        )
    # but it might still be ragged, as in differnt length per neuron.
    # thats fine though.

    log.debug(
        f"Binning spiketimes: dtype {type(spiketimes)} with shape {spiketimes.shape}"
    )
    return _binned_spike_count(spiketimes, bin_size)


@jit(nopython=True, parallel=False, fastmath=False, cache=True)
def _binned_spike_count(spiketimes, bin_size):
    """
    lower level, here we rely on a 2d array of spiketimes,
    first dim neuron, second dim spiketimes, nan-padded.
    """
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
            # we have nan-padding. stop when that starts
            if not np.isfinite(t):
                break
            # align to the block-level t min (not the global one)
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
