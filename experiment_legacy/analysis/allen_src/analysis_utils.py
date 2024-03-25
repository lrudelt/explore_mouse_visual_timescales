import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd
import yaml
import random
import scipy.stats as st
import arviz as az
import h5py
import os

# necessary cont boxplot
import matplotlib.cbook as cbook
from matplotlib import rcParams

import logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s",
    level=logging.ERROR,
)
log = logging.getLogger(__name__)
log.setLevel("INFO")

# end

region_dict = {
    "cortex": [
        "VISp",
        "VISl",
        "VISrl",
        "VISam",
        "VISpm",
        "VIS",
        "VISal",
        "VISmma",
        "VISmmp",
        "VISli",
    ],
    "thalamus": [
        "LGd",
        "LD",
        "LP",
        "VPM",
        "TH",
        "MGm",
        "MGv",
        "MGd",
        "PO",
        "LGv",
        "VL",
        "VPL",
        "POL",
        "Eth",
        "PoT",
        "PP",
        "PIL",
        "IntG",
        "IGL",
        "SGN",
        "VPL",
        "PF",
        "RT",
    ],
    "hippocampus": ["CA1", "CA2", "CA3", "DG", "SUB", "POST", "PRE", "ProS", "HPF"],
    "midbrain": [
        "MB",
        "SCig",
        "SCiw",
        "SCsg",
        "SCzo",
        "PPT",
        "APN",
        "NOT",
        "MRN",
        "OP",
        "LT",
        "RPF",
        "CP",
    ],
}


def get_structures_map():
    color_palette = sns.color_palette()
    structures = ["VISp", "VISl", "VISrl", "VISal", "VISpm", "VISam", "LGd", "LP"]
    structures_names = ["V1", "LM", "RL", "AL", "PM", "AM", "LGN", "LP"]
    structures_colors = [
        color_palette[4],
        color_palette[0],
        color_palette[9],
        color_palette[8],
        color_palette[1],
        color_palette[3],
        color_palette[6],
        color_palette[2],
    ]
    # hierarchy score from https://github.com/AllenInstitute/neuropixels_platform_paper/blob/master/Figure3/Figure3.py
    hierarchy_scores = [-0.357, -0.093, -0.059, 0.152, 0.327, 0.441, -0.515, 0.105]

    # region_dict = {'cortex' : ['VISp', 'VISl', 'VISrl', 'VISam', 'VISpm', 'VIS', 'VISal','VISmma','VISmmp','VISli'],
    #              'thalamus' : ['LGd','LD', 'LP', 'VPM', 'TH', 'MGm','MGv','MGd','PO','LGv','VL',
    #               'VPL','POL','Eth','PoT','PP','PIL','IntG','IGL','SGN','VPL','PF','RT'],
    #              'hippocampus' : ['CA1', 'CA2','CA3', 'DG', 'SUB', 'POST','PRE','ProS','HPF'],
    #              'midbrain': ['MB','SCig','SCiw','SCsg','SCzo','PPT','APN','NOT','MRN','OP','LT','RPF','CP']}

    structures_map = {}
    for structure, name, color, hierarchy_score in zip(
        structures, structures_names, structures_colors, hierarchy_scores
    ):
        parent_structure = "none"
        for ps in region_dict:
            if structure in region_dict[ps]:
                parent_structure = ps
                break
        structures_map[structure] = {
            "name": name,
            "color": color,
            "parent_structure": ps,
            "hierarchy_score": hierarchy_score,
        }

    return structures_map


def get_stimuli_map(bo_analysis=False):
    if bo_analysis:  # allen_bo
        stimuli = ["natural_movie_three"]
        stimuli_names = ["movie"]
        stimuli_markers = ["*"]

        stimuli_map = {}
        for stimulus, name, marker in zip(stimuli, stimuli_names, stimuli_markers):
            stimuli_map[stimulus] = {"name": name, "marker": marker}
    else:  # allen_fc
        stimuli = ["natural_movie_one_more_repeats", "spontaneous"]
        stimuli_names = ["movie", "spont."]
        stimuli_markers = ["*", "s"]

        stimuli_map = {}
        for stimulus, name, marker in zip(stimuli, stimuli_names, stimuli_markers):
            stimuli_map[stimulus] = {"name": name, "marker": marker}

    return stimuli_map


def get_stimulus_blocks_map(bo_analysis=False):
    if bo_analysis:  # allen_bo
        stimulus_blocks = ["3.0", "6.0"]
        stimulus_blocks_names = ["Nat. Mov. Block 1", "Nat. Mov. Block 2"]
        stimuli_markers = ["*", "*"]

        stimulus_blocks_map = {}
        for _stimulus_blocks, name, marker in zip(
            stimulus_blocks, stimulus_blocks_names, stimuli_markers
        ):
            stimulus_blocks_map[_stimulus_blocks] = {"name": name, "marker": marker}
    else:  # allen_fc
        stimulus_blocks = ["3.0,8.0", "null"]
        stimulus_blocks_names = ["Natural Movie", "No Stimulus"]
        stimuli_markers = ["*", "s"]

        stimulus_blocks_map = {}
        for _stimulus_blocks, name, marker in zip(
            stimulus_blocks, stimulus_blocks_names, stimuli_markers
        ):
            stimulus_blocks_map[_stimulus_blocks] = {"name": name, "marker": marker}

    return stimulus_blocks_map


def get_quality_metrics(analysis):
    if analysis == "allen_fc" or analysis == "allen_bo":
        presence_ratio_minimum = 0.9
        amplitude_cutoff_maximum = 0.01
        isi_violations_maximum = 0.5
    else:
        presence_ratio_minimum = -np.inf
        amplitude_cutoff_maximum = 0.01
        isi_violations_maximum = 0.5

    return presence_ratio_minimum, amplitude_cutoff_maximum, isi_violations_maximum


def get_analysis_data(
    csv_file_name,
    analysis,
    estimation_method="shuffling",
    analysis_metrics=None,
    mre_stats_file_name=None,
    pp_stats_file_name=None,
    m_ar_stats_file_name=None,
    fit_selection_criterion="bic",
):

    data = pd.read_csv(csv_file_name, na_values="-")
    data.rename(columns={"#analysis_num": "analysis_num"}, inplace=True)
    labels = np.array([label.split(";") for label in data["label"].values])
    if analysis.startswith("allen_fc"):
        labels = np.array(
            [
                label.replace("3.0;8.0", "3.0,8.0").split(";")
                for label in data["label"].values
            ]
        )
        data.insert(2, "session_id", labels[:, 0])
        data.insert(3, "unit", np.array(labels[:, 1], dtype=int))
        data.insert(4, "stimulus", labels[:, 2])
        data.insert(5, "stimulus_blocks", labels[:, 3])
        data.insert(6, "ecephys_structure_acronym", labels[:, 4])
    elif analysis == "allen_bo":
        data.insert(2, "session_id", labels[:, 0])
        data.insert(3, "unit", np.array(labels[:, 1], dtype=int))
        data.insert(4, "stimulus", labels[:, 2])
        data.insert(
            5,
            "stimulus_blocks",
            np.array(
                [
                    "{},{}".format(block1, block2)
                    for block1, block2 in zip(labels[:, 3], labels[:, 4])
                ]
            ),
        )
        data.insert(6, "ecephys_structure_acronym", labels[:, 5])
    elif analysis == "hx":
        data.insert(2, "data_file_name", labels[:, 0])
        data.insert(3, "neuron_num", np.array(labels[:, 1], dtype=int))
        data.insert(4, "kin", np.array(labels[:, 2], dtype=float))
        data.insert(5, "seed", np.array(labels[:, 3], dtype=int))
        data.insert(6, "sigma_stim", np.array(labels[:, 4], dtype=float))
    elif analysis == "brian":
        data.insert(
            2,
            "data_file_name",
            np.array(
                [
                    f_name[len("/data.nst/bcramer/simulation/timescales/") :]
                    for f_name in labels[:, 0]
                ]
            ),
        )
        data.insert(3, "neuron_num", np.array(labels[:, 1], dtype=int))
        data.insert(4, "kin", np.array(labels[:, 2], dtype=float))
        data.insert(5, "seed", np.array(labels[:, 3], dtype=int))
        data.insert(6, "sigma_stim", np.array(labels[:, 4], dtype=float))
    else:
        data.insert(2, "session_id", labels[:, 0])
        data.insert(3, "unit", np.array(labels[:, 1], dtype=int))
        data.insert(4, "stimulus", labels[:, 2])
        data.insert(5, "ecephys_structure_acronym", labels[:, 3])
    data.drop("label", axis=1, inplace=True)

    Ts, Rs, tau_Rs = data[
        [
            "T_D_{}".format(estimation_method),
            "R_tot_{}".format(estimation_method),
            "tau_R_{}".format(estimation_method),
        ]
    ].values.T
    data.insert(6, "T_D", Ts)
    data.insert(7, "R_tot", Rs)
    data.insert(8, "tau_R", tau_Rs)
    data.insert(9, "log_T", np.log10(Ts))
    data.insert(10, "log_R", np.log10(Rs))
    data.insert(11, "log_tau_R", np.log10(tau_Rs))

    fr = data["firing_rate"].values
    data.insert(12, "log_fr", np.log10(fr))

    # cast object columns to string
    for col in data.columns:
        if data[col].dtype == object:
            log.debug("casting column {} to string".format(col))
            data[col] = data[col].astype(str)

    if analysis.startswith("allen"):

        def get_parent_structure(structure):
            parent_structure = np.nan
            for ps in region_dict:
                if structure in region_dict[ps]:
                    parent_structure = ps
                    break
            return parent_structure

        data.insert(
            13,
            "parent_structure",
            data["ecephys_structure_acronym"].apply(get_parent_structure),
        )

        def get_structure_group(structure):
            structure_group_dict = {
                "Thalamus": ["LGd", "LP"],
                "V1": ["VISp"],
                "higher cortical": ["VISl", "VISrl", "VISal", "VISpm", "VISam"],
            }
            structure_group = np.nan
            for sg in structure_group_dict:
                if structure in structure_group_dict[sg]:
                    structure_group = sg
                    break
            return structure_group

        data.insert(
            14,
            "structure_group",
            data["ecephys_structure_acronym"].apply(get_structure_group),
        )

        def get_hierarchy_score(structure):
            hierarchy_score_dict = {
                "VISp": -0.357,
                "VISl": -0.093,
                "VISrl": -0.059,
                "VISal": 0.152,
                "VISpm": 0.327,
                "VISam": 0.441,
                "LGd": -0.515,
                "LP": 0.105,
            }
            hierarchy_score = np.nan
            if structure in hierarchy_score_dict:
                hierarchy_score = hierarchy_score_dict[structure]
            return hierarchy_score

        data.insert(
            15,
            "hierarchy_score",
            data["ecephys_structure_acronym"].apply(get_hierarchy_score),
        )

        if "spontaneous" in data["stimulus"].unique():
            data_sp = data[df_filter(data, stimuli="spontaneous")]
            R_sp = dict(zip(data_sp["unit"].values, data_sp["R_tot"].values))
            T_sp = dict(zip(data_sp["unit"].values, data_sp["T_D"].values))

            for unit in data.unit.values:
                if not unit in R_sp:
                    R_sp[unit] = np.nan
                if not unit in T_sp:
                    T_sp[unit] = np.nan

            data["R_sp"] = data.unit.replace(R_sp).values
            data["T_sp"] = data.unit.replace(T_sp).values

            data["deltaR_sp"] = data["R_sp"].values - data["R_tot"].values
            data["deltaT_sp"] = data["T_sp"].values - data["T_D"].values

        if not analysis_metrics is None:
            # ['waveform_PT_ratio' 'waveform_amplitude' 'amplitude_cutoff'
            #  'cumulative_drift' 'd_prime' 'waveform_duration' 'ecephys_channel_id'
            #  'firing_rate' 'waveform_halfwidth' 'isi_violations' 'isolation_distance'
            #  'L_ratio' 'max_drift' 'nn_hit_rate' 'nn_miss_rate' 'presence_ratio'
            #  'waveform_recovery_slope' 'waveform_repolarization_slope'
            #  'silhouette_score' 'snr' 'waveform_spread' 'waveform_velocity_above'
            #  'waveform_velocity_below' 'ecephys_probe_id' 'local_index'
            #  'probe_horizontal_position' 'probe_vertical_position'
            #  'anterior_posterior_ccf_coordinate' 'dorsal_ventral_ccf_coordinate'
            #  'left_right_ccf_coordinate' 'ecephys_structure_id'
            #  'ecephys_structure_acronym' 'ecephys_session_id' 'lfp_sampling_rate'
            #  'name' 'phase' 'sampling_rate' 'has_lfp_data' 'date_of_acquisition'
            #  'published_at' 'specimen_id' 'session_type' 'age_in_days' 'sex'
            #  'genotype' 'c50_dg' 'area_rf' 'fano_dg' 'fano_fl' 'fano_ns' 'fano_rf'
            #  'fano_sg' 'f1_f0_dg' 'g_dsi_dg' 'g_osi_dg' 'g_osi_sg' 'width_rf'
            #  'height_rf' 'azimuth_rf' 'mod_idx_dg' 'p_value_rf' 'pref_sf_sg'
            #  'pref_tf_dg' 'run_mod_dg' 'run_mod_fl' 'run_mod_ns' 'run_mod_rf'
            #  'run_mod_sg' 'pref_ori_dg' 'pref_ori_sg' 'run_pval_dg' 'run_pval_fl'
            #  'run_pval_ns' 'run_pval_rf' 'run_pval_sg' 'elevation_rf' 'on_screen_rf'
            #  'pref_image_ns' 'pref_phase_sg' 'firing_rate_dg' 'firing_rate_fl'
            #  'firing_rate_ns' 'firing_rate_rf' 'firing_rate_sg' 'on_off_ratio_fl'
            #  'time_to_peak_fl' 'time_to_peak_ns' 'time_to_peak_rf' 'time_to_peak_sg'
            #  'pref_sf_multi_sg' 'pref_tf_multi_dg' 'sustained_idx_fl'
            #  'pref_ori_multi_dg' 'pref_ori_multi_sg' 'pref_phase_multi_sg'
            #  'image_selectivity_ns' 'pref_images_multi_ns' 'lifetime_sparseness_dg'
            #  'lifetime_sparseness_fl' 'lifetime_sparseness_ns'
            #  'lifetime_sparseness_rf' 'lifetime_sparseness_sg' 'fano_dm' 'run_mod_dm'
            #  'pref_dir_dm' 'run_pval_dm' 'pref_speed_dm' 'firing_rate_dm'
            #  'time_to_peak_dm' 'pref_dir_multi_dm' 'pref_speed_multi_dm'
            #  'lifetime_sparseness_dm']
            merge_df = analysis_metrics[
                ["p_value_rf", "fano_dg", "area_rf", "snr", "firing_rate_dg"]
            ]

            def get_sign_rf(x):
                if x < 0.01:
                    return True
                elif x >= 0.01:
                    return False
                else:
                    return np.nan

            merge_df.insert(2, "sign_rf", merge_df["p_value_rf"].apply(get_sign_rf))

            # print(sum((merge_df['p_value_rf'] < 0.01) &
            #           ~(merge_df['p_value_rf'] >= 0.01))) # 9973
            # print(sum((merge_df['p_value_rf'] >= 0.01) &
            #           ~( merge_df['p_value_rf'] < 0.01))) # 76

            data = data.merge(merge_df, left_on="unit", right_index=True)

    def blocks_formatter(x):
        _x = x.split(";")
        if type(_x) == list:
            return ",".join(str(x).split(";"))
        else:
            return str(x)

    if not mre_stats_file_name is None:
        #session_id,unit,stimulus,stimulus_blocks,mre_tau,mre_A,mre_O,mre_m,mre_ssres,mre_bic_passed,bin_size,dtunit,tmin,tmax

        if analysis.startswith('allen'):
            _data = pd.read_csv(
                mre_stats_file_name,
                na_values='-',
                converters={'session_id': str, 'unit':int},
            )#, 'stimulus_blocks': blocks_formatter})
            # _data.rename(columns={'#session_id': 'session_id'}, inplace=True)
            # _data['session_id'] = _data['session_id'].astype(str)
            # _data['unit'] = _data['unit'].astype(int)
            # _data['stimulus_blocks'] = _data['stimulus_blocks'].astype(str)
            _data["stimulus_blocks"]= _data["stimulus_blocks"].fillna("null")
            # depending on the pandas version (?) it might pick up the `#` character
            # in `#session_id`
            if "#session_id" in _data.columns:
                _data.rename(columns={"#session_id": "session_id"}, inplace=True)

            # workaround for unmatching datatypes
            # (session_id as object in data and int in _data)
            data["session_id"] = data["session_id"].astype(int)

            # stimuls block naming inconsistencies
            _data["stimulus_blocks"] = _data["stimulus_blocks"].apply(blocks_formatter)

            # cast objects to string
            for col in _data.columns:
                if _data[col].dtype == object:
                    log.debug(f"casting {col} to string")
                    _data[col] = _data[col].astype(str)

            log.debug(f"merging mre_stats_file_name file: {mre_stats_file_name}")
            log.debug(f"mre_stats_file_name has length {len(_data)}")
            log.debug(f"length of data before merge: {len(data)}")

            data = data.merge(
                _data,
                left_on=['session_id', 'unit', 'stimulus', 'stimulus_blocks'],
                right_on=['session_id', 'unit', 'stimulus', 'stimulus_blocks'],
                how='left',
                validate='one_to_one',
            )

            log.debug(f"length of data after merge: {len(data)}")

            # print("merged")
            # print(data[["tau_two_timescales", "stimulus"]].dtypes)
            # if mre_stats_file_name.endswith('_two_timescales.csv'):
                # if fit_selection_criterion == 'aic':
                #     # select units where the the offset fit has better aic score
                #     offset_selection = data['aic_offset'] < data['aic_two_timescales']
                #     # use the aic score of the better offset fits to compare against constant fit
                #     # data.loc[offset_selection,'aic_passed_two_timescales'] = data[offset_selection]['aic_passed_offset'].values
                #     data[offset_selection]['aic_passed_two_timescales'] = data[offset_selection]['aic_passed_offset']
                # elif fit_selection_criterion == 'bic':
                #     # select units where the the offset fit has better bic score
                #     offset_selection = data['bic_offset'] < data['bic_two_timescales']
                #     # use the bic score of the better offset fits to compare against constant fit
                #     # data.loc[offset_selection,'bic_passed_two_timescales'] = data[offset_selection]['bic_passed_offset'].values
                #     data[offset_selection]['bic_passed_two_timescales'] = data[offset_selection]['bic_passed_offset']
                # print("For {} out of {} intrinsic timescale fits a single timescale and constant offset were selected.".format(len(data[offset_selection]['tau_two_timescales']), len(data)))
                # data.loc[offset_selection,'tau_two_timescales'] = data[offset_selection]['tau_offset'].values
                # data[offset_selection]['tau_two_timescales'] = data[offset_selection]['tau_offset']
            data.insert(12, 'log_mre_tau', data['tau_two_timescales'].apply(np.log10))
            # else:
                # data.insert(12, 'log_mre_tau', data['mre_tau'].apply(np.log10))
        elif analysis == 'hx' or analysis == 'brian':
            with pd.HDFStore(mre_stats_file_name) as store:
                _data = store["data"]
            _data.rename(columns={"unit": "neuron_num"}, inplace=True)
            _data.insert(
                4, "mre_tau", _data["tau"].apply(lambda x: x / 1000)
            )  # data in the h5 file are in ms
            _data.drop("tau", axis=1, inplace=True)
            _data.insert(
                5, "mre_bic_passed", _data["BIC test"].apply(lambda x: x == "passed")
            )
            _data.drop("BIC test", axis=1, inplace=True)
            _data.insert(6, "log_mre_tau", _data["mre_tau"].apply(np.log10))

            data = data.merge(
                _data,
                left_on=["neuron_num", "kin", "seed"],
                right_on=["neuron_num", "kin", "seed"],
                how="right",
            )

    if not pp_stats_file_name is None:
        # session_id,unit,stimulus,stimulus_blocks,firing_rate_1,median_ISI_1,CV_1,firing_rate_2,median_ISI_2,CV_2,firing_rate,median_ISI,CV

        _data = pd.read_csv(
            pp_stats_file_name,
            na_values="-",
            converters={
                "#session_id": str,
                "unit": int,
                "stimulus_blocks": blocks_formatter,
            },
        )
        _data.rename(columns={"#session_id": "session_id"}, inplace=True)
        _data.drop("firing_rate", axis=1, inplace=True)

        data = data.merge(
            _data,
            left_on=["session_id", "unit", "stimulus", "stimulus_blocks"],
            right_on=["session_id", "unit", "stimulus", "stimulus_blocks"],
            how="left",
        )

        data.insert(13, "log_median_ISI", data["median_ISI"].apply(np.log10))

    if not m_ar_stats_file_name is None:
        _data = pd.read_csv(m_ar_stats_file_name, na_values="-")

        data = data.merge(
            _data[["kin", "seed", "m_ar"]],
            left_on=["kin", "seed"],
            right_on=["kin", "seed"],
            how="left",
        )

    if analysis.startswith("allen"):
        _units = []
        _stimulus_blocks = np.unique(data["stimulus_blocks"].values)
        for unit in np.unique(data["unit"].values):
            if len(data[data["unit"] == unit]) == len(_stimulus_blocks):
                _units += [unit]
        data = data[data["unit"].isin(_units)]

    return data


def get_histdep_data(histdep_file_name, analysis_data):
    histdep_d = pd.read_csv(histdep_file_name, na_values="-")
    histdep_d.rename(columns={"#T": "T"}, inplace=True)
    histdep_d = histdep_d[
        histdep_d["analysis_num"].isin(np.unique(analysis_data["analysis_num"]))
    ]
    histdep_d = histdep_d.merge(
        analysis_data[["analysis_num", "unit"]],
        left_on="analysis_num",
        right_on="analysis_num",
        how="left",
    )

    return histdep_d


def get_analysis_metrics(cache, analysis):
    (
        presence_ratio_minimum,
        amplitude_cutoff_maximum,
        isi_violations_maximum,
    ) = get_quality_metrics(analysis)

    analysis_metrics1 = cache.get_unit_analysis_metrics_by_session_type(
        "brain_observatory_1.1",
        amplitude_cutoff_maximum=amplitude_cutoff_maximum,
        presence_ratio_minimum=presence_ratio_minimum,
        isi_violations_maximum=isi_violations_maximum,
    )

    analysis_metrics2 = cache.get_unit_analysis_metrics_by_session_type(
        "functional_connectivity",
        amplitude_cutoff_maximum=amplitude_cutoff_maximum,
        presence_ratio_minimum=presence_ratio_minimum,
        isi_violations_maximum=isi_violations_maximum,
    )

    analysis_metrics = pd.concat([analysis_metrics1, analysis_metrics2], sort=False)

    return analysis_metrics


def get_timescale_metrics():
    # timescale metrics downloaded from
    # https://github.com/AllenInstitute/neuropixels_platform_paper/blob/master/data/timescale_metrics.csv
    return pd.read_csv("timescale_metrics.csv")


def get_expected_number_of_neurons(analysis, structures=[], stimuli=[], measure=None):
    num_neurons = 0
    if analysis.startswith("allen_fc"):
        if "cortex" in structures:
            num_neurons += 4763
        if "thalamus" in structures:
            num_neurons += 785
        if analysis == "allen_fc_single_blocks":
            num_neurons *= 2
        num_neurons *= len(stimuli)
    elif analysis == "allen_bo":
        if "cortex" in structures:
            num_neurons += 4882
        if "thalamus" in structures:
            num_neurons += 1311
        num_neurons *= len(stimuli)
    elif analysis == "hx":
        if measure == "tau_C":
            num_neurons += 255784
        else:
            num_neurons += 12162
    elif analysis == "brian":
        if measure == "tau_C":
            num_neurons += 140584
        else:
            num_neurons += 6864

    return num_neurons


def get_data_filter(data, measure, analysis="allen", fit_selection_criterion="aic"):
    if measure == "tau_R":
        selection = data[measure] >= 0
    elif measure == "log_tau_R":
        selection = data[measure] > -np.inf
    elif measure == "R_tot":
        selection = data[measure] >= 0
    elif measure == "mre_tau":
        selection = data[measure] > 0
        selection &= data[measure] < 10
        if fit_selection_criterion == "bic":
            selection &= data["mre_bic_passed"].values.astype(bool)
        # selection &= data[measure] < 2.5
        # selection &= (data[measure] > np.nanpercentile(data[measure].values, 2.5)) \
        #     & (data[measure] < np.nanpercentile(data[measure].values, 97.5))
    elif measure == "tau_two_timescales":
        selection = data[measure] > 0
        selection &= data[measure] < 10
        if fit_selection_criterion == "aic":
            selection &= data["aic_passed_two_timescales"].values.astype(bool)
        elif fit_selection_criterion == "bic":
            selection &= data["bic_passed_two_timescales"].values.astype(bool)
    elif measure == "log_mre_tau":
        # selection = data['mre_bic_passed'].values
        # if analysis.startswith("allen"):
        selection = data[measure] > -np.inf
        selection &= data[measure] < np.log10(10)
        if fit_selection_criterion == "aic":
            selection &= data["aic_passed_two_timescales"].values.astype(bool)
        elif fit_selection_criterion == "bic":
            selection &= data["bic_passed_two_timescales"].values.astype(bool)
    elif measure == "CV":
        selection = data[measure] < 5
        # selection &= data[measure] < np.log10(2.5)
        # selection &= (data[measure] > np.nanpercentile(data[measure].values, 2.5)) \
        #     & (data[measure] < np.nanpercentile(data[measure].values, 97.5))
    else:
        selection = np.array([True] * len(data))
        print("warning, no filter found for measure")

    data_len = len(selection)
    filtered_data_len = sum(selection)

    print(
        "{} data points dropped by filter ({:.2f}% of data) for measure {}".format(
            data_len - filtered_data_len,
            (1 - filtered_data_len / data_len) * 100,
            measure,
        )
    )
    return selection


def df_filter(
    data,
    structures=None,
    sessions=None,
    units=None,
    stimuli=None,
    sign_rf=None,
    T0=None,
    tmin=None,
):
    df_filter = np.array([True] * len(data))

    filters = {
        "ecephys_structure_acronym": structures,
        "session_id": sessions,
        "unit": units,
        "stimulus": stimuli,
        "sign_rf": sign_rf,
        "timescale_minimum_past_range": T0,
        "tmin": tmin,
    }

    log.debug(f"length before filtering: {len(data)}")
    for name, values in filters.items():
        if not values == None:
            if type(values) == list or type(values) == np.ndarray:
                df_filter &= data[name].isin(values)
            else:
                df_filter &= data[name] == values
        log.debug(f"filtering {name}, new length: {len(data[df_filter])}")

    # if not sign_rf is None:
    #     if sign_rf == True:
    #         df_filter &= (data['p_value_rf'] < 0.01)
    #     if sign_rf == False:
    #         df_filter &= (data['p_value_rf'] >= 0.01)

    return df_filter


def get_default_plot_settings():
    plot_settings = {}

    plot_settings["imgdir"] = "../img"
    plot_settings["img_format"] = "pdf"
    plot_settings["textwidth"] = 5.787402103
    plot_settings["panel_width"] = 0.37 * plot_settings["textwidth"]
    plot_settings["panel_height"] = 2.9
    plot_settings["panel_size"] = (
        plot_settings["panel_width"],
        plot_settings["panel_height"],
    )
    plot_settings["rcparams"] = {
        "axes.labelsize": 11,
        "font.size": 11,
        "font.family": "serif",
        "font.serif": "Computer Modern Roman",
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        # transparent figure bg
        "savefig.facecolor" : (0.0, 0.0, 0.0, 0.0),
        "axes.facecolor" : (1.0, 0.0, 0.0, 0.0),
        # 'text.usetex': True,
            # 'figure.figsize': [4.6299216824, 3]  # 0.8 * width
        "figure.figsize": [plot_settings["panel_width"], plot_settings["panel_height"]],
    }


    return plot_settings


def save_plot(plot_settings, file_name, allen_bo=False, stimulus=None, measure=""):
    if allen_bo:
        _bo = "_bo"
    else:
        _bo = ""

    if stimulus == "movie":
        _stimulus = "_nm"
    elif stimulus == "spontaneous":
        _stimulus = "_sp"
    else:
        _stimulus = ""

    if measure == "":
        _measure = ""
    else:
        _measure = f"_{measure}"

    plt.savefig(
        "{}/{}{}{}{}.{}".format(
            plot_settings["imgdir"],
            file_name,
            _bo,
            _stimulus,
            _measure,
            plot_settings["img_format"],
        ),
        bbox_inches="tight",
        dpi=300,
    )


def make_plot_pretty(ax, grid=True):
    ax.tick_params(axis="x", top=False)
    ax.tick_params(axis="y", right=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(axis="x", direction="out")
    ax.tick_params(axis="y", length=0)

    for spine in ax.spines.values():
        spine.set_position(("outward", 5))

    ax.set_axisbelow(True)
    if grid == True:
        ax.grid(axis="y", color="0.9", linestyle="-", linewidth=1, zorder=0)


def make_boxplot_pretty(ax):
    ax.tick_params(axis="x", top=False)
    ax.tick_params(axis="y", right=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(axis="x", direction="out")
    ax.tick_params(axis="y", length=0)

    for spine in ax.spines.values():
        spine.set_position(("outward", 5))

    ax.set_axisbelow(True)

    ax.grid(axis="y", color="0.9", linestyle="-", linewidth=1)

    for tick in ax.get_xticklabels():
        tick.set_rotation(25)
        tick.set_horizontalalignment("right")


def get_outliers(data):
    med = np.quantile(data, 0.5)
    q1 = np.quantile(data, 0.25)
    q3 = np.quantile(data, 0.75)
    IQR = q3 - q1
    left = q1 - 1.5 * IQR
    right = q3 + 1.5 * IQR
    is_outlier_range = [
        not left < x < right for x in data
    ]  # for each datum: 1 if outlier else 0
    data_outliers = [
        datum for datum, is_outlier in zip(data, is_outlier_range) if is_outlier
    ]
    data_no_outliers = [
        datum for datum, is_outlier in zip(data, is_outlier_range) if not is_outlier
    ]

    return data_outliers, data_no_outliers


def plot_boxplots(
    ax, data, labels, colors, plot_data=True, plot_outliers=True, alpha_data=1
):
    bplot = ax.boxplot(
        data,
        positions=range(len(data)),
        patch_artist=True,
        labels=labels,
        showfliers=False,
        zorder=3,
    )

    for i, dataset in enumerate(data):
        data_outliers, data_no_outliers = get_outliers(dataset)

        if plot_data:
            ax.plot(
                [i] * len(data_no_outliers),  # same index for all data points
                data_no_outliers,
                "x",
                ms=10,
                markeredgewidth=3,
                color=colors[i],
                zorder=2,
                alpha=alpha_data,
            )
        if plot_outliers:
            ax.plot(
                [i] * len(data_outliers),
                data_outliers,
                "o",
                ms=10,
                markeredgewidth=3,
                color=colors[i],
                zorder=2,
                markerfacecolor="None",
                alpha=alpha_data,
            )

    for patch, color in zip(bplot["boxes"], colors):
        # patch.set_facecolor(color)
        patch.set_facecolor("white")
        # patch.set_alpha(0.4)
    for element in bplot["medians"]:
        # element.set_color('white')
        element.set_color("black")


def get_whisker_top(d):
    q1 = np.quantile(d, 0.25)
    q3 = np.quantile(d, 0.75)
    IQR = q3 - q1
    return q3 + 1.5 * IQR


def plot_hdi(
    y,
    pos=None,
    hdi_prob=0.95,
    colors=None,
    labels=None,
    structures_map=None,
    linewidth=2,
    ax=None,
    vertical=False,
):
    if ax is None:
        _, ax = plt.subplots()

    if pos is None:
        if not structures_map is None:
            pos = {}
            for _y in y:
                pos[_y] = structures_map[_y]["hierarchy_score"]
        elif not type(pos) == dict:
            pos = {}
            for _y, _pos in zip(y, range(len(y))):
                pos[_y] = _pos

    if colors is None:
        if not structures_map is None:
            colors = {}
            for _y in y:
                colors[_y] = structures_map[_y]["color"]
        elif not type(colors) == dict:
            colors = {}
            for _y in y:
                colors[_y] = "k"

    if labels is None:
        if not structures_map is None:
            labels = {}
            for _y in y:
                labels[_y] = structures_map[_y]["name"]
        elif not type(labels) == dict:
            labels = {}
            for _y in y:
                labels[_y] = _y

    for _y in y:
        values = az.hdi(y[_y], hdi_prob, multimodal=False)
        med = np.percentile(y[_y], 50)

        if vertical:
            ax.plot(
                values,
                [pos[_y]] * len(values),
                lw=linewidth,
                color=colors[_y],
                # solid_capstyle="butt",
                label=labels[_y],
            )
            ax.plot(
                med,
                pos[_y],
                "o",
                color="white",
                mec=colors[_y]
                # solid_capstyle="butt",
            )
        else:
            ax.plot(
                [pos[_y]] * len(values),
                values,
                lw=linewidth,
                color=colors[_y],
                # solid_capstyle="butt",
                label=labels[_y],
            )
            ax.plot(
                pos[_y],
                med,
                "o",
                color="white",
                mec=colors[_y]
                # solid_capstyle="butt",
            )

    return ax


# cf https://github.com/arviz-devs/arviz
def plot_posterior(
    data,
    var_names=None,
    filter_vars=None,
    transform=None,
    coords=None,
    figsize=None,
    textsize=None,
    hdi=True,
    hdi_prob=0.95,
    multimodal=False,
    round_to=2,
    point_estimate="median",
    group="posterior",
    rope=None,
    ref_val=None,
    kind="kde",
    bw=4.5,
    bins=None,
    ax=None,
    show=None,
    credible_interval=None,
    **kwargs,
):

    data = az.data.convert_to_dataset(data, group="posterior")
    if transform is not None:
        data = transform(data)
    coords = {}

    plotters = az.plots.plot_utils.filter_plotters_list(
        list(
            az.plots.plot_utils.xarray_var_iter(
                az.utils.get_coords(data, coords), var_names=var_names, combined=True
            )
        ),
        "plot_posterior",
    )
    length_plotters = len(plotters)
    rows, cols = az.plots.plot_utils.default_grid(length_plotters)

    (
        figsize,
        ax_labelsize,
        titlesize,
        xt_labelsize,
        linewidth,
        _,
    ) = az.plots.plot_utils._scale_fig_size(figsize, textsize, rows, cols)
    kwargs = az.plots.plot_utils.matplotlib_kwarg_dealiaser(kwargs, "plot")
    kwargs.setdefault("linewidth", linewidth)

    for (var_name, selection, x), _ax in zip(plotters, np.ravel(ax)):
        values = x.flatten()

        # density

        az.plot_kde(
            values,
            bw=bw,
            fill_kwargs={"alpha": kwargs.pop("fill_alpha", 0)},
            plot_kwargs=kwargs,
            ax=_ax,
            rug=False,
            show=False,
        )

        _ax.yaxis.set_ticks([])
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)
        _ax.spines["left"].set_visible(False)
        _ax.spines["bottom"].set_visible(True)
        _ax.xaxis.set_ticks_position("bottom")
        _ax.tick_params(
            axis="x",
            direction="out",
            width=1,
            length=3,
            color="0.5",
            labelsize=xt_labelsize,
        )
        _ax.spines["bottom"].set_color("0.5")

        plot_height = _ax.get_ylim()[1]

        # hdi
        if hdi:
            hdi_probs = az.stats.stats.hdi(
                values, hdi_prob=hdi_prob, multimodal=multimodal
            )  # type: np.ndarray

            for hdi_i in np.atleast_2d(hdi_probs):
                _ax.plot(
                    hdi_i,
                    (plot_height * 0.02, plot_height * 0.02),
                    lw=linewidth * 2,
                    color="k",
                    solid_capstyle="butt",
                )

            # median

            point_value = az.plots.plot_utils.calculate_point_estimate(
                point_estimate, values, bw
            )

            _ax.plot(point_value, plot_height * 0.02, "o", color="red")


def plot_pairplot(
    data,
    pairplot_measures_x,
    axes,
    lims,
    ticks,
    labels,
    fmts,
    pairplot_measures_y=None,
    levels=5,
):
    if pairplot_measures_y == None:
        pairplot_measures_y = pairplot_measures_x

    assert len(pairplot_measures_x) == len(pairplot_measures_y)

    for y, axes_row in zip(pairplot_measures_y, axes):
        for x, ax in zip(pairplot_measures_x, axes_row):
            if x == y:
                vals = data[x].values
                x_lo = min(vals[vals > -np.inf])
                ax.hist(
                    vals,
                    bins=np.linspace(x_lo, max(vals), 20),
                    density=True,
                    color="k",
                    lw=0,
                    zorder=-1,
                    rwidth=0.9,
                )
                make_plot_pretty(ax)
                ax.set_xlim(lims[x])
            else:
                x_vals = data[x].values[
                    (data[x].values > -np.inf) & (data[y].values > -np.inf)
                ]
                y_vals = data[y].values[
                    (data[x].values > -np.inf) & (data[y].values > -np.inf)
                ]
                ax.scatter(
                    x_vals,
                    y_vals,
                    marker=".",
                    color="k",
                    edgecolors="0.1",
                    alpha=0.01,
                    rasterized=True,
                )
                sns.kdeplot(x_vals, y_vals, ax=ax, color="white", levels=levels)
                make_plot_pretty(ax)
                ax.grid(axis="both", color="0.9", linestyle="-", linewidth=1)
                ax.set_xlim(lims[x])
                ax.set_ylim(lims[y])
                ax.set_yticks(ticks=ticks[y])
            ax.set_xticks(ticks=ticks[x])
        # format_ax(axes_row[0], y, 'y', tight_layout=True)

        axes_row[0].set_ylabel(labels[y])
        if not fmts[y] is None:
            axes_row[0].yaxis.set_major_formatter(ticker.FuncFormatter(fmts[y]))

        axes_row[1].set_yticklabels([])
        axes_row[2].set_yticklabels([])

    for ax, x in zip(axes[-1], pairplot_measures_x):  # was: axes[2]
        #     format_ax(ax, x, 'x', tight_layout=True)
        ax.set_xlabel(labels[x])
        if not fmts[x] is None:
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmts[x]))
        ax.set_xticks(ticks=ticks[x])

    for axes_row in axes[:-1]:  # was: axes[:2]
        for ax in axes_row:
            ax.set_xticklabels([])

    if pairplot_measures_x == pairplot_measures_y:
        # hack for diagonals
        for ax, y in zip(
            [axes[i][i] for i in range(len(pairplot_measures_y))], pairplot_measures_y
        ):
            y_lo, y_hi = ax.get_ylim()
            lim_lo, lim_hi = lims[y]
            fake_ticks = (
                (np.array(ticks[y]) - (lim_lo - y_lo)) * (y_hi - y_lo) / (lim_hi - lim_lo)
            )
            ax.set_yticks(ticks=fake_ticks)
            ax.grid(False)
        # axes[0][0].set_yticklabels(ticks[pairplot_measures[0]])
        y = pairplot_measures_y[0]
        if not fmts[y] is None:
            _labels = [fmts[y](tick, None) for tick in ticks[y]]
        else:
            _labels = ticks[y]
        axes[0][0].set_yticklabels(_labels)


# adapted from https://github.com/AllenInstitute/AllenSDK/blob/467a9725d716bb8435cda65e0fe5dce757a5a4da/allensdk/brain_observatory/ecephys/visualization/__init__.py
class _VlPlotter:
    def __init__(self, ax, num_objects, colors):
        self.ii = 0
        self.ax = ax
        self.num_objects = num_objects
        self.colors = colors

    def __call__(self, gb):
        # print(self.ii, gb['unit_id'])
        low = self.ii
        high = self.ii + 1

        # color = self.colors[self.ii % self.num_objects]
        color = self.colors[self.ii]

        self.ax.vlines(gb.index.values, low, high, colors=color)
        self.ii += 1


def raster_plot(spike_times, colors, title="spike raster", ax=None, figsize=(8, 8)):

    if ax == None:
        fig, ax = plt.subplots(figsize=figsize)

    plotter = _VlPlotter(
        ax, num_objects=len(spike_times["unit_id"].unique()), colors=colors
    )
    # spike_times.groupby('unit_id').agg(plotter)
    spike_times[["stimulus_presentation_id", "unit_id"]].groupby("unit_id").agg(plotter)

    # print(len(np.unique(spike_times['unit_id'])))
    # print(len(ax.collections))

    ax.set_xlabel("time (s)", fontsize=16)
    ax.set_ylabel("unit", fontsize=16)
    ax.set_title(title, fontsize=20)

    # plt.yticks([])
    # plt.axis('tight')


def format_label(x, pos):
    return "{:.0f}".format(x)


def format_label_p1(x, pos):
    return "{:.1f}".format(x)


def format_label_p2(x, pos):
    return "{:.2f}".format(x)


def format_label_p3(x, pos):
    return "{:.3f}".format(x)


def format_label_in_ms(x, pos):
    return "{:.0f}".format(x * 1000)


def format_label_log(x, pos):
    x10 = 10**x
    if x10 >= 1:
        return "{:.0f}".format(x10)
    if x10 >= 0.1:
        return "{:.1f}".format(x10)
    elif x10 >= 0.01:
        return "{:.2f}".format(x10)
    elif x10 >= 0.001:
        return "{:.3f}".format(x10)


def format_label_log_in_ms(x, pos):
    x10 = (10**x) * 1000
    if x10 >= 1:
        return "{:.0f}".format(x10)
    if x10 >= 0.1:
        return "{:.1f}".format(x10)
    elif x10 >= 0.01:
        return "{:.2f}".format(x10)
    elif x10 >= 0.001:
        return "{:.3f}".format(x10)

def format_base_ten(x):
    """
    Input a float, get a string representation that is formatted
    to 3.5 x 10^3
    """
    if x == 0:
        return "0"
    else:
        a = np.floor(np.log10(np.abs(x)))
        b = x / 10 ** a
        if abs(a) < 3:
            return "{:.3f}".format(x)

        return r"${:.1f} \times 10^{{{}}}$".format(b, int(a))

def format_ax(ax, measure, axis, set_ticks=True, tight_layout=False):
    if measure == "tau_R":
        label = "information\ntimescale $τ_R$ (ms)"
        if tight_layout:
            label = "$τ_R$ (ms)"
        fmt = format_label_in_ms
        ticks = None
    elif measure == "log_tau_R":
        label = "information\ntimescale $τ_R$ (ms)"
        if tight_layout:
            label = "$τ_R$ (ms)"
        fmt = format_label_log_in_ms
        ticks = [-3.0, -2.0, -1.0, 0.0]
    elif measure == "R_tot":
        label = r"predictability $R_{\mathregular{tot}}$"
        if tight_layout:
            label = r"$R_{\mathregular{tot}}$"
        fmt = None
        ticks = None
    elif measure == "log_R":
        label = r"predictability $R_{\mathregular{tot}}$"
        if tight_layout:
            label = r"$R_{\mathregular{tot}}$"
        fmt = format_label_log
        ticks = [-3.0, -2.0, -1.0]
    elif measure == "log_fr":
        label = "firing rate [Hz]"
        if tight_layout:
            label = "fir. rate [Hz]"
        fmt = format_label_log
        ticks = [-1.0, 0.0, 1.0, 2.0]
    elif measure == "firing_rate":
        label = "firing rate (Hz)"
        if tight_layout:
            label = "fir. rate (Hz)"
        fmt = None
        ticks = None
    elif measure == "median_ISI":
        label = "ISI (ms)"
        if tight_layout:
            label = "ISI (ms)"
        fmt = None
        ticks = None
    elif measure == "CV":
        label = "CV"
        if tight_layout:
            label = "CV"
        fmt = None
        ticks = None
    elif measure == "mre_tau":
        label = "intrinsic\n" + r"timescale $τ_{\mathregular{C}}$ (ms)"
        if tight_layout:
            label = "$τ_{\mathregular{C}}$ (ms)"
        fmt = format_label_in_ms
        ticks = None
    elif measure == "tau_two_timescales":
        label = "intrinsic\n" + r"timescale $τ_{\mathregular{C}}$ (ms)"
        if tight_layout:
            label = "$τ_{\mathregular{C}}$ (ms)"
        fmt = format_label_in_ms
        ticks = None
    elif measure == "log_mre_tau":
        label = "intrinsic\n" + r"timescale $τ_{\mathregular{C}}$ (ms)"
        if tight_layout:
            label = "$τ_{\mathregular{C}}$ (ms)"
        fmt = format_label_log_in_ms
        ticks = None

    if axis == "x":
        ax.set_xlabel(label)
        if not fmt is None:
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt))
        if set_ticks and not ticks is None:
            ax.set_xticks(ticks=ticks)
    elif axis == "y":
        ax.set_ylabel(label)
        if not fmt is None:
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt))
        if set_ticks and not ticks is None:
            ax.set_yticks(ticks=ticks)


def share_x_range(ax0, ax1, ax0_log=False, ax1_log=False):
    x0_lo, x0_hi = ax0.get_xlim()
    x1_lo, x1_hi = ax1.get_xlim()

    if ax0_log:
        x0_lo = 10**x0_lo
        x0_hi = 10**x0_hi
    if ax1_log:
        x1_lo = 10**x1_lo
        x1_hi = 10**x1_hi

    new_lo = min(x0_lo, x1_lo)
    new_hi = max(x0_hi, x1_hi)

    if ax0_log:
        x0_lo = np.log10(new_lo)
        x0_hi = np.log10(new_hi)
    if ax1_log:
        x1_lo = np.log10(new_lo)
        x1_hi = np.log10(new_hi)

    ax0.set_xlim(x0_lo, x0_hi)
    ax1.set_xlim(x1_lo, x1_hi)


def get_effect_size(data_1, data_2):
    mu_1 = np.mean(data_1)
    mu_2 = np.mean(data_2)
    std_1 = np.std(data_1)
    std_2 = np.std(data_2)
    n_1 = len(data_1)
    n_2 = len(data_2)
    s = np.sqrt(((n_1 - 1) * std_1**2 + (n_2 - 1) * std_2**2) / (n_1 + n_2 - 2))
    return (mu_1 - mu_2) / s


def print_stat_tests(data, labels):
    assert len(data) == len(labels)
    print("ANOVA: ", st.f_oneway(*data))
    print("Kruskal-Wallis: ", st.kruskal(*data))

    print("Welch's t-test: ")
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            print(
                "  {} - {} : ".format(labels[i], labels[j]),
                st.ttest_ind(data[i], data[j], equal_var=False),
            )
            print("    effect size: {}".format(get_effect_size(data[i], data[j])))

    print("Mann-Whitney: ")
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            print(
                "  {} - {} : ".format(labels[i], labels[j]),
                st.mannwhitneyu(data[i], data[j], alternative="two-sided"),
            )


def get_sd(data, statistic, num_bootstraps=250):
    assert statistic in ["median"]

    stats = []
    for i in range(num_bootstraps):
        bs_data = random.choices(data, k=len(data))

        if statistic == "median":
            stats += [np.nanmedian(bs_data)]

    return np.std(stats)


def get_center(data, measure_of_central_tendency):
    if measure_of_central_tendency == "median":
        if type(data) == pd.DataFrame or type(data) == pd.core.series.Series:
            return data.median()
        else:
            return np.nanmedian(data)


def get_CI_median(data):
    N_data = len(data)
    median_data = np.sort(np.nanmedian(np.random.choice(data, size=(10000, N_data)), axis=1))
    CI_lo = median_data[249]
    CI_hi = median_data[9749]
    return CI_lo, CI_hi


def plot_stacked_area(num_neurons_evol, ax, structures, structures_map):
    # plot stacked area, top to bottom
    # want:
    # - labels in same order as colours in stack
    # - choose own colours

    num_neurons_left = num_neurons_evol.agg(sum, axis=1).values

    for structure in structures:
        num_neurons_structure = num_neurons_evol[structure].values

        ax.fill_between(
            x=range(len(num_neurons_structure)),
            y1=num_neurons_left,
            y2=num_neurons_left - num_neurons_structure,
            color=structures_map[structure]["color"],
            label=structures_map[structure]["name"],
        )

        num_neurons_left -= num_neurons_structure
    # print(num_neurons_evol.columns.values)
    # print(num_neurons_evol.head())
    # print()


# FIXME
# code from jonathan sedar
# def plot_tsne(dftsne, fts_num=fts_num, ft_hue='mfr_is_vw'):
#     """ customised scatterplot for t-sne representation """

#     pal = 'cubehelix'
#     leg = True

#     if ft_hue in fts_num:
#         pal = 'BuPu'
#         leg = False

#     g = sns.lmplot('x', 'y', dftsne.sort_values(by=ft_hue), hue=ft_hue, palette=pal
#                    ,fit_reg=False, size=9, legend=leg
#                    ,scatter_kws={'alpha':0.7,'s':100, 'edgecolor':'w', 'lw':0.4})
#     _ = g.axes.flat[0].set_title('t-SNE rep colored by {}'.format(ft_hue))


def get_sample_units(data=None, size=500, fc_analysis=False):
    if data is None:
        if fc_analysis:
            return [
                951138094,
                951138121,
                951138166,
                951138185,
                951138201,
                951138212,
                951138220,
                951138247,
                951138363,
                951138456,
                951138468,
                951138485,
                951138494,
                951138501,
                951138714,
                951138732,
                951138785,
                951138813,
                951139070,
                951139193,
                951139475,
                951139549,
                951139768,
                951139871,
                951139905,
                951139917,
                951139967,
                951140017,
                951140107,
                951140134,
                951141089,
                951141184,
                951141754,
                951141797,
                951141815,
                951142023,
                951142095,
                951142171,
                951142289,
                951142300,
                951142333,
                951146166,
                951146185,
                951146340,
                951146390,
                951146590,
                951146809,
                951147040,
                951147070,
                951147112,
                951147135,
                951147225,
                951147404,
                951147450,
                951147518,
                951147592,
                951147638,
                951147669,
                951147713,
                951147774,
                951147908,
                951148000,
                951148085,
                951148199,
                951148381,
                951148476,
                951148500,
                951148526,
                951148616,
                951148634,
                951148942,
                951149199,
                951149445,
                951149457,
                951149596,
                951149619,
                951149637,
                951149650,
                951149703,
                951149778,
                951149992,
                951150519,
                951150578,
                951150649,
                951150772,
                951150811,
                951150877,
                951150912,
                951150929,
                951150966,
                951150993,
                951151048,
                951151074,
                951151150,
                951151167,
                951151250,
                951151370,
                951151404,
                951155285,
                951155297,
                951155371,
                951155382,
                951155399,
                951155468,
                951155565,
                951155570,
                951155575,
                951155581,
                951155591,
                951155609,
                951155614,
                951155628,
                951155644,
                951155650,
                951155655,
                951155670,
                951155682,
                951155686,
                951155696,
                951155712,
                951155717,
                951155722,
                951155743,
                951155758,
                951155810,
                951155829,
                951155844,
                951155892,
                951155932,
                951155960,
                951155965,
                951155993,
                951156000,
                951156059,
                951156070,
                951156081,
                951156142,
                951156153,
                951156193,
                951156263,
                951156278,
                951156534,
                951156673,
                951157864,
                951157885,
                951157924,
                951157929,
                951157943,
                951157957,
                951158000,
                951158009,
                951158014,
                951158022,
                951158057,
                951158064,
                951158096,
                951158108,
                951158120,
                951158132,
                951158137,
                951158160,
                951158184,
                951158188,
                951158193,
                951158206,
                951158217,
                951158240,
                951158246,
                951158257,
                951158292,
                951158303,
                951158310,
                951158320,
                951158379,
                951158394,
                951158406,
                951158427,
                951158445,
                951158681,
                951158693,
                951158723,
            ]
        else:
            return [
                951026542,
                951135959,
                951133578,
                951147667,
                951026546,
                950999222,
                950940612,
                951092413,
                951146185,
                950932369,
                951128891,
                951186563,
                951172262,
                951085154,
                951012060,
                951085741,
                951087860,
                951166084,
                950923928,
                951168950,
                951059120,
                951016855,
                950930787,
                951174330,
                951147881,
                951104696,
                950917679,
                951030647,
                950989619,
                951011497,
                950912493,
                950928621,
                951088389,
                950946360,
                950940870,
                951031344,
                951186543,
                950913945,
                951134668,
                951042798,
                951006463,
                950921529,
                951131689,
                951167825,
                951186688,
                951017243,
                951027209,
                951150578,
                951168730,
                951035312,
                950960738,
                950950280,
                951015564,
                951006966,
                950922377,
                951190848,
                951024089,
                951187234,
                950997109,
                950940465,
                950939393,
                950998365,
                951098617,
                951057401,
                951186614,
                950942749,
                951171609,
                951166382,
                951025622,
                951059060,
                951156059,
                951168592,
                950935740,
                951146590,
                951141850,
                951021354,
                951013044,
                951090382,
                951061011,
                951158193,
                951089890,
                950996519,
                951142395,
                951128407,
                951042707,
                951084877,
                951176042,
                951027194,
                951141215,
                951057498,
                951141476,
                950912441,
                951093823,
                951105205,
                951104547,
                951140656,
                951103584,
                950923903,
                951017481,
                951171942,
                951148762,
                950937945,
                951139582,
                951091037,
                951003972,
                951086189,
                950923495,
                951007573,
                950913856,
                950912847,
                950915005,
                950921447,
                950924392,
                951013894,
                951098518,
                951015810,
                951085491,
                951024131,
                951172680,
                951189413,
                951190458,
                951013127,
                951149457,
                951174382,
                951137963,
                951083953,
                951128181,
                951042127,
                951035513,
                950929836,
                950945527,
                951100251,
                950917714,
                951005106,
                951095181,
                950923751,
                951149650,
                951022323,
                951167844,
                951143575,
                951005238,
                951150966,
                950994899,
                950921711,
                951087680,
                951004716,
                950941406,
                951171873,
                951167293,
                950937337,
                950960789,
                950912200,
                950944215,
                951186670,
                951137290,
                951175952,
                951003277,
                950920669,
                951014690,
                951003932,
                951093194,
                951022530,
                951026509,
                951096820,
                951090512,
                951166084,
                951059056,
                951093973,
                951144783,
                951016108,
                951003792,
                951185558,
                951003136,
                951168692,
                950998503,
                951140966,
                950998637,
                950994260,
                951167953,
                950923187,
                951092948,
                951011699,
                951150929,
                951005262,
                951168952,
                951185612,
                951104387,
                951084917,
                951022706,
                951016303,
                950923630,
                951188500,
                951021300,
                950998503,
                951128233,
                951141054,
                951137007,
                951094157,
                951017028,
                951004814,
                951098224,
                951013647,
                950993126,
                951185741,
                951101549,
                951141643,
                951172367,
                951091664,
                951012989,
                951167195,
                951085636,
                951083730,
                951011264,
                950955049,
                951006114,
                951015681,
                951005873,
                951014305,
                951019395,
                951100741,
                951139541,
                951006966,
                951085678,
                951133771,
                950941553,
                951091989,
                951168915,
                951147823,
                951189703,
                950953874,
                951140306,
                951141219,
                951186515,
                951037202,
                950945928,
                951142246,
                951006277,
                951104824,
                950913241,
                951188327,
                951171690,
                950998563,
                951085407,
                951140218,
                951104915,
                950953425,
                951084120,
                951139478,
                950959886,
                950923237,
                951187268,
                950945645,
                950992713,
                950914676,
                951154618,
                950990374,
                951105144,
                951086118,
                951127345,
                950914025,
                950999820,
                950993237,
                950946531,
                950943194,
                951043740,
                950989428,
                951152080,
                950941847,
                951090512,
                951017028,
                951132066,
                951032197,
                951127390,
                951087652,
                951173222,
                951083712,
                951002160,
                951158057,
                950989676,
                951006117,
                951142354,
                951186601,
                951014105,
                951141818,
                951172391,
                951045035,
                951011645,
                951083976,
                951022589,
                951147435,
                950939334,
                950955049,
                951167491,
                950921243,
                951104915,
                951133835,
                951085341,
                951132189,
                951012680,
                951097419,
                951146847,
                950931063,
                951154594,
                951015869,
                950993827,
                951169027,
                951014093,
                951167959,
                951141022,
                951015774,
                951142151,
                951145840,
                951046989,
                951148757,
                951014116,
                951005747,
                951021884,
                950953928,
                951171509,
                951141203,
                951139839,
                951005905,
                950918746,
                951023452,
                951189757,
                950924396,
                951089646,
                950924175,
                951145850,
                951083420,
                950913386,
                951015822,
                951032356,
                951062244,
                951015950,
                951172411,
                950941929,
                950923360,
                950989680,
                951169001,
                951131562,
                951011131,
                950994005,
                951002939,
                951005616,
                951141129,
                951137610,
                951158723,
                951187421,
                950941033,
                951167839,
                951002101,
                951035267,
                951147070,
                951141528,
                951012456,
                950997265,
                951017060,
                951150772,
                951105297,
                951083775,
                951020698,
                950990364,
                951134112,
                951020912,
                950950707,
                951190391,
                951105252,
                951154486,
                951027043,
                951013073,
                951171852,
                950945940,
                951035976,
                951188718,
                951172455,
                951002333,
                950937168,
                951003100,
                951035976,
                951125872,
                950988826,
                951166211,
                950937198,
                951042429,
                950923955,
                951140346,
                951136947,
                951019883,
                951007841,
                951087437,
                951019408,
                951139871,
                951166284,
                951046916,
                951091077,
                951168810,
                950991331,
                951085884,
                951090338,
                951133736,
                951168726,
                951106267,
                950999981,
                951139726,
                950913885,
                950924336,
                951046568,
                951083705,
                951132308,
                951133606,
                951022720,
                951013974,
                951145960,
                950943746,
                951142315,
                951013932,
                951014305,
                950923616,
                951093935,
                951023063,
                951035277,
                950929753,
                951083398,
                951105339,
                950914787,
                951084289,
                951042585,
                950938890,
                950998493,
                951032230,
                951085467,
                951167263,
                951006026,
                951174384,
                951167098,
                951023220,
                951168173,
                951155468,
                951098224,
                951006955,
                951158108,
                951093979,
                951188390,
                950941929,
                951167182,
                951139732,
                951171942,
                951127308,
                951189466,
                951032585,
                951139635,
                950915073,
                951125822,
                951147592,
                951141219,
                951027194,
                951089857,
                950937855,
                951021387,
                950936588,
                950998625,
                951006397,
                951155644,
                951142293,
                950914676,
                951019165,
                951187212,
                951134021,
                950941509,
                951185616,
                950922657,
                951125948,
                950950688,
                951188504,
                950940070,
                951101237,
                950941536,
                951013542,
                951143494,
                951133736,
                951005845,
                951085916,
                951105705,
                950938840,
                951012456,
                951023111,
                951093114,
                950992977,
                950922489,
                951010954,
                951136997,
                951148423,
                950932684,
                951002605,
                951085916,
                950939830,
                951105205,
                951126817,
                951086580,
            ]
    else:
        indices = np.random.choice(np.arange(len(data)), size, replace = False)
        units = data['unit'].values[indices]
        session_ids = data['session_id'].values[indices]
        return units, session_ids
        # return np.random.choice(np.unique(data['unit']), size, replace = False)



def plot_cont_boxplots(
    ax,
    data,
    positions,
    colors,
    plot_data=True,
    plot_outliers=True,
    alpha_data=1,
    widths=None,
):
    bplot = cont_boxplot(
        ax,
        data,
        positions=positions,
        patch_artist=True,
        showfliers=False,
        zorder=3,
        manage_ticks=False,
        widths=widths,
    )

    for i, pos, dataset in zip(range(len(positions)), positions, data):
        data_outliers, data_no_outliers = get_outliers(dataset)

        if plot_data:
            ax.plot(
                [pos] * len(data_no_outliers),  # same position for all data points
                data_no_outliers,
                "x",
                ms=10,
                markeredgewidth=3,
                color=colors[i],
                zorder=2,
                alpha=alpha_data,
            )
        if plot_outliers:
            ax.plot(
                [pos] * len(data_outliers),
                data_outliers,
                "o",
                ms=10,
                markeredgewidth=3,
                color=colors[i],
                zorder=2,
                markerfacecolor="None",
                alpha=alpha_data,
            )

    for patch, color in zip(bplot["boxes"], colors):
        # patch.set_facecolor(color)
        patch.set_facecolor("white")
        # patch.set_alpha(0.4)
    for element in bplot["medians"]:
        # element.set_color('white')
        element.set_color("black")


# adapted from matplotlib
def cont_boxplot(
    ax,
    x,
    notch=None,
    sym=None,
    vert=None,
    whis=None,
    positions=None,
    widths=None,
    patch_artist=None,
    bootstrap=None,
    usermedians=None,
    conf_intervals=None,
    meanline=None,
    showmeans=None,
    showcaps=None,
    showbox=None,
    showfliers=None,
    boxprops=None,
    labels=None,
    flierprops=None,
    medianprops=None,
    meanprops=None,
    capprops=None,
    whiskerprops=None,
    manage_ticks=True,
    autorange=False,
    zorder=None,
):
    """
    Make a box and whisker plot.
    Make a box and whisker plot for each column of *x* or each
    vector in sequence *x*.  The box extends from the lower to
    upper quartile values of the data, with a line at the median.
    The whiskers extend from the box to show the range of the
    data.  Flier points are those past the end of the whiskers.
    Parameters
    ----------
    x : Array or a sequence of vectors.
        The input data.
    notch : bool, default: False
        Whether to draw a notched box plot (`True`), or a rectangular box
        plot (`False`).  The notches represent the confidence interval (CI)
        around the median.  The documentation for *bootstrap* describes how
        the locations of the notches are computed by default, but their
        locations may also be overridden by setting the *conf_intervals*
        parameter.
        .. note::
            In cases where the values of the CI are less than the
            lower quartile or greater than the upper quartile, the
            notches will extend beyond the box, giving it a
            distinctive "flipped" appearance. This is expected
            behavior and consistent with other statistical
            visualization packages.
    sym : str, optional
        The default symbol for flier points.  An empty string ('') hides
        the fliers.  If `None`, then the fliers default to 'b+'.  More
        control is provided by the *flierprops* parameter.
    vert : bool, default: True
        If `True`, draws vertical boxes.
        If `False`, draw horizontal boxes.
    whis : float or (float, float), default: 1.5
        The position of the whiskers.
        If a float, the lower whisker is at the lowest datum above
        ``Q1 - whis*(Q3-Q1)``, and the upper whisker at the highest datum
        below ``Q3 + whis*(Q3-Q1)``, where Q1 and Q3 are the first and
        third quartiles.  The default value of ``whis = 1.5`` corresponds
        to Tukey's original definition of boxplots.
        If a pair of floats, they indicate the percentiles at which to
        draw the whiskers (e.g., (5, 95)).  In particular, setting this to
        (0, 100) results in whiskers covering the whole range of the data.
        "range" is a deprecated synonym for (0, 100).
        In the edge case where ``Q1 == Q3``, *whis* is automatically set
        to (0, 100) (cover the whole range of the data) if *autorange* is
        True.
        Beyond the whiskers, data are considered outliers and are plotted
        as individual points.
    bootstrap : int, optional
        Specifies whether to bootstrap the confidence intervals
        around the median for notched boxplots. If *bootstrap* is
        None, no bootstrapping is performed, and notches are
        calculated using a Gaussian-based asymptotic approximation
        (see McGill, R., Tukey, J.W., and Larsen, W.A., 1978, and
        Kendall and Stuart, 1967). Otherwise, bootstrap specifies
        the number of times to bootstrap the median to determine its
        95% confidence intervals. Values between 1000 and 10000 are
        recommended.
    usermedians : array-like, optional
        A 1D array-like of length ``len(x)``.  Each entry that is not
        `None` forces the value of the median for the corresponding
        dataset.  For entries that are `None`, the medians are computed
        by Matplotlib as normal.
    conf_intervals : array-like, optional
        A 2D array-like of shape ``(len(x), 2)``.  Each entry that is not
        None forces the location of the corresponding notch (which is
        only drawn if *notch* is `True`).  For entries that are `None`,
        the notches are computed by the method specified by the other
        parameters (e.g., *bootstrap*).
    positions : array-like, optional
        The positions of the boxes. The ticks and limits are
        automatically set to match the positions. Defaults to
        ``range(1, N+1)`` where N is the number of boxes to be drawn.
    widths : float or array-like
        The widths of the boxes.  The default is 0.5, or ``0.15*(distance
        between extreme positions)``, if that is smaller.
    patch_artist : bool, default: False
        If `False` produces boxes with the Line2D artist. Otherwise,
        boxes and drawn with Patch artists.
    labels : sequence, optional
        Labels for each dataset (one per dataset).
    manage_ticks : bool, default: True
        If True, the tick locations and labels will be adjusted to match
        the boxplot positions.
    autorange : bool, default: False
        When `True` and the data are distributed such that the 25th and
        75th percentiles are equal, *whis* is set to (0, 100) such
        that the whisker ends are at the minimum and maximum of the data.
    meanline : bool, default: False
        If `True` (and *showmeans* is `True`), will try to render the
        mean as a line spanning the full width of the box according to
        *meanprops* (see below).  Not recommended if *shownotches* is also
        True.  Otherwise, means will be shown as points.
    zorder : float, default: ``Line2D.zorder = 2``
        The zorder of the boxplot.
    Returns
    -------
    dict
      A dictionary mapping each component of the boxplot to a list
      of the `.Line2D` instances created. That dictionary has the
      following keys (assuming vertical boxplots):
      - ``boxes``: the main body of the boxplot showing the
        quartiles and the median's confidence intervals if
        enabled.
      - ``medians``: horizontal lines at the median of each box.
      - ``whiskers``: the vertical lines extending to the most
        extreme, non-outlier data points.
      - ``caps``: the horizontal lines at the ends of the
        whiskers.
      - ``fliers``: points representing data that extend beyond
        the whiskers (fliers).
      - ``means``: points or lines representing the means.
    Other Parameters
    ----------------
    showcaps : bool, default: True
        Show the caps on the ends of whiskers.
    showbox : bool, default: True
        Show the central box.
    showfliers : bool, default: True
        Show the outliers beyond the caps.
    showmeans : bool, default: False
        Show the arithmetic means.
    capprops : dict, default: None
        The style of the caps.
    boxprops : dict, default: None
        The style of the box.
    whiskerprops : dict, default: None
        The style of the whiskers.
    flierprops : dict, default: None
        The style of the fliers.
    medianprops : dict, default: None
        The style of the median.
    meanprops : dict, default: None
        The style of the mean.
    """

    # Missing arguments default to rcParams.
    if whis is None:
        whis = rcParams["boxplot.whiskers"]
    if bootstrap is None:
        bootstrap = rcParams["boxplot.bootstrap"]

    bxpstats = cbook.boxplot_stats(
        x, whis=whis, bootstrap=bootstrap, labels=labels, autorange=autorange
    )
    if notch is None:
        notch = rcParams["boxplot.notch"]
    if vert is None:
        vert = rcParams["boxplot.vertical"]
    if patch_artist is None:
        patch_artist = rcParams["boxplot.patchartist"]
    if meanline is None:
        meanline = rcParams["boxplot.meanline"]
    if showmeans is None:
        showmeans = rcParams["boxplot.showmeans"]
    if showcaps is None:
        showcaps = rcParams["boxplot.showcaps"]
    if showbox is None:
        showbox = rcParams["boxplot.showbox"]
    if showfliers is None:
        showfliers = rcParams["boxplot.showfliers"]

    if boxprops is None:
        boxprops = {}
    if whiskerprops is None:
        whiskerprops = {}
    if capprops is None:
        capprops = {}
    if medianprops is None:
        medianprops = {}
    if meanprops is None:
        meanprops = {}
    if flierprops is None:
        flierprops = {}

    if patch_artist:
        boxprops["linestyle"] = "solid"  # Not consistent with bxp.
        if "color" in boxprops:
            boxprops["edgecolor"] = boxprops.pop("color")

    # if non-default sym value, put it into the flier dictionary
    # the logic for providing the default symbol ('b+') now lives
    # in bxp in the initial value of final_flierprops
    # handle all of the *sym* related logic here so we only have to pass
    # on the flierprops dict.
    if sym is not None:
        # no-flier case, which should really be done with
        # 'showfliers=False' but none-the-less deal with it to keep back
        # compatibility
        if sym == "":
            # blow away existing dict and make one for invisible markers
            flierprops = dict(linestyle="none", marker="", color="none")
            # turn the fliers off just to be safe
            showfliers = False
        # now process the symbol string
        else:
            # process the symbol string
            # discarded linestyle
            _, marker, color = _process_plot_format(sym)
            # if we have a marker, use it
            if marker is not None:
                flierprops["marker"] = marker
            # if we have a color, use it
            if color is not None:
                # assume that if color is passed in the user want
                # filled symbol, if the users want more control use
                # flierprops
                flierprops["color"] = color
                flierprops["markerfacecolor"] = color
                flierprops["markeredgecolor"] = color

    # replace medians if necessary:
    if usermedians is not None:
        if len(np.ravel(usermedians)) != len(bxpstats) or np.shape(usermedians)[0] != len(
            bxpstats
        ):
            raise ValueError("'usermedians' and 'x' have different lengths")
        else:
            # reassign medians as necessary
            for stats, med in zip(bxpstats, usermedians):
                if med is not None:
                    stats["med"] = med

    if conf_intervals is not None:
        if len(conf_intervals) != len(bxpstats):
            raise ValueError("'conf_intervals' and 'x' have different lengths")
        else:
            for stats, ci in zip(bxpstats, conf_intervals):
                if ci is not None:
                    if len(ci) != 2:
                        raise ValueError("each confidence interval must have two values")
                    else:
                        if ci[0] is not None:
                            stats["cilo"] = ci[0]
                        if ci[1] is not None:
                            stats["cihi"] = ci[1]

    artists = ax.bxp(
        bxpstats,
        positions=positions,
        widths=widths,
        vert=vert,
        patch_artist=patch_artist,
        shownotches=notch,
        showmeans=showmeans,
        showcaps=showcaps,
        showbox=showbox,
        boxprops=boxprops,
        flierprops=flierprops,
        medianprops=medianprops,
        meanprops=meanprops,
        meanline=meanline,
        showfliers=showfliers,
        capprops=capprops,
        whiskerprops=whiskerprops,
        manage_ticks=manage_ticks,
        zorder=zorder,
    )
    return artists


def bxp(
    self,
    bxpstats,
    positions=None,
    widths=None,
    vert=True,
    patch_artist=False,
    shownotches=False,
    showmeans=False,
    showcaps=True,
    showbox=True,
    showfliers=True,
    boxprops=None,
    whiskerprops=None,
    flierprops=None,
    medianprops=None,
    capprops=None,
    meanprops=None,
    meanline=False,
    manage_ticks=True,
    zorder=None,
):
    """
    Drawing function for box and whisker plots.
    Make a box and whisker plot for each column of *x* or each
    vector in sequence *x*.  The box extends from the lower to
    upper quartile values of the data, with a line at the median.
    The whiskers extend from the box to show the range of the
    data.  Flier points are those past the end of the whiskers.
    Parameters
    ----------
    bxpstats : list of dicts
      A list of dictionaries containing stats for each boxplot.
      Required keys are:
      - ``med``: The median (scalar float).
      - ``q1``: The first quartile (25th percentile) (scalar
        float).
      - ``q3``: The third quartile (75th percentile) (scalar
        float).
      - ``whislo``: Lower bound of the lower whisker (scalar
        float).
      - ``whishi``: Upper bound of the upper whisker (scalar
        float).
      Optional keys are:
      - ``mean``: The mean (scalar float). Needed if
        ``showmeans=True``.
      - ``fliers``: Data beyond the whiskers (sequence of floats).
        Needed if ``showfliers=True``.
      - ``cilo`` & ``cihi``: Lower and upper confidence intervals
        about the median. Needed if ``shownotches=True``.
      - ``label``: Name of the dataset (string). If available,
        this will be used a tick label for the boxplot
    positions : array-like, default: [1, 2, ..., n]
      The positions of the boxes. The ticks and limits
      are automatically set to match the positions.
    widths : array-like, default: None
      Either a scalar or a vector and sets the width of each
      box. The default is ``0.15*(distance between extreme
      positions)``, clipped to no less than 0.15 and no more than
      0.5.
    vert : bool, default: True
      If `True` (default), makes the boxes vertical.  If `False`,
      makes horizontal boxes.
    patch_artist : bool, default: False
      If `False` produces boxes with the `.Line2D` artist.
      If `True` produces boxes with the `~matplotlib.patches.Patch` artist.
    shownotches : bool, default: False
      If `False` (default), produces a rectangular box plot.
      If `True`, will produce a notched box plot
    showmeans : bool, default: False
      If `True`, will toggle on the rendering of the means
    showcaps  : bool, default: True
      If `True`, will toggle on the rendering of the caps
    showbox  : bool, default: True
      If `True`, will toggle on the rendering of the box
    showfliers : bool, default: True
      If `True`, will toggle on the rendering of the fliers
    boxprops : dict or None (default)
      If provided, will set the plotting style of the boxes
    whiskerprops : dict or None (default)
      If provided, will set the plotting style of the whiskers
    capprops : dict or None (default)
      If provided, will set the plotting style of the caps
    flierprops : dict or None (default)
      If provided will set the plotting style of the fliers
    medianprops : dict or None (default)
      If provided, will set the plotting style of the medians
    meanprops : dict or None (default)
      If provided, will set the plotting style of the means
    meanline : bool, default: False
      If `True` (and *showmeans* is `True`), will try to render the mean
      as a line spanning the full width of the box according to
      *meanprops*. Not recommended if *shownotches* is also True.
      Otherwise, means will be shown as points.
    manage_ticks : bool, default: True
      If True, the tick locations and labels will be adjusted to match the
      boxplot positions.
    zorder : float, default: ``Line2D.zorder = 2``
      The zorder of the resulting boxplot.
    Returns
    -------
    dict
      A dictionary mapping each component of the boxplot to a list
      of the `.Line2D` instances created. That dictionary has the
      following keys (assuming vertical boxplots):
      - ``boxes``: the main body of the boxplot showing the
        quartiles and the median's confidence intervals if
        enabled.
      - ``medians``: horizontal lines at the median of each box.
      - ``whiskers``: the vertical lines extending to the most
        extreme, non-outlier data points.
      - ``caps``: the horizontal lines at the ends of the
        whiskers.
      - ``fliers``: points representing data that extend beyond
        the whiskers (fliers).
      - ``means``: points or lines representing the means.
    Examples
    --------
    .. plot:: gallery/statistics/bxp.py
    """
    # lists of artists to be output
    whiskers = []
    caps = []
    boxes = []
    medians = []
    means = []
    fliers = []

    # empty list of xticklabels
    datalabels = []

    # Use default zorder if none specified
    if zorder is None:
        zorder = mlines.Line2D.zorder

    zdelta = 0.1

    def line_props_with_rcdefaults(subkey, explicit, zdelta=0, use_marker=True):
        d = {
            k.split(".")[-1]: v
            for k, v in rcParams.items()
            if k.startswith(f"boxplot.{subkey}")
        }
        d["zorder"] = zorder + zdelta
        if not use_marker:
            d["marker"] = ""
        d.update(cbook.normalize_kwargs(explicit, mlines.Line2D))
        return d

    # box properties
    if patch_artist:
        final_boxprops = {
            "linestyle": rcParams["boxplot.boxprops.linestyle"],
            "linewidth": rcParams["boxplot.boxprops.linewidth"],
            "edgecolor": rcParams["boxplot.boxprops.color"],
            "facecolor": (
                "white"
                if rcParams["_internal.classic_mode"]
                else rcParams["patch.facecolor"]
            ),
            "zorder": zorder,
            **cbook.normalize_kwargs(boxprops, mpatches.PathPatch),
        }
    else:
        final_boxprops = line_props_with_rcdefaults(
            "boxprops", boxprops, use_marker=False
        )
    final_whiskerprops = line_props_with_rcdefaults(
        "whiskerprops", whiskerprops, use_marker=False
    )
    final_capprops = line_props_with_rcdefaults("capprops", capprops, use_marker=False)
    final_flierprops = line_props_with_rcdefaults("flierprops", flierprops)
    final_medianprops = line_props_with_rcdefaults(
        "medianprops", medianprops, zdelta, use_marker=False
    )
    final_meanprops = line_props_with_rcdefaults("meanprops", meanprops, zdelta)
    removed_prop = "marker" if meanline else "linestyle"
    # Only remove the property if it's not set explicitly as a parameter.
    if meanprops is None or removed_prop not in meanprops:
        final_meanprops[removed_prop] = ""

    def patch_list(xs, ys, **kwargs):
        path = mpath.Path(
            # Last vertex will have a CLOSEPOLY code and thus be ignored.
            np.append(np.column_stack([xs, ys]), [(0, 0)], 0),
            closed=True,
        )
        patch = mpatches.PathPatch(path, **kwargs)
        self.add_artist(patch)
        return [patch]

    # vertical or horizontal plot?
    if vert:

        def doplot(*args, **kwargs):
            return self.plot(*args, **kwargs)

        def dopatch(xs, ys, **kwargs):
            return patch_list(xs, ys, **kwargs)

    else:

        def doplot(*args, **kwargs):
            shuffled = []
            for i in range(0, len(args), 2):
                shuffled.extend([args[i + 1], args[i]])
            return self.plot(*shuffled, **kwargs)

        def dopatch(xs, ys, **kwargs):
            xs, ys = ys, xs  # flip X, Y
            return patch_list(xs, ys, **kwargs)

    # input validation
    N = len(bxpstats)
    datashape_message = (
        "List of boxplot statistics and `{0}` values must have same the length"
    )
    # check position
    if positions is None:
        positions = list(range(1, N + 1))
    elif len(positions) != N:
        raise ValueError(datashape_message.format("positions"))

    positions = np.array(positions)
    if len(positions) > 0 and not isinstance(positions[0], Number):
        raise TypeError("positions should be an iterable of numbers")

    # width
    if widths is None:
        widths = [np.clip(0.15 * np.ptp(positions), 0.15, 0.5)] * N
    elif np.isscalar(widths):
        widths = [widths] * N
    elif len(widths) != N:
        raise ValueError(datashape_message.format("widths"))

    for pos, width, stats in zip(positions, widths, bxpstats):
        # try to find a new label
        datalabels.append(stats.get("label", pos))

        # whisker coords
        whisker_x = np.ones(2) * pos
        whiskerlo_y = np.array([stats["q1"], stats["whislo"]])
        whiskerhi_y = np.array([stats["q3"], stats["whishi"]])

        # cap coords
        cap_left = pos - width * 0.25
        cap_right = pos + width * 0.25
        cap_x = np.array([cap_left, cap_right])
        cap_lo = np.ones(2) * stats["whislo"]
        cap_hi = np.ones(2) * stats["whishi"]

        # box and median coords
        box_left = pos - width * 0.5
        box_right = pos + width * 0.5
        med_y = [stats["med"], stats["med"]]

        # notched boxes
        if shownotches:
            box_x = [
                box_left,
                box_right,
                box_right,
                cap_right,
                box_right,
                box_right,
                box_left,
                box_left,
                cap_left,
                box_left,
                box_left,
            ]
            box_y = [
                stats["q1"],
                stats["q1"],
                stats["cilo"],
                stats["med"],
                stats["cihi"],
                stats["q3"],
                stats["q3"],
                stats["cihi"],
                stats["med"],
                stats["cilo"],
                stats["q1"],
            ]
            med_x = cap_x

        # plain boxes
        else:
            box_x = [box_left, box_right, box_right, box_left, box_left]
            box_y = [stats["q1"], stats["q1"], stats["q3"], stats["q3"], stats["q1"]]
            med_x = [box_left, box_right]

        # maybe draw the box:
        if showbox:
            if patch_artist:
                boxes.extend(dopatch(box_x, box_y, **final_boxprops))
            else:
                boxes.extend(doplot(box_x, box_y, **final_boxprops))

        # draw the whiskers
        whiskers.extend(doplot(whisker_x, whiskerlo_y, **final_whiskerprops))
        whiskers.extend(doplot(whisker_x, whiskerhi_y, **final_whiskerprops))

        # maybe draw the caps:
        if showcaps:
            caps.extend(doplot(cap_x, cap_lo, **final_capprops))
            caps.extend(doplot(cap_x, cap_hi, **final_capprops))

        # draw the medians
        medians.extend(doplot(med_x, med_y, **final_medianprops))

        # maybe draw the means
        if showmeans:
            if meanline:
                means.extend(
                    doplot(
                        [box_left, box_right],
                        [stats["mean"], stats["mean"]],
                        **final_meanprops,
                    )
                )
            else:
                means.extend(doplot([pos], [stats["mean"]], **final_meanprops))

        # maybe draw the fliers
        if showfliers:
            # fliers coords
            flier_x = np.full(len(stats["fliers"]), pos, dtype=np.float64)
            flier_y = stats["fliers"]

            fliers.extend(doplot(flier_x, flier_y, **final_flierprops))

    if manage_ticks:
        axis_name = "x" if vert else "y"
        interval = getattr(self.dataLim, f"interval{axis_name}")
        axis = getattr(self, f"{axis_name}axis")
        positions = axis.convert_units(positions)
        # The 0.5 additional padding ensures reasonable-looking boxes
        # even when drawing a single box.  We set the sticky edge to
        # prevent margins expansion, in order to match old behavior (back
        # when separate calls to boxplot() would completely reset the axis
        # limits regardless of what was drawn before).  The sticky edges
        # are attached to the median lines, as they are always present.
        interval[:] = (
            min(interval[0], min(positions) - 0.5),
            max(interval[1], max(positions) + 0.5),
        )
        for median, position in zip(medians, positions):
            getattr(median.sticky_edges, axis_name).extend(
                [position - 0.5, position + 0.5]
            )
        # Modified from Axis.set_ticks and Axis.set_ticklabels.
        locator = axis.get_major_locator()
        if not isinstance(axis.get_major_locator(), mticker.FixedLocator):
            locator = mticker.FixedLocator([])
            axis.set_major_locator(locator)
        locator.locs = np.array([*locator.locs, *positions])
        formatter = axis.get_major_formatter()
        if not isinstance(axis.get_major_formatter(), mticker.FixedFormatter):
            formatter = mticker.FixedFormatter([])
            axis.set_major_formatter(formatter)
        formatter.seq = [*formatter.seq, *datalabels]

        self._request_autoscale_view(scalex=self._autoscaleXon, scaley=self._autoscaleYon)

    return dict(
        whiskers=whiskers,
        caps=caps,
        boxes=boxes,
        medians=medians,
        fliers=fliers,
        means=means,
    )


def pd_nested_bootstrap(
    df,
    grouping_col,
    obs,
    num_boot=500,
    func=np.nanmean,
    resample_group_col=False,
    percentiles=None,
):

    """
    bootstrap across rows of a dataframe to get the mean across
    many samples and standard error of this estimate.

    uses `grouping_col` to filter the dataframe into subframes and permute
    in those groups, see below.

    # Parameters:
    obs : str, the column to estimate for
    grouping_col : str,
        bootstrap samples are generated independantly for every unique entry
        of this column.
    resample_group_col : bool, default False.
        Per default, we draw for each "experiment" in `grouping_col` as many
        rows as in the original frame, and create one large list of rows from
        all those experiments. on this list, the bs estimator is calculated.
        When True, we also draw with replacement the experiments. This should
        yield the most conservative error estimate.
    num_boot : int, how many bootstrap samples to generate
    func : function, default np.nanmean is used to calculate the estimate
        for each sample

    # Returns:
    mean : mean across all drawn bootstrap samples
    std : std
    """

    if percentiles is None:
        percentiles = [2.5, 50, 97.5]

    candidates = df[grouping_col].unique()

    resampled_estimates = []

    sub_dfs = dict()
    for candidate in candidates:
        sub_df = df.query(f"`{grouping_col}` == '{candidate}'")
        # this is a hacky way to remove rows where the observable is nan,
        # such as could be for inter-burst-intervals at the end of the experiment
        sub_df = sub_df.query(f"`{obs}` == `{obs}`")
        sub_dfs[candidate] = sub_df

    for idx in tqdm(range(0, num_boot), desc="Bootstrapping dataframe", leave=False):
        merged_obs = []

        if resample_group_col:
            candidates_resampled = np.random.choice(
                candidates, size=len(candidates), replace=True
            )
        else:
            candidates_resampled = candidates

        for candidate in candidates_resampled:
            sub_df = sub_dfs[candidate]
            sample_size = np.fmin(len(sub_df), 10_000)

            # log.debug(f"{candidate}: {sample_size} entries for {obs}")

            # make sure to use different seeds
            sample_df = sub_df.sample(n=sample_size, replace=True)  # , ignore_index=True)
            merged_obs.extend(sample_df[obs])

        estimate = func(merged_obs)
        resampled_estimates.append(estimate)

    # log.debug(resampled_estimates)

    mean = np.mean(resampled_estimates)
    sem = np.std(resampled_estimates, ddof=1)
    q = np.percentile(resampled_estimates, percentiles)

    return mean, sem, q


def pd_bootstrap(
    df, obs, sample_size=None, num_boot=500, func=np.nanmean, percentiles=None
):
    """
    bootstrap across all rows of a dataframe to get the mean across
    many samples and standard error of this estimate.
    query the dataframe first to filter for the right conditions.

    # Parameters:
    obs : str, the column to estimate for
    sample_size: int or None, default (None) for samples that are as large
        as the original dataframe (number of rows)
    num_boot : int, how many bootstrap samples to generate
    func : function, default np.nanmean is used to calculate the estimate
        for each sample
    percentiles : list of floats
        the percentiles to return. default is [2.5, 50, 97.5]

    # Returns:
    mean : mean across all drawn bootstrap samples
    std : std

    """

    if sample_size is None:
        sample_size = np.fmin(len(df), 10_000)

    if percentiles is None:
        percentiles = [2.5, 50, 97.5]

    # drop nans, i.e. for ibis we have one nan-row at the end of every burst
    df = df.query(f"`{obs}` == `{obs}`")

    resampled_estimates = []
    for idx in range(0, num_boot):
        sample_df = df.sample(n=sample_size, replace=True)  # , ignore_index=True)

        resampled_estimates.append(func(sample_df[obs]))

    mean = np.mean(resampled_estimates)
    std = np.std(resampled_estimates, ddof=1)
    q = np.percentile(resampled_estimates, percentiles)

    return mean, std, q


def _draw_error_stick(
    ax,
    center,
    mid,
    errors,
    outliers=None,
    orientation="v",
    linewidth=1.5,
    **kwargs,
):
    """
    Use this to draw errors likes seaborns nice error bars, but using our own
    error estimates.

    note: seaborn v 0.12 broke my swarmplot hack. `conda install seaborn=0.11.2`

    # Parameters
    ax : axis element
    center : number,
        where to align in off-data direction.
        seaborn uses integers 0, 1, 2, ... when plotting categorial violins.
    mid : float,
        middle data point to draw (white dot)
    errors : array like, length 2
        thick bar corresponding to errors
    outliers : array like, length 2
        thin (longer) bar corresponding to outliers
    orientation : "v" or "h"
    **kwargs are passed through to `ax.plot`
    """

    kwargs = kwargs.copy()
    kwargs.setdefault("color", "black")
    kwargs.setdefault("zorder", 3)
    kwargs.setdefault("clip_on", False)

    if outliers is not None:
        assert len(outliers) == 2
        if orientation == "h":
            ax.plot(outliers, [center, center], linewidth=linewidth, **kwargs)
        else:
            ax.plot([center, center], outliers, linewidth=linewidth, **kwargs)

    assert len(errors) == 2
    if orientation == "h":
        ax.plot(errors, [center, center], linewidth=linewidth * 3, **kwargs)
    else:
        ax.plot([center, center], errors, linewidth=linewidth * 3, **kwargs)

    kwargs["zorder"] += 1
    kwargs["edgecolor"] = kwargs["color"]
    kwargs["color"] = "white"
    if orientation == "h":
        ax.scatter(mid, center, s=np.square(linewidth * 2), **kwargs)
    else:
        ax.scatter(center, mid, s=np.square(linewidth * 2), **kwargs)


def _unit_bins(low=0, high=1, num_bins=20):
    bw = (high - low) / num_bins
    return np.arange(low, high + 0.1 * bw, bw)


def fancy_violins(
    df,
    category,
    observable,
    ax=None,
    num_swarm_points=400,
    same_points_per_swarm=True,
    replace=False,
    palette=None,
    seed=42,
    violin_kwargs=dict(),
    swarm_kwargs=dict(),
):
    # log.info(f'|{"":-^75}|')
    # print(f"## Pooled violins for {observable}")
    # log.info(f'|{"":-^65}|')
    # print(f"| Condition | 2.5% percentile | 50% percentile | 97.5% percentile |")
    # print(f"| --------- | --------------- | -------------- | ---------------- |")
    violin_kwargs = violin_kwargs.copy()
    swarm_kwargs = swarm_kwargs.copy()

    np.random.seed(seed)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    ax.set_rasterization_zorder(-5)

    categories = df[category].unique()

    # we want half violins, so hack the data and abuse seaborns "hue" and "split"
    df["fake_hue"] = 0
    for cat in categories:
        dummy = pd.Series([np.nan] * len(df.columns), index=df.columns)
        dummy["fake_hue"] = 1
        dummy[category] = cat
        df = df.append(dummy, ignore_index=True)

    # lets use that seaborn looks up the `hue` variable as the key in palette dict,
    # maybe matching our global colors
    if palette is None:
        palette = {
            "cortex": sns.color_palette().as_hex()[0],
            "thalamus": sns.color_palette().as_hex()[1],
            "natural_movie_one_more_repeats": sns.color_palette().as_hex()[0],
            "spontaneous": sns.color_palette().as_hex()[1],
            "tau_offset_accepted": sns.color_palette().as_hex()[0],
            "tau_offset_rejected": sns.color_palette().as_hex()[0],
            "tau_two_timescales": sns.color_palette().as_hex()[1],
            "tau_rejected": sns.color_palette().as_hex()[1],
        }

    light_palette = dict()
    for key in palette.keys():
        try:
            from bitsandbobs.plt import alpha_to_solid_on_bg
            light_palette[key] = alpha_to_solid_on_bg(palette[key], 0.5)
        except:
            # if we have keys that are not colors
            # pass
            light_palette[key] = palette[key]

    violin_defaults = dict(
        scale_hue=False,
        cut=0,
        scale="width",
        inner=None,
        bw=0.1,
        # palette=light_palette,
        hue="fake_hue",
        split=True,
    )

    for key in violin_defaults.keys():
        violin_kwargs.setdefault(key, violin_defaults[key])

    sns.violinplot(x=category, y=observable, data=df, ax=ax, **violin_kwargs)

    ylim = ax.get_ylim()

    # prequerry the data frames via category
    sub_dfs = dict()
    max_points = 0
    for idx, cat in enumerate(categories):
        if type(cat) == str:
            df_for_cat = df.query(f"`{category}` == '{cat}'")
        else:
            df_for_cat = df.query(f"`{category}` == {cat}")
        sub_dfs[cat] = df_for_cat

        # for the swarm plot, fetch max height so we could tweak number of points and size
        hist, bins = np.histogram(df_for_cat[observable], _unit_bins(ylim[0], ylim[1]))
        max_points = np.max([max_points, np.max(hist)])

    for idx, cat in enumerate(categories):
        ax.collections[idx].set_color(light_palette[cat])
        ax.collections[idx].set_edgecolor(palette[cat])
        ax.collections[idx].set_linewidth(1.0)

        # custom error estimates
        df_for_cat = sub_dfs[cat]
        # log.debug("bootstrapping")
        try:
            raise KeyError
            # we ended up not using nested bootstrapping
            mid, error, percentiles = pd_nested_bootstrap(
                df_for_cat,
                grouping_col="Trial",
                obs=observable,
                num_boot=500,
                func=np.nanmedian,
                resample_group_col=True,
                percentiles=[2.5, 50, 97.5],
            )
        except:
            # log.warning("Nested bootstrap failed")
            # this may also happen when category variable is not defined.
            mid, std, percentiles = pd_bootstrap(
                df_for_cat,
                obs=observable,
                num_boot=500,
                func=np.nanmedian,
                percentiles=[2.5, 50, 97.5],
            )

        # log.debug(f"{cat}: median {mid:.3g}, std {std:.3g}")

        p_str = f"| {cat:>9} "
        p_str += f"| {percentiles[0]:15.4f} "  # 2.5%
        p_str += f"| {percentiles[1]:14.4f} "  # 50%
        p_str += f"| {percentiles[2]:16.4f} |"  # 97.5%

        # print(p_str)

        _draw_error_stick(
            ax,
            center=idx,
            mid=percentiles[1],
            errors=[percentiles[0], percentiles[2]],
            orientation="v",
            color=palette[cat],
            zorder=2,
        )

    # swarms
    swarm_defaults = dict(
        size=1.4,
        # palette=light_palette,
        palette=palette,
        order=categories,
        zorder=1,
        edgecolor=(1.0, 1.0, 1.0, 1.0),
        linewidth=0.2,
    )

    for key in swarm_defaults.keys():
        swarm_kwargs.setdefault(key, swarm_defaults[key])

    # so, for the swarms, we may want around the same number of points per swarm
    # then, apply subsampling for each category!
    if same_points_per_swarm:
        merged_df = []
        for idx, cat in enumerate(categories):
            if type(cat) == str:
                sub_df = df.query(f"`{category}` == '{cat}'")
            else:
                sub_df = df.query(f"`{category}` == {cat}")
            if not replace:
                num_samples = np.min([num_swarm_points, len(sub_df)])
            else:
                num_samples = num_swarm_points
            merged_df.append(
                sub_df.sample(
                    n=num_samples,
                    replace=replace,
                    # ignore_index=True,
                )
            )
        merged_df = pd.concat(merged_df, ignore_index=True)


    else:
        if not replace:
            num_samples = np.min([num_swarm_points, len(df)])
        else:
            num_samples = num_swarm_points
        merged_df = df.sample(n=num_samples, replace=replace)  # , ignore_index=True)

    print(f"plotting {len(merged_df)} points for cat {category}")

    sns.swarmplot(
        x=category,
        y=observable,
        data=merged_df,
        ax=ax,
        **swarm_kwargs,
    )

    # move the swarms slightly to the right and throw away left half
    for idx, cat in enumerate(categories):
        c = ax.collections[-len(categories) + idx]
        offsets = c.get_offsets()
        odx = np.where(offsets[:, 0] > idx)
        offsets = offsets[odx]
        offsets[:, 0] += 0.05
        c.set_offsets(offsets)

    # for idx in range(len(ax.collections)):
    #     try:
    #         offsets = ax.collections[idx].get_offsets()
    #         odx = np.where(offsets[:, 0] > idx)
    #         offsets = offsets[odx]
    #         offsets[:, 0] += 0.1
    #         ax.collections[idx].set_offsets(offsets)
    #     except:
    #         pass

    ax.get_legend().set_visible(False)
    # log.info(f'|{"":-^65}|')

    return ax


def myround(x):
    if x < 0.1:
        exponent = 0
        while x < 1:
            exponent += 1
            x = x * 10
        x = np.ceil(x)
        x = x * 10 ** (-exponent)
    else:
        x = np.ceil(x * 100)
        x = x / 100.0
    # if (x < 0.0001 and x> 0.00001):
    #     x = 0.0001
    return x

def get_data_for_stimulus_conditions(data, stimuli, measure):
    stim1_data = []
    stim2_data = []
    diff_data = []
    valid_units = []
    for unit in np.unique(data.unit.values):
        unit_selection = data["unit"]==unit
        if len(data[unit_selection]["stimulus"]) == 2:
            stim1_selection = data["unit"]==unit
            stim1_selection &= data["stimulus"] == stimuli[0]
            stim1_data_unit = data[stim1_selection][measure].values[0]
            stim1_data += [stim1_data_unit]
            stim2_selection = data["unit"]==unit
            stim2_selection &= data["stimulus"] == stimuli[1]
            stim2_data_unit = data[stim2_selection][measure].values[0]
            stim2_data += [stim2_data_unit]
            diff_data += [stim2_data_unit - stim1_data_unit]
            valid_units += [unit]
    return stim1_data, stim2_data, diff_data, valid_units

def load_spike_data_hdf5(filepath, session_id, stimulus, stimulus_block, unit_index):
    filename = f"{filepath}/session_{session_id}/session_{session_id}_spike_data.h5"
    f = h5py.File(os.path.expanduser(filename), "r", libver="latest")
    spike_data = f[f"session_{session_id}_stimulus_{stimulus}_stimulus_block_{stimulus_block}_spiketimes"]
    spike_times_unit = spike_data[unit_index]
    spike_times_unit = spike_times_unit[np.isfinite(spike_times_unit)].astype(float)
    f.close()
    metadata = pd.read_hdf(filename, f"session_{session_id}_stimulus_{stimulus}_stimulus_block_{stimulus_block}_metadata")
    metadata_unit = metadata.loc[unit_index]
    return spike_times_unit, metadata_unit
