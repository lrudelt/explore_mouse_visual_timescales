import os
import shutil
from sys import argv, path, modules

import numpy as np
import pandas as pd

from scipy.stats import pearsonr
import matplotlib
from matplotlib import rc
import pylab as plt
import pickle
import seaborn as sns
path.insert(1, "../../allen_src/")
import analysis_utils as utl

# TODO: Change data to the statistics in ../data, and apply filter before getting valid units in terms of metrics_data

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

"""CUSTOM FUNCTIONS"""

debug_units_of_interest = None
debug_metrics_data = None
def get_valid_units(analysis_data, metrics_data, area, stimulus, stimulus_blocks, metric):
    global debug_units_of_interest
    global debug_metrics_data
    query = (((analysis_data["stimulus"] == stimulus) & (analysis_data["ecephys_structure_acronym"] == area)) & (analysis_data["stimulus_blocks"] == stimulus_blocks))
    units_of_interest = analysis_data[query].index
    this_metrics_data = metrics_data[metric].reindex(units_of_interest)
    valid_units = this_metrics_data[np.isfinite(this_metrics_data)].index
    return valid_units, query

def myround(x):
    if x < 0.1:
        exponent = 0
        while x<1:
            exponent+=1
            x = x*10
        x = np.ceil(x)
        x = x*10**(-exponent)
    else:
        x = np.ceil(x * 100)
        x = x/100.
    if (x < 0.0001 and x> 0.00001):
        x = 0.0001
    return x

"""PLOT PARAMETERS"""
# metric = argv[1]
measure = argv[1]
experiment = 'brain_observatory'
if measure == 'tau_C':
    measure = 'tau_two_timescales'
# experiment = argv[1]
# experiment = "brain_observatory"
# # experiment = "functional_connectivity"
# metric = 'g_dsi_dg'
# if len(argv) > 3:
#     stimulus_name = argv[3]
#     if len(argv) > 4:
#         stimulus_blocks = argv[4]
#     else:
#         stimulus_blocks = "3.0,6.0"

"""LOAD DATA"""
# analysis_dir = '/data.nst/lucas/history_dependence/signatures_of_temporal_processing_paper_code'
stats_dir = '../data/stats'

metric_name_dict = {"pref_speed_dm": "preferred speed dot motion", "mod_idx_dg": "modulation index", "p_value_rf": "p value rf", "fano_dg": "fano factor", "area_rf" : "receptive field area", "snr": "SNR", "firing_rate_dg" : "firing rate (gratings)", "firing_rate_ns": "firing rate (scenes)", "g_dsi_dg": "direction selectivity", "g_osi_dg": "orientation selectivity (drifting)", "g_osi_sg" : "orientation selectivity (static)", "pref_sf_sg" : "preferred spatial frequency" , "pref_tf_dg" : "preferred temporal frequency", "image_selectivity_ns": "image selectivity", "c50_dg": "c50 contrast tuning"}

if experiment == "functional_connectivity":
    metrics_of_interest = ["pref_speed_dm", "mod_idx_dg", "p_value_rf", "fano_dg", "area_rf", "snr", "firing_rate_dg", "c50_dg"]
    stimulus_dict = {'natural_scenes': 'natural_movie_one_more_repeats', 'spontaneous_activity':'spontaneous'}
    stimulus = stimulus_dict[stimulus_name]

if experiment == 'brain_observatory':
    metrics_of_interest = ["g_dsi_dg", "g_osi_dg", "g_osi_sg", "pref_sf_sg" , "pref_tf_dg", "image_selectivity_ns", "mod_idx_dg", "firing_rate_ns", "firing_rate_dg", "p_value_rf", "fano_dg", "area_rf", "snr"]
    metrics_of_interest = ["g_dsi_dg", "image_selectivity_ns", "mod_idx_dg"]
    stimulus_dict = {'natural_scenes': 'natural_movie_three'}
    stimulus_name  = 'natural_scenes'
    stimulus = stimulus_dict[stimulus_name]
    stimulus_blocks = '3.0,6.0'

# assert metric in metrics_of_interest
# assert metric2 in metrics_of_interest
for metric in metrics_of_interest:
    metric_name = metric_name_dict[metric]
    # metric2_name = metric_name_dict[metric2]

    area_dict = {'VISp': 'V1', 'VISl' : 'LM', 'VISal': 'AL',  'VISpm': 'PM', 'VISam' : 'AM', 'VISrl': 'RL', 'LGd': 'LGN', 'LP': 'LP'}
    areas = list(area_dict.keys())
    areas = ['VISp','VISl','VISrl','VISal','VISpm','VISam','LGd','LP']
    areas_map = utl.get_structures_map()
    areas_cortex = [area for area in areas
                         if areas_map[area]['parent_structure'] == 'cortex']

    analysis =  'allen_bo'
    csv_file_name = "{}/{}_statistics_T0_30.csv".format(stats_dir, analysis)
    # mre_stats_file_name = "{}/{}_mre_statistics_T0_30_Tmax_2500.csv".format(stats_dir, analysis)
    mre_stats_file_name = "{}/{}_mre_statistics_T0_30_Tmax_10000_two_timescales.csv".format(stats_dir, analysis)
    metrics_data = pd.read_csv('%s/allen_metrics_%s.csv'%(stats_dir, experiment))
    analysis_data = utl.get_analysis_data(csv_file_name, analysis, mre_stats_file_name=mre_stats_file_name)
    analysis_data = analysis_data.set_index("unit")
    metrics_data = metrics_data.set_index("unit")

    measure_selected_data = np.zeros(0)
    metric_selected_data = np.zeros(0)
    for area in areas_cortex:
        valid_units, query = get_valid_units(analysis_data, metrics_data, area, stimulus, stimulus_blocks, metric)
        valid_data = analysis_data.loc[valid_units]
        valid_metrics_data = metrics_data.loc[valid_units]
        selection = utl.get_data_filter(valid_data, measure)
        measure_selected_data = np.append(measure_selected_data, valid_data[measure][selection])
        metric_selected_data = np.append(metric_selected_data, valid_metrics_data[metric][selection])
        # measure_selected_data = valid_data[measure][selection]
    N_units = len(measure_selected_data)

    """PLOT DATA"""
    plot_settings = utl.get_default_plot_settings()
    plot_settings['panel_width'] = 0.6 * plot_settings['textwidth']
    plot_settings['panel_height'] = 1.8
    plot_settings['rcparams']['figure.figsize'] =  [plot_settings['panel_width'], plot_settings['panel_height']]
    plot_settings['panel_size'] =  [plot_settings['panel_width'], plot_settings['panel_height']]
    plt.rcParams.update(plot_settings['rcparams'])
    color = "#223954"

    fig, ax = plt.subplots(figsize=plot_settings["panel_size"])
    fig.subplots_adjust(right=0.98, left=0.1, bottom=0.1)

    x_min = np.amin(metric_selected_data)
    x_max = np.amax(metric_selected_data)
    if metric_name == "direction selectivity":
        x_min = 0
        x_max = 1
    x_range = x_max-x_min

    x_text = 0.95
    if measure == "tau_two_timescales":
        y_text = 0.8
        y_max = 1501/1000
    elif measure == "tau_R":
        y_text = 0.8
        y_max = 201/1000
    elif measure == "R_tot":
        y_text = 0.2
        y_max = 0.3

    # 1% below zero to avoid clipping of grid
    y_min = -y_max/100

    y_range = y_max-y_min

    ax.set_ylim((y_min, y_max))
    utl.format_ax(ax, measure, 'y', set_ticks=False, tight_layout=False)
    utl.make_plot_pretty(ax)
    # Get x range for metric

    ax.spines['top'].set_bounds(0, 0)
    ax.spines['right'].set_bounds(0, 0)
    ax.set_xlabel(r"%s"%metric_name)
    ax.set_xlim((x_min, x_max))

    # Use formatting of y-axis as for other plots
    r , p_val = pearsonr(measure_selected_data, metric_selected_data)
    m, b = np.polyfit(metric_selected_data, measure_selected_data, 1)
    x = np.linspace(x_min, x_max, 10)
    ax.plot(x, m*x + b, color = ".2")
    #
    print("r = %s, p = %s"%(r, p_val))
    if p_val >= 1e-8:
        p_val_text = "$p$ = {}".format(utl.format_base_ten(p_val))
    else:
        p_val_text = "$p < 10^{-8}$"


    ax.text(
        x_text,
        y_text,
        r"$r = %s$"%(f'{r:.2f}') + "\n" + p_val_text ,
        ha='right',
        va='center',
        transform=ax.transAxes,
    )

    ax.scatter(metric_selected_data, measure_selected_data, s=1.5, lw=0, alpha = 0.5, color = color)

    fig.tight_layout(pad=1.0, w_pad=1., h_pad=1.0)
    # fig.suptitle(r'$\beta_I = {}$, $\Delta I_0$ = {}$'.format(settings["beta_I"], settings["white_input_component"]), fontsize=16)


    # plt.savefig('../../figs/%s_vs_%s_%s_natural_scenes.png'%(metric, metric2, experiment),
                    # format="png", dpi = 600, bbox_inches='tight')
    # fig.text(.25,1,r"\textbf{Natural scenes}", usetex = True)
    # fig.text(.7,1,r"\textbf{Spontaneous activity}", usetex = True)
    utl.save_plot(plot_settings, f"{__file__[:-3]}"+ "_" + metric, measure=argv[1])
    # plt.show()
    # plt.close()
