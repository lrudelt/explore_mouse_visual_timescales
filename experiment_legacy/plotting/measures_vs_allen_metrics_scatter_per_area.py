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

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

"""CUSTOM FUNCTIONS"""

def get_valid_units(analysis_data, metrics_data, area, stimulus, stimulus_blocks, metric):
    query = (((analysis_data["stimulus"] == stimulus) & (analysis_data["ecephys_structure_acronym"] == area)) & (analysis_data["stimulus_blocks"] == stimulus_blocks))
    units_of_interest = analysis_data[query].index
    valid_units = units_of_interest[np.where(np.isnan(metrics_data[metric][units_of_interest]) == False)]
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
    # if (x < 0.0001 and x> 0.00001):
    #     x = 0.0001
    return x

def get_range_metric(analysis_data, metrics_data, areas, area_dict, stimulus, stimulus_blocks, metric, experiment):
    for i, area in enumerate(areas):
        valid_units, query = get_valid_units(analysis_data, metrics_data, area, stimulus, stimulus_blocks, metric)
        metric_vals = metrics_data[metric][valid_units]
        if i == 0:
            x_min = np.amin(metric_vals)
            x_max = np.amax(metric_vals)
        else:
            x_min = np.amin(np.append(metric_vals,x_min))
            x_max = np.amax(np.append(metric_vals,x_max))
    return x_min, x_max

"""Plot Parameters"""
plot_settings = utl.get_default_plot_settings()
plt.rcParams.update(plot_settings['rcparams'])

color_palette = sns.color_palette().as_hex()
hex_area_colors = [color_palette[4],
                         color_palette[0],
                         color_palette[9],
                         color_palette[8],
                         color_palette[1],
                         color_palette[3],
                         color_palette[6],
                         color_palette[2]]

"""ANALYSIS PARAMETERS"""
experiment = 'brain_observatory'
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

# assert metric in metrics_of_interest
# assert metric2 in metrics_of_interest
# metric = 'g_dsi_dg'
for metric in metrics_of_interest:
    metric_name = metric_name_dict[metric]
    # metric2_name = metric_name_dict[metric2]

    fig, axes = plt.subplots(8, 3, figsize=(1.5 * plot_settings['textwidth'], 15))
    fig.subplots_adjust(left=0.01, right=0.9, top=0.95, bottom=0.1, wspace = 0.5, hspace = 0.48)
    # Get x range for metric
    x_min, x_max = get_range_metric(analysis_data, metrics_data, areas, area_dict, stimulus, stimulus_blocks, metric, experiment)
    x_range = x_max-x_min

    i = 0
    for i, rows in enumerate(axes):
        # this iterates over brain areas
        area = areas[i]
        area_name = area_dict[area]
        color = hex_area_colors[i]

        ax0 = rows[0]
        ax1 = rows[1]
        ax2 = rows[2]

        # Set subtitle
        if i == 0:
            ax0.set_title(r"intrinsic timescale", pad = 15)#, usetex = True, fontsize = 16)
            ax1.set_title(r"information timescale", pad = 15)#, usetex = True, fontsize = 16)
            ax2.set_title(r"predictability", pad = 15)#, usetex = True, fontsize = 16)

        x_text = x_min - .8* x_range
        # Set x-axis
        for ax, measure in zip([ax0,ax1,ax2],['tau_two_timescales', 'tau_R', 'R_tot']):
            valid_units, query = get_valid_units(analysis_data, metrics_data, area, stimulus, stimulus_blocks, metric)
            valid_measure_data = analysis_data.loc[valid_units]
            valid_metric_data = metrics_data.loc[valid_units]
            selection = utl.get_data_filter(valid_measure_data, measure)
            measure_selected_data = valid_measure_data[measure][selection]
            metric_selected_data = valid_metric_data[metric][selection]
            N_units = len(measure_selected_data)

            if measure == "tau_two_timescales":
                # Indication of area and sample size
                ax.text(x_text,0.75, area_name, weight='bold')#, usetex = True )
                ax.text(x_text,0.5, r"$n = %d$"%N_units)#, usetex = True)
                y_min = 0.0
                y_max = 1.5

            elif measure == "tau_R":
                y_min = 0.0
                y_max = 0.2

            elif measure == "R_tot":
                y_min = 0
                y_max = 0.3

            y_range = y_max-y_min
            y_text = y_min + 0.85 * y_range
            x_text = x_min + 0.15* x_range
            ax.set_ylim((y_min, y_max))
            utl.format_ax(ax, measure, 'y', set_ticks=False, tight_layout=True)
            if i ==7:
                ax.set_xlabel(r"%s"%metric_name)
            ax.set_xlim((x_min, x_max))
            utl.make_plot_pretty(ax)

            r , p_val = pearsonr(measure_selected_data, metric_selected_data)
            m, b = np.polyfit(metric_selected_data, measure_selected_data, 1)
            x = np.linspace(x_min,x_max,10)
            ax.plot(x, m*x + b, color = ".2")
            if p_val < 10**(-5):
                ax.text(x_text, y_text, r"$r_P \approx %s$; $P_P < 10^{-5}$"%(f'{r:.2f}'))
            else:
                if (myround(p_val) <0.001 and myround(p_val) > 0.0001):
                    ax.text(x_text, y_text, r"$r_P \approx %s$; $P_P = %s$"%(f'{r:.2f}', f'{myround(p_val):.4f}'))
                else:
                    ax.text(x_text, y_text, r"$r_P \approx %s$; $P_P = %s$"%(f'{r:.2f}', f'{myround(p_val)}'))
            ax.scatter(metric_selected_data, measure_selected_data, s=.6, alpha = 1.0, color = color)

    # fig.tight_layout(pad=1.0, w_pad=1., h_pad=1.0)

    utl.save_plot(plot_settings, f"{__file__[:-3]}"+ "_" + metric)

# OUTDATED
#
# fig, axes = plt.subplots(8, 2, figsize=(9., 18.))
# # Get x range for metric
# x_min, x_max = get_range_metric(analysis_data, metrics_data, areas, area_dict, stimulus, stimulus_blocks, metric, experiment)
# x_range = x_max-x_min
# # y_min, y_max = get_range_metric(analysis_data, metrics_data, areas, area_dict, stimulus, stimulus_blocks, metric2, experiment)
# # y_range = y_max-y_min
# i = 0
# for i, rows in enumerate(axes):
#     # this iterates over brain areas
#     area = areas[i]
#     area_name = area_dict[area]
#
#     ax0 = rows[0]
#     ax1 = rows[1]
#     # Set x-axis
#     for ax in [ax0,ax1]:
#         ax.spines['top'].set_bounds(0, 0)
#         ax.spines['right'].set_bounds(0, 0)
#         ax.set_xlabel(r"%s"%metric_name)
#         ax.set_xlim((x_min, x_max))
#         # ax.spines['bottom'].set_bounds(-.5, 10)
#     # ax.set_xscale('log')
#     if experiment == "brain_observatory":
#         valid_units, query = get_valid_units(analysis_data, metrics_data, area, stimulus, stimulus_blocks, metric)
#     if experiment == "functional_connectivity":
#         valid_units, query = get_valid_units(analysis_data, metrics_data, area, stimulus, stimulus_blocks, metric)
#     valid_data = analysis_data.loc[valid_units]
#     valid_metrics_data = metrics_data.loc[valid_units]
#     selection = utl.get_data_filter(valid_data, measure)
#     measure_selected_data = valid_data[measure][selection]
#     metrics_selected_data = valid_metrics_data[metric][selection]
#     # metric2_vals = metrics_data[metric2][valid_units]
#     N_units = len(measure_selected_data)
#
#     # Indication of area and sample size
#     x_text = x_min - 0.7* x_range
#     rows[0].text(x_text,20, r"\textbf{%s}"%area_name, usetex = True )
#     rows[0].text(x_text,12, r"$n = %d$"%N_units, usetex = True )
#
#     # Set subtitle
#     if i == 0:
#         ax0.set_title(r"\textbf{predictable information}", usetex = True, fontsize = 16)
#         ax1.set_title(r"\textbf{information timescale}", usetex = True, fontsize = 16)
#
#     # R_tot
#     ax0.set_ylabel(r"$R_{\mathrm{tot}}$ (\%)")
#     ax0.set_ylim((0.0, 45))
#     ax0.set_yticks([0.0, 20, 40])
#     ax0.spines['left'].set_bounds(.0, 40)
#     r , p_val = pearsonr(R_tot, metric_vals)
#     m, b = np.polyfit(metric_vals, R_tot, 1)
#     x = np.linspace(x_min,x_max,10)
#     ax0.plot(x, m*x + b, color = ".2")
#     x_text = x_min + 0.05* x_range
#     if p_val < 10**(-5):
#         ax0.text(x_text,37,r"$r_p \approx %s$; $p < 10^{-5}$"%(f'{r:.2f}'), usetex = True)
#     else:
#         ax0.text(x_text,37,r"$r_p \approx %s$; $p = %s$"%(f'{r:.2f}', f'{myround(p_val)}'), usetex = True)
#     ax0.scatter(metric_vals, R_tot, s=.5, alpha = 0.5, color = main_blue)
#
#     # tau_R
#     ax1.set_ylabel(r"$\tau_R$ (ms)")
#     y_min = 0
#     y_max = 500
#     y_range = y_max-y_min
#     # ax1.set_ylim((y_min, y_max))
#     ax1.set_yticks([0.0, 200, 400])
#     ax1.spines['left'].set_bounds(.0, 500)
#     r , p_val = pearsonr(tau_R, metric_vals)
#     m, b = np.polyfit(metric_vals, tau_R, 1)
#     x_text = x_min + 0.05* x_range
#     y_text = y_min + 0.8 * y_range
#     if p_val < 10**(-5):
#         ax1.text(x_text,y_text,r"$r_p \approx %s$; $p < 10^{-5}$"%(f'{r:.2f}'), usetex = True)
#     else:
#         ax1.text(x_text,y_text,r"$r_p \approx %s$; $p = %s$"%(f'{r:.2f}', f'{myround(p_val)}'), usetex = True)
#     ax1.scatter(metric_vals, tau_R, s=.5, alpha = 0.5, color = main_blue)
#     # plot fitting line
#     # x = np.linspace(0,10,10)
#     ax1.plot(x, m*x + b, color = ".2")
#
# fig.tight_layout(pad=1.0, w_pad=1., h_pad=1.0)
# # fig.suptitle(r'$\beta_I = {}$, $\Delta I_0$ = {}$'.format(settings["beta_I"], settings["white_input_component"]), fontsize=16)
#
#
# # plt.savefig('../../figs/%s_vs_%s_%s_natural_scenes.png'%(metric, metric2, experiment),
#                 # format="png", dpi = 600, bbox_inches='tight')
# # fig.text(.25,1,r"\textbf{Natural scenes}", usetex = True)
# # fig.text(.7,1,r"\textbf{Spontaneous activity}", usetex = True)
#
# plt.show()
# plt.close()
